#!/usr/bin/env python3
"""Adversarial audit of the problem bank.

selftest.py proves every reference passes its own checks. This proves the other
direction: wrong answers must FAIL. Three attacks per task:

  identity      return the first argument unchanged
  cheat         call the oracle the statement forbids (now blocked by `banned`)
  mutant[...]   the reference solution with one classic bug injected

Any attack that comes back ACCEPTED is a hole in that task's checks.

    ./.venv/bin/python audit.py            # everything (a few minutes)
    ./.venv/bin/python audit.py gpt        # tasks whose id contains "gpt"
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import runner
import tasks

# (needle, replacement, label) — applied to the torch reference, first match only
MUTATIONS = [
    ("keepdim=True", "keepdim=False", "dropped keepdim"),
    ("unbiased=False", "unbiased=True", "wrong variance divisor"),
    ("math.sqrt(d_h)", "d_h", "scale by d, not sqrt(d)"),
    ("x - m", "x * 1.0", "no max-shift"),
    ("logits - m", "logits * 1.0", "no max-shift"),
    (".mean()", ".sum()", "sum for mean"),
    ("torch.finfo(scores.dtype).min", "float('-inf')", "-inf mask"),
    ("torch.finfo(logits.dtype).min", "float('-inf')", "-inf mask"),
    ("tril", "triu", "wrong triangle"),
    ("diagonal=S - L", "diagonal=0", "no decode offset"),
    ("logits[:, :-1]", "logits[:, 1:]", "shifted the wrong way"),
    ("descending=True", "descending=False", "ascending sort"),
    ("cum - probs", "cum", "nucleus off-by-one"),
    ("(1 - b1)", "(1.0)", "undamped first moment"),
    ("b2 ** t", "b2", "wrong bias correction"),
    ("2 * batch", "1 * batch", "forgot K and V"),
    ("2.0 * (n_ranks - 1)", "1.0 * (n_ranks - 1)", "reduce-scatter only"),
    ("* corr", "* 1.0", "no running rescale"),
    ("0.5 * logvar", "logvar", "sigma vs variance"),
    ("+ eps", "+ 0 * eps", "dropped eps"),
    (" / (n_layers", " / (1", None),  # noise; skipped via label None
    ("interval / 2.0", "interval * 1.0", "full interval redo"),
    ("6.0 *", "2.0 *", "2ND instead of 6ND"),
    ("6 *", "2 *", "2ND instead of 6ND"),
    ("- logr", "+ logr", "sign of log-ratio"),
    ("torch.minimum", "torch.maximum", "max instead of min"),
    ("1 - eps, 1 + eps", "0.0, 2.0", "wrong clip band"),
    ("G - 1", "G", "baseline includes self"),
    ("head.weight = self.tok.weight", "head.weight = nn.Parameter(self.tok.weight.clone())",
     "copied instead of tied"),
    ("x + self.attn", "self.attn", "dropped residual"),
    ("scatter_add_", "scatter_", "overwrite, not accumulate"),
    ("index_add_", "index_copy_", "overwrite, not accumulate"),
]


# Handwritten wrong solutions implementing the traps each task itself lists,
# where no mutation operator already covers them. Every one must be rejected.
TRAPS = {
    "cross-entropy": [("unstable -log(softmax)", """
def cross_entropy(logits, target):
    p = torch.softmax(logits, -1)
    return -torch.log(p.gather(1, target[:, None]).squeeze(1)).mean()
""")],
    "log-softmax": [("softmax then log", """
def log_softmax(x, dim=-1):
    m = x.max(dim, keepdim=True).values
    e = torch.exp(x - m)
    return torch.log(e / e.sum(dim, keepdim=True))
""")],
    "perplexity-bpb": [("inverted token/byte ratio", """
def report(loss, n_tokens, n_bytes):
    return {'ppl': math.exp(loss),
            'bpb': (loss / math.log(2)) * (n_bytes / n_tokens)}
""")],
    "training-flops": [("2ND as the total", """
def flops(n_params, n_tokens):
    total = 2 * n_params * n_tokens
    return {'forward': total / 3, 'backward': 2 * total / 3, 'total': total}
""")],
    "split-heads": [("reshape without transpose (interleaves heads)", """
def split_heads(x, n_heads):
    B, L, D = x.shape
    return x.reshape(B, n_heads, L, D // n_heads)
""")],
    "gqa": [("tiled, not grouped ([0,1,0,1] instead of [0,0,1,1])", """
def expand_kv(kv, n_q):
    return torch.cat([kv] * (n_q // kv.shape[1]), dim=1)
""")],
    "grad-clip": [("clips each tensor separately", """
def clip_grads(grads, max_norm):
    total = sum((g * g).sum() for g in grads) ** 0.5
    out = []
    for g in grads:
        n = g.norm()
        out.append(g * (max_norm / n) if float(n) > max_norm else g)
    return out, total
""")],
    "power-law-fit": [("fit in linear space", """
def fit_power_law(xs, ys):
    b, a0 = np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)
    return {'a': float(a0), 'b': float(b),
            'predict': lambda x: a0 + b * np.asarray(x)}
""")],
    "gmm-estep": [("ignores the mixing weights", """
def e_step(X, means, variances, weights):
    D = X.shape[1]
    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    logp = -0.5 * (D * torch.log(2 * math.pi * variances)[None, :]
                   + d2 / variances[None, :])
    return torch.softmax(logp, dim=-1)
""")],
    "temperature": [("returns one-hot probabilities at tau=0", """
def apply_temperature(logits, tau):
    if tau == 0:
        out = torch.zeros_like(logits)
        return out.scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
    return logits / tau
""")],
    "gpt-loss": [("no shift: trains the model to copy its input", """
def lm_loss(logits, idx):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))
""")],
    "sgd-momentum": [("EMA-damped momentum (the non-PyTorch convention)", """
def sgd_step(p, g, buf, lr=0.1, mu=0.9):
    buf = mu * buf + (1 - mu) * g
    return p - lr * buf, buf
""")],
    "linear-attention": [("no normaliser", """
def linear_attention(q, k, v):
    B, H, L, Dh = q.shape
    S = torch.zeros(B, H, Dh, Dh)
    outs = []
    for t in range(L):
        S = S + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        outs.append(torch.einsum('bhd,bhde->bhe', q[:, :, t], S))
    return torch.stack(outs, dim=2)
""")],
    "mla-absorb": [("absorbs the value projection into the query", """
def absorbed_attention(q, c, W_uk, W_uv):
    d_h = q.shape[-1]
    q_lat = torch.einsum('bhld,hcd->bhlc', q, W_uv)
    scores = torch.einsum('bhlc,bsc->bhls', q_lat, c) / math.sqrt(d_h)
    w = torch.softmax(scores, -1)
    return torch.einsum('bhls,bsc,hcd->bhld', w, c, W_uk)
""")],
    "moe-aux-loss": [("uses the soft probabilities twice", """
def aux_load_loss(probs, assign):
    T, E = probs.shape
    P = probs.mean(0)
    return E * (P * P).sum()
""")],
    "kahan-summation": [("the naive running sum", """
def kahan_sum(x):
    s = torch.zeros((), dtype=x.dtype)
    for xi in x:
        s = s + xi
    return s
""")],
    "gae": [("mask on the bootstrap but not the carried advantage", """
def gae(rewards, values, gamma=0.99, lam=0.95, dones=None):
    T = rewards.shape[0]
    if dones is None:
        dones = torch.zeros(T)
    adv = torch.zeros(T)
    last = 0.0
    for t in range(T - 1, -1, -1):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv
""")],
    "distill-loss": [("forgets the T^2 factor", """
def kd_loss(student, teacher, T=2.0):
    logp_s = F.log_softmax(student / T, -1)
    p_t = torch.softmax(teacher / T, -1)
    return F.kl_div(logp_s, p_t, reduction='batchmean')
""")],
    "beam-search": [("pure greedy: expands only the best beam", """
def beam_search(step_fn, k, steps, vocab):
    beams = torch.zeros(1, 0, dtype=torch.long)
    scores = torch.zeros(1)
    for _ in range(steps):
        logp = step_fn(beams[:1])
        tok = int(logp[0].argmax())
        beams = torch.cat([beams[:1], torch.tensor([[tok]])], dim=1)
        scores = scores[:1] + logp[0, tok]
    return beams.repeat(k, 1), scores.repeat(k)
""")],
    "pass-at-k": [("the biased with-replacement formula", """
def pass_at_k(n, c, k):
    return 1.0 - (1.0 - c / n) ** k
""")],
    "local-sgd": [("each worker converges alone; average once at the end", """
def local_sgd(p0, A, b, lr=0.02, H=5, rounds=200):
    K = A.shape[0]
    p = p0.clone().unsqueeze(0).repeat(K, 1)
    for _ in range(rounds * H):
        r = torch.einsum('kmd,kd->km', A, p) - b
        g = torch.einsum('kmd,km->kd', A, r)
        p = p - lr * g
    return p.mean(0)
""")],
    "gram-schmidt": [("classical Gram-Schmidt (coefficients against original A)", """
def gram_schmidt(A):
    Q = A.clone()
    m, n = Q.shape
    for j in range(n):
        v = A[:, j].clone()
        for i in range(j):
            v = v - (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / v.norm()
    return Q
""")],
    "gaussian-conditioning": [("inverts the free block instead of the observed one", """
def condition(mu, S, obs_idx, obs_val):
    D = mu.shape[0]
    b = list(obs_idx)
    a = [i for i in range(D) if i not in b]
    ia, ib = torch.tensor(a), torch.tensor(b)
    S_aa = S[ia][:, ia]
    S_ab = S[ia][:, ib]
    inv = torch.linalg.inv(S_aa)
    mean = mu[ia] + S_ab @ (obs_val - mu[ib]) if S_ab.shape[1] == len(b) else mu[ia]
    cov = S_aa - inv @ S_ab @ S_ab.T if S_aa.shape == inv.shape else S_aa
    return {'mean': mean, 'cov': cov}
""")],
    "importance-sampling": [("plain IS: breaks on an unnormalised target", """
def is_mean(f, target_logpdf, proposal_logpdf, xs):
    w = torch.exp(target_logpdf(xs) - proposal_logpdf(xs))
    return (w * f(xs)).mean()
""")],
    "kfold-split": [("drops the remainder", """
def kfold(n, k):
    size = n // k
    out = []
    for i in range(k):
        val = list(range(i * size, (i + 1) * size))
        train = [j for j in range(n) if j not in val]
        out.append((train, val))
    return out
""")],
    "naive-bayes": [("multiplies probabilities (underflows)", """
def gnb_predict(Xtr, ytr, Xte):
    C = int(ytr.max()) + 1
    scores = []
    for c in range(C):
        Xc = Xtr[ytr == c]
        prior = Xc.shape[0] / Xtr.shape[0]
        m = Xc.mean(0)
        v = Xc.var(0, unbiased=False) + 1e-9
        dens = torch.exp(-0.5 * (Xte - m) ** 2 / v) / torch.sqrt(2 * math.pi * v)
        scores.append(prior * dens.prod(-1))
    return torch.stack(scores, -1).argmax(-1)
""")],
    "inverted-dropout": [("no 1/(1-p) rescale", """
def dropout(x, p, training=True):
    if not training or p == 0:
        return x
    mask = (torch.rand_like(x) >= p).to(x.dtype)
    return x * mask
""")],
    "gmm-mstep": [("variance divided by N_k instead of N_k * D", """
def m_step(X, resp):
    N, D = X.shape
    Nk = resp.sum(0)
    weights = Nk / N
    means = resp.T @ X / Nk[:, None]
    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    variances = (resp * d2).sum(0) / Nk
    return {'weights': weights, 'means': means, 'variances': variances}
""")],
    "viterbi": [("per-step emission argmax (ignores transitions)", """
def viterbi(log_pi, log_A, log_B, obs):
    path = log_B[:, obs].argmax(0)
    s = float(log_pi[path[0]] + log_B[path[0], obs[0]])
    for t in range(1, len(obs)):
        s += float(log_A[path[t - 1], path[t]] + log_B[path[t], obs[t]])
    return path, s
""")],
    "kalman-1d": [("gain built from the pre-prediction variance", """
def kalman_1d(ys, a, c, q, r, mu0, p0):
    mu, P = mu0, p0
    means, variances = [], []
    for y in ys:
        K = P * c / (c * c * P + r)
        mu_pred = a * mu
        P_pred = a * a * P + q
        mu = mu_pred + K * (float(y) - c * mu_pred)
        P = (1 - K * c) * P_pred
        means.append(mu)
        variances.append(P)
    return {'means': torch.tensor(means), 'vars': torch.tensor(variances)}
""")],
    "metropolis-hastings": [("compares u to the log-ratio without the log", """
def mh_sample(logpdf, n, step=1.0, x0=0.0, burn=1000):
    x = torch.tensor(float(x0))
    lp = logpdf(x)
    out = []
    for i in range(burn + n):
        prop = x + step * torch.randn(())
        lp_new = logpdf(prop)
        if float(torch.rand(())) < float(lp_new - lp):
            x, lp = prop, lp_new
        if i >= burn:
            out.append(float(x))
    return torch.tensor(out)
""")],
    "gini-split": [("unweighted average of child impurities", """
def best_split(x, y):
    xs = torch.unique(x).sort().values
    best_t, best_g = None, float('inf')
    def gini(labels):
        if labels.numel() == 0:
            return 0.0
        p1 = float(labels.float().mean())
        return 1.0 - p1 ** 2 - (1 - p1) ** 2
    for i in range(len(xs) - 1):
        t = float((xs[i] + xs[i + 1]) / 2)
        g = (gini(y[x <= t]) + gini(y[x > t])) / 2
        if g < best_g:
            best_t, best_g = t, g
    return {'threshold': best_t, 'impurity': best_g}
""")],
    "knn": [("votes among the FARTHEST k", """
def knn_predict(Xtr, ytr, Xte, k=3):
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    idx = d2.topk(k, dim=-1, largest=True).indices
    return ytr[idx].mode(dim=-1).values
""")],
    "adaboost-round": [("no renormalisation", """
def adaboost_round(w, h, y):
    e = float(w[(h != y)].sum())
    alpha = 0.5 * math.log((1 - e) / e)
    return {'alpha': alpha, 'weights': w * torch.exp(-alpha * y.float() * h.float())}
""")],
    "roc-auc": [("per-example staircase, ties ignored", """
def auc(scores, labels):
    order = torch.argsort(scores, descending=True)
    l = labels[order].float()
    P = float(l.sum()); Nn = float((1 - l).sum())
    tp = fp = 0.0
    area = 0.0
    prev_fpr = 0.0; prev_tpr = 0.0
    for li in l:
        if float(li) == 1:
            tp += 1
        else:
            fp += 1
        area += (fp / Nn - prev_fpr) * prev_tpr
        prev_fpr, prev_tpr = fp / Nn, tp / P
    return area
""")],
    "welford": [("one-pass sum of squares", """
def online_stats(xs):
    n = 0; s = 0.0; s2 = 0.0
    for x in xs:
        x = float(x); n += 1; s += x; s2 += x * x
    mean = s / n
    return {'mean': mean, 'var': (s2 / n - mean * mean) * n / (n - 1)}
""")],
    "reservoir-sampling": [("always replaces a random slot", """
def reservoir(stream, k):
    res = [stream[i] for i in range(k)]
    for i in range(k, len(stream)):
        j = int(torch.randint(0, k, (1,)))
        res[j] = stream[i]
    return torch.stack([torch.as_tensor(r) for r in res])
""")],
    "grad-check": [("forward difference", """
def numerical_grad(f, x, eps=1e-5):
    g = torch.zeros_like(x)
    f0 = float(f(x))
    for i in range(x.numel()):
        xp = x.clone(); xp.view(-1)[i] += eps
        g.view(-1)[i] = (float(f(xp)) - f0) / eps
    return g
""")],
    "vae-elbo": [("KL averaged over latent dims", """
def vae_loss(x, x_recon, mu, logvar):
    recon = F.binary_cross_entropy(x_recon, x, reduction='none').sum(-1).mean()
    kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(-1)).mean()
    return {'recon': recon, 'kl': kl, 'loss': recon + kl}
""")],
    "conv2d": [("flips the kernel (true convolution)", """
def conv2d(x, w, b, stride=1, padding=0):
    return F.conv2d(x, torch.flip(w, dims=(-2, -1)), b, stride=stride, padding=padding)
""")],
    "batchnorm2d": [("running var without Bessel's correction", """
def batchnorm2d(x, gamma, beta, run_mean, run_var, momentum=0.1, eps=1e-5):
    dims = (0, 2, 3)
    mean = x.mean(dims)
    var_b = x.var(dims, unbiased=False)
    xhat = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var_b.view(1, -1, 1, 1) + eps)
    out = gamma.view(1, -1, 1, 1) * xhat + beta.view(1, -1, 1, 1)
    return {'out': out,
            'run_mean': (1 - momentum) * run_mean + momentum * mean,
            'run_var': (1 - momentum) * run_var + momentum * var_b}
""")],
    "lstm-cell": [("gate order i, g, f, o", """
def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):
    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
    i, g, f, o = gates.chunk(4, dim=1)
    c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    return torch.sigmoid(o) * torch.tanh(c_new), c_new
""")],
    "focal-loss": [("plain cross-entropy, gamma ignored", """
def focal_loss(logits, target, gamma=2.0):
    return F.cross_entropy(logits, target)
""")],
    "label-smoothing": [("smooths over C-1 wrong classes", """
def ls_cross_entropy(logits, target, eps=0.1):
    C = logits.shape[-1]
    logp = torch.log_softmax(logits, -1)
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    others = -(logp.sum(-1) - logp.gather(1, target[:, None]).squeeze(1)) / (C - 1)
    return ((1 - eps) * nll + eps * others).mean()
""")],
    "infonce": [("no L2 normalisation", """
def info_nce(z1, z2, temp=0.1):
    logits = z1 @ z2.T / temp
    return F.cross_entropy(logits, torch.arange(z1.shape[0]))
""")],
    "sinusoidal-pe": [("sin and cos swapped", """
def sinusoidal_pe(L, d):
    i = torch.arange(d // 2, dtype=torch.float32)
    w = 10000.0 ** (-2 * i / d)
    ang = torch.arange(L, dtype=torch.float32)[:, None] * w[None, :]
    pe = torch.zeros(L, d)
    pe[:, 0::2] = torch.cos(ang)
    pe[:, 1::2] = torch.sin(ang)
    return pe
""")],
    "grad-accumulation": [("uniform mean over unequal micro-batches", """
def accumulate_grads(loss_fn, params, batches):
    total = torch.zeros_like(params)
    for X, y in batches:
        (g,) = torch.autograd.grad(loss_fn(params, X, y), params)
        total = total + g / len(batches)
    return total
""")],
    "newton-schulz": [("no initial normalisation", """
def newton_schulz(G, steps=40):
    X = G.clone()
    for _ in range(steps):
        X = 1.5 * X - 0.5 * (X @ X.T @ X)
    return X
""")],
    "mcts-nim": [("no negamax sign flip", """
def uct_best_move(n, n_sims=3000, c=1.4):
    stats = {}
    def moves(s):
        return [m for m in (1, 2, 3) if m <= s]
    def rollout(s):
        sign = 1
        while s > 0:
            m = int(torch.randint(1, min(3, s) + 1, (1,)))
            s -= m
            sign = -sign
        return -sign
    def search(s):
        if s == 0:
            return -1
        if s not in stats:
            stats[s] = [0, 0.0]
            v = rollout(s)
        else:
            best, best_u = None, -1e18
            for m in moves(s):
                ch = s - m
                vis = stats.get(ch, [0, 0.0])[0]
                if vis == 0:
                    best = m
                    break
                q = stats[ch][1] / vis
                u = q + c * math.sqrt(math.log(stats[s][0]) / vis)
                if u > best_u:
                    best, best_u = m, u
            v = search(s - best)
        stats[s][0] += 1
        stats[s][1] += v
        return v
    for _ in range(n_sims):
        search(n)
    return max(moves(n), key=lambda m: stats.get(n - m, [0, 0.0])[0])
""")],
    "bpe-train": [("ties broken by lexicographically LARGEST pair", """
def bpe_merges(word_freqs, num_merges):
    words = {w: list(w) for w in word_freqs}
    merges = []
    for _ in range(num_merges):
        counts = {}
        for w, syms in words.items():
            f = word_freqs[w]
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                counts[p] = counts.get(p, 0) + f
        if not counts:
            break
        top = max(counts.values())
        best = max(p for p, cnt in counts.items() if cnt == top)
        merges.append(best)
        for w, syms in words.items():
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:
                    out.append(syms[i] + syms[i + 1]); i += 2
                else:
                    out.append(syms[i]); i += 1
            words[w] = out
    return merges
""")],
    "lora-forward": [("no alpha/r scaling", """
def lora_forward(x, W, A, B, alpha=16.0):
    return x @ W.T + (x @ A.T) @ B.T
""")],
    "diffusion-forward": [("signal and noise coefficients swapped", """
def forward_diffusion(x0, t, abar, eps):
    a = abar[t].unsqueeze(-1)
    return torch.sqrt(1 - a) * x0 + torch.sqrt(a) * eps
""")],
    "ddpm-step": [("samples with variance beta instead of btilde", """
def ddpm_step(x_t, eps_hat, t, betas, z):
    alphas = 1 - betas
    abar = torch.cumprod(alphas, dim=0)
    mu = (x_t - betas[t] / torch.sqrt(1 - abar[t]) * eps_hat) / torch.sqrt(alphas[t])
    if t == 0:
        return mu
    return mu + torch.sqrt(betas[t]) * z
""")],
    "ddpm-schedule": [("cumsum instead of cumprod", """
def make_schedule(T=50, beta_start=1e-4, beta_end=0.25):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    return {'betas': betas, 'alphas': alphas,
            'abar': torch.cumsum(alphas, dim=0)}
""")],
    "ddpm-q-sample": [("coefficients swapped", """
def q_sample(x0, t, abar, eps):
    a = abar[t].unsqueeze(-1)
    return torch.sqrt(1 - a) * x0 + torch.sqrt(a) * eps
""")],
    "ddpm-time-emb": [("interleaved instead of half-split", """
def time_embedding(t, dim=32):
    i = torch.arange(dim // 2, dtype=torch.float32)
    w = 10000.0 ** (-2 * i / dim)
    ang = t.float()[:, None] * w[None, :]
    out = torch.zeros(t.shape[0], dim)
    out[:, 0::2] = torch.sin(ang)
    out[:, 1::2] = torch.cos(ang)
    return out
""")],
    "ddpm-denoiser": [("ignores the time embedding", """
class Denoiser(nn.Module):
    def __init__(self, tdim=32, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x, temb):
        return self.net(x)
""")],
    "ddpm-loss": [("regresses on x0 instead of the noise", """
def ddpm_loss(model, x0, sched, t, eps):
    x_t = _q_sample(x0, t, sched['abar'], eps)
    return ((model(x_t, _time_emb(t)) - x0) ** 2).mean()
""")],
    "ddpm-train": [("no zero_grad: gradients accumulate", """
def train_ddpm(model, data_fn, sched, steps=2500, lr=2e-3, batch=256):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x0 = data_fn(batch)
        t = torch.randint(0, len(sched['betas']), (batch,))
        eps = torch.randn_like(x0)
        loss = _loss(model, x0, sched, t, eps)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
""")],
    "ddpm-sample": [("noise scaled by beta instead of btilde", """
def ddpm_sample(denoise, sched, n):
    betas, alphas, abar = sched['betas'], sched['alphas'], sched['abar']
    T = len(betas)
    x = torch.randn(n, 2)
    for t in range(T - 1, -1, -1):
        with torch.no_grad():
            eps_hat = denoise(x, t)
        mu = (x - betas[t] / torch.sqrt(1 - abar[t]) * eps_hat) / torch.sqrt(alphas[t])
        if t > 0:
            x = mu + torch.sqrt(betas[t]) * torch.randn_like(x)
        else:
            x = mu
    return x
"""), ("starts the chain from zeros", """
def ddpm_sample(denoise, sched, n):
    betas, alphas, abar = sched['betas'], sched['alphas'], sched['abar']
    T = len(betas)
    x = torch.zeros(n, 2)
    for t in range(T - 1, -1, -1):
        with torch.no_grad():
            eps_hat = denoise(x, t)
        mu = (x - betas[t] / torch.sqrt(1 - abar[t]) * eps_hat) / torch.sqrt(alphas[t])
        if t > 0:
            btilde = (1 - abar[t - 1]) / (1 - abar[t]) * betas[t]
            x = mu + torch.sqrt(btilde) * torch.randn_like(x)
        else:
            x = mu
    return x
""")],
    "ddpm-pipeline": [("samples from the untrained model", """
def two_mode_pipeline(train_steps=2500):
    sched = _schedule()
    model = _Denoiser()
    losses = _train(model, _two_gaussians, sched, train_steps)
    fresh = _Denoiser()
    def denoise(x, t):
        return fresh(x, _time_emb(torch.full((x.shape[0],), t)))
    return {'samples': _sample(denoise, sched, 2000), 'losses': losses}
""")],
    "ring-allreduce": [("gather to root and broadcast", """
def ring_allreduce(xs):
    P = len(xs)
    if P == 1:
        return {'results': [xs[0].clone()], 'sends_per_rank': [0]}
    C = xs[0].shape[0] // P
    total = xs[0].clone()
    sends = [0] * P
    for r in range(1, P):
        total = total + xs[r]
        sends[r] += P                      # rank r sends its full vector (P chunks)
    for r in range(1, P):
        sends[0] += P                      # root broadcasts the full vector
    return {'results': [total.clone() for _ in range(P)], 'sends_per_rank': sends}
""")],
    "tensor-parallel-mlp": [("all-reduce before the nonlinearity", """
def tp_mlp(x, W1, W2, P):
    # sums the per-rank pre-activations first, then applies GELU once —
    # but gelu(a + b) != gelu(a) + gelu(b), so this cannot match
    H = W1.shape[1]
    h = H // P
    pre = torch.zeros(x.shape[0], h)
    for r in range(P):
        pre = pre + x @ W1[:, r * h:(r + 1) * h]
    return F.gelu(pre) @ W2[:h, :]
""")],
    "fsdp-forward": [("gathers every layer up front", """
def fsdp_forward(x, shards):
    P = len(shards[0])
    Ws = [torch.cat(layer, dim=0) for layer in shards]
    peak = sum(W.numel() for W in Ws)
    h = x
    for li, W in enumerate(Ws):
        h = h @ W.T
        if li < len(shards) - 1:
            h = torch.relu(h)
    return {'out': h, 'peak_params': peak}
""")],
    "pipeline-schedule": [("sequential: no pipelining at all", """
def pipeline_schedule(p, m):
    timeline = [(s, j, j * p + s) for j in range(m) for s in range(p)]
    finish = m * p
    return {'timeline': timeline, 'finish': finish,
            'bubble': (p - 1) / (m + p - 1)}
""")],
    "ring-attention-combine": [("uniform average of shard outputs", """
def combine_attention(ms, ls, os):
    out = torch.zeros_like(os[0])
    for o in os:
        out = out + o / len(os)
    return out
""")],
    "kmeans-pp": [("uniform seeding", """
def kmeans_pp(X, k):
    idx = torch.randperm(X.shape[0])[:k]
    return X[idx]
""")],
    "masked-softmax": [("no post-multiply: masked entries keep tiny weight", """
def masked_softmax(x, mask):
    x = x.masked_fill(~mask, -1e9)
    return torch.softmax(x, -1)
""")],
}


def variants(t):
    yield "identity", f"def {t['entry']}(*args, **kwargs):\n    return args[0]\n"
    for label, code in TRAPS.get(t["id"], []):
        yield f"trap[{label}]", code
    sol = t["solution"]
    for old, new, label in MUTATIONS:
        if label and old in sol:
            m = sol.replace(old, new, 1)
            if m != sol:
                yield f"mutant[{label}]", m


# Mutants that are behaviourally equivalent to the reference, verified by hand:
# a causal row always contains its own position, so -inf and finfo.min cannot be
# distinguished there (same in generate, where top-k >= 1 keeps an entry), and
# dropping keepdim on a mean over axis 0 broadcasts identically in PCA.
EQUIVALENT = {
    ("causal-mask", "mutant[-inf mask]"),
    ("gpt-generate", "mutant[-inf mask]"),
    ("pca", "mutant[dropped keepdim]"),
    # mean(0) then .repeat(K, 1) on the 1-D result rebuilds the identical (K, D)
    # tensor, so keepdim=False changes nothing here
    ("local-sgd", "mutant[dropped keepdim]"),
    # the task returns class labels; an N vs N-1 variance divisor moves the
    # densities ~1% at 100+ samples per class and never flips a prediction
    ("naive-bayes", "mutant[wrong variance divisor]"),
}


def attack(job):
    t, name, code = job
    try:
        r = runner.run(t["id"], code, "torch")
    except Exception as exc:
        return (t["id"], name, f"audit error: {exc}")
    if r.get("error"):
        return (t["id"], name, None)          # rejected before running: fine
    if r.get("accepted"):
        return (t["id"], name, "ACCEPTED")
    return (t["id"], name, None)


def coverage():
    """Which tasks does the audit actually exercise? Pure enumeration, no runs."""
    thin = []
    for t in tasks.TASKS:
        n = sum(1 for _ in variants(t))
        kinds = [name.split("[")[0] for name, _ in variants(t)]
        if "trap" not in kinds and n <= 2:
            thin.append((t["id"], n))
    print(f"  {len(tasks.TASKS)} tasks; {len(thin)} defended only by "
          f"identity/1 generic mutant and no written trap:")
    for tid, n in sorted(thin):
        print(f"    {tid:24s} {n} attack(s)")
    print("  A new task should ship with a trap implementation (see AUTHORING.md).")
    return 0


def main():
    if "--coverage" in sys.argv:
        return coverage()
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = [(t, name, code)
            for t in tasks.TASKS if only in t["id"]
            for name, code in variants(t)]
    print(f"  {len(jobs)} attacks across "
          f"{len({t['id'] for t, _, _ in jobs})} tasks")
    holes = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, res in enumerate(pool.map(attack, jobs), 1):
            tid, name, verdict = res
            if verdict and (tid, name) in EQUIVALENT:
                print(f"  ok    {tid:22s} {name}: equivalent mutant (see EQUIVALENT)")
            elif verdict:
                holes.append(res)
                print(f"  HOLE  {tid:22s} {name}: {verdict}")
            if i % 25 == 0:
                print(f"  ... {i}/{len(jobs)}")
    print(f"\n  {len(jobs) - len(holes)}/{len(jobs)} attacks correctly rejected")
    if holes:
        print(f"  {len(holes)} hole(s) — each is a wrong answer the checks accept")
    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main())
