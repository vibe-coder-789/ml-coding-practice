"""Additions to the llm-math volume — sections the first pass left uncovered.

Chapter strings match the existing modules exactly, so these tasks merge into
the existing sidebar groups.
"""
from .schema import task

CH2 = "2 · The transformer, term by term"
CH4 = "4 · Optimisation"
CH6 = "6 · Alignment and reinforcement learning"
CH7 = "7 · Inference"
CH8 = "8 · Infrastructure: parallelism, reliability, serving"

TASKS = [

task(
    id="linear-attention",
    title="Causal linear attention as a recurrence",
    chapter=CH2,
    section="2.5 Linear attention: one recurrence, many transition matrices",
    level=3,
    entry="linear_attention",
    statement=(
        "Implement causal linear attention as a recurrence over time: maintain a "
        "state S_t = sum of k_s v_s^T and a normaliser z_t = sum of k_s over "
        "s <= t, and emit out_t = (q_t S_t) / (q_t . z_t). The inputs are already "
        "positive (a feature map has been applied), so no softmax appears "
        "anywhere. This is the O(L d^2) form that makes linear attention an RNN "
        "at decode time — it must agree exactly with the O(L^2) quadratic form."
    ),
    shapes="q, k, v (B, H, L, Dh), q and k positive  ->  (B, H, L, Dh)",
    stub=("def linear_attention(q, k, v):\n"
          "    # recurrence over t: S += k_t v_t^T, z += k_t\n"
          "    # out_t = (q_t @ S) / (q_t . z)\n    pass\n"),
    hints=[
        "The state S has shape (B, H, Dh, Dh); the normaliser z is (B, H, Dh).",
        "At each step: S += k_t[..., :, None] * v_t[..., None, :] and z += k_t.",
        "out_t = einsum('bhd,bhde->bhe', q_t, S) divided by (q_t * z).sum(-1, "
        "keepdim=True). Stack the outputs along the time axis.",
    ],
    solution=(
        "def linear_attention(q, k, v):\n"
        "    B, H, L, Dh = q.shape\n"
        "    S = torch.zeros(B, H, Dh, Dh)\n"
        "    z = torch.zeros(B, H, Dh)\n"
        "    outs = []\n"
        "    for t in range(L):\n"
        "        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]\n"
        "        S = S + kt.unsqueeze(-1) * vt.unsqueeze(-2)\n"
        "        z = z + kt\n"
        "        num = torch.einsum('bhd,bhde->bhe', qt, S)\n"
        "        den = (qt * z).sum(-1, keepdim=True)\n"
        "        outs.append(num / den)\n"
        "    return torch.stack(outs, dim=2)\n"
    ),
    solution_np=(
        "def linear_attention(q, k, v):\n"
        "    B, H, L, Dh = q.shape\n"
        "    S = np.zeros((B, H, Dh, Dh))\n"
        "    z = np.zeros((B, H, Dh))\n"
        "    outs = []\n"
        "    for t in range(L):\n"
        "        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]\n"
        "        S = S + kt[..., :, None] * vt[..., None, :]\n"
        "        z = z + kt\n"
        "        num = np.einsum('bhd,bhde->bhe', qt, S)\n"
        "        den = (qt * z).sum(-1, keepdims=True)\n"
        "        outs.append(num / den)\n"
        "    return np.stack(outs, axis=2)\n"
    ),
    traps=[
        "Dropping the normaliser z, so the outputs are unnormalised sums rather "
        "than convex combinations of the values.",
        "Building the state as v k^T instead of k v^T, which transposes the "
        "output projection.",
        "Applying a softmax anywhere — the whole point is that none appears.",
    ],
    tests='''
def checks(fn, check):
    B, H, L, Dh = 2, 2, 6, 4
    q = F.elu(torch.randn(B, H, L, Dh)) + 1
    k = F.elu(torch.randn(B, H, L, Dh)) + 1
    v = torch.randn(B, H, L, Dh)

    def quadratic():
        # the O(L^2) form: causal-masked scores q.k, normalised per row
        scores = torch.einsum('bhld,bhsd->bhls', q, k)
        mask = torch.tril(torch.ones(L, L)).bool()
        scores = scores * mask
        w = scores / scores.sum(-1, keepdim=True)
        return torch.einsum('bhls,bhsd->bhld', w, v)
    check("matches the quadratic form exactly", lambda: close(fn(q, k, v), quadratic(), 1e-4))
    check("output shape", lambda: shape(fn(q, k, v)) == (B, H, L, Dh))
    def causal():
        v2 = v.clone(); v2[:, :, -1] += 10.
        return close(fn(q, k, v)[:, :, :-1], fn(q, k, v2)[:, :, :-1], 1e-5)
    check("a later value cannot change an earlier output", causal)
    check("weights form a convex combination (v = ones -> ones)",
          lambda: close(fn(q, k, torch.ones(B, H, L, Dh)), torch.ones(B, H, L, Dh), 1e-4))
    check("first position attends only to itself",
          lambda: close(fn(q, k, v)[:, :, 0], v[:, :, 0], 1e-4))
''',
),

task(
    id="mla-absorb",
    title="Latent attention with weight absorption (MLA)",
    chapter=CH2,
    section="2.3 The KV-cache problem: MHA → MQA → GQA → MLA",
    level=3,
    entry="absorbed_attention",
    statement=(
        "Attend against a compressed latent cache directly. The cache stores one "
        "latent c_s of width d_c per position, shared by every head; per-head "
        "keys and values are k_h = c W_uk[h] and v_h = c W_uv[h]. The MLA trick "
        "is that k never needs to be materialised: absorb W_uk into the query "
        "(q~ = q W_uk^T), so scores are q~ . c — computed in latent space. "
        "Implement attention that way; it must equal the decompressed version "
        "exactly."
    ),
    shapes=("q (B,H,L,Dh) · c (B,S,dc) · W_uk (H,dc,Dh) · W_uv (H,dc,Dh)"
            "  ->  (B,H,L,Dh), scaled by 1/sqrt(Dh)"),
    stub=("def absorbed_attention(q, c, W_uk, W_uv):\n"
          "    # scores against the latent cache; never materialise k\n    pass\n"),
    hints=[
        "q~ = einsum('bhld,hcd->bhlc', q, W_uk) — the up-projection folds into "
        "the query.",
        "scores = einsum('bhlc,bsc->bhls', q~, c) / sqrt(Dh) — same numbers as "
        "q . k, because (q W_uk^T) . c = q . (c W_uk).",
        "The output weights the latents, then up-projects once: "
        "einsum('bhls,bsc,hcd->bhld', softmax(scores), c, W_uv).",
    ],
    solution=(
        "def absorbed_attention(q, c, W_uk, W_uv):\n"
        "    d_h = q.shape[-1]\n"
        "    q_lat = torch.einsum('bhld,hcd->bhlc', q, W_uk)\n"
        "    scores = torch.einsum('bhlc,bsc->bhls', q_lat, c) / math.sqrt(d_h)\n"
        "    w = torch.softmax(scores, -1)\n"
        "    return torch.einsum('bhls,bsc,hcd->bhld', w, c, W_uv)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Absorbing W_uv into the query instead of W_uk — the shapes agree, the "
        "numbers do not.",
        "Scaling by sqrt(d_c) instead of sqrt(Dh); the score is still a "
        "Dh-dimensional inner product, just computed in latent coordinates.",
        "Materialising per-head k and v anyway, which throws away the cache "
        "saving that motivates MLA.",
    ],
    tests='''
def checks(fn, check):
    B, H, L, S, Dh, dc = 2, 3, 4, 6, 8, 5
    q = torch.randn(B, H, L, Dh)
    c = torch.randn(B, S, dc)
    W_uk = torch.randn(H, dc, Dh)
    W_uv = torch.randn(H, dc, Dh)

    def decompressed():
        k = torch.einsum('bsc,hcd->bhsd', c, W_uk)
        v = torch.einsum('bsc,hcd->bhsd', c, W_uv)
        scores = torch.einsum('bhld,bhsd->bhls', q, k) / math.sqrt(Dh)
        return torch.softmax(scores, -1) @ v
    check("equals attention over the decompressed k and v",
          lambda: close(fn(q, c, W_uk, W_uv), decompressed(), 1e-4))
    check("output shape", lambda: shape(fn(q, c, W_uk, W_uv)) == (B, H, L, Dh))
    check("handles S != L", lambda: shape(fn(q, c, W_uk, W_uv)) == (B, H, L, Dh))
    def uses_uk_for_scores():
        # swapping the two up-projections must change the answer
        return not close(fn(q, c, W_uk, W_uv), fn(q, c, W_uv, W_uk), 1e-3)
    check("the two up-projections are not interchangeable", uses_uk_for_scores)
    def heads_differ():
        out = fn(q, c, W_uk, W_uv)
        return not close(out[:, 0], out[:, 1], 1e-3)
    check("heads use their own projections", heads_differ)
''',
),

task(
    id="moe-aux-loss",
    title="MoE load-balancing auxiliary loss",
    chapter=CH2,
    section="2.9 Mixture of experts",
    level=2,
    entry="aux_load_loss",
    statement=(
        "Compute the Switch-style load-balancing loss: E times the sum over "
        "experts of f_e · P_e, where f_e is the fraction of tokens whose top-1 "
        "assignment is expert e, and P_e is the mean router probability given to "
        "e. Balanced routing gives exactly 1.0; a collapsed router scores near "
        "E. The two factors are different objects — one comes from the hard "
        "assignments, one from the soft probabilities — and using either twice "
        "gives a quantity that no longer penalises collapse correctly."
    ),
    shapes="probs (T, E) rows sum to 1 · assign (T,) int64  ->  scalar",
    stub=("def aux_load_loss(probs, assign):\n"
          "    # E * sum_e f_e * P_e\n    pass\n"),
    hints=[
        "f_e is a count of assignments divided by T — one-hot the assignment and "
        "take the mean over tokens.",
        "P_e is probs.mean(0).",
        "Multiply elementwise, sum over experts, scale by E.",
    ],
    solution=(
        "def aux_load_loss(probs, assign):\n"
        "    T, E = probs.shape\n"
        "    f = F.one_hot(assign, E).float().mean(0)\n"
        "    P = probs.mean(0)\n"
        "    return E * (f * P).sum()\n"
    ),
    solution_np=(
        "def aux_load_loss(probs, assign):\n"
        "    T, E = probs.shape\n"
        "    f = np.eye(E)[assign].mean(0)\n"
        "    P = probs.mean(0)\n"
        "    return E * (f * P).sum()\n"
    ),
    traps=[
        "Using P twice (E · sum P_e^2), which also equals 1 at uniform but "
        "mis-scores collapsed routing.",
        "Using f twice, which is not differentiable in the router at all — the "
        "soft factor is what carries the gradient.",
        "Dropping the factor E, so the loss shrinks as experts are added.",
    ],
    tests='''
def checks(fn, check):
    check("balanced routing scores exactly 1",
          lambda: close(fn(torch.full((4, 2), 0.5), torch.tensor([0, 1, 0, 1])),
                        torch.tensor(1.0), 1e-6))
    # hand case: probs [[.9,.1],[.8,.2]], both assigned to 0
    # f = [1, 0], P = [.85, .15]  ->  2 * .85 = 1.7
    check("hand-computed collapsed case",
          lambda: close(fn(torch.tensor([[.9, .1], [.8, .2]]), torch.tensor([0, 0])),
                        torch.tensor(1.7), 1e-6))
    check("same probs, balanced assignment scores 1.0",
          lambda: close(fn(torch.tensor([[.9, .1], [.8, .2]]), torch.tensor([0, 1])),
                        torch.tensor(1.0), 1e-6))
    check("collapse scores higher than balance",
          lambda: float(fn(torch.tensor([[.9, .1]] * 6), torch.zeros(6, dtype=torch.long))) >
                  float(fn(torch.full((6, 2), 0.5),
                           torch.tensor([0, 1] * 3))))
    check("scalar output", lambda: shape(fn(torch.full((4, 4), 0.25),
                                            torch.tensor([0, 1, 2, 3]))) == ())
''',
),

task(
    id="kahan-summation",
    title="Kahan compensated summation",
    chapter=CH4,
    section="4.5 Numerical precision",
    level=2,
    entry="kahan_sum",
    statement=(
        "Sum a float32 vector without losing the small terms. Adding 1.0 to 1e8 "
        "in float32 returns 1e8 — the ulp there is 8 — so a running sum silently "
        "drops every small addend near a large partial sum. Kahan's trick keeps a "
        "compensation term c holding what the last addition lost, and feeds it "
        "back. Stay in float32 throughout: upcasting to float64 is the answer to "
        "a different question, and it is off-limits here."
    ),
    shapes="x (N,) float32  ->  scalar float32, accurate to float64 reference",
    stub=("def kahan_sum(x):\n"
          "    # running sum s, compensation c; float32 throughout\n    pass\n"),
    hints=[
        "Per element: y = x_i - c; t = s + y; c = (t - s) - y; s = t.",
        "c captures exactly the low-order bits (t - s) discarded by the add.",
        "Every intermediate must stay in the input's dtype — the compiler-level "
        "point of the algorithm is that it works without extra precision.",
    ],
    solution=(
        "def kahan_sum(x):\n"
        "    s = torch.zeros((), dtype=x.dtype)\n"
        "    c = torch.zeros((), dtype=x.dtype)\n"
        "    for xi in x:\n"
        "        y = xi - c\n"
        "        t = s + y\n"
        "        c = (t - s) - y\n"
        "        s = t\n"
        "    return s\n"
    ),
    solution_np=(
        "def kahan_sum(x):\n"
        "    s = x.dtype.type(0.0)\n"
        "    c = x.dtype.type(0.0)\n"
        "    for xi in x:\n"
        "        y = xi - c\n"
        "        t = s + y\n"
        "        c = (t - s) - y\n"
        "        s = t\n"
        "    return s\n"
    ),
    traps=[
        "The naive running sum, which loses every small term next to a large "
        "partial sum.",
        "Upcasting to float64 internally — banned here, because the algorithm "
        "exists precisely for when wider precision is not available.",
        "Simplifying c = (t - s) - y algebraically to zero, which is exactly "
        "what an optimising compiler must be prevented from doing.",
        "Expecting it to survive terms LARGER than the running sum — massive "
        "alternating cancellation defeats plain Kahan too; that case needs "
        "Neumaier's variant, which swaps the roles when |x_i| > |s|.",
    ],
    tests='''
def checks(fn, check):
    # the statement's failure mode exactly: 9999 ones drowned by a large head.
    # a sequential float32 running sum drops every single one of them.
    xs = torch.cat([torch.tensor([1e8]), torch.ones(9999)]).to(torch.float32)
    truth = float(xs.double().sum())
    def naive_sequential():
        s = torch.zeros((), dtype=torch.float32)
        for xi in xs:
            s = s + xi
        return float(s)
    check("recovers the drowned addends to within one ulp of the head",
          lambda: abs(float(fn(xs)) - truth) <= 8.0)
    def beats_naive():
        naive_err = abs(naive_sequential() - truth)
        return naive_err > 5000 and abs(float(fn(xs)) - truth) < naive_err / 100
    check("beats a sequential float32 running sum by orders of magnitude", beats_naive)
    check("exact on exact data",
          lambda: abs(float(fn(torch.tensor([1.5, 2.5, -1.0], dtype=torch.float32))) - 3.0) < 1e-7)
    def random_data():
        r = torch.randn(5000, dtype=torch.float32)
        return abs(float(fn(r)) - float(r.double().sum())) < 1e-3
    check("accurate on ordinary data", random_data)
    check("stays in float32",
          lambda: fn(torch.ones(3, dtype=torch.float32)).dtype == torch.float32)
''',
),

task(
    id="gae",
    title="Generalized advantage estimation",
    chapter=CH6,
    section="6.2 Policy gradient and PPO",
    level=2,
    entry="gae",
    statement=(
        "Compute GAE advantages by the backward recurrence: delta_t = r_t + "
        "gamma·V_{t+1}·(1-done_t) - V_t, and A_t = delta_t + gamma·lambda·"
        "(1-done_t)·A_{t+1}. done_t = 1 means the episode ends at step t: no "
        "bootstrapping across the boundary, and no advantage flowing back across "
        "it. Forgetting the mask on the recursive term — while remembering it on "
        "the bootstrap — is the classic bug, and it leaks value across episodes."
    ),
    shapes="rewards (T,) · values (T+1,) · gamma, lam float · dones (T,) in {0,1}  ->  (T,)",
    stub=("def gae(rewards, values, gamma=0.99, lam=0.95, dones=None):\n"
          "    # backward recurrence with the episode mask on BOTH terms\n    pass\n"),
    hints=[
        "Walk t from T-1 down to 0, carrying the running advantage.",
        "mask = 1 - done_t multiplies the bootstrap value AND the carried "
        "advantage.",
        "lam = 0 must reduce to the one-step TD residuals.",
    ],
    solution=(
        "def gae(rewards, values, gamma=0.99, lam=0.95, dones=None):\n"
        "    T = rewards.shape[0]\n"
        "    if dones is None:\n"
        "        dones = torch.zeros(T)\n"
        "    adv = torch.zeros(T)\n"
        "    last = 0.0\n"
        "    for t in range(T - 1, -1, -1):\n"
        "        mask = 1.0 - float(dones[t])\n"
        "        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]\n"
        "        last = delta + gamma * lam * mask * last\n"
        "        adv[t] = last\n"
        "    return adv\n"
    ),
    solution_np=(
        "def gae(rewards, values, gamma=0.99, lam=0.95, dones=None):\n"
        "    T = rewards.shape[0]\n"
        "    if dones is None:\n"
        "        dones = np.zeros(T)\n"
        "    adv = np.zeros(T)\n"
        "    last = 0.0\n"
        "    for t in range(T - 1, -1, -1):\n"
        "        mask = 1.0 - float(dones[t])\n"
        "        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]\n"
        "        last = delta + gamma * lam * mask * last\n"
        "        adv[t] = last\n"
        "    return adv\n"
    ),
    traps=[
        "Masking the bootstrap but not the carried advantage, which leaks "
        "advantage across episode boundaries.",
        "Iterating forward, which turns a two-line recurrence into an O(T^2) "
        "double loop or a wrong answer.",
        "Off-by-one on values — it has T+1 entries, and V_{t+1} is the bootstrap.",
    ],
    extra=(
        "def _brute_gae(r, v, gamma, lam, d):\n"
        "    T = len(r)\n"
        "    deltas = [float(r[t] + gamma * v[t + 1] * (1 - d[t]) - v[t]) for t in range(T)]\n"
        "    out = []\n"
        "    for t in range(T):\n"
        "        acc, coef = 0.0, 1.0\n"
        "        for u in range(t, T):\n"
        "            acc += coef * deltas[u]\n"
        "            if d[u]:\n"
        "                break\n"
        "            coef *= gamma * lam\n"
        "        out.append(acc)\n"
        "    return torch.tensor(out)\n"
    ),
    tests='''
def checks(fn, check):
    T = 8
    r = torch.randn(T)
    v = torch.randn(T + 1)
    z = torch.zeros(T)
    check("matches brute force without dones",
          lambda: close(fn(r, v, 0.9, 0.8, z), _brute_gae(r, v, 0.9, 0.8, z), 1e-5))
    d = torch.tensor([0., 0., 1., 0., 0., 0., 1., 0.])
    check("matches brute force across episode boundaries",
          lambda: close(fn(r, v, 0.9, 0.8, d), _brute_gae(r, v, 0.9, 0.8, d), 1e-5))
    check("lam = 0 gives the TD residuals",
          lambda: close(fn(r, v, 0.9, 0.0, z),
                        r + 0.9 * v[1:] - v[:-1], 1e-5))
    def telescopes():
        a = fn(r, v, 1.0, 1.0, z)
        want = torch.tensor([float(r[t:].sum() + v[-1] - v[t]) for t in range(T)])
        return close(a, want, 1e-4)
    check("gamma = lam = 1 telescopes to returns minus values", telescopes)
    def no_leak():
        r2 = r.clone(); r2[4:] += 100.0        # perturb only after the done at t=2
        a1 = fn(r, v, 0.9, 0.8, d)
        a2 = fn(r2, v, 0.9, 0.8, d)
        return close(a1[:3], a2[:3], 1e-5)
    check("rewards beyond a done cannot reach advantages before it", no_leak)
''',
),

task(
    id="distill-loss",
    title="Distillation loss with temperature",
    chapter=CH6,
    section="6.5 Distillation",
    level=2,
    entry="kd_loss",
    statement=(
        "Compute the knowledge-distillation loss: T^2 times the KL divergence "
        "from the temperature-softened teacher to the temperature-softened "
        "student, averaged over the batch. The T^2 factor is not decoration: "
        "softening by T scales the logit gradients by 1/T^2, and without the "
        "correction the distillation term silently vanishes from the total loss "
        "as T grows."
    ),
    shapes="student (B, V) · teacher (B, V) logits · T float  ->  scalar",
    stub=("def kd_loss(student, teacher, T=2.0):\n"
          "    # T^2 * KL( softmax(teacher/T) || softmax(student/T) ), batch mean\n    pass\n"),
    hints=[
        "Soften both by dividing the logits by T before any softmax.",
        "F.kl_div expects LOG-probabilities for the input (student) and plain "
        "probabilities for the target (teacher); reduction='batchmean'.",
        "Multiply the result by T*T.",
    ],
    solution=(
        "def kd_loss(student, teacher, T=2.0):\n"
        "    logp_s = F.log_softmax(student / T, -1)\n"
        "    p_t = torch.softmax(teacher / T, -1)\n"
        "    return (T * T) * F.kl_div(logp_s, p_t, reduction='batchmean')\n"
    ),
    frameworks=["torch"],
    traps=[
        "Forgetting the T^2, which makes the distillation gradient shrink as "
        "1/T^2 and the term effectively disappear at high temperature.",
        "Passing probabilities where F.kl_div expects log-probabilities — it "
        "runs, and the number is wrong.",
        "Reversing the direction: the teacher is the target distribution, the "
        "student supplies the log-probabilities.",
    ],
    tests='''
def checks(fn, check):
    s = torch.randn(4, 6)
    t = torch.randn(4, 6)
    def manual(T):
        p_t = torch.softmax(t / T, -1)
        logp_t = torch.log_softmax(t / T, -1)
        logp_s = torch.log_softmax(s / T, -1)
        return (T * T) * (p_t * (logp_t - logp_s)).sum(-1).mean()
    check("matches the hand-written KL at T=3", lambda: close(fn(s, t, 3.0), manual(3.0), 1e-5))
    check("T=1 is the plain batch-mean KL", lambda: close(fn(s, t, 1.0), manual(1.0), 1e-5))
    check("identical logits give zero", lambda: abs(float(fn(s, s.clone(), 4.0))) < 1e-6)
    check("non-negative", lambda: float(fn(s, t, 2.0)) >= -1e-7)
    def grad_survives_high_T():
        s1 = s.clone().requires_grad_(True)
        fn(s1, t, 1.0).backward()
        g1 = s1.grad.norm()
        s8 = s.clone().requires_grad_(True)
        fn(s8, t, 8.0).backward()
        g8 = s8.grad.norm()
        return float(g8) > 0.05 * float(g1)
    check("the T^2 factor keeps the gradient alive at high temperature",
          grad_survives_high_T)
''',
),

task(
    id="beam-search",
    title="Beam search",
    chapter=CH7,
    section="7.1 Decoding",
    level=3,
    entry="beam_search",
    statement=(
        "Decode with beam search: keep the k highest-scoring prefixes, expand "
        "every one over the whole vocabulary, and keep the global top k of the "
        "k·V candidates — not the best child of each beam, which is k parallel "
        "greedy searches wearing a disguise. Scores are sums of log-probabilities. "
        "step_fn(prefixes) returns the (N, V) next-token log-probs for a batch of "
        "prefixes; return the final (k, steps) sequences and their scores, best "
        "first."
    ),
    shapes=("step_fn: (N, t) int64 -> (N, V) log-probs · k, steps, vocab int"
            "  ->  (beams (k, steps) int64, scores (k,)), sorted descending"),
    stub=("def beam_search(step_fn, k, steps, vocab):\n"
          "    # global top-k over all k*V candidates at every step\n    pass\n"),
    hints=[
        "Start from one empty prefix with score 0.",
        "Candidate scores are scores[:, None] + logp; flatten to (N*V,) and take "
        "one global topk.",
        "Recover parents and tokens with idx // vocab and idx % vocab, then "
        "gather-and-extend the prefixes.",
    ],
    solution=(
        "def beam_search(step_fn, k, steps, vocab):\n"
        "    beams = torch.zeros(1, 0, dtype=torch.long)\n"
        "    scores = torch.zeros(1)\n"
        "    for _ in range(steps):\n"
        "        logp = step_fn(beams)\n"
        "        cand = (scores[:, None] + logp).flatten()\n"
        "        n = min(k, cand.numel())\n"
        "        top, idx = cand.topk(n)\n"
        "        parent, tok = idx // vocab, idx % vocab\n"
        "        beams = torch.cat([beams[parent], tok[:, None]], dim=1)\n"
        "        scores = top\n"
        "    return beams, scores\n"
    ),
    frameworks=["torch"],
    traps=[
        "Keeping the best child of each beam instead of the global top k — that "
        "is greedy search run k times, and it misses the sequences beam search "
        "exists to find.",
        "Multiplying probabilities instead of adding log-probabilities, which "
        "underflows on any real sequence length.",
        "Forgetting to reorder the prefixes by parent index when extending, "
        "which grafts tokens onto the wrong histories.",
    ],
    extra=(
        "# A model where greedy decoding is suboptimal: the safe first token\n"
        "# leads to a flat continuation, the risky one to a sharp win.\n"
        "_T0 = torch.log(torch.tensor([0.50, 0.40, 0.10]))\n"
        "def _step(prefixes):\n"
        "    N, t = prefixes.shape\n"
        "    if t == 0:\n"
        "        return _T0.expand(N, 3).clone()\n"
        "    out = torch.empty(N, 3)\n"
        "    for i in range(N):\n"
        "        if prefixes[i, -1] == 1:\n"
        "            out[i] = torch.log(torch.tensor([0.90, 0.05, 0.05]))\n"
        "        else:\n"
        "            out[i] = torch.log(torch.tensor([0.35, 0.35, 0.30]))\n"
        "    return out\n"
        "\n"
        "def _brute(steps, vocab=3):\n"
        "    import itertools\n"
        "    best, best_seq = -1e30, None\n"
        "    for seq in itertools.product(range(vocab), repeat=steps):\n"
        "        pref = torch.zeros(1, 0, dtype=torch.long)\n"
        "        s = 0.0\n"
        "        for tok in seq:\n"
        "            s += float(_step(pref)[0, tok])\n"
        "            pref = torch.cat([pref, torch.tensor([[tok]])], dim=1)\n"
        "        if s > best:\n"
        "            best, best_seq = s, seq\n"
        "    return torch.tensor(best_seq), best\n"
    ),
    tests='''
def checks(fn, check):
    def rescore(seq):
        pref = torch.zeros(1, 0, dtype=torch.long)
        s = 0.0
        for tok in seq:
            s += float(_step(pref)[0, int(tok)])
            pref = torch.cat([pref, torch.tensor([[int(tok)]])], dim=1)
        return s

    def exact_with_full_beam():
        beams, scores = fn(_step, 9, 2, 3)      # k = V^steps: beam search is exact
        bseq, bscore = _brute(2)
        return close(beams[0], bseq) and abs(float(scores[0]) - bscore) < 1e-5
    check("k = V^steps recovers the brute-force optimum", exact_with_full_beam)

    def beats_greedy():
        _, s1 = fn(_step, 1, 2, 3)              # greedy
        _, s2 = fn(_step, 2, 2, 3)
        return float(s2[0]) > float(s1[0]) + 1e-6
    check("k=2 finds the optimum greedy misses on this model", beats_greedy)

    def scores_are_real():
        beams, scores = fn(_step, 3, 3, 3)
        return all(abs(float(scores[i]) - rescore(beams[i])) < 1e-5
                   for i in range(beams.shape[0]))
    check("every returned score equals the resummed score of its sequence",
          scores_are_real)

    def sorted_desc():
        _, scores = fn(_step, 4, 3, 3)
        return bool((scores[:-1] >= scores[1:] - 1e-8).all())
    check("beams come back best first", sorted_desc)
    check("shapes", lambda: shape(fn(_step, 3, 4, 3)[0]) == (3, 4))
''',
),

task(
    id="pass-at-k",
    title="The unbiased pass@k estimator",
    chapter=CH7,
    section="7.3 Test-time scaling",
    level=2,
    entry="pass_at_k",
    statement=(
        "Given n sampled solutions of which c passed, estimate the probability "
        "that at least one of k samples passes: pass@k = 1 - C(n-c, k)/C(n, k). "
        "This is the unbiased estimator from the Codex paper; the tempting "
        "1 - (1 - c/n)^k treats the samples as drawn with replacement and is "
        "biased upward — visibly so at small n. Compute it in product form so "
        "large n does not overflow."
    ),
    shapes="n, c, k int, 0 <= c <= n  ->  float in [0, 1]",
    stub=("def pass_at_k(n, c, k):\n"
          "    # 1 - C(n-c, k) / C(n, k), stably\n    pass\n"),
    hints=[
        "If n - c < k, a failure-only draw is impossible: the answer is 1.",
        "C(n-c,k)/C(n,k) = prod over i in (n-c, n] of (i-k)/i — a product of "
        "numbers in (0,1), so nothing overflows.",
        "c = 0 must return exactly 0, and k = 1 must return exactly c/n.",
    ],
    solution=(
        "def pass_at_k(n, c, k):\n"
        "    if n - c < k:\n"
        "        return 1.0\n"
        "    prod = 1.0\n"
        "    for i in range(n - c + 1, n + 1):\n"
        "        prod *= (i - k) / i\n"
        "    return 1.0 - prod\n"
    ),
    solution_np=(
        "def pass_at_k(n, c, k):\n"
        "    if n - c < k:\n"
        "        return 1.0\n"
        "    prod = 1.0\n"
        "    for i in range(n - c + 1, n + 1):\n"
        "        prod *= (i - k) / i\n"
        "    return 1.0 - prod\n"
    ),
    traps=[
        "1 - (1 - c/n)^k — the with-replacement version, biased upward at small n.",
        "Evaluating the binomial coefficients directly in floating point, which "
        "overflows long before n reaches realistic sample counts.",
        "Not short-circuiting n - c < k, where the formula's C(n-c, k) is zero "
        "and a naive product walks through a negative factor.",
    ],
    tests='''
def checks(fn, check):
    check("c = 0 gives 0", lambda: abs(fn(10, 0, 3)) < 1e-12)
    check("c = n gives 1", lambda: abs(fn(10, 10, 3) - 1.0) < 1e-12)
    check("k = 1 is c/n", lambda: abs(fn(20, 7, 1) - 7 / 20) < 1e-12)
    check("matches the exact combinatorial formula",
          lambda: all(abs(fn(n, c, k) - (1 - math.comb(n - c, k) / math.comb(n, k))) < 1e-12
                      for n, c, k in [(10, 3, 4), (25, 10, 5), (7, 2, 3)]
                      if n - c >= k))
    check("survives large n without overflow",
          lambda: 0.0 <= fn(100000, 5000, 100) <= 1.0)
    def unbiased_vs_simulation():
        n, c, k = 12, 4, 5
        torch.manual_seed(0)
        hits = 0
        trials = 4000
        for _ in range(trials):
            draw = torch.randperm(n)[:k]
            hits += int(bool((draw < c).any()))    # first c indices are the passes
        return abs(fn(n, c, k) - hits / trials) < 0.03
    check("agrees with drawing k of n without replacement", unbiased_vs_simulation)
    check("differs from the biased with-replacement formula",
          lambda: abs(fn(4, 2, 3) - (1 - (1 - 0.5) ** 3)) > 0.05)
''',
),

task(
    id="local-sgd",
    title="Local SGD with periodic averaging",
    chapter=CH8,
    section="8.4 Training across datacenters",
    level=3,
    entry="local_sgd",
    statement=(
        "K workers each hold a shard of a least-squares problem (A_k, b_k). Run "
        "local SGD: every worker takes H gradient steps on its own shard, then "
        "all workers average their PARAMETERS and continue from the average, for "
        "a number of rounds. This is the communication pattern behind "
        "cross-datacenter training: with H=1 it collapses to synchronous "
        "data-parallel SGD, and with enough rounds it still converges to the "
        "global least-squares solution — which is what distinguishes it from "
        "letting each worker converge alone and averaging once at the end."
    ),
    shapes="p0 (D,) · A (K, M, D) · b (K, M) · lr float · H, rounds int  ->  (D,)",
    stub=("def local_sgd(p0, A, b, lr=0.02, H=5, rounds=200):\n"
          "    # per round: H local full-gradient steps per worker, then average params\n"
          "    pass\n"),
    hints=[
        "Worker k's gradient at p is A_k^T (A_k p - b_k).",
        "Keep a (K, D) matrix of per-worker parameters; einsum does the batched "
        "gradients in two lines.",
        "After H steps, replace every row with the row-mean — parameters are "
        "averaged, not gradients.",
    ],
    solution=(
        "def local_sgd(p0, A, b, lr=0.02, H=5, rounds=200):\n"
        "    K = A.shape[0]\n"
        "    p = p0.clone().unsqueeze(0).repeat(K, 1)\n"
        "    for _ in range(rounds):\n"
        "        for _ in range(H):\n"
        "            r = torch.einsum('kmd,kd->km', A, p) - b\n"
        "            g = torch.einsum('kmd,km->kd', A, r)\n"
        "            p = p - lr * g\n"
        "        p = p.mean(0, keepdim=True).repeat(K, 1)\n"
        "    return p[0]\n"
    ),
    frameworks=["torch"],
    traps=[
        "Averaging once at the end: each worker converges to ITS OWN shard's "
        "solution, and the average of those is not the global solution.",
        "Averaging gradients instead of parameters every step — that is plain "
        "synchronous SGD, and it erases the H-step communication saving.",
        "Forgetting to restart every worker from the averaged parameters.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    K, M, D = 3, 8, 4
    A = torch.randn(K, M, D)
    w_true = torch.randn(D)
    b = torch.einsum('kmd,d->km', A, w_true)      # consistent system

    def converges_to_global():
        out = fn(torch.zeros(D), A, b, 0.02, 5, 300)
        want = torch.linalg.lstsq(A.reshape(K * M, D),
                                  b.reshape(K * M, 1)).solution.squeeze(-1)
        return close(out, want, 1e-2)
    check("converges to the GLOBAL least-squares solution", converges_to_global)

    def h1_is_sync():
        p = torch.zeros(D)
        for _ in range(40):
            g = torch.stack([Ak.T @ (Ak @ p - bk) for Ak, bk in zip(A, b)]).mean(0)
            p = p - 0.02 * g
        return close(fn(torch.zeros(D), A, b, 0.02, 1, 40), p, 1e-5)
    check("H = 1 equals synchronous SGD on the averaged gradient", h1_is_sync)

    def k1_is_plain():
        p = torch.zeros(D)
        for _ in range(60):
            p = p - 0.02 * (A[0].T @ (A[0] @ p - b[0]))
        return close(fn(torch.zeros(D), A[:1], b[:1], 0.02, 60, 1), p, 1e-5)
    check("K = 1 is plain SGD", k1_is_plain)

    check("output shape", lambda: shape(fn(torch.zeros(D), A, b, 0.02, 2, 3)) == (D,))
    def h_matters():
        a1 = fn(torch.zeros(D), A, b, 0.05, 1, 6)
        a5 = fn(torch.zeros(D), A, b, 0.05, 5, 6)
        return not close(a1, a5, 1e-4)
    check("H > 1 genuinely changes the trajectory", h_matters)
''',
),

]
