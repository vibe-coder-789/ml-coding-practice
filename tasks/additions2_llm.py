"""Batch A additions to llm-math: loss variants, positions, optimisation, search."""
from .schema import task

CH1 = "1 · Probability, information, and the loss"
CH2 = "2 · The transformer, term by term"
CH4 = "4 · Optimisation"
CH7 = "7 · Inference"

TASKS = [

task(
    id="focal-loss",
    title="Focal loss",
    chapter=CH1,
    section="1.4 Softmax, its Jacobian, and the loss gradient",
    level=2,
    entry="focal_loss",
    statement=(
        "Implement multi-class focal loss: cross-entropy modulated by "
        "(1 - p_t)^gamma, where p_t is the probability the model assigns to the "
        "TRUE class. Confident-correct examples get down-weighted toward zero, "
        "which is the point — on heavily imbalanced data the easy majority "
        "class otherwise supplies almost the entire gradient. gamma = 0 must "
        "reduce to plain cross-entropy exactly."
    ),
    shapes="logits (B, C) · target (B,) int64 · gamma float  ->  scalar mean",
    stub=("def focal_loss(logits, target, gamma=2.0):\n"
          "    # mean of (1 - p_t)^gamma * (-log p_t)\n    pass\n"),
    hints=[
        "log p_t = log_softmax(logits) gathered at the target.",
        "The modulator is (1 - exp(log p_t))^gamma — compute p_t from the "
        "stable log form, not a separate softmax.",
        "Multiply per-example, then mean over the batch.",
    ],
    solution=(
        "def focal_loss(logits, target, gamma=2.0):\n"
        "    logp = torch.log_softmax(logits, -1)\n"
        "    logp_t = logp.gather(1, target[:, None]).squeeze(1)\n"
        "    p_t = logp_t.exp()\n"
        "    return (-(1 - p_t) ** gamma * logp_t).mean()\n"
    ),
    solution_np=(
        "def focal_loss(logits, target, gamma=2.0):\n"
        "    m = logits.max(-1, keepdims=True)\n"
        "    z = logits - m\n"
        "    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))\n"
        "    logp_t = np.take_along_axis(logp, target[:, None], 1).squeeze(1)\n"
        "    p_t = np.exp(logp_t)\n"
        "    return (-(1 - p_t) ** gamma * logp_t).mean()\n"
    ),
    traps=[
        "Dropping the modulation and returning plain cross-entropy — passes "
        "every gamma = 0 test and nothing else.",
        "Modulating by (1 - p) of the PREDICTED class rather than the true "
        "class.",
        "Computing p_t through a second, unstabilised softmax.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    logits = torch.randn(6, 4)
    tgt = torch.randint(0, 4, (6,))
    check("gamma = 0 is exactly cross-entropy",
          lambda: close(fn(logits, tgt, 0.0), F.cross_entropy(logits, tgt), 1e-6))
    def manual_gamma2():
        logp = torch.log_softmax(logits, -1)
        lp_t = logp.gather(1, tgt[:, None]).squeeze(1)
        want = (-(1 - lp_t.exp()) ** 2 * lp_t).mean()
        return close(fn(logits, tgt, 2.0), want, 1e-6)
    check("matches the formula at gamma = 2", manual_gamma2)
    def downweights_easy():
        easy = torch.tensor([[8.0, 0.0, 0.0, 0.0]])
        t0 = torch.tensor([0])
        return float(fn(easy, t0, 2.0)) < 0.01 * float(F.cross_entropy(easy, t0))
    check("a confident-correct example is down-weighted hard", downweights_easy)
    check("hard examples keep most of their loss",
          lambda: float(fn(torch.tensor([[0.0, 8.0, 0.0, 0.0]]), torch.tensor([0]), 2.0))
                  > 0.9 * float(F.cross_entropy(torch.tensor([[0.0, 8.0, 0.0, 0.0]]),
                                                torch.tensor([0]))))
    check("stable at large logits",
          lambda: bool(torch.isfinite(fn(torch.tensor([[0., 1000., 0., 0.]]),
                                         torch.tensor([0]), 2.0)).all()))
''',
),

task(
    id="label-smoothing",
    title="Label-smoothed cross-entropy",
    chapter=CH1,
    section="1.4 Softmax, its Jacobian, and the loss gradient",
    level=2,
    entry="ls_cross_entropy",
    statement=(
        "Cross-entropy against a smoothed target: mass 1 - eps on the true "
        "class and eps spread uniformly over ALL C classes (torch's "
        "convention — the true class receives 1 - eps + eps/C in total). Must "
        "match F.cross_entropy(label_smoothing=eps). The trap is the "
        "convention: spreading eps over only the C-1 wrong classes is a "
        "different, also-published formula that silently disagrees with every "
        "torch-trained baseline."
    ),
    shapes="logits (B, C) · target (B,) int64 · eps float  ->  scalar mean",
    stub=("def ls_cross_entropy(logits, target, eps=0.1):\n"
          "    # smoothed target: (1 - eps) one-hot + eps * uniform over C\n    pass\n"),
    hints=[
        "loss_i = -(1 - eps) * logp[target_i] - (eps / C) * sum_c logp[c].",
        "Both terms come from one log_softmax; no explicit one-hot needed.",
        "eps = 0 must reduce to plain cross-entropy exactly.",
    ],
    solution=(
        "def ls_cross_entropy(logits, target, eps=0.1):\n"
        "    C = logits.shape[-1]\n"
        "    logp = torch.log_softmax(logits, -1)\n"
        "    nll = -logp.gather(1, target[:, None]).squeeze(1)\n"
        "    smooth = -logp.mean(-1)\n"
        "    return ((1 - eps) * nll + eps * smooth).mean()\n"
    ),
    solution_np=(
        "def ls_cross_entropy(logits, target, eps=0.1):\n"
        "    C = logits.shape[-1]\n"
        "    m = logits.max(-1, keepdims=True)\n"
        "    z = logits - m\n"
        "    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))\n"
        "    nll = -np.take_along_axis(logp, target[:, None], 1).squeeze(1)\n"
        "    smooth = -logp.mean(-1)\n"
        "    return ((1 - eps) * nll + eps * smooth).mean()\n"
    ),
    traps=[
        "Spreading eps over C-1 classes instead of all C — a different "
        "convention that mismatches torch's.",
        "Smoothing the LOGITS instead of the target distribution.",
        "Forgetting that eps = 0 must recover plain cross-entropy bit-for-bit.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    logits = torch.randn(5, 7)
    tgt = torch.randint(0, 7, (5,))
    check("matches F.cross_entropy(label_smoothing=0.1)",
          lambda: close(fn(logits, tgt, 0.1),
                        F.cross_entropy(logits, tgt, label_smoothing=0.1), 1e-6))
    check("matches at a different eps",
          lambda: close(fn(logits, tgt, 0.3),
                        F.cross_entropy(logits, tgt, label_smoothing=0.3), 1e-6))
    check("eps = 0 is plain cross-entropy",
          lambda: close(fn(logits, tgt, 0.0), F.cross_entropy(logits, tgt), 1e-6))
    check("eps = 1 ignores the target entirely",
          lambda: close(fn(logits, tgt, 1.0),
                        (-torch.log_softmax(logits, -1).mean(-1)).mean(), 1e-6))
    check("smoothing raises the loss of a perfect prediction",
          lambda: float(fn(torch.tensor([[50., 0., 0.]]), torch.tensor([0]), 0.1))
                  > float(fn(torch.tensor([[50., 0., 0.]]), torch.tensor([0]), 0.0)))
''',
),

task(
    id="infonce",
    title="InfoNCE contrastive loss",
    chapter=CH1,
    section="1.1 Entropy, cross-entropy, KL — mutual information",
    level=2,
    entry="info_nce",
    statement=(
        "The contrastive loss behind SimCLR and CLIP: L2-normalise both views, "
        "form the (B, B) cosine-similarity matrix scaled by 1/temperature, and "
        "apply cross-entropy with the diagonal as the targets — each example's "
        "positive is its own counterpart, every other row is a negative. The "
        "normalisation is part of the loss, not preprocessing: without it the "
        "model can cheat by inflating embedding norms instead of aligning "
        "directions."
    ),
    shapes="z1, z2 (B, D) · temp float  ->  scalar",
    stub=("def info_nce(z1, z2, temp=0.1):\n"
          "    # normalise, similarity matrix / temp, CE against the diagonal\n    pass\n"),
    hints=[
        "F.normalize(z, dim=-1) on both views.",
        "logits = z1n @ z2n.T / temp; targets are arange(B).",
        "Return F.cross_entropy(logits, targets).",
    ],
    solution=(
        "def info_nce(z1, z2, temp=0.1):\n"
        "    a = F.normalize(z1, dim=-1)\n"
        "    b = F.normalize(z2, dim=-1)\n"
        "    logits = a @ b.T / temp\n"
        "    return F.cross_entropy(logits, torch.arange(z1.shape[0]))\n"
    ),
    frameworks=["torch"],
    traps=[
        "Skipping the L2 normalisation, so the loss is minimised by norm "
        "inflation rather than alignment.",
        "Multiplying by the temperature instead of dividing.",
        "Targets other than the diagonal — the pairing IS the supervision.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    def manual(temp):
        a = F.normalize(z1, dim=-1); b = F.normalize(z2, dim=-1)
        return F.cross_entropy(a @ b.T / temp, torch.arange(8))
    check("matches the reference composition", lambda: close(fn(z1, z2, 0.1), manual(0.1), 1e-5))
    check("scale-invariant in either input (normalisation is inside)",
          lambda: close(fn(7 * z1, z2, 0.1), fn(z1, z2, 0.1), 1e-5)
                  and close(fn(z1, 0.3 * z2, 0.1), fn(z1, z2, 0.1), 1e-5))
    check("aligned views at low temperature drive the loss to ~0",
          lambda: float(fn(z1, z1.clone(), 0.02)) < 0.05)
    def alignment_helps():
        return float(fn(z1, z1.clone(), 0.1)) < float(fn(z1, z2, 0.1))
    check("aligned pairs score lower than random pairs", alignment_helps)
    check("permuting the negatives changes nothing for a perfect diagonal",
          lambda: float(fn(z1, z1.clone(), 0.02)) < 0.05)
''',
),

task(
    id="sinusoidal-pe",
    title="Sinusoidal positional encoding",
    chapter=CH2,
    section="2.6 Rotary position embeddings — the fixed-encoding ancestor",
    level=1,
    entry="sinusoidal_pe",
    statement=(
        "The original transformer's fixed position table: "
        "PE[p, 2i] = sin(p / 10000^(2i/d)) and PE[p, 2i+1] = cos of the same "
        "angle. The property that motivated it — and that the tests check — is "
        "that PE[p+k] is a fixed LINEAR function of PE[p]: each (sin, cos) pair "
        "advances by a rotation that depends only on the offset k, which is "
        "the same idea RoPE later applied to queries and keys directly."
    ),
    shapes="L int · d even int  ->  (L, d) float",
    stub=("def sinusoidal_pe(L, d):\n"
          "    # PE[p, 2i] = sin(p * w_i), PE[p, 2i+1] = cos(p * w_i)\n    pass\n"),
    hints=[
        "Frequencies: w_i = 10000^(-2i/d) for i in 0..d/2-1.",
        "Outer product positions x frequencies gives every angle at once.",
        "Interleave: even columns sin, odd columns cos.",
    ],
    solution=(
        "def sinusoidal_pe(L, d):\n"
        "    i = torch.arange(d // 2, dtype=torch.float32)\n"
        "    w = 10000.0 ** (-2 * i / d)\n"
        "    ang = torch.arange(L, dtype=torch.float32)[:, None] * w[None, :]\n"
        "    pe = torch.zeros(L, d)\n"
        "    pe[:, 0::2] = torch.sin(ang)\n"
        "    pe[:, 1::2] = torch.cos(ang)\n"
        "    return pe\n"
    ),
    solution_np=(
        "def sinusoidal_pe(L, d):\n"
        "    i = np.arange(d // 2, dtype=np.float64)\n"
        "    w = 10000.0 ** (-2 * i / d)\n"
        "    ang = np.arange(L, dtype=np.float64)[:, None] * w[None, :]\n"
        "    pe = np.zeros((L, d))\n"
        "    pe[:, 0::2] = np.sin(ang)\n"
        "    pe[:, 1::2] = np.cos(ang)\n"
        "    return pe\n"
    ),
    traps=[
        "Swapping sin and cos, which every downstream check of position 0 "
        "catches: row 0 must be [0, 1, 0, 1, ...].",
        "Using exponent i/d instead of 2i/d, compressing the frequency range.",
        "Concatenating the sin half and cos half instead of interleaving.",
    ],
    tests='''
def checks(fn, check):
    pe = fn(10, 8)
    check("shape", lambda: shape(pe) == (10, 8))
    check("position 0 is [0, 1, 0, 1, ...]",
          lambda: close(pe[0], torch.tensor([0., 1.] * 4), 1e-6))
    check("first pair advances at frequency 1",
          lambda: close(pe[3, 0], torch.tensor(math.sin(3.0)), 1e-5)
                  and close(pe[3, 1], torch.tensor(math.cos(3.0)), 1e-5))
    check("last pair uses frequency 10000^(-6/8)",
          lambda: close(pe[5, 6], torch.tensor(math.sin(5 * 10000 ** (-0.75))), 1e-5))
    check("values bounded by 1", lambda: bool((pe.abs() <= 1 + 1e-6).all()))
    def relative_shift_is_linear():
        # PE[p+k] pair = rotation by k*w applied to PE[p] pair
        L, d, k = 12, 8, 3
        P = fn(L, d)
        for pair in range(d // 2):
            w = 10000.0 ** (-2 * pair / d)
            cw, sw = math.cos(k * w), math.sin(k * w)
            for p in range(L - k):
                s, c = float(P[p, 2 * pair]), float(P[p, 2 * pair + 1])
                s2 = s * cw + c * sw
                c2 = c * cw - s * sw
                if abs(s2 - float(P[p + k, 2 * pair])) > 1e-4:
                    return False
                if abs(c2 - float(P[p + k, 2 * pair + 1])) > 1e-4:
                    return False
        return True
    check("a position shift is a fixed rotation of each pair", relative_shift_is_linear)
''',
),

task(
    id="grad-accumulation",
    title="Gradient accumulation over micro-batches",
    chapter=CH4,
    section="4.3 Schedules, batch size, and gradient clipping",
    level=2,
    entry="accumulate_grads",
    statement=(
        "Compute the gradient of the mean loss over a large batch by "
        "processing micro-batches one at a time — the standard trick for "
        "training with more examples than fit in memory. The result must equal "
        "the full-batch gradient EXACTLY, which forces the detail everyone "
        "fumbles: with unequal micro-batch sizes, each micro-gradient must be "
        "weighted by n_b / N, not averaged uniformly."
    ),
    shapes=("loss_fn(params, X, y) -> scalar mean loss · params (D,) requires_grad "
            "· batches list of (X, y)  ->  (D,) gradient"),
    stub=("def accumulate_grads(loss_fn, params, batches):\n"
          "    # per-micro-batch grads, combined to the exact full-batch grad\n    pass\n"),
    hints=[
        "torch.autograd.grad(loss, params) per micro-batch, no .backward() "
        "bookkeeping needed.",
        "The full-batch mean is sum_b (n_b / N) * mean_b, so the gradients "
        "combine with the same weights.",
        "N is the TOTAL example count across the micro-batches.",
    ],
    solution=(
        "def accumulate_grads(loss_fn, params, batches):\n"
        "    N = sum(X.shape[0] for X, _ in batches)\n"
        "    total = torch.zeros_like(params)\n"
        "    for X, y in batches:\n"
        "        loss = loss_fn(params, X, y)\n"
        "        (g,) = torch.autograd.grad(loss, params)\n"
        "        total = total + (X.shape[0] / N) * g\n"
        "    return total\n"
    ),
    frameworks=["torch"],
    traps=[
        "Uniformly averaging the micro-gradients — exact only when every "
        "micro-batch has the same size, silently wrong otherwise.",
        "Summing the micro-gradients, which scales the step by the number of "
        "micro-batches.",
        "Calling .backward() repeatedly and forgetting that .grad accumulates "
        "on its own.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    D = 4
    params = torch.randn(D, requires_grad=True)
    X = torch.randn(9, D)
    y = torch.randn(9)
    def loss_fn(p, Xb, yb):
        return ((Xb @ p - yb) ** 2).mean()

    def full_grad():
        (g,) = torch.autograd.grad(loss_fn(params, X, y), params)
        return g

    check("equal micro-batches reproduce the full gradient exactly",
          lambda: close(fn(loss_fn, params, [(X[:3], y[:3]), (X[3:6], y[3:6]),
                                             (X[6:], y[6:])]), full_grad(), 1e-6))
    check("UNEQUAL micro-batches still reproduce it exactly",
          lambda: close(fn(loss_fn, params, [(X[:2], y[:2]), (X[2:9], y[2:9])]),
                        full_grad(), 1e-6))
    check("a single micro-batch is the plain gradient",
          lambda: close(fn(loss_fn, params, [(X, y)]), full_grad(), 1e-7))
    check("output shape", lambda: shape(fn(loss_fn, params, [(X, y)])) == (D,))
    check("params keep their gradient requirement intact",
          lambda: (fn(loss_fn, params, [(X, y)]), params.requires_grad)[-1])
''',
),

task(
    id="newton-schulz",
    title="Orthogonalisation by Newton–Schulz iteration",
    chapter=CH4,
    section="4.2 Muon: orthogonalised updates",
    level=3,
    entry="newton_schulz",
    statement=(
        "Compute the orthogonal polar factor of G — the same U V^T the SVD "
        "gives — WITHOUT an SVD, by the cubic Newton–Schulz iteration: "
        "normalise G by its Frobenius norm, then repeat X <- 1.5 X - 0.5 X X^T "
        "X. This is how Muon orthogonalises its updates inside a training "
        "step, where a MatMul-only iteration is fast on accelerators and an "
        "SVD is not. The normalisation is what makes it converge: the "
        "iteration's basin is singular values below sqrt(3)."
    ),
    shapes="G (m, n) · steps int  ->  (m, n) with singular values ~1",
    stub=("def newton_schulz(G, steps=40):\n"
          "    # normalise by ||G||_F, then X <- 1.5 X - 0.5 X X^T X\n    pass\n"),
    hints=[
        "X0 = G / G.norm() puts every singular value in (0, 1].",
        "Each sweep maps each singular value s to 1.5 s - 0.5 s^3, a fixed "
        "point at 1.",
        "40 steps is plenty for well-conditioned G; the check tolerance "
        "assumes it.",
    ],
    solution=(
        "def newton_schulz(G, steps=40):\n"
        "    X = G / G.norm()\n"
        "    for _ in range(steps):\n"
        "        X = 1.5 * X - 0.5 * (X @ X.T @ X)\n"
        "    return X\n"
    ),
    solution_np=(
        "def newton_schulz(G, steps=40):\n"
        "    X = G / np.linalg.norm(G)\n"
        "    for _ in range(steps):\n"
        "        X = 1.5 * X - 0.5 * (X @ X.T @ X)\n"
        "    return X\n"
    ),
    traps=[
        "Skipping the initial normalisation — singular values above sqrt(3) "
        "diverge instead of converging to 1.",
        "Writing X X X^T or X^T X X; only X X^T X preserves the singular "
        "vectors while pushing the values toward 1.",
        "Using it on a rank-deficient matrix and expecting orthogonality — "
        "zero singular values stay at zero.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    G = torch.randn(6, 4, dtype=torch.float64)
    O = fn(G, 60)
    def matches_svd():
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        return close(O, U @ Vh, 1e-4)
    check("converges to the SVD's polar factor U V^T", matches_svd)
    check("singular values are all ~1",
          lambda: close(torch.linalg.svdvals(O), torch.ones(4, dtype=torch.float64), 1e-4))
    check("columns are orthonormal",
          lambda: close(O.T @ O, torch.eye(4, dtype=torch.float64), 1e-4))
    check("scale invariant (normalisation is inside)",
          lambda: close(fn(37.0 * G, 60), O, 1e-4))
    check("shape preserved for wide matrices",
          lambda: shape(fn(torch.randn(3, 7, dtype=torch.float64), 60)) == (3, 7))
    check("an orthogonal input is a fixed point",
          lambda: (lambda Q: close(fn(Q * 5.0, 60), Q, 1e-4))(
              torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64))[0]))
''',
),

task(
    id="mcts-nim",
    title="Monte Carlo tree search with UCT",
    chapter=CH7,
    section="7.3 Test-time scaling",
    level=3,
    entry="uct_best_move",
    statement=(
        "Choose a move in the game of Nim (n stones, take 1–3, taking the last "
        "stone WINS) by UCT search: selection by the UCB rule "
        "Q + c*sqrt(ln(N_parent)/N_child), random-playout evaluation, and "
        "negamax backup — a win for the player to move at a node is a loss for "
        "the player above it, and forgetting that sign flip makes the search "
        "actively prefer losing moves. Nim has a known optimal policy (leave a "
        "multiple of 4), so the search is checked against ground truth."
    ),
    shapes="n int stones · n_sims int · c float  ->  int move in {1, 2, 3}",
    stub=("def uct_best_move(n, n_sims=3000, c=1.4):\n"
          "    # UCT: select, expand, random rollout, negamax backup\n    pass\n"),
    hints=[
        "A node's value is always from the perspective of the player TO MOVE "
        "there; backing up flips the sign at every level.",
        "Terminal rule: if a node has 0 stones, the player to move has LOST "
        "(the previous player took the last stone).",
        "After the simulations, pick the root child with the most visits.",
    ],
    solution=(
        "def uct_best_move(n, n_sims=3000, c=1.4):\n"
        "    stats = {}                                    # state -> [visits, wins]\n"
        "    def moves(s):\n"
        "        return [m for m in (1, 2, 3) if m <= s]\n"
        "    def rollout(s):\n"
        "        # returns +1 if the player to move at s wins under random play\n"
        "        sign = 1\n"
        "        while s > 0:\n"
        "            m = int(torch.randint(1, min(3, s) + 1, (1,)))\n"
        "            s -= m\n"
        "            sign = -sign\n"
        "        return -sign                              # mover at s=0 has lost\n"
        "    def search(s):\n"
        "        if s == 0:\n"
        "            return -1                             # to-move loses\n"
        "        if s not in stats:\n"
        "            stats[s] = [0, 0.0]\n"
        "            v = rollout(s)\n"
        "        else:\n"
        "            best, best_u = None, -1e18\n"
        "            for m in moves(s):\n"
        "                ch = s - m\n"
        "                vis = stats.get(ch, [0, 0.0])[0]\n"
        "                if vis == 0:\n"
        "                    best = m\n"
        "                    break\n"
        "                q = stats[ch][1] / vis\n"
        "                u = -q + c * math.sqrt(math.log(stats[s][0]) / vis)\n"
        "                if u > best_u:\n"
        "                    best, best_u = m, u\n"
        "            v = -search(s - best)\n"
        "        stats[s][0] += 1\n"
        "        stats[s][1] += v\n"
        "        return v\n"
        "    for _ in range(n_sims):\n"
        "        search(n)\n"
        "    return max(moves(n), key=lambda m: stats.get(n - m, [0, 0.0])[0])\n"
    ),
    frameworks=["torch"],
    traps=[
        "Backing up values without the negamax sign flip — the search then "
        "maximises the OPPONENT's win rate and recommends losing moves.",
        "Selecting the final move by raw value instead of visit count, which "
        "is noisy for rarely-visited children.",
        "Getting the terminal convention backwards: at 0 stones the player to "
        "move has already lost.",
    ],
    tests='''
def checks(fn, check):
    # optimal play leaves a multiple of 4
    check("from 5 stones, take 1", lambda: fn(5, 3000) == 1)
    check("from 6 stones, take 2", lambda: fn(6, 3000) == 2)
    check("from 7 stones, take 3", lambda: fn(7, 3000) == 3)
    check("from 9 stones, take 1", lambda: fn(9, 4000) == 1)
    check("from 10 stones, take 2", lambda: fn(10, 4000) == 2)
    check("returns a legal move from 2 stones", lambda: fn(2, 500) in (1, 2))
''',
),

]
