"""Off-book topics folded into the existing volumes — the bank is not limited
to what the two source books cover. BPE joins the llm-math data chapter, LoRA
the alignment/fine-tuning chapter, and diffusion the ml-basics variational
chapter, where a hierarchical latent-variable model belongs.
"""
from .schema import task

TASKS = [

task(
    id="bpe-train",
    title="Train a BPE merge table",
    chapter="9 · Data",
    section="Tokenisation — byte-pair encoding (off-book)",
    level=3,
    entry="bpe_merges",
    statement=(
        "Learn byte-pair-encoding merges from a word-frequency table: start "
        "from characters, repeatedly find the adjacent symbol pair with the "
        "highest total frequency, record it, and merge every occurrence. Break "
        "ties by the lexicographically smallest pair, so the result is "
        "deterministic. The tests hand-trace the classic low/lower/newest/"
        "widest corpus merge by merge — this is the algorithm behind every "
        "GPT-style tokeniser, shrunk to where you can verify it on paper."
    ),
    shapes="word_freqs dict[str, int] · num_merges int  ->  list of (str, str) pairs, in order",
    stub=("def bpe_merges(word_freqs, num_merges):\n"
          "    # highest-count adjacent pair, ties -> lexicographically smallest\n    pass\n"),
    hints=[
        "Represent each word as a list of symbols, starting from its "
        "characters; keep the frequency alongside.",
        "Count adjacent pairs weighted by word frequency; pick "
        "max((count, ...)) with the tie-break, then rewrite every word, "
        "replacing the pair with its concatenation.",
        "Overlapping occurrences merge left to right: after merging (a, a) in "
        "'aaa' you get 'aa' + 'a'.",
    ],
    solution=(
        "def bpe_merges(word_freqs, num_merges):\n"
        "    words = {w: list(w) for w in word_freqs}\n"
        "    merges = []\n"
        "    for _ in range(num_merges):\n"
        "        counts = {}\n"
        "        for w, syms in words.items():\n"
        "            f = word_freqs[w]\n"
        "            for i in range(len(syms) - 1):\n"
        "                p = (syms[i], syms[i + 1])\n"
        "                counts[p] = counts.get(p, 0) + f\n"
        "        if not counts:\n"
        "            break\n"
        "        top = max(counts.values())\n"
        "        best = min(p for p, c in counts.items() if c == top)\n"
        "        merges.append(best)\n"
        "        for w, syms in words.items():\n"
        "            out, i = [], 0\n"
        "            while i < len(syms):\n"
        "                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:\n"
        "                    out.append(syms[i] + syms[i + 1])\n"
        "                    i += 2\n"
        "                else:\n"
        "                    out.append(syms[i])\n"
        "                    i += 1\n"
        "            words[w] = out\n"
        "    return merges\n"
    ),
    solution_np=(
        "def bpe_merges(word_freqs, num_merges):\n"
        "    words = {w: list(w) for w in word_freqs}\n"
        "    merges = []\n"
        "    for _ in range(num_merges):\n"
        "        counts = {}\n"
        "        for w, syms in words.items():\n"
        "            f = word_freqs[w]\n"
        "            for i in range(len(syms) - 1):\n"
        "                p = (syms[i], syms[i + 1])\n"
        "                counts[p] = counts.get(p, 0) + f\n"
        "        if not counts:\n"
        "            break\n"
        "        top = max(counts.values())\n"
        "        best = min(p for p, c in counts.items() if c == top)\n"
        "        merges.append(best)\n"
        "        for w, syms in words.items():\n"
        "            out, i = [], 0\n"
        "            while i < len(syms):\n"
        "                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == best:\n"
        "                    out.append(syms[i] + syms[i + 1])\n"
        "                    i += 2\n"
        "                else:\n"
        "                    out.append(syms[i])\n"
        "                    i += 1\n"
        "            words[w] = out\n"
        "    return merges\n"
    ),
    traps=[
        "Counting pairs per unique word instead of weighting by word frequency.",
        "A different tie-break (or dict-order nondeterminism), which diverges "
        "from the specified table on the third merge.",
        "Re-scanning left to right without skipping after a merge, so 'aaa' "
        "yields two overlapping merges.",
    ],
    tests='''
def checks(fn, check):
    corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    # hand trace: ('e','s') 9 -> ('es','t') 9 -> ('l','o') 7 -> ('lo','w') 7
    #             -> tie at 6 between ('n','e'), ('e','w'), ('w','est'):
    #                lexicographically smallest is ('e','w')
    want5 = [("e", "s"), ("es", "t"), ("l", "o"), ("lo", "w"), ("e", "w")]
    check("reproduces the hand-traced classic corpus, merge by merge",
          lambda: [tuple(m) for m in fn(corpus, 5)] == want5)
    check("prefix property: fewer merges are a prefix of more",
          lambda: [tuple(m) for m in fn(corpus, 3)] == want5[:3])
    def encode_lowest():
        merges = [tuple(m) for m in fn(corpus, 4)]
        syms = list("lowest")
        for a, b in merges:
            out, i = [], 0
            while i < len(syms):
                if i + 1 < len(syms) and (syms[i], syms[i + 1]) == (a, b):
                    out.append(a + b); i += 2
                else:
                    out.append(syms[i]); i += 1
            syms = out
        return syms == ["low", "est"]
    check("the learned merges tokenise 'lowest' as [low, est]", encode_lowest)
    def overlap():
        m = fn({"aaaa": 1}, 1)
        return [tuple(x) for x in m] == [("a", "a")]
    check("overlapping pairs are handled", overlap)
    check("stops early when nothing is left to merge",
          lambda: len(fn({"ab": 1}, 10)) == 1)
''',
),

task(
    id="lora-forward",
    title="LoRA forward pass",
    chapter="6 · Alignment and reinforcement learning",
    section="Parameter-efficient fine-tuning — LoRA (off-book)",
    level=2,
    entry="lora_forward",
    statement=(
        "A LoRA-adapted linear layer: out = x W^T + (alpha / r) * x A^T B^T, "
        "with A (r, d_in) and B (d_out, r) the trainable low-rank pair and W "
        "frozen. Two identities define correctness: with B initialised to "
        "zero the adapter is EXACTLY the base layer (which is why training "
        "starts from the pretrained behaviour), and the adapter can be merged "
        "into a single weight W + (alpha/r) B A with no behaviour change "
        "(which is why serving costs nothing extra)."
    ),
    shapes="x (N, d_in) · W (d_out, d_in) · A (r, d_in) · B (d_out, r) · alpha  ->  (N, d_out)",
    stub=("def lora_forward(x, W, A, B, alpha=16.0):\n"
          "    # base + (alpha/r) * low-rank path\n    pass\n"),
    hints=[
        "r is A.shape[0]; the scale is alpha / r.",
        "Low-rank path: (x @ A.T) @ B.T — never materialise B @ A in the "
        "forward, that is the memory the trick saves.",
        "Merged equivalence: x @ (W + (alpha/r) * B @ A).T must give the same "
        "numbers.",
    ],
    solution=(
        "def lora_forward(x, W, A, B, alpha=16.0):\n"
        "    r = A.shape[0]\n"
        "    return x @ W.T + (alpha / r) * ((x @ A.T) @ B.T)\n"
    ),
    solution_np=(
        "def lora_forward(x, W, A, B, alpha=16.0):\n"
        "    r = A.shape[0]\n"
        "    return x @ W.T + (alpha / r) * ((x @ A.T) @ B.T)\n"
    ),
    traps=[
        "Dropping the alpha/r scale, so changing the rank silently changes the "
        "adapter's effective learning rate.",
        "Ordering the factors as A B instead of B A, which only type-checks "
        "when r happens to equal a layer dimension.",
        "Materialising the full (d_out, d_in) delta every forward pass, "
        "defeating the memory saving.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    N, d_in, d_out, r = 5, 8, 6, 2
    x = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in)
    A = torch.randn(r, d_in)
    B = torch.randn(d_out, r)
    check("B = 0 reproduces the base layer exactly",
          lambda: close(fn(x, W, A, torch.zeros(d_out, r)), x @ W.T, 1e-6))
    check("equals the merged weight",
          lambda: close(fn(x, W, A, B, 16.0),
                        x @ (W + (16.0 / r) * B @ A).T, 1e-4))
    check("alpha scales the delta linearly",
          lambda: close(fn(x, W, A, B, 32.0) - x @ W.T,
                        2 * (fn(x, W, A, B, 16.0) - x @ W.T), 1e-4))
    check("the scale divides by the rank",
          lambda: close(fn(x, W, torch.cat([A, torch.zeros(r, d_in)]),\
                           torch.cat([B, torch.zeros(d_out, r)], dim=1), 16.0),
                        x @ W.T + 0.5 * (16.0 / r) * ((x @ A.T) @ B.T), 1e-4))
    check("output shape", lambda: shape(fn(x, W, A, B)) == (N, d_out))
''',
),

task(
    id="diffusion-forward",
    title="The forward diffusion process",
    book="ml-basics", chapter="Variational inference and sampling",
    section="Diffusion — the forward process (off-book)",
    level=2,
    entry="forward_diffusion",
    statement=(
        "Jump straight to any noise level in closed form: "
        "x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps, where abar is the "
        "cumulative product of the alphas and eps is standard normal. This "
        "closed form — not the step-by-step chain — is what makes diffusion "
        "training practical: any (x_0, t) pair becomes a training example in "
        "one line. The coefficients are a signal/noise split: their squares "
        "sum to 1, so swapping them is the classic silent bug."
    ),
    shapes=("x0 (B, D) · t (B,) int64 · abar (T,) in (0, 1] · eps (B, D)"
            "  ->  (B, D)"),
    stub=("def forward_diffusion(x0, t, abar, eps):\n"
          "    # sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, per-sample t\n    pass\n"),
    hints=[
        "Gather abar at each sample's t and unsqueeze to broadcast over "
        "features.",
        "Signal coefficient sqrt(abar_t), noise coefficient sqrt(1 - abar_t).",
        "The noise is an INPUT here so the tests can check exact identities.",
    ],
    solution=(
        "def forward_diffusion(x0, t, abar, eps):\n"
        "    a = abar[t].unsqueeze(-1)\n"
        "    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps\n"
    ),
    solution_np=(
        "def forward_diffusion(x0, t, abar, eps):\n"
        "    a = abar[t][:, None]\n"
        "    return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps\n"
    ),
    traps=[
        "Swapping the coefficients — the marginal variance still looks "
        "plausible, and training silently learns the wrong schedule.",
        "Using alpha_t instead of the CUMULATIVE product abar_t, which is the "
        "one-step formula applied where the closed form belongs.",
        "Sharing one t across the batch when each sample carries its own.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    T, B, D = 10, 6, 4
    betas = torch.linspace(1e-4, 0.2, T)
    abar = torch.cumprod(1 - betas, dim=0)
    x0 = torch.randn(B, D)
    eps = torch.randn(B, D)
    t = torch.randint(0, T, (B,))
    xt = fn(x0, t, abar, eps)
    def noise_recoverable():
        a = abar[t].unsqueeze(-1)
        eps_back = (xt - torch.sqrt(a) * x0) / torch.sqrt(1 - a)
        return close(eps_back, eps, 1e-5)
    check("the injected noise is exactly recoverable (coefficients correct)",
          noise_recoverable)
    def signal_recoverable():
        a = abar[t].unsqueeze(-1)
        x0_back = (xt - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
        return close(x0_back, x0, 1e-5)
    check("x0 is exactly recoverable given the noise", signal_recoverable)
    check("abar ~ 1 returns nearly x0",
          lambda: close(fn(x0, torch.zeros(B, dtype=torch.long),
                           torch.tensor([1.0 - 1e-9] + [0.5] * (T - 1)), eps), x0, 1e-3))
    check("per-sample t is honoured",
          lambda: not close(fn(x0, torch.zeros(B, dtype=torch.long), abar, eps),
                            fn(x0, torch.full((B,), T - 1), abar, eps), 1e-3))
    def variance_is_one_minus_abar():
        tt = torch.full((20000,), 5)
        big_eps = torch.randn(20000, 1)
        out = fn(torch.zeros(20000, 1), tt, abar, big_eps)
        return abs(float(out.var()) - float(1 - abar[5])) < 0.02
    check("for x0 = 0 the marginal variance is 1 - abar_t", variance_is_one_minus_abar)
''',
),

task(
    id="ddpm-step",
    title="One DDPM reverse step",
    book="ml-basics", chapter="Variational inference and sampling",
    section="Diffusion — reverse sampling (off-book)",
    level=3,
    entry="ddpm_step",
    statement=(
        "One step of DDPM ancestral sampling: from x_t and the model's noise "
        "estimate, the posterior mean is mu = (x_t - beta_t/sqrt(1-abar_t) * "
        "eps_hat) / sqrt(alpha_t), and x_{t-1} = mu + sqrt(btilde_t) * z with "
        "btilde_t = (1 - abar_{t-1})/(1 - abar_t) * beta_t and z the given "
        "noise (zero at t = 0). The mean has a second, algebraically equal "
        "form — reconstruct x0-hat from eps_hat and mix it with x_t by the "
        "posterior weights — and the tests verify your step against that "
        "independent formula."
    ),
    shapes=("x_t (B, D) · eps_hat (B, D) · t int · betas (T,) · z (B, D)"
            "  ->  (B, D)"),
    stub=("def ddpm_step(x_t, eps_hat, t, betas, z):\n"
          "    # posterior mean + sqrt(btilde) * z; no noise at t == 0\n    pass\n"),
    hints=[
        "alpha_t = 1 - beta_t; abar is the running cumprod of the alphas; "
        "abar_{t-1} is 1 when t == 0.",
        "mu = (x_t - beta_t / sqrt(1 - abar_t) * eps_hat) / sqrt(alpha_t).",
        "Variance btilde_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t; add "
        "sqrt(btilde) * z only when t > 0.",
    ],
    solution=(
        "def ddpm_step(x_t, eps_hat, t, betas, z):\n"
        "    alphas = 1 - betas\n"
        "    abar = torch.cumprod(alphas, dim=0)\n"
        "    a_t = alphas[t]\n"
        "    ab_t = abar[t]\n"
        "    ab_prev = abar[t - 1] if t > 0 else torch.tensor(1.0)\n"
        "    mu = (x_t - betas[t] / torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(a_t)\n"
        "    if t == 0:\n"
        "        return mu\n"
        "    btilde = (1 - ab_prev) / (1 - ab_t) * betas[t]\n"
        "    return mu + torch.sqrt(btilde) * z\n"
    ),
    frameworks=["torch"],
    traps=[
        "Using beta_t as the sampling variance instead of the posterior "
        "btilde_t — close at large t, badly wrong near t = 0.",
        "Adding noise at t = 0, so the final sample never sharpens.",
        "Dividing by sqrt(abar_t) instead of sqrt(alpha_t) in the mean — the "
        "one-step and closed-form coefficients belong to different formulas.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    T, B, D = 10, 5, 3
    betas = torch.linspace(1e-4, 0.2, T)
    alphas = 1 - betas
    abar = torch.cumprod(alphas, dim=0)
    x_t = torch.randn(B, D)
    eps_hat = torch.randn(B, D)
    z = torch.randn(B, D)

    def dual_form_mean(t):
        # independent identity: mu = w0 * x0hat + wt * x_t with the posterior weights
        ab_t = abar[t]
        ab_prev = abar[t - 1] if t > 0 else torch.tensor(1.0)
        x0_hat = (x_t - torch.sqrt(1 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        w0 = torch.sqrt(ab_prev) * betas[t] / (1 - ab_t)
        wt = torch.sqrt(alphas[t]) * (1 - ab_prev) / (1 - ab_t)
        return w0 * x0_hat + wt * x_t

    check("mean matches the independent x0-hat mixture formula (t = 6)",
          lambda: close(fn(x_t, eps_hat, 6, betas, torch.zeros(B, D)),
                        dual_form_mean(6), 1e-4))
    check("mean matches at t = 1",
          lambda: close(fn(x_t, eps_hat, 1, betas, torch.zeros(B, D)),
                        dual_form_mean(1), 1e-4))
    check("no noise is added at t = 0",
          lambda: close(fn(x_t, eps_hat, 0, betas, z),
                        fn(x_t, eps_hat, 0, betas, torch.zeros(B, D)), 1e-6))
    def noise_scale_is_btilde():
        t = 6
        ab_prev, ab_t = abar[t - 1], abar[t]
        btilde = (1 - ab_prev) / (1 - ab_t) * betas[t]
        diff = fn(x_t, eps_hat, t, betas, z) - fn(x_t, eps_hat, t, betas, torch.zeros(B, D))
        return close(diff, torch.sqrt(btilde) * z, 1e-5)
    check("the added noise is scaled by sqrt(btilde), not sqrt(beta)",
          noise_scale_is_btilde)
    check("output shape", lambda: shape(fn(x_t, eps_hat, 3, betas, z)) == (B, D))
''',
),

]
