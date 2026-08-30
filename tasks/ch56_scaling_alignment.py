"""Chapters 5 and 6 — Scaling laws · Alignment and reinforcement learning."""
from .schema import task

CH5 = "5 · Scaling laws"
CH6 = "6 · Alignment and reinforcement learning"

TASKS = [

task(
    id="chinchilla",
    title="Chinchilla-optimal allocation",
    chapter=CH5,
    section="5.1 Chinchilla",
    level=2,
    entry="allocate",
    statement=(
        "Given a compute budget C in FLOPs, split it into a parameter count N and "
        "a token count D under the compute constraint C = 6ND and the "
        "Chinchilla-optimal ratio D = r·N (r ≈ 20). Substituting gives "
        "6·r·N² = C. The result is the finding that models of that era were "
        "systematically undertrained: fixed compute buys a smaller model on more "
        "tokens than intuition suggests."
    ),
    shapes="compute float (FLOPs) · ratio float  ->  dict 'n_params', 'n_tokens'",
    stub=("def allocate(compute, ratio=20.0):\n"
          "    # -> {'n_params': float, 'n_tokens': float}\n    pass\n"),
    hints=[
        "Two equations: C = 6ND and D = r·N.",
        "Substitute the second into the first: C = 6·N·(r·N) = 6r·N².",
        "N = sqrt(C / (6r)), then D = r·N.",
    ],
    solution=(
        "def allocate(compute, ratio=20.0):\n"
        "    n = math.sqrt(compute / (6.0 * ratio))\n"
        "    return {'n_params': n, 'n_tokens': ratio * n}\n"
    ),
    solution_np=(
        "def allocate(compute, ratio=20.0):\n"
        "    n = math.sqrt(compute / (6.0 * ratio))\n"
        "    return {'n_params': n, 'n_tokens': ratio * n}\n"
    ),
    traps=[
        "Forgetting the 6 in C = 6ND and getting a model sqrt(6) too large.",
        "Solving for N and D independently rather than under the joint constraint.",
        "Assuming N and D each scale linearly with C — both scale with its square "
        "root, so ten times the compute buys about three times the model.",
    ],
    tests='''
def checks(fn, check):
    C = 6e23
    o = fn(C)
    check("satisfies the compute constraint",
          lambda: abs(6 * o["n_params"] * o["n_tokens"] - C) / C < 1e-9)
    check("honours the 20:1 ratio",
          lambda: abs(o["n_tokens"] / o["n_params"] - 20.0) < 1e-9)
    check("both scale as sqrt(C)",
          lambda: abs(fn(4 * C)["n_params"] / o["n_params"] - 2.0) < 1e-9)
    check("a different ratio still satisfies the constraint",
          lambda: abs(6 * fn(C, 100.)["n_params"] * fn(C, 100.)["n_tokens"] - C) / C < 1e-9)
    check("larger ratio means a smaller model",
          lambda: fn(C, 100.)["n_params"] < o["n_params"])
    check("reproduces the Chinchilla paper's own point (5.76e23 -> ~70B / ~1.4T)",
          lambda: abs(fn(5.76e23)["n_params"] - 70e9) / 70e9 < 0.05
                  and abs(fn(5.76e23)["n_tokens"] - 1.4e12) / 1.4e12 < 0.05)
''',
),

task(
    id="power-law-fit",
    title="Fit a scaling law",
    chapter=CH5,
    section="5.1 Chinchilla · 5.2 Data-limited, sparse, and RL scaling",
    level=2,
    entry="fit_power_law",
    statement=(
        "Fit L(x) = a·x^b to observed losses by least squares in log space, and "
        "return a, b and a predictor. Taking logs turns the power law into a "
        "straight line, log L = log a + b·log x, which is why scaling plots are "
        "always drawn log-log — the exponent is a slope you can read off."
    ),
    shapes="xs (N,) · ys (N,) positive  ->  dict 'a', 'b', 'predict' (callable)",
    stub=("def fit_power_law(xs, ys):\n"
          "    # -> {'a': float, 'b': float, 'predict': callable}\n    pass\n"),
    hints=[
        "Regress log(y) on log(x) with an ordinary least-squares straight line.",
        "The slope is b; the intercept is log a, so a = exp(intercept).",
        "np.polyfit(log x, log y, 1) returns [slope, intercept].",
    ],
    solution=(
        "def fit_power_law(xs, ys):\n"
        "    lx = np.log(np.asarray(xs, dtype=float))\n"
        "    ly = np.log(np.asarray(ys, dtype=float))\n"
        "    b, loga = np.polyfit(lx, ly, 1)\n"
        "    a = float(np.exp(loga))\n"
        "    return {'a': a, 'b': float(b), 'predict': lambda x: a * np.asarray(x) ** b}\n"
    ),
    solution_np=(
        "def fit_power_law(xs, ys):\n"
        "    lx = np.log(np.asarray(xs, dtype=float))\n"
        "    ly = np.log(np.asarray(ys, dtype=float))\n"
        "    b, loga = np.polyfit(lx, ly, 1)\n"
        "    a = float(np.exp(loga))\n"
        "    return {'a': a, 'b': float(b), 'predict': lambda x: a * np.asarray(x) ** b}\n"
    ),
    traps=[
        "Fitting in linear space, where the largest x dominates the residual and "
        "the exponent comes out wrong.",
        "Returning the intercept as a without exponentiating it.",
        "Assuming a negative exponent — loss decreasing with scale means b < 0, "
        "but the fit should not hard-code the sign.",
    ],
    tests='''
def checks(fn, check):
    xs = np.array([1e2, 1e3, 1e4, 1e5, 1e6])
    ys = 3.0 * xs ** (-0.1)
    o = fn(xs, ys)
    check("recovers the exponent", lambda: abs(o["b"] - (-0.1)) < 1e-6)
    check("recovers the coefficient", lambda: abs(o["a"] - 3.0) < 1e-4)
    check("predict reproduces the data",
          lambda: close(np.asarray(o["predict"](xs)), ys, 1e-5))
    check("extrapolates on the fitted law",
          lambda: abs(float(o["predict"](1e7)) - 3.0 * 1e7 ** -0.1) < 1e-6)
    check("handles a positive exponent too",
          lambda: abs(fn(xs, 2.0 * xs ** 0.5)["b"] - 0.5) < 1e-6)
''',
),

# ------------------------------------------------------------------ chapter 6
task(
    id="policy-entropy",
    title="Policy entropy",
    chapter=CH6,
    section="6.7 Entropy dynamics and collapse",
    level=1,
    entry="entropy",
    statement=(
        "Return the mean entropy of a batch of categorical policies given their "
        "logits, in nats. Entropy is monitored during RLHF because a policy whose "
        "entropy collapses has stopped exploring — it will keep scoring well on "
        "the reward model while the outputs degenerate."
    ),
    shapes="logits (B, V)  ->  scalar mean entropy in nats",
    stub="def entropy(logits):\n    # -> mean over the batch of -sum p log p\n    pass\n",
    hints=[
        "H = -sum_i p_i log p_i, with p the softmax of the logits.",
        "Use log-softmax for the log term rather than log(softmax(x)), so an "
        "underflowed probability does not become -inf.",
        "p·log p is 0 where p is 0, which the log-softmax form handles naturally.",
    ],
    solution=(
        "def entropy(logits):\n"
        "    logp = torch.log_softmax(logits, -1)\n"
        "    p = logp.exp()\n"
        "    return -(p * logp).sum(-1).mean()\n"
    ),
    solution_np=(
        "def entropy(logits):\n"
        "    m = logits.max(-1, keepdims=True)\n"
        "    z = logits - m\n"
        "    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))\n"
        "    p = np.exp(logp)\n"
        "    return (-(p * logp).sum(-1)).mean()\n"
    ),
    traps=[
        "Using log(softmax(x)), which gives -inf·0 = NaN in the tail.",
        "Returning per-example entropy when the caller wants the batch mean.",
        "Reporting in bits when the convention is nats, a factor of ln 2.",
    ],
    tests='''
def checks(fn, check):
    check("uniform over V has entropy log V",
          lambda: abs(float(fn(torch.zeros(1, 8))) - math.log(8)) < 1e-5)
    check("a one-hot policy has entropy 0",
          lambda: abs(float(fn(torch.tensor([[100., 0., 0., 0.]])))) < 1e-4)
    check("matches the reference formula",
          lambda: (lambda lg: close(fn(lg),
                   -(torch.softmax(lg, -1) * torch.log_softmax(lg, -1)).sum(-1).mean(), 1e-5))
                  (torch.randn(4, 7)))
    check("no NaN on a peaked policy",
          lambda: bool(torch.isfinite(fn(torch.tensor([[0., -800.]]))).all()))
    check("averages over the batch",
          lambda: abs(float(fn(torch.cat([torch.zeros(1, 4),
                                          torch.tensor([[100., 0., 0., 0.]])])))
                      - math.log(4) / 2) < 1e-4)
''',
),

task(
    id="ppo-clip",
    title="PPO clipped surrogate objective",
    chapter=CH6,
    section="6.2 Policy gradient and PPO",
    level=3,
    entry="ppo_loss",
    statement=(
        "Return the PPO clipped surrogate loss: with ratio r = exp(logp - "
        "logp_old), the objective is the mean of min(r·A, clip(r, 1-eps, 1+eps)·A), "
        "and the loss is its negation. The min is what makes the clip one-sided in "
        "effect — it removes the incentive to move further once the ratio has left "
        "the trust region, but never rewards moving back."
    ),
    shapes="logp (N,) · logp_old (N,) · adv (N,) · eps float  ->  scalar loss",
    stub=("def ppo_loss(logp, logp_old, adv, eps=0.2):\n"
          "    # -> scalar (a loss, so negated objective)\n    pass\n"),
    hints=[
        "The ratio is exp(logp - logp_old), not a difference of probabilities.",
        "Form both terms — r·A and clamp(r, 1-eps, 1+eps)·A — then take the "
        "elementwise minimum.",
        "Negate the mean, because optimisers minimise.",
    ],
    solution=(
        "def ppo_loss(logp, logp_old, adv, eps=0.2):\n"
        "    r = (logp - logp_old).exp()\n"
        "    unclipped = r * adv\n"
        "    clipped = r.clamp(1 - eps, 1 + eps) * adv\n"
        "    return -torch.minimum(unclipped, clipped).mean()\n"
    ),
    solution_np=(
        "def ppo_loss(logp, logp_old, adv, eps=0.2):\n"
        "    r = np.exp(logp - logp_old)\n"
        "    unclipped = r * adv\n"
        "    clipped = np.clip(r, 1 - eps, 1 + eps) * adv\n"
        "    return -np.minimum(unclipped, clipped).mean()\n"
    ),
    traps=[
        "Taking the maximum instead of the minimum, which inverts the trust region.",
        "Clipping the advantage rather than the ratio.",
        "Forgetting the negation and reporting the objective as a loss.",
    ],
    tests='''
def checks(fn, check):
    lp = torch.zeros(4); lpo = torch.zeros(4); adv = torch.tensor([1., -1., 2., -2.])
    check("at ratio 1 the loss is -mean(A)",
          lambda: close(fn(lp, lpo, adv), -adv.mean(), 1e-6))
    check("it is a mean, not a sum",
          lambda: close(fn(torch.zeros(2), torch.zeros(2), torch.tensor([1., 2.])),
                        torch.tensor(-1.5), 1e-6))
    def clip_binds_positive():
        # ratio well above 1+eps with positive advantage: the clip should bind
        big = torch.log(torch.tensor([2.0]))
        return close(fn(big, torch.zeros(1), torch.tensor([1.0]), 0.2),
                     -torch.tensor([1.2]), 1e-5)
    check("clip caps the gain on a positive advantage", clip_binds_positive)
    def no_clip_negative():
        # ratio above 1+eps with negative advantage: min picks the unclipped term
        big = torch.log(torch.tensor([2.0]))
        return close(fn(big, torch.zeros(1), torch.tensor([-1.0]), 0.2),
                     torch.tensor([2.0]), 1e-5)
    check("the bound is one-sided: no cap when the advantage is negative", no_clip_negative)
    check("uses a ratio of exponentials, not a difference",
          lambda: not close(fn(torch.full((4,), 0.5), lpo, adv), -adv.mean(), 1e-3))
    check("returns a scalar", lambda: fn(lp, lpo, adv).ndim == 0)
''',
),

task(
    id="dpo-loss",
    title="Direct preference optimisation loss",
    chapter=CH6,
    section="6.3 Direct preference optimisation",
    level=3,
    entry="dpo_loss",
    statement=(
        "Return the DPO loss from the policy and reference log-probabilities of a "
        "chosen and a rejected response: "
        "-logsigmoid(beta·((lp_c - ref_c) - (lp_r - ref_r))). Each bracket is an "
        "implicit reward — the log-ratio against the frozen reference — so DPO "
        "trains a preference model and a policy with one expression and no "
        "separate reward network."
    ),
    shapes="lp_c · lp_r · ref_c · ref_r (N,) · beta float  ->  scalar",
    stub=("def dpo_loss(lp_c, lp_r, ref_c, ref_r, beta=0.1):\n"
          "    # -> scalar loss\n    pass\n"),
    hints=[
        "Form the two implicit rewards first: (lp_c - ref_c) and (lp_r - ref_r).",
        "The margin is their difference, scaled by beta.",
        "Loss is -logsigmoid(margin), averaged. Use logsigmoid, not log(sigmoid).",
    ],
    solution=(
        "def dpo_loss(lp_c, lp_r, ref_c, ref_r, beta=0.1):\n"
        "    margin = beta * ((lp_c - ref_c) - (lp_r - ref_r))\n"
        "    return -F.logsigmoid(margin).mean()\n"
    ),
    solution_np=(
        "def dpo_loss(lp_c, lp_r, ref_c, ref_r, beta=0.1):\n"
        "    margin = beta * ((lp_c - ref_c) - (lp_r - ref_r))\n"
        "    return -(-np.logaddexp(0.0, -margin)).mean()\n"
    ),
    traps=[
        "Dropping the reference terms, which turns it into plain likelihood "
        "training on the chosen response.",
        "Using log(sigmoid(x)), which underflows for confidently wrong pairs.",
        "Subtracting in the wrong order, so the loss rewards the rejected response.",
    ],
    tests='''
def checks(fn, check):
    z = torch.zeros(3)
    check("no preference signal gives -log(0.5)",
          lambda: abs(float(fn(z, z, z, z, 0.1)) - math.log(2)) < 1e-6)
    check("a correct large margin drives the loss toward 0",
          lambda: float(fn(torch.full((1,), 50.), torch.zeros(1),
                           torch.zeros(1), torch.zeros(1), 1.0)) < 1e-6)
    check("an inverted margin is penalised heavily",
          lambda: float(fn(torch.zeros(1), torch.full((1,), 50.),
                           torch.zeros(1), torch.zeros(1), 1.0)) > 10)
    check("the reference cancels a shared shift",
          lambda: close(fn(torch.tensor([2.]), torch.tensor([1.]),
                           torch.tensor([2.]), torch.tensor([1.]), 0.1),
                        torch.tensor(math.log(2)), 1e-6))
    check("stable at an extreme margin",
          lambda: bool(torch.isfinite(fn(torch.tensor([-500.]), torch.tensor([500.]),
                                         torch.zeros(1), torch.zeros(1), 1.0)).all()))
''',
),

task(
    id="grpo-advantage",
    title="GRPO group-normalised advantages",
    chapter=CH6,
    section="6.4 The GRPO family · 6.8 Baselines: RLOO",
    level=2,
    entry="group_advantages",
    statement=(
        "Given rewards for G sampled completions of each of B prompts, return "
        "advantages normalised within each group: subtract the group mean and "
        "divide by the group standard deviation. Normalising within the group is "
        "what removes the need for a learned value network — the other samples of "
        "the same prompt are the baseline."
    ),
    shapes="rewards (B, G) float · eps float  ->  (B, G) float, each row zero-mean",
    stub=("def group_advantages(rewards, eps=1e-4):\n"
          "    # -> (B, G) advantages normalised within each row\n    pass\n"),
    hints=[
        "Reduce along the group axis with keepdim so the result broadcasts back.",
        "A = (r - mean) / (std + eps).",
        "Use the population standard deviation for consistency across group sizes.",
    ],
    solution=(
        "def group_advantages(rewards, eps=1e-4):\n"
        "    mu = rewards.mean(-1, keepdim=True)\n"
        "    sd = rewards.std(-1, unbiased=False, keepdim=True)\n"
        "    return (rewards - mu) / (sd + eps)\n"
    ),
    solution_np=(
        "def group_advantages(rewards, eps=1e-4):\n"
        "    mu = rewards.mean(-1, keepdims=True)\n"
        "    sd = rewards.std(-1, keepdims=True)\n"
        "    return (rewards - mu) / (sd + eps)\n"
    ),
    traps=[
        "Normalising across the whole batch, which reintroduces cross-prompt "
        "coupling and makes easy prompts dominate.",
        "Dividing by zero when every completion in a group scores identically.",
        "Omitting keepdim, so the subtraction broadcasts along the wrong axis.",
    ],
    tests='''
def checks(fn, check):
    r = torch.tensor([[1., 2., 3., 4.], [10., 10., 10., 10.]])
    a = fn(r)
    check("shape is preserved", lambda: shape(a) == (2, 4))
    check("each group is zero-mean", lambda: close(a[0].mean(), torch.tensor(0.), 1e-5))
    check("each group has unit scale", lambda: abs(float(a[0].std(unbiased=False)) - 1.0) < 1e-3)
    check("a constant group does not produce NaN", lambda: bool(torch.isfinite(a[1]).all()))
    check("normalised within, not across, groups",
          lambda: close(fn(torch.tensor([[1., 2., 3., 4.]])), a[0:1], 1e-5))
    check("ordering is preserved", lambda: bool((a[0].argsort() == r[0].argsort()).all()))
''',
),

task(
    id="rloo-baseline",
    title="RLOO leave-one-out baseline",
    chapter=CH6,
    section="6.8 Baselines: RLOO and what group normalisation costs",
    level=2,
    entry="rloo",
    statement=(
        "Compute leave-one-out advantages: for each sample, the baseline is the "
        "mean of the other samples in its group, so A_i = r_i - mean_{j≠i} r_j. "
        "Unlike dividing by the group standard deviation, this baseline is "
        "unbiased — it does not use sample i's own reward, so it cannot leak into "
        "its own advantage."
    ),
    shapes="rewards (B, G) float, G >= 2  ->  (B, G) float",
    stub="def rloo(rewards):\n    # A_i = r_i - mean of the other G-1 rewards\n    pass\n",
    hints=[
        "Avoid a loop: the sum of the others is (row sum) - r_i.",
        "The baseline is that divided by G-1.",
        "A_i = r_i - (S - r_i)/(G-1), which simplifies to (G·r_i - S)/(G-1).",
    ],
    solution=(
        "def rloo(rewards):\n"
        "    G = rewards.shape[-1]\n"
        "    S = rewards.sum(-1, keepdim=True)\n"
        "    return (G * rewards - S) / (G - 1)\n"
    ),
    solution_np=(
        "def rloo(rewards):\n"
        "    G = rewards.shape[-1]\n"
        "    S = rewards.sum(-1, keepdims=True)\n"
        "    return (G * rewards - S) / (G - 1)\n"
    ),
    traps=[
        "Using the full group mean as the baseline, which includes the sample "
        "itself and biases the estimate toward zero.",
        "Dividing by G rather than G-1.",
        "Writing a Python loop over the group when the closed form is one line.",
    ],
    tests='''
def checks(fn, check):
    r = torch.tensor([[1., 2., 3., 4.]])
    a = fn(r)
    check("matches the explicit leave-one-out mean",
          lambda: close(a[0, 0], r[0, 0] - r[0, 1:].mean(), 1e-5))
    check("last element too", lambda: close(a[0, 3], r[0, 3] - r[0, :3].mean(), 1e-5))
    check("shape is preserved", lambda: shape(a) == (1, 4))
    check("a constant group gives zero advantage",
          lambda: close(fn(torch.full((1, 5), 7.)), torch.zeros(1, 5), 1e-5))
    check("differs from the plain group-mean baseline",
          lambda: not close(a, r - r.mean(-1, keepdim=True), 1e-3))
    check("G=2 is a simple difference",
          lambda: close(fn(torch.tensor([[1., 5.]])), torch.tensor([[-4., 4.]]), 1e-5))
    check("works for a batch of groups (B=2)",
          lambda: close(fn(torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]]))[1, 0],
                        torch.tensor(4.) - torch.tensor([3., 2., 1.]).mean(), 1e-5))
''',
),

]
