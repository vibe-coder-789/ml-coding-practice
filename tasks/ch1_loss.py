"""Chapter 1 — Probability, information, and the loss."""
from .schema import task

CH = "1 · Probability, information, and the loss"

TASKS = [

task(
    id="stable-softmax",
    title="Numerically stable softmax",
    chapter=CH,
    section="1.4 Softmax, its Jacobian, and the loss gradient",
    level=1,
    entry="softmax",
    statement=(
        "Implement softmax along a given axis so that it survives large logits. "
        "The direct formula overflows: float32 exp saturates to infinity at about "
        "88.7, so a logit of 1000 gives inf/inf = NaN. Do not call torch.softmax."
    ),
    shapes="x (..., N) float · dim int  ->  (..., N) float, sums to 1 along dim",
    stub="def softmax(x, dim=-1):\n    # (..., N) -> same shape, sums to 1 along dim\n    pass\n",
    hints=[
        "Softmax is invariant to adding a constant to every score: the factor e^c "
        "appears in both numerator and denominator and cancels.",
        "Choose that constant so the largest exponent becomes exactly zero.",
        "Subtract x.max(dim, keepdim=True).values before exponentiating. Without "
        "keepdim the subtraction broadcasts against the wrong axis.",
    ],
    solution=(
        "def softmax(x, dim=-1):\n"
        "    m = x.max(dim, keepdim=True).values\n"
        "    e = torch.exp(x - m)\n"
        "    return e / e.sum(dim, keepdim=True)\n"
    ),
    traps=[
        "Omitting keepdim, so the subtraction broadcasts into a larger tensor.",
        "Subtracting a global max instead of a per-row max.",
        "Worrying about underflow of the small terms — they were negligible "
        "relative to the maximum by construction.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(4, 6)
    check("matches torch.softmax", lambda: close(fn(x), torch.softmax(x, -1)))
    check("rows sum to 1", lambda: close(fn(x).sum(-1), torch.ones(4)))
    check("shift invariant",
          lambda: close(fn(torch.tensor([[1., 2., 3.]])),
                        fn(torch.tensor([[101., 102., 103.]]))))
    big = torch.tensor([[1000., 1001., 1002.]])
    check("survives logits of 1000", lambda: bool(torch.isfinite(fn(big)).all()))
    check("correct on the overflow case", lambda: close(fn(big), torch.softmax(big, -1)))
    check("respects dim=0", lambda: close(fn(torch.randn(3, 4), 0).sum(0), torch.ones(4)))
''',
),

task(
    id="log-softmax",
    title="log-softmax via log-sum-exp",
    chapter=CH,
    section="1.4 Softmax, its Jacobian, and the loss gradient",
    level=2,
    entry="log_softmax",
    statement=(
        "Compute log-softmax without ever forming the sum of exponentials "
        "directly. Taking log(softmax(x)) loses precision exactly where it "
        "matters — in the tail, where the probability underflows to zero and the "
        "logarithm becomes -inf."
    ),
    shapes="x (..., N) float · dim int  ->  (..., N) float",
    stub="def log_softmax(x, dim=-1):\n    # (..., N) -> same shape\n    pass\n",
    hints=[
        "log sum_j exp(z_j) = m + log sum_j exp(z_j - m), with m the row max.",
        "log softmax(z)_i = z_i - logsumexp(z).",
        "Shift by the max first, then subtract the log of the shifted sum.",
    ],
    solution=(
        "def log_softmax(x, dim=-1):\n"
        "    m = x.max(dim, keepdim=True).values\n"
        "    z = x - m\n"
        "    return z - z.exp().sum(dim, keepdim=True).log()\n"
    ),
    traps=[
        "Computing softmax then log, which returns -inf for underflowed entries.",
        "Forgetting to subtract the max, overflowing on large logits.",
        "Dropping the shift term m when reassembling the result.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(4, 6)
    check("matches F.log_softmax", lambda: close(fn(x), F.log_softmax(x, -1)))
    check("exp sums to 1", lambda: close(fn(x).exp().sum(-1), torch.ones(4)))
    big = torch.tensor([[1000., 1001., 1002.]])
    check("stable at large logits", lambda: bool(torch.isfinite(fn(big)).all()))
    tail = torch.tensor([[0., -200.]])
    check("tail stays finite where log(softmax) would be -inf",
          lambda: bool(torch.isfinite(fn(tail)).all()))
    check("respects dim=0",
          lambda: close(fn(torch.randn(3, 4), 0).exp().sum(0), torch.ones(4)))
''',
),

task(
    id="cross-entropy",
    title="Cross-entropy from logits",
    chapter=CH,
    section="1.4 Softmax, its Jacobian, and the loss gradient",
    level=2,
    entry="cross_entropy",
    statement=(
        "Compute mean cross-entropy directly from logits and integer targets, "
        "without forming probabilities. Must match F.cross_entropy. The reason "
        "the interface takes logits is that log-sum-exp minus the true logit "
        "never divides by a probability."
    ),
    shapes="logits (B, C) float · target (B,) int64  ->  scalar",
    stub="def cross_entropy(logits, target):\n    # (B, C), (B,) -> scalar\n    pass\n",
    hints=[
        "L = log(sum_j exp(z_j)) - z_c. Both terms are computable stably.",
        "The second term is one gather of the true class's logit.",
        "Build log-softmax stably, gather at the target, negate, take the mean.",
    ],
    solution=(
        "def cross_entropy(logits, target):\n"
        "    m = logits.max(-1, keepdim=True).values\n"
        "    z = logits - m\n"
        "    lp = z - z.exp().sum(-1, keepdim=True).log()\n"
        "    return -lp.gather(1, target[:, None]).squeeze(1).mean()\n"
    ),
    traps=[
        "Applying softmax first, reintroducing the 1/p term the fused form avoids.",
        "Passing probabilities to F.cross_entropy when comparing — it applies "
        "log-softmax internally and would double-softmax.",
        "Summing instead of averaging over the batch.",
    ],
    tests='''
def checks(fn, check):
    logits = torch.randn(4, 6)
    tgt = torch.randint(0, 6, (4,))
    check("matches F.cross_entropy",
          lambda: close(fn(logits, tgt), F.cross_entropy(logits, tgt)))
    check("matches on a fixed case",
          lambda: close(fn(torch.tensor([[2., 1., 0.], [0., 3., 1.]]), torch.tensor([0, 1])),
                        F.cross_entropy(torch.tensor([[2., 1., 0.], [0., 3., 1.]]),
                                        torch.tensor([0, 1]))))
    check("returns a scalar", lambda: fn(logits, tgt).ndim == 0)
    check("stable at large logits",
          lambda: bool(torch.isfinite(fn(torch.tensor([[1000., 0., 0.]]), torch.tensor([0]))).all()))
    check("stable when the target is NOT the max class",
          lambda: close(fn(torch.tensor([[0., 1000., 0.]]), torch.tensor([0])),
                        torch.tensor(1000.), 1e-2))

    def grad_is_p_minus_onehot():
        lg = torch.randn(3, 4, requires_grad=True)
        t = torch.tensor([0, 2, 1])
        (fn(lg, t) * 3).backward()
        want = torch.softmax(lg.detach(), -1) - F.one_hot(t, 4).float()
        return close(lg.grad, want, 1e-4)
    check("gradient equals p - onehot", grad_is_p_minus_onehot)

    def perfect_is_zero():
        lg = torch.tensor([[100., 0., 0.]])
        return fn(lg, torch.tensor([0])).item() < 1e-5
    check("confident correct prediction gives ~0 loss", perfect_is_zero)
''',
),

task(
    id="softmax-jacobian",
    title="The softmax Jacobian",
    chapter=CH,
    section="1.4 Softmax, its Jacobian, and the loss gradient",
    level=3,
    entry="softmax_jacobian",
    statement=(
        "Return the Jacobian of softmax with respect to its input, for a single "
        "vector. The identity is J = diag(s) - s s^T, where s is the softmax "
        "output. Verify your understanding rather than autograd's: build it "
        "directly from s, without calling torch.autograd."
    ),
    shapes="z (N,) float  ->  (N, N) float",
    stub="def softmax_jacobian(z):\n    # (N,) -> (N, N)\n    pass\n",
    hints=[
        "ds_i/dz_j = s_i (delta_ij - s_j). Split that into two terms.",
        "The delta term is a diagonal matrix of s; the other is an outer product.",
        "torch.diag(s) - torch.outer(s, s)",
    ],
    solution=(
        "def softmax_jacobian(z):\n"
        "    s = torch.softmax(z, -1)\n"
        "    return torch.diag(s) - torch.outer(s, s)\n"
    ),
    traps=[
        "Writing diag(s) - s^T s, which is a scalar times identity, not an outer "
        "product.",
        "Forgetting the Jacobian is symmetric — a useful self-check.",
        "Expecting it to be invertible; rows sum to zero, so it is singular.",
    ],
    tests='''
def checks(fn, check):
    z = torch.randn(5)
    check("shape is (N, N)", lambda: shape(fn(z)) == (5, 5))
    check("matches autograd (the tests may use the oracle; submissions may not)",
          lambda: close(fn(z), torch.autograd.functional.jacobian(
              lambda t: torch.softmax(t, -1), z), 1e-4))
    check("is symmetric", lambda: (lambda J: close(J, J.T, 1e-6))(fn(z)))
    check("rows sum to zero", lambda: close(fn(z).sum(-1), torch.zeros(5), 1e-5))
    check("diagonal is s(1-s)",
          lambda: close(torch.diagonal(fn(z)),
                        torch.softmax(z, -1) * (1 - torch.softmax(z, -1)), 1e-6))
''',
),

task(
    id="kl-divergence",
    title="KL divergence and its estimators",
    chapter=CH,
    section="1.1 Entropy, cross-entropy, KL · 1.3 Forward vs. reverse KL",
    level=2,
    entry="kl_estimators",
    statement=(
        "Given log-probabilities of two distributions under samples from p, "
        "return the three standard Monte-Carlo estimators of KL(p||q) used in "
        "RLHF: k1 = -log r, k2 = (log r)^2 / 2, k3 = (r - 1) - log r, where "
        "r = q/p. k1 is unbiased but high variance and can go negative; k2 is "
        "low variance but biased; k3 is unbiased and never negative."
    ),
    shapes="logp (N,) · logq (N,)  ->  dict with keys 'k1', 'k2', 'k3', each scalar",
    stub=("def kl_estimators(logp, logq):\n"
          "    # -> {'k1': scalar, 'k2': scalar, 'k3': scalar}\n    pass\n"),
    hints=[
        "log r = logq - logp. Every estimator is a function of that one quantity.",
        "k1 = -log r; k2 = (log r)^2 / 2; k3 = (r - 1) - log r with r = exp(log r).",
        "Each estimator is the mean over samples.",
    ],
    solution=(
        "def kl_estimators(logp, logq):\n"
        "    logr = logq - logp\n"
        "    r = logr.exp()\n"
        "    return {'k1': (-logr).mean(),\n"
        "            'k2': (logr.pow(2) / 2).mean(),\n"
        "            'k3': ((r - 1) - logr).mean()}\n"
    ),
    traps=[
        "Getting the direction of log r backwards, flipping the sign of k1.",
        "Expecting k1 to be non-negative on a finite sample — it can be negative.",
        "Using k2 where an unbiased estimate is required.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    logp = torch.log_softmax(torch.randn(2000), -1)
    logq = torch.log_softmax(torch.randn(2000), -1)
    out = fn(logp, logq)
    check("returns all three keys",
          lambda: set(out.keys()) == {"k1", "k2", "k3"})
    check("k3 is never negative", lambda: float(out["k3"]) >= -1e-6)
    check("k2 is never negative", lambda: float(out["k2"]) >= -1e-6)
    check("identical distributions give zero",
          lambda: all(abs(float(v)) < 1e-6 for v in fn(logp, logp.clone()).values()))

    def k1_unbiased():
        # under q == p the estimator is exactly zero; perturb and check k1 == -mean(logr)
        o = fn(logp, logq)
        return close(o["k1"], -(logq - logp).mean(), 1e-5)
    check("k1 equals -mean(log r)", k1_unbiased)

    def k3_formula():
        o = fn(logp, logq)
        logr = logq - logp
        return close(o["k3"], ((logr.exp() - 1) - logr).mean(), 1e-5)
    check("k3 matches (r-1) - log r", k3_formula)
''',
),

task(
    id="perplexity-bpb",
    title="Perplexity and bits-per-byte",
    chapter=CH,
    section="1.2 Perplexity and bits-per-byte",
    level=1,
    entry="report",
    statement=(
        "Convert a mean cross-entropy in nats into the two numbers people "
        "actually compare: perplexity, and bits-per-byte. Perplexity is "
        "exp(loss). Bits-per-byte converts to bits and rescales by the token-to-"
        "byte ratio, which is what makes it comparable across tokenisers."
    ),
    shapes="loss float (nats/token) · n_tokens int · n_bytes int  ->  dict 'ppl', 'bpb'",
    stub=("def report(loss, n_tokens, n_bytes):\n"
          "    # -> {'ppl': float, 'bpb': float}\n    pass\n"),
    hints=[
        "Perplexity is exp of the mean loss in nats.",
        "Bits per token is loss / ln 2.",
        "Bits per byte multiplies bits-per-token by tokens-per-byte, i.e. "
        "n_tokens / n_bytes.",
    ],
    solution=(
        "def report(loss, n_tokens, n_bytes):\n"
        "    return {'ppl': math.exp(loss),\n"
        "            'bpb': (loss / math.log(2)) * (n_tokens / n_bytes)}\n"
    ),
    traps=[
        "Using log2 of the loss instead of dividing by ln 2.",
        "Inverting the token/byte ratio, which makes a better tokeniser look worse.",
        "Comparing perplexity across different tokenisers — it is not comparable; "
        "that is the reason bits-per-byte exists.",
    ],
    tests='''
def checks(fn, check):
    out = fn(2.0, 1000, 4000)
    check("perplexity is exp(loss)", lambda: abs(out["ppl"] - math.exp(2.0)) < 1e-9)
    check("bpb uses tokens per byte",
          lambda: abs(out["bpb"] - (2.0 / math.log(2)) * 0.25) < 1e-9)
    check("zero loss gives perplexity 1", lambda: abs(fn(0.0, 10, 10)["ppl"] - 1.0) < 1e-9)
    check("one bit per byte at loss=ln2 and 1 token per byte",
          lambda: abs(fn(math.log(2), 100, 100)["bpb"] - 1.0) < 1e-9)
    check("bpb halves when a token covers twice the bytes",
          lambda: abs(fn(2.0, 500, 4000)["bpb"] - fn(2.0, 1000, 4000)["bpb"] / 2) < 1e-9)
''',
),

]

# ---------------------------------------------------------------------------
# NumPy reference solutions. A task only advertises the numpy backend once a
# reference exists here, so `tasks.selftest` can hold it to the same checks.
# cross-entropy stays torch-only: one of its checks calls .backward().
NUMPY = {
 "stable-softmax":
    "def softmax(x, dim=-1):\n"
    "    m = np.max(x, axis=dim, keepdims=True)\n"
    "    e = np.exp(x - m)\n"
    "    return e / e.sum(axis=dim, keepdims=True)\n",
 "log-softmax":
    "def log_softmax(x, dim=-1):\n"
    "    m = np.max(x, axis=dim, keepdims=True)\n"
    "    z = x - m\n"
    "    return z - np.log(np.exp(z).sum(axis=dim, keepdims=True))\n",
 "softmax-jacobian":
    "def softmax_jacobian(z):\n"
    "    e = np.exp(z - z.max())\n"
    "    s = e / e.sum()\n"
    "    return np.diag(s) - np.outer(s, s)\n",
 "kl-divergence":
    "def kl_estimators(logp, logq):\n"
    "    logr = logq - logp\n"
    "    r = np.exp(logr)\n"
    "    return {'k1': (-logr).mean(),\n"
    "            'k2': (logr ** 2 / 2).mean(),\n"
    "            'k3': ((r - 1) - logr).mean()}\n",
 "perplexity-bpb":
    "def report(loss, n_tokens, n_bytes):\n"
    "    return {'ppl': math.exp(loss),\n"
    "            'bpb': (loss / math.log(2)) * (n_tokens / n_bytes)}\n",
}

for _t in TASKS:
    if _t["id"] in NUMPY:
        _t["solution_np"] = NUMPY[_t["id"]]
        if "numpy" not in _t["frameworks"]:
            _t["frameworks"] = _t["frameworks"] + ["numpy"]
