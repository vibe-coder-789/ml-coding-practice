"""Chapters 7 and 9 — Inference · Data."""
from .schema import task

CH7 = "7 · Inference"
CH9 = "9 · Data"

TASKS = [

task(
    id="temperature",
    title="Temperature scaling",
    chapter=CH7,
    section="7.1 Decoding",
    level=1,
    entry="apply_temperature",
    statement=(
        "Scale logits by a temperature before sampling. As tau falls to 0 the "
        "distribution approaches the argmax; as it grows it approaches uniform.\n\n"
        "Return LOGITS at every temperature, never probabilities. That is what "
        "makes this composable: the result feeds straight into top-k, top-p, or a "
        "softmax, exactly like the unscaled logits did. So tau = 0 does not return "
        "a one-hot distribution — it returns logits whose softmax IS one-hot, "
        "which means the argmax entry sits far above every other. Handle tau = 0 "
        "as that greedy case rather than dividing by zero; callers pass it to mean "
        "deterministic decoding."
    ),
    shapes="logits (..., V) · tau float >= 0  ->  (..., V) logits, not probabilities",
    stub=("def apply_temperature(logits, tau):\n"
          "    # returns LOGITS. tau == 0 -> logits whose softmax is one-hot\n"
          "    pass\n"),
    hints=[
        "For tau > 0 this is a plain division.",
        "tau = 0 cannot be expressed by division; branch on it.",
        "The greedy branch must still return logits. Put a very negative but "
        "FINITE value everywhere except the argmax — torch.finfo(dtype).min — so "
        "that softmax of the result is one-hot. Returning the one-hot vector "
        "itself is the common wrong answer: softmax of [0, 1, 0] is "
        "[0.21, 0.58, 0.21], not [0, 1, 0].",
    ],
    solution=(
        "def apply_temperature(logits, tau):\n"
        "    if tau == 0:\n"
        "        out = torch.full_like(logits, torch.finfo(logits.dtype).min)\n"
        "        return out.scatter(-1, logits.argmax(-1, keepdim=True), 0.0)\n"
        "    return logits / tau\n"
    ),
    solution_np=(
        "def apply_temperature(logits, tau):\n"
        "    if tau == 0:\n"
        "        out = np.full_like(logits, np.finfo(logits.dtype).min)\n"
        "        idx = logits.argmax(-1, keepdims=True)\n"
        "        np.put_along_axis(out, idx, 0.0, axis=-1)\n"
        "        return out\n"
        "    return logits / tau\n"
    ),
    traps=[
        "Returning a one-hot probability vector at tau = 0. It looks right, but "
        "it is a distribution where every other temperature returns logits, and "
        "softmaxing it gives 0.58 on the argmax rather than 1.",
        "Dividing by zero and returning inf or NaN.",
        "Masking with -inf instead of a large finite value, which NaNs the "
        "softmax if the row is later fully masked.",
        "Applying temperature after the softmax instead of to the logits.",
    ],
    tests='''
def checks(fn, check):
    lg = torch.randn(3, 6)

    check("tau = 0 returns logits whose softmax is one-hot",
          lambda: close(torch.softmax(fn(lg, 0.0), -1).max(-1).values,
                        torch.ones(3), 1e-4))
    check("tau = 0 keeps the argmax",
          lambda: torch.softmax(fn(lg, 0.0), -1).argmax(-1).tolist() == lg.argmax(-1).tolist())
    check("tau = 0 stays finite (no -inf, no NaN)",
          lambda: bool(torch.isfinite(fn(lg, 0.0)).all()))
    check("tau = 1 is the identity", lambda: close(fn(lg, 1.0), lg))
    check("small tau concentrates on the argmax",
          lambda: torch.softmax(fn(lg, 0.01), -1).argmax(-1).tolist() == lg.argmax(-1).tolist())
    check("large tau approaches uniform",
          lambda: close(torch.softmax(fn(lg, 1e6), -1), torch.full_like(lg, 1/6), 1e-3))
    def not_mutated():
        c = lg.clone(); fn(lg, 2.0); return close(lg, c)
    check("input is not mutated", not_mutated)
''',
),

task(
    id="top-k",
    title="Top-k filtering",
    chapter=CH7,
    section="7.1 Decoding",
    level=2,
    entry="top_k_filter",
    statement=(
        "Mask all but the k highest logits, leaving the survivors unchanged so the "
        "result can be softmaxed and sampled. Masked entries must be finite, not "
        "-inf, so a later fully-masked row cannot turn the whole batch into NaN."
    ),
    shapes="logits (..., V) · k int  ->  (..., V)",
    stub="def top_k_filter(logits, k):\n    # keep the k largest per row, mask the rest\n    pass\n",
    hints=[
        "Find the k-th largest value per row; everything strictly below is masked.",
        "logits.topk(k, -1).values[..., -1:] is that threshold, kept as a column "
        "so it broadcasts.",
        "masked_fill with torch.finfo(logits.dtype).min.",
    ],
    solution=(
        "def top_k_filter(logits, k):\n"
        "    if k <= 0:\n"
        "        return logits\n"
        "    kth = logits.topk(k, dim=-1).values[..., -1:]\n"
        "    return logits.masked_fill(logits < kth, torch.finfo(logits.dtype).min)\n"
    ),
    solution_np=(
        "def top_k_filter(logits, k):\n"
        "    if k <= 0:\n"
        "        return logits\n"
        "    kth = np.sort(logits, axis=-1)[..., -k:][..., :1]\n"
        "    return np.where(logits < kth, np.finfo(logits.dtype).min, logits)\n"
    ),
    traps=[
        "Masking with -inf, which NaNs a fully masked row after softmax.",
        "Dropping the keepdim on the threshold so it broadcasts along the wrong axis.",
        "Using a strict > comparison that can drop ties and keep fewer than k.",
    ],
    tests='''
def checks(fn, check):
    lg = torch.randn(2, 10)
    MIN = torch.finfo(torch.float32).min
    check("keeps exactly k", lambda: int((fn(lg, 3) > MIN).sum(-1)[0]) == 3)
    check("keeps the largest k",
          lambda: close(fn(lg, 3).topk(3, -1).values, lg.topk(3, -1).values, 1e-5))
    check("masked entries stay finite", lambda: bool(torch.isfinite(fn(lg, 3)).all()))
    check("k >= V is a no-op", lambda: close(fn(lg, 10), lg))
    def survivors_unchanged():
        out = fn(lg, 3)
        keep = out > MIN
        return close(out[keep], lg[keep], 1e-5) and int(keep.sum()) == 6
    check("survivors keep their original values and positions", survivors_unchanged)
''',
),

task(
    id="top-p",
    title="Top-p (nucleus) filtering",
    chapter=CH7,
    section="7.1 Decoding",
    level=3,
    entry="top_p_filter",
    statement=(
        "Keep the smallest set of tokens whose cumulative probability reaches p "
        "and mask the rest, returning logits in the original vocabulary order. The "
        "first token must never be dropped, even when its own probability already "
        "exceeds p — otherwise a peaked distribution leaves nothing to sample from."
    ),
    shapes="logits (..., V) · p float in (0, 1]  ->  (..., V)",
    stub="def top_p_filter(logits, p):\n    # nucleus filtering, original order preserved\n    pass\n",
    hints=[
        "Sort descending, softmax the sorted logits, take a cumulative sum.",
        "Test the mass strictly BEFORE each token: (cum - probs) >= p. Testing "
        "cum >= p drops the first token on a peaked distribution.",
        "Undo the sort by scattering back with the sort indices.",
    ],
    solution=(
        "def top_p_filter(logits, p):\n"
        "    srt, idx = logits.sort(dim=-1, descending=True)\n"
        "    probs = torch.softmax(srt, dim=-1)\n"
        "    cum = probs.cumsum(dim=-1)\n"
        "    drop = (cum - probs) >= p\n"
        "    srt = srt.masked_fill(drop, torch.finfo(logits.dtype).min)\n"
        "    return srt.scatter(-1, idx, srt)\n"
    ),
    solution_np=(
        "def top_p_filter(logits, p):\n"
        "    idx = np.argsort(-logits, axis=-1)\n"
        "    srt = np.take_along_axis(logits, idx, axis=-1)\n"
        "    e = np.exp(srt - srt.max(-1, keepdims=True))\n"
        "    probs = e / e.sum(-1, keepdims=True)\n"
        "    cum = probs.cumsum(-1)\n"
        "    srt = np.where((cum - probs) >= p, np.finfo(logits.dtype).min, srt)\n"
        "    out = np.empty_like(srt)\n"
        "    np.put_along_axis(out, idx, srt, axis=-1)\n"
        "    return out\n"
    ),
    traps=[
        "Using cum >= p, which can mask every token.",
        "Forgetting to invert the sort, returning logits in sorted order.",
        "Masking with -inf rather than finfo.min.",
    ],
    tests='''
def checks(fn, check):
    lg = torch.randn(4, 10)
    MIN = torch.finfo(torch.float32).min
    check("always keeps at least one", lambda: bool(((fn(lg, 0.9) > MIN).sum(-1) >= 1).all()))
    check("keeps exactly one on a peaked distribution",
          lambda: int((fn(torch.tensor([[10., 0., 0., 0.]]), 0.5) > MIN).sum()) == 1)
    check("p = 1 keeps everything", lambda: close(fn(lg, 1.0), lg, 1e-4))
    check("larger p keeps at least as many",
          lambda: int((fn(lg, 0.5) > MIN).sum()) <= int((fn(lg, 0.95) > MIN).sum()))
    def order_preserved():
        out = fn(lg, 0.9)
        keep = out > MIN
        return close(out[keep], lg[keep], 1e-5)
    check("surviving logits keep their vocabulary positions", order_preserved)
    check("masked entries are finite, not -inf",
          lambda: bool(torch.isfinite(fn(lg, 0.5)).all()))
''',
),

task(
    id="speculative-accept",
    title="Speculative decoding acceptance",
    chapter=CH7,
    section="7.2 Speculative decoding",
    level=3,
    entry="accept",
    statement=(
        "Implement one speculative-decoding acceptance test. Given the target "
        "distribution p, the draft distribution q, the drafted token and a uniform "
        "random u, accept the token when u <= p[token]/q[token]; otherwise reject "
        "and resample from the normalised residual max(p - q, 0). This exact rule "
        "is what makes the output distribution identical to sampling from p — the "
        "speedup is free of any quality cost."
    ),
    shapes=("p (V,) · q (V,) probabilities · token int · u float in [0,1)"
            "  ->  dict 'accepted' (bool), 'token' (int)"),
    stub=("def accept(p, q, token, u):\n"
          "    # -> {'accepted': bool, 'token': int}\n    pass\n"),
    hints=[
        "The acceptance probability is min(1, p[token]/q[token]); compare u to it.",
        "On rejection the replacement comes from the residual max(p - q, 0), "
        "renormalised to sum to 1.",
        "Return the drafted token when accepted, the resampled one when not.",
    ],
    solution=(
        "def accept(p, q, token, u):\n"
        "    ratio = float(p[token] / q[token]) if float(q[token]) > 0 else 1.0\n"
        "    if u <= min(1.0, ratio):\n"
        "        return {'accepted': True, 'token': int(token)}\n"
        "    resid = torch.clamp(p - q, min=0)\n"
        "    total = float(resid.sum())\n"
        "    resid = resid / total if total > 0 else torch.ones_like(p) / p.numel()\n"
        "    return {'accepted': False, 'token': int(torch.multinomial(resid, 1))}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Resampling from p instead of the residual, which biases the output "
        "distribution away from p.",
        "Forgetting to renormalise the residual before sampling.",
        "Clamping the ratio but not the residual, leaving negative probabilities.",
    ],
    tests='''
def checks(fn, check):
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.2, 0.3, 0.5])
    check("accepts when the target likes the token more",
          lambda: fn(p, q, 0, 0.9)["accepted"] is True)
    check("accepts when the ratio is exactly 1",
          lambda: fn(p, q, 1, 0.999)["accepted"] is True)
    check("rejects when u exceeds the ratio",
          lambda: fn(p, q, 2, 0.9)["accepted"] is False)
    check("an accepted result returns the drafted token",
          lambda: fn(p, q, 0, 0.1)["token"] == 0)
    def resample_from_residual():
        # residual is max(p-q,0) = [0.3, 0, 0] -> normalised all mass on token 0
        outs = {fn(p, q, 2, 0.99)["token"] for _ in range(20)}
        return outs == {0}
    check("rejection resamples from the normalised residual", resample_from_residual)
    def distribution_is_exact():
        # empirically: accepted+resampled tokens follow p
        counts = torch.zeros(3)
        torch.manual_seed(1)
        for _ in range(20000):
            tok = int(torch.multinomial(q, 1))
            r = fn(p, q, tok, float(torch.rand(1)))
            counts[r["token"]] += 1
        return close(counts / counts.sum(), p, 0.02)
    check("the accept/reject rule reproduces p exactly", distribution_is_exact)
''',
),

# ------------------------------------------------------------------ chapter 9
task(
    id="minhash",
    title="MinHash signature",
    chapter=CH9,
    section="9.1 Near-duplicate detection: MinHash and LSH",
    level=2,
    entry="signature",
    statement=(
        "Build a MinHash signature: for each of the given hash seeds, return the "
        "minimum hash value over the set's elements. Two sets agree at a given "
        "position with probability exactly equal to their Jaccard similarity, "
        "which is what turns an expensive set comparison into a cheap vector "
        "comparison for deduplicating a training corpus."
    ),
    shapes="items (iterable of str) · seeds (K,) int  ->  (K,) int signature",
    stub=("def signature(items, seeds):\n"
          "    # -> one minimum hash per seed\n    pass\n"),
    hints=[
        "For each seed, hash every item with that seed and keep the minimum.",
        "Use a deterministic hash: the provided helper `h(item, seed)` is stable "
        "across processes, unlike Python's built-in hash for strings.",
        "Return the K minima as an array, in seed order.",
    ],
    solution=(
        "def signature(items, seeds):\n"
        "    items = list(items)\n"
        "    return np.array([min(h(it, int(s)) for it in items) for s in seeds],\n"
        "                    dtype=np.int64)\n"
    ),
    solution_np=(
        "def signature(items, seeds):\n"
        "    items = list(items)\n"
        "    return np.array([min(h(it, int(s)) for it in items) for s in seeds],\n"
        "                    dtype=np.int64)\n"
    ),
    extra=(
        "import hashlib\n"
        "def h(item, seed):\n"
        "    \"\"\"Stable 32-bit hash of a string under a seed.\"\"\"\n"
        "    d = hashlib.blake2b(f'{seed}:{item}'.encode(), digest_size=8).digest()\n"
        "    return int.from_bytes(d, 'big') & 0xFFFFFFFF\n"
    ),
    traps=[
        "Using Python's built-in hash(), which is salted per process, so "
        "signatures do not compare across runs.",
        "Taking the minimum over seeds instead of over items.",
        "Returning the item that hashed lowest rather than the hash value.",
    ],
    tests='''
def checks(fn, check):
    seeds = np.arange(16)
    a = ["the", "quick", "brown", "fox"]
    sig_a = fn(a, seeds)
    check("one value per seed", lambda: len(np.asarray(sig_a)) == 16)
    check("order of items does not matter",
          lambda: close(np.asarray(fn(list(reversed(a)), seeds)), np.asarray(sig_a)))
    check("identical sets give identical signatures",
          lambda: close(np.asarray(fn(a, seeds)), np.asarray(sig_a)))
    check("each entry is a minimum over items",
          lambda: int(np.asarray(sig_a)[0]) == min(h(x, 0) for x in a))
    def estimates_jaccard():
        b = ["the", "quick", "brown", "dog"]          # true Jaccard = 3/5
        agree = (np.asarray(fn(a, np.arange(400))) ==
                 np.asarray(fn(b, np.arange(400)))).mean()
        return abs(agree - 0.6) < 0.10
    check("agreement rate estimates the Jaccard similarity", estimates_jaccard)
''',
),

task(
    id="lsh-bands",
    title="LSH banding",
    chapter=CH9,
    section="9.1 Near-duplicate detection: MinHash and LSH",
    level=3,
    entry="candidates",
    statement=(
        "Given MinHash signatures, split each into b bands of r rows and return "
        "the set of candidate duplicate pairs — those agreeing on at least one "
        "whole band. Banding turns similarity search into hashing: the "
        "probability a pair becomes a candidate is 1-(1-s^r)^b, an S-curve whose "
        "threshold you tune by choosing b and r."
    ),
    shapes="sigs (N, K) int · b int (bands) · r int (rows), b·r == K  ->  set of (i, j) with i < j",
    stub=("def candidates(sigs, b, r):\n"
          "    # -> {(i, j), ...} pairs sharing at least one identical band\n    pass\n"),
    hints=[
        "Reshape or slice each signature into b contiguous blocks of r values.",
        "Bucket by (band index, the band's contents as a tuple) — a shared bucket "
        "means a shared band.",
        "Emit every pair within each bucket, ordered i < j, deduplicated by using "
        "a set.",
    ],
    solution=(
        "def candidates(sigs, b, r):\n"
        "    sigs = np.asarray(sigs)\n"
        "    buckets = {}\n"
        "    out = set()\n"
        "    for i, sig in enumerate(sigs):\n"
        "        for bi in range(b):\n"
        "            key = (bi, tuple(int(v) for v in sig[bi * r:(bi + 1) * r]))\n"
        "            buckets.setdefault(key, []).append(i)\n"
        "    for members in buckets.values():\n"
        "        for x in range(len(members)):\n"
        "            for y in range(x + 1, len(members)):\n"
        "                out.add((members[x], members[y]))\n"
        "    return out\n"
    ),
    solution_np=(
        "def candidates(sigs, b, r):\n"
        "    sigs = np.asarray(sigs)\n"
        "    buckets = {}\n"
        "    out = set()\n"
        "    for i, sig in enumerate(sigs):\n"
        "        for bi in range(b):\n"
        "            key = (bi, tuple(int(v) for v in sig[bi * r:(bi + 1) * r]))\n"
        "            buckets.setdefault(key, []).append(i)\n"
        "    for members in buckets.values():\n"
        "        for x in range(len(members)):\n"
        "            for y in range(x + 1, len(members)):\n"
        "                out.add((members[x], members[y]))\n"
        "    return out\n"
    ),
    traps=[
        "Requiring every band to match, which is exact duplication, not "
        "near-duplication.",
        "Forgetting the band index in the bucket key, so identical contents in "
        "different bands collide spuriously.",
        "Emitting both (i, j) and (j, i), or a pair with itself.",
    ],
    tests='''
def checks(fn, check):
    sigs = np.array([
        [1, 2, 3, 4, 5, 6],      # 0
        [1, 2, 9, 9, 9, 9],      # 1: shares band 0 with row 0
        [7, 7, 7, 7, 5, 6],      # 2: shares band 2 with row 0
        [0, 0, 0, 0, 0, 0],      # 3: shares nothing
    ])
    out = fn(sigs, 3, 2)
    check("finds a pair sharing the first band", lambda: (0, 1) in out)
    check("finds a pair sharing a later band", lambda: (0, 2) in out)
    check("excludes a row sharing no band",
          lambda: not any(3 in pair for pair in out))
    check("pairs are ordered and not self-paired",
          lambda: all(i < j for i, j in out))
    check("identical signatures are candidates",
          lambda: (0, 1) in fn(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), 2, 2))
    check("one band per row is the strictest setting",
          lambda: len(fn(sigs, 1, 6)) == 0)
''',
),

]
