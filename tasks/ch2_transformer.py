"""Chapter 2 — The transformer, term by term."""
from .schema import task

CH = "2 · The transformer, term by term"

TASKS = [

task(
    id="sdpa",
    title="Scaled dot-product attention",
    chapter=CH,
    section="2.2 Attention, and where sqrt(d) comes from",
    level=2,
    entry="sdpa",
    statement=(
        "Implement attention over batched multi-head tensors, unmasked. Must "
        "match F.scaled_dot_product_attention. Get the scale right: a dot product "
        "of two d-dimensional vectors with variance s^2 per entry has variance "
        "d·s^4, so without the 1/sqrt(d) the softmax saturates as heads widen and "
        "the gradient through it vanishes."
    ),
    shapes="q (B,H,L,Dh) · k (B,H,S,Dh) · v (B,H,S,Dh)  ->  (B,H,L,Dh)",
    stub="def sdpa(q, k, v):\n    # q (B,H,L,Dh), k/v (B,H,S,Dh) -> (B,H,L,Dh)\n    pass\n",
    hints=[
        "Scores contract q against k over the head dimension, giving (B,H,L,S).",
        "Transpose only the last two axes of k — .mT, not .T, which reverses all four.",
        "Divide by sqrt(Dh) before the softmax, then weight v by the result.",
    ],
    solution=(
        "def sdpa(q, k, v):\n"
        "    d_h = q.shape[-1]\n"
        "    scores = q @ k.mT / math.sqrt(d_h)\n"
        "    return torch.softmax(scores, -1) @ v\n"
    ),
    solution_np=(
        "def sdpa(q, k, v):\n"
        "    d_h = q.shape[-1]\n"
        "    scores = q @ np.swapaxes(k, -2, -1) / math.sqrt(d_h)\n"
        "    e = np.exp(scores - scores.max(-1, keepdims=True))\n"
        "    return (e / e.sum(-1, keepdims=True)) @ v\n"
    ),
    traps=[
        "Using .T on a 4-D tensor; it reverses every axis and is deprecated.",
        "Dividing by Dh instead of its square root.",
        "Softmaxing over the query axis instead of the key axis.",
    ],
    tests='''
def checks(fn, check):
    q, k, v = (torch.randn(2, 3, 7, 16) for _ in range(3))
    check("matches F.scaled_dot_product_attention",
          lambda: close(fn(q, k, v), F.scaled_dot_product_attention(q, k, v), 1e-4))
    check("handles S != L",
          lambda: shape(fn(torch.randn(1,2,3,8), torch.randn(1,2,5,8),
                           torch.randn(1,2,5,8))) == (1,2,3,8))
    check("weights form a convex combination",
          lambda: close(fn(torch.randn(1,1,4,8), torch.randn(1,1,4,8),
                           torch.ones(1,1,4,1)), torch.ones(1,1,4,1), 1e-4))
    def scale_is_right():
        qq, kk, vv = (torch.randn(1,1,4,64) for _ in range(3))
        want = torch.softmax(qq @ kk.mT / math.sqrt(64), -1) @ vv
        return close(fn(qq, kk, vv), want, 1e-4)
    check("scale is 1/sqrt(Dh), not 1/Dh", scale_is_right)
''',
),

task(
    id="causal-mask",
    title="Causal masking",
    chapter=CH,
    section="2.2 Attention, and where sqrt(d) comes from",
    level=3,
    entry="sdpa_causal",
    statement=(
        "Add a causal mask so query i attends only to keys j <= i. Must match "
        "F.scaled_dot_product_attention(..., is_causal=True). Two details decide "
        "correctness: the value you mask with, and how the triangle is offset "
        "when S != L, which is the incremental-decoding case."
    ),
    shapes="q (B,H,L,Dh) · k (B,H,S,Dh) · v (B,H,S,Dh)  ->  (B,H,L,Dh)",
    stub="def sdpa_causal(q, k, v):\n    # causal: query i sees keys j <= i\n    pass\n",
    hints=[
        "Build a (L,S) boolean triangle of positions to keep, then fill the rest "
        "of the scores before the softmax.",
        "Mask with torch.finfo(dtype).min, not -inf: a fully masked row of -inf "
        "softmaxes to NaN, and padding-only rows do occur.",
        "When S != L the triangle must be offset: torch.ones(L,S).tril(diagonal=S-L).",
    ],
    solution=(
        "def sdpa_causal(q, k, v):\n"
        "    d_h = q.shape[-1]\n"
        "    scores = q @ k.mT / math.sqrt(d_h)\n"
        "    L, S = q.shape[-2], k.shape[-2]\n"
        "    keep = torch.ones(L, S, dtype=torch.bool).tril(diagonal=S - L)\n"
        "    scores = scores.masked_fill(~keep, torch.finfo(scores.dtype).min)\n"
        "    return torch.softmax(scores, -1) @ v\n"
    ),
    solution_np=(
        "def sdpa_causal(q, k, v):\n"
        "    d_h = q.shape[-1]\n"
        "    scores = q @ np.swapaxes(k, -2, -1) / math.sqrt(d_h)\n"
        "    L, S = q.shape[-2], k.shape[-2]\n"
        "    keep = np.tril(np.ones((L, S), dtype=bool), k=S - L)\n"
        "    scores = np.where(keep, scores, np.finfo(scores.dtype).min)\n"
        "    e = np.exp(scores - scores.max(-1, keepdims=True))\n"
        "    return (e / e.sum(-1, keepdims=True)) @ v\n"
    ),
    traps=[
        "Masking with -inf, which yields NaN on a fully masked row.",
        "Omitting the diagonal offset, which breaks the moment L != S.",
        "Masking after the softmax, leaving the weights unnormalised.",
    ],
    tests='''
def checks(fn, check):
    q, k, v = (torch.randn(2, 3, 7, 16) for _ in range(3))
    check("matches torch's causal attention",
          lambda: close(fn(q, k, v),
                        F.scaled_dot_product_attention(q, k, v, is_causal=True), 1e-4))
    def future_cannot_leak():
        v2 = v.clone(); v2[:, :, -1] += 10.
        return close(fn(q, k, v)[:, :, :-1], fn(q, k, v2)[:, :, :-1], 1e-4)
    check("a later value cannot change an earlier output", future_cannot_leak)
    check("first position attends only to itself",
          lambda: close(fn(torch.randn(1,1,4,8), torch.randn(1,1,4,8),
                           v[:1,:1,:4,:8])[:, :, 0], v[:1,:1,0,:8], 1e-4))
    check("no NaN anywhere", lambda: bool(torch.isfinite(fn(q, k, v)).all()))
    def offset_for_decode():
        # one query against a 5-long cache must attend to everything
        qq = torch.randn(1,1,1,8); kk = torch.randn(1,1,5,8); vv = torch.randn(1,1,5,8)
        want = torch.softmax(qq @ kk.mT / math.sqrt(8), -1) @ vv
        return close(fn(qq, kk, vv), want, 1e-4)
    check("L=1 against a cache keeps the whole row", offset_for_decode)
''',
),

task(
    id="split-heads",
    title="Split activations into heads",
    chapter=CH,
    section="2.2 Attention, and where sqrt(d) comes from",
    level=1,
    entry="split_heads",
    statement=(
        "Reshape a (B,L,D) activation into H independent heads of width D/H, laid "
        "out as (B,H,L,Dh). The order of view and transpose matters: reversing "
        "them interleaves features belonging to different heads, which produces "
        "the right shape and the wrong model."
    ),
    shapes="x (B, L, D) · n_heads int  ->  (B, H, L, D//H)",
    stub="def split_heads(x, n_heads):\n    # (B, L, D) -> (B, H, L, Dh)\n    pass\n",
    hints=[
        "First expose the head axis by viewing D as (H, Dh), giving (B,L,H,Dh).",
        "Then move the head axis next to the batch with a transpose.",
        "view before transpose — never the reverse.",
    ],
    solution=(
        "def split_heads(x, n_heads):\n"
        "    B, L, D = x.shape\n"
        "    return x.view(B, L, n_heads, D // n_heads).transpose(1, 2)\n"
    ),
    solution_np=(
        "def split_heads(x, n_heads):\n"
        "    B, L, D = x.shape\n"
        "    return x.reshape(B, L, n_heads, D // n_heads).swapaxes(1, 2)\n"
    ),
    traps=[
        "Transposing first, then viewing — silently interleaves heads.",
        "Calling view on a non-contiguous tensor without contiguous().",
        "Assuming the inverse needs no contiguous(): merging a freshly "
        "materialised attention output does.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(2, 5, 12)
    check("output shape", lambda: shape(fn(x, 4)) == (2, 4, 5, 3))
    check("head 0 holds the first Dh features", lambda: close(fn(x, 4)[:, 0], x[:, :, :3]))
    check("head 1 holds the next Dh features", lambda: close(fn(x, 4)[:, 1], x[:, :, 3:6]))
    check("round-trips back",
          lambda: close(fn(x, 4).transpose(1, 2).reshape(2, 5, 12), x))
    check("works with a single head", lambda: close(fn(x, 1)[:, 0], x))
''',
),

task(
    id="gqa",
    title="Grouped-query attention",
    chapter=CH,
    section="2.3 The KV-cache problem: MHA → MQA → GQA → MLA",
    level=2,
    entry="expand_kv",
    statement=(
        "Share each key/value head across a group of query heads, expanding "
        "(B,H_kv,S,Dh) to (B,H_q,S,Dh). Query heads 0..g-1 map to kv head 0, the "
        "next g to kv head 1, and so on. This is what shrinks the KV cache by "
        "H/H_kv. Choosing the wrong repeat gives the right shape and the wrong "
        "grouping, with no error."
    ),
    shapes="kv (B, H_kv, S, Dh) · n_q int  ->  (B, n_q, S, Dh)",
    stub="def expand_kv(kv, n_q):\n    # (B, H_kv, S, Dh) -> (B, n_q, S, Dh)\n    pass\n",
    hints=[
        "Each kv head serves n_q // H_kv adjacent query heads.",
        "repeat gives [0,1,0,1,...]; repeat_interleave gives [0,0,1,1,...].",
        "kv.repeat_interleave(n_q // kv.shape[1], dim=1)",
    ],
    solution=(
        "def expand_kv(kv, n_q):\n"
        "    return kv.repeat_interleave(n_q // kv.shape[1], dim=1)\n"
    ),
    solution_np=(
        "def expand_kv(kv, n_q):\n"
        "    return np.repeat(kv, n_q // kv.shape[1], axis=1)\n"
    ),
    traps=[
        "Using repeat instead of repeat_interleave — correct shape, wrong grouping.",
        "Trying to use expand, which cannot produce this layout.",
        "Not handling H_kv = 1 (multi-query) or H_kv = H (plain multi-head).",
    ],
    tests='''
def checks(fn, check):
    kv = torch.randn(2, 2, 5, 8)
    check("output shape", lambda: shape(fn(kv, 6)) == (2, 6, 5, 8))
    check("adjacent query heads share a kv head",
          lambda: close(fn(kv, 6)[:, 0], fn(kv, 6)[:, 1]) and close(fn(kv, 6)[:, 1], fn(kv, 6)[:, 2]))
    check("different groups differ", lambda: not close(fn(kv, 6)[:, 0], fn(kv, 6)[:, 3]))
    check("group 1 is kv head 1", lambda: close(fn(kv, 6)[:, 3], kv[:, 1]))
    mq = torch.randn(1, 1, 4, 8)
    check("H_kv = 1 is multi-query: every head shares the one kv head",
          lambda: close(fn(mq, 4)[:, 0], fn(mq, 4)[:, 3]))
    mh = torch.randn(1, 3, 4, 8)
    check("H_kv = H is a no-op", lambda: close(fn(mh, 3), mh))
''',
),

task(
    id="kv-cache",
    title="KV cache decode step",
    chapter=CH,
    section="2.3 The KV-cache problem: MHA → MQA → GQA → MLA",
    level=3,
    entry="decode_step",
    statement=(
        "Implement one incremental decoding step: append the new key and value to "
        "the cache, attend the single new query over everything cached, and return "
        "the output alongside the grown cache. Stepped over a whole sequence this "
        "must reproduce the parallel causal forward exactly — that equivalence is "
        "the test that catches almost every generation bug."
    ),
    shapes=("q_t (B,H,1,Dh) · k_cache/v_cache (B,H,t,Dh) · k_t/v_t (B,H,1,Dh)"
            "  ->  (out (B,H,1,Dh), k_cache (B,H,t+1,Dh), v_cache (B,H,t+1,Dh))"),
    stub=("def decode_step(q_t, k_cache, v_cache, k_t, v_t):\n"
          "    # -> (out, k_cache, v_cache)\n    pass\n"),
    hints=[
        "Append along the sequence axis, which is dim=2 for (B,H,S,Dh).",
        "No causal mask is needed inside the loop: the cache holds only positions "
        "up to now, so the constraint is enforced by what is present.",
        "Return all three values — the caller threads the grown cache onward.",
    ],
    solution=(
        "def decode_step(q_t, k_cache, v_cache, k_t, v_t):\n"
        "    k_cache = torch.cat([k_cache, k_t], dim=2)\n"
        "    v_cache = torch.cat([v_cache, v_t], dim=2)\n"
        "    d_h = q_t.shape[-1]\n"
        "    scores = q_t @ k_cache.mT / math.sqrt(d_h)\n"
        "    out = torch.softmax(scores, -1) @ v_cache\n"
        "    return out, k_cache, v_cache\n"
    ),
    solution_np=(
        "def decode_step(q_t, k_cache, v_cache, k_t, v_t):\n"
        "    k_cache = np.concatenate([k_cache, k_t], axis=2)\n"
        "    v_cache = np.concatenate([v_cache, v_t], axis=2)\n"
        "    d_h = q_t.shape[-1]\n"
        "    scores = q_t @ np.swapaxes(k_cache, -2, -1) / math.sqrt(d_h)\n"
        "    e = np.exp(scores - scores.max(-1, keepdims=True))\n"
        "    out = (e / e.sum(-1, keepdims=True)) @ v_cache\n"
        "    return out, k_cache, v_cache\n"
    ),
    traps=[
        "Concatenating along the wrong axis — dim=1 would stack heads.",
        "Applying a causal mask anyway, which masks out the current token.",
        "Recomputing keys and values for the whole prefix, defeating the cache.",
    ],
    tests='''
def checks(fn, check):
    B, H, L, Dh = 1, 2, 6, 8
    q, k, v = (torch.randn(B, H, L, Dh) for _ in range(3))
    z = torch.zeros(B, H, 0, Dh)

    check("returns three values",
          lambda: len(fn(q[:, :, :1], z, z, k[:, :, :1], v[:, :, :1])) == 3)
    check("cache grows by one along the sequence axis",
          lambda: shape(fn(q[:, :, :1], z, z, k[:, :, :1], v[:, :, :1])[1]) == (B, H, 1, Dh))
    check("first step returns the first value",
          lambda: close(fn(q[:, :, :1], z, z, k[:, :, :1], v[:, :, :1])[0], v[:, :, :1], 1e-4))

    def matches_parallel():
        kc, vc, outs = z, z, []
        for t in range(L):
            o, kc, vc = fn(q[:, :, t:t+1], kc, vc, k[:, :, t:t+1], v[:, :, t:t+1])
            outs.append(o)
        want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return close(torch.cat(outs, dim=2), want, 1e-4)
    check("stepping the sequence equals the parallel causal forward", matches_parallel)

    def cache_is_correct():
        kc, vc = z, z
        for t in range(L):
            _, kc, vc = fn(q[:, :, t:t+1], kc, vc, k[:, :, t:t+1], v[:, :, t:t+1])
        return close(kc, k, 1e-5) and close(vc, v, 1e-5)
    check("the accumulated cache equals the full k and v", cache_is_correct)
''',
),

task(
    id="online-softmax",
    title="Online softmax (the FlashAttention identity)",
    chapter=CH,
    section="2.4 Online softmax: the identity behind FlashAttention",
    level=3,
    entry="online_softmax",
    statement=(
        "Compute a softmax-weighted sum in a single streaming pass over blocks, "
        "never materialising the full score row. Maintain a running max m, a "
        "running denominator l, and a running output o; on each block, rescale "
        "what you have by exp(m_old - m_new) and fold in the new terms. This "
        "identity is what lets FlashAttention avoid an (L,S) matrix in memory."
    ),
    shapes="scores (N,) float · values (N, D) float · block int  ->  (D,) float",
    stub=("def online_softmax(scores, values, block=4):\n"
          "    # streaming softmax(scores) @ values, one block at a time\n    pass\n"),
    hints=[
        "Track m (running max), l (running sum of exp), o (running weighted sum).",
        "On a new block with max mb: m_new = max(m, mb); the correction factor for "
        "everything accumulated so far is exp(m - m_new).",
        "l_new = l*corr + sum(exp(s_block - m_new)); "
        "o_new = o*corr + exp(s_block - m_new) @ v_block. Divide o by l only at the end.",
    ],
    solution=(
        "def online_softmax(scores, values, block=4):\n"
        "    N, D = values.shape\n"
        "    m = torch.tensor(float('-inf'))\n"
        "    l = torch.zeros(())\n"
        "    o = torch.zeros(D)\n"
        "    for i in range(0, N, block):\n"
        "        s = scores[i:i+block]\n"
        "        v = values[i:i+block]\n"
        "        m_new = torch.maximum(m, s.max())\n"
        "        corr = torch.exp(m - m_new) if torch.isfinite(m) else torch.zeros(())\n"
        "        w = torch.exp(s - m_new)\n"
        "        l = l * corr + w.sum()\n"
        "        o = o * corr + w @ v\n"
        "        m = m_new\n"
        "    return o / l\n"
    ),
    solution_np=(
        "def online_softmax(scores, values, block=4):\n"
        "    N, D = values.shape\n"
        "    m, l, o = -np.inf, 0.0, np.zeros(D)\n"
        "    for i in range(0, N, block):\n"
        "        s = scores[i:i+block]\n"
        "        v = values[i:i+block]\n"
        "        m_new = max(m, float(s.max()))\n"
        "        corr = np.exp(m - m_new) if np.isfinite(m) else 0.0\n"
        "        w = np.exp(s - m_new)\n"
        "        l = l * corr + w.sum()\n"
        "        o = o * corr + w @ v\n"
        "        m = m_new\n"
        "    return o / l\n"
    ),
    traps=[
        "Normalising inside the loop instead of once at the end.",
        "Forgetting to rescale the accumulated output when the max moves.",
        "Starting m at 0 rather than -inf, which biases the first block.",
    ],
    tests='''
def checks(fn, check):
    s = torch.randn(17)
    v = torch.randn(17, 5)
    want = torch.softmax(s, -1) @ v
    check("matches the one-shot softmax", lambda: close(fn(s, v, 4), want, 1e-4))
    check("independent of block size",
          lambda: close(fn(s, v, 1), fn(s, v, 16), 1e-4))
    check("block larger than N still works", lambda: close(fn(s, v, 100), want, 1e-4))
    big = torch.tensor([0., 1000., 2., 3.])
    check("stable against a huge score",
          lambda: close(fn(big, torch.ones(4, 2), 2), torch.ones(2), 1e-4))
    check("output shape", lambda: shape(fn(s, v, 4)) == (5,))
''',
),

task(
    id="rope",
    title="Rotary position embedding",
    chapter=CH,
    section="2.6 Rotary position embeddings",
    level=3,
    entry="rope",
    statement=(
        "Rotate each consecutive coordinate pair (2t, 2t+1) of x by an angle "
        "proportional to its position, given precomputed cos and sin of shape "
        "(L, Dh/2). The property that must hold is that the inner product of two "
        "rotated vectors depends only on their offset — absolute position goes in, "
        "relative position is what the attention score sees."
    ),
    shapes="x (..., L, Dh) · cos (L, Dh/2) · sin (L, Dh/2)  ->  (..., L, Dh)",
    stub="def rope(x, cos, sin):\n    # pairs (2t, 2t+1); cos/sin are (L, Dh/2)\n    pass\n",
    hints=[
        "Split into even and odd coordinates: x[..., 0::2] and x[..., 1::2].",
        "Apply the plane rotation: o1 = x1*cos - x2*sin, o2 = x1*sin + x2*cos.",
        "Re-interleave by stacking on a new last axis and flattening the last two.",
    ],
    solution=(
        "def rope(x, cos, sin):\n"
        "    x1, x2 = x[..., 0::2], x[..., 1::2]\n"
        "    o1 = x1 * cos - x2 * sin\n"
        "    o2 = x1 * sin + x2 * cos\n"
        "    return torch.stack([o1, o2], dim=-1).flatten(-2)\n"
    ),
    solution_np=(
        "def rope(x, cos, sin):\n"
        "    x1, x2 = x[..., 0::2], x[..., 1::2]\n"
        "    o1 = x1 * cos - x2 * sin\n"
        "    o2 = x1 * sin + x2 * cos\n"
        "    return np.stack([o1, o2], axis=-1).reshape(x.shape)\n"
    ),
    traps=[
        "Concatenating instead of interleaving, producing the half-split (Llama/HF) "
        "layout — also valid, but not the same, and mixing the two with a trained "
        "checkpoint degrades output with no error.",
        "Rotating v as well as q and k.",
        "Forgetting that cos and sin are (L, Dh/2), not (L, Dh).",
    ],
    extra=(
        "def _angles(L, d_h, base=10000.0):\n"
        "    t = torch.arange(d_h // 2, dtype=torch.float32)\n"
        "    theta = base ** (-2 * t / d_h)\n"
        "    pos = torch.arange(L, dtype=torch.float32)\n"
        "    ang = pos[:, None] * theta[None, :]\n"
        "    return ang.cos(), ang.sin()\n"
    ),
    tests='''
def checks(fn, check):
    L, Dh = 5, 8
    cos, sin = _angles(L, Dh)
    x = torch.randn(1, 1, L, Dh)
    check("shape is preserved", lambda: shape(fn(x, cos, sin)) == (1, 1, L, Dh))
    check("position 0 is the identity", lambda: close(fn(x, cos, sin)[:, :, 0], x[:, :, 0], 1e-5))
    check("norm is preserved (rotations are orthogonal)",
          lambda: close(fn(x, cos, sin).norm(dim=-1), x.norm(dim=-1), 1e-4))

    def relative():
        q, k = torch.randn(1,1,L,Dh), torch.randn(1,1,L,Dh)
        qr, kr = fn(q, cos, sin), fn(k, cos, sin)
        ip = torch.einsum('bhid,bhjd->bhij', qr, kr)[0, 0]
        t = torch.arange(Dh // 2, dtype=torch.float32)
        theta = 10000.0 ** (-2 * t / Dh)
        for i in range(L):
            for j in range(L):
                a = (j - i) * theta
                kj = fn(k[:, :, j:j+1], a.cos()[None], a.sin()[None])
                if abs(ip[i, j].item() - (q[0,0,i] * kj[0,0,0]).sum().item()) > 1e-3:
                    return False
        return True
    check("inner product depends only on the offset", relative)

    def exact_rotation():
        # x1=1, x2=0 in every pair: the rotation must return (cos, sin) exactly.
        # An identity function returns (1, 0) and fails; so does the half-split
        # layout, whose pairing differs.
        x = torch.zeros(1, 1, L, Dh)
        x[..., 0::2] = 1.0
        out = fn(x, cos, sin)
        return close(out[0, 0, :, 0::2], cos, 1e-5) and close(out[0, 0, :, 1::2], sin, 1e-5)
    check("rotates by exactly the given angles", exact_rotation)

    check("is the interleaved layout, not the half-split one",
          lambda: not close(fn(x, cos, sin),
                            x * torch.cat([cos, cos], -1) +
                            torch.cat([-x.chunk(2, -1)[1], x.chunk(2, -1)[0]], -1)
                            * torch.cat([sin, sin], -1), 1e-3))
''',
),

task(
    id="rms-norm",
    title="RMSNorm",
    chapter=CH,
    section="2.7 Normalisation",
    level=1,
    entry="rms_norm",
    statement=(
        "Divide each feature vector by its root mean square and apply a learned "
        "scale. No mean subtraction, no shift. Must match F.rms_norm. It costs one "
        "fewer reduction and one fewer parameter tensor than LayerNorm, and is "
        "what most recent large language models use."
    ),
    shapes="x (..., D) float · gamma (D,) · eps float  ->  (..., D)",
    stub="def rms_norm(x, gamma, eps=1e-6):\n    # (..., D), (D,) -> (..., D)\n    pass\n",
    hints=[
        "RMS(x) = sqrt(mean(x^2) + eps), taken over the last axis.",
        "There is no centring step — the vector's mean component survives.",
        "Keep the reduced axis so the division broadcasts back.",
    ],
    solution=(
        "def rms_norm(x, gamma, eps=1e-6):\n"
        "    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)\n"
        "    return gamma * x / rms\n"
    ),
    solution_np=(
        "def rms_norm(x, gamma, eps=1e-6):\n"
        "    rms = np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)\n"
        "    return gamma * x / rms\n"
    ),
    traps=[
        "Subtracting the mean, which turns it into LayerNorm without a shift.",
        "Using sum instead of mean, making the scale depend on D.",
        "Putting eps outside the square root.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(3, 5, 8)
    g = torch.randn(8)
    check("matches F.rms_norm", lambda: close(fn(x, g), F.rms_norm(x, (8,), g), 1e-5))
    check("not mean-centred",
          lambda: not close(fn(x + 3.0, torch.ones(8)).mean(-1), torch.zeros(3, 5), 1e-4))
    check("identity gamma gives unit RMS",
          lambda: close(fn(x, torch.ones(8)).pow(2).mean(-1), torch.ones(3, 5), 1e-3))
    check("scale invariant up to gamma",
          lambda: close(fn(x, torch.ones(8)), fn(x * 7.0, torch.ones(8)), 1e-3))
    check("all-zero input stays finite (eps is inside the root)",
          lambda: bool(torch.isfinite(fn(torch.zeros(2, 8), torch.ones(8))).all()))
''',
),

task(
    id="layer-norm",
    title="LayerNorm",
    chapter=CH,
    section="2.7 Normalisation",
    level=2,
    entry="layer_norm",
    statement=(
        "Normalise each token's feature vector by its own mean and variance, then "
        "apply a learned scale and shift. Must match F.layer_norm. The divisor for "
        "the variance is D — PyTorch's var() defaults to D-1, which is the single "
        "most common way this is got wrong."
    ),
    shapes="x (..., D) · gamma (D,) · beta (D,) · eps float  ->  (..., D)",
    stub=("def layer_norm(x, gamma, beta, eps=1e-5):\n"
          "    # (..., D), (D,), (D,) -> (..., D)\n    pass\n"),
    hints=[
        "Statistics are taken over the last axis only — every token uses its own.",
        "torch.var defaults to the unbiased estimator, dividing by D-1. "
        "Normalisation wants the population variance.",
        "Pass unbiased=False, keep the reduced axis, and put eps inside the root.",
    ],
    solution=(
        "def layer_norm(x, gamma, beta, eps=1e-5):\n"
        "    mu = x.mean(-1, keepdim=True)\n"
        "    var = x.var(-1, unbiased=False, keepdim=True)\n"
        "    return gamma * (x - mu) / torch.sqrt(var + eps) + beta\n"
    ),
    solution_np=(
        "def layer_norm(x, gamma, beta, eps=1e-5):\n"
        "    mu = x.mean(-1, keepdims=True)\n"
        "    var = x.var(-1, keepdims=True)\n"
        "    return gamma * (x - mu) / np.sqrt(var + eps) + beta\n"
    ),
    traps=[
        "Leaving unbiased at its default, scaling every activation by sqrt(D/(D-1)).",
        "Adding eps outside the square root, so a near-constant vector has an "
        "unbounded derivative.",
        "Normalising over the batch axis — that is BatchNorm, which couples "
        "examples and behaves differently at train and test time.",
    ],
    tests='''
def checks(fn, check):
    x, g, b = torch.randn(3, 5, 8), torch.randn(8), torch.randn(8)
    check("matches F.layer_norm", lambda: close(fn(x, g, b), F.layer_norm(x, (8,), g, b), 1e-5))
    check("zero mean with identity affine",
          lambda: close(fn(x, torch.ones(8), torch.zeros(8)).mean(-1), torch.zeros(3, 5), 1e-4))
    check("unit variance with identity affine",
          lambda: close(fn(x, torch.ones(8), torch.zeros(8)).var(-1, unbiased=False),
                        torch.ones(3, 5), 1e-3))
    check("uses the biased variance (divisor D, not D-1)",
          lambda: not close(fn(x, g, b),
                            g * (x - x.mean(-1, keepdim=True)) /
                            torch.sqrt(x.var(-1, unbiased=True, keepdim=True) + 1e-5) + b, 1e-4))
    check("constant input stays finite",
          lambda: bool(torch.isfinite(fn(torch.zeros(1, 8), torch.ones(8), torch.zeros(8))).all()))
''',
),

task(
    id="swiglu",
    title="SwiGLU feed-forward",
    chapter=CH,
    section="2.8 Feed-forward block and SwiGLU",
    level=2,
    entry="swiglu",
    statement=(
        "Implement the gated feed-forward block used in most modern transformers: "
        "SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down, where SiLU(z) = z·sigmoid(z). "
        "Two projections go up, one comes back down, and the gate multiplies "
        "elementwise — which is why the hidden width is usually shrunk to keep the "
        "parameter count comparable to a plain two-matrix MLP."
    ),
    shapes="x (..., D) · w_gate (D, H) · w_up (D, H) · w_down (H, D)  ->  (..., D)",
    stub=("def swiglu(x, w_gate, w_up, w_down):\n"
          "    # (..., D) -> (..., D)\n    pass\n"),
    hints=[
        "SiLU, also called swish, is z * sigmoid(z).",
        "Gate and up are two separate projections of the same input, multiplied "
        "elementwise — not concatenated.",
        "Only the gate branch passes through the nonlinearity.",
    ],
    solution=(
        "def swiglu(x, w_gate, w_up, w_down):\n"
        "    g = x @ w_gate\n"
        "    return (g * torch.sigmoid(g) * (x @ w_up)) @ w_down\n"
    ),
    solution_np=(
        "def swiglu(x, w_gate, w_up, w_down):\n"
        "    g = x @ w_gate\n"
        "    return ((g / (1 + np.exp(-g))) * (x @ w_up)) @ w_down\n"
    ),
    traps=[
        "Applying the nonlinearity to the up branch as well as the gate.",
        "Using ReLU or GELU where SiLU is specified — they are not interchangeable "
        "when matching a reference implementation.",
        "Adding a bias; these projections are usually bias-free.",
    ],
    tests='''
def checks(fn, check):
    D, H = 6, 16
    x = torch.randn(2, 4, D)
    wg, wu, wd = torch.randn(D, H), torch.randn(D, H), torch.randn(H, D)
    want = (F.silu(x @ wg) * (x @ wu)) @ wd
    check("matches the reference composition", lambda: close(fn(x, wg, wu, wd), want, 1e-4))
    check("output shape", lambda: shape(fn(x, wg, wu, wd)) == (2, 4, D))
    check("zero input gives zero output",
          lambda: close(fn(torch.zeros(1, D), wg, wu, wd), torch.zeros(1, D), 1e-6))
    check("gate uses SiLU, not ReLU",
          lambda: not close(fn(x, wg, wu, wd), (F.relu(x @ wg) * (x @ wu)) @ wd, 1e-3))
    check("nonlinearity is on the gate branch only",
          lambda: not close(fn(x, wg, wu, wd), (F.silu(x @ wg) * F.silu(x @ wu)) @ wd, 1e-3))
''',
),

task(
    id="moe-routing",
    title="Mixture-of-experts top-k routing",
    chapter=CH,
    section="2.9 Mixture of experts",
    level=3,
    entry="route",
    statement=(
        "Route each token to its top-k experts. Given router logits, return the "
        "expert indices and their normalised weights. The weights must be "
        "renormalised over the selected k only — softmaxing over all experts and "
        "then slicing leaves the weights summing to less than one, which quietly "
        "scales down every routed token."
    ),
    shapes="logits (N, E) float · k int  ->  (idx (N, k) int64, w (N, k) float summing to 1)",
    stub="def route(logits, k):\n    # -> (idx (N,k), weights (N,k))\n    pass\n",
    hints=[
        "topk gives both the values and the indices you need, in one call.",
        "Renormalise after selection, not before.",
        "Softmax over the k selected logits — equivalently, divide the selected "
        "softmax weights by their own sum.",
    ],
    solution=(
        "def route(logits, k):\n"
        "    vals, idx = logits.topk(k, dim=-1)\n"
        "    w = torch.softmax(vals, dim=-1)\n"
        "    return idx, w\n"
    ),
    solution_np=(
        "def route(logits, k):\n"
        "    idx = np.argsort(-logits, axis=-1)[:, :k]\n"
        "    vals = np.take_along_axis(logits, idx, axis=-1)\n"
        "    e = np.exp(vals - vals.max(-1, keepdims=True))\n"
        "    return idx, e / e.sum(-1, keepdims=True)\n"
    ),
    traps=[
        "Softmaxing over all E experts and slicing k — the weights then sum to "
        "less than 1.",
        "Returning unsorted indices when the caller assumes descending order.",
        "Forgetting that k=1 must still return a (N,1) weight of exactly 1.",
    ],
    tests='''
def checks(fn, check):
    logits = torch.randn(7, 6)
    idx, w = fn(logits, 2)
    check("index shape", lambda: shape(idx) == (7, 2))
    check("weight shape", lambda: shape(w) == (7, 2))
    check("weights sum to 1 over the selected k", lambda: close(w.sum(-1), torch.ones(7), 1e-5))
    check("selects the largest logits",
          lambda: close(torch.gather(logits, 1, idx.long()).sort(-1, descending=True).values,
                        logits.topk(2, -1).values, 1e-5))
    check("k=1 gives weight exactly 1",
          lambda: close(fn(logits, 1)[1], torch.ones(7, 1), 1e-6))
    check("renormalised over k, not over all experts",
          lambda: not close(fn(logits, 2)[1],
                            torch.softmax(logits, -1).topk(2, -1).values, 1e-3))
''',
),

]
