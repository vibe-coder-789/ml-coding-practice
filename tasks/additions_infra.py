"""Distributed-training drills — the parallelism axes, simulated single-process.

Every task here carries the anchor that defines distributed correctness: the
sharded computation must equal the unsharded one exactly. Ranks are simulated
as plain Python lists of tensors; no torch.distributed anywhere.
"""
from .schema import task

CH = "8 · Infrastructure: parallelism, reliability, serving"

TASKS = [

task(
    id="ring-allreduce",
    title="Ring all-reduce, the actual algorithm",
    chapter=CH,
    section="8.1 What each parallelism axis costs — the ring algorithm",
    level=3,
    entry="ring_allreduce",
    statement=(
        "Implement ring all-reduce over P simulated ranks: split each rank's "
        "vector into P chunks, run P-1 reduce-scatter steps (each rank sends "
        "one chunk to its right neighbour, which ADDS it), then P-1 all-gather "
        "steps (same ring, the receiver now COPIES). Every rank must end with "
        "the full sum, and — the property the algorithm exists for — every "
        "rank sends exactly 2(P-1) chunks, no rank more than any other. A "
        "gather-to-root-and-broadcast gets the same answer while concentrating "
        "P(P-1) chunk-sends on the root, which is precisely what the "
        "per-rank send count check rejects."
    ),
    shapes=("xs list of P tensors (D,), D divisible by P"
            "  ->  dict 'results' list of P tensors, 'sends_per_rank' list of P ints"),
    stub=("def ring_allreduce(xs):\n"
          "    # reduce-scatter then all-gather around the ring;\n"
          "    # count every chunk each rank sends\n    pass\n"),
    hints=[
        "Work on clones; chunk c of rank r's buffer is buf[r][c*C:(c+1)*C].",
        "Reduce-scatter step s: rank r sends chunk (r - s) mod P to rank "
        "(r+1) mod P, which adds it. After P-1 steps, rank r owns the fully "
        "reduced chunk (r+1) mod P.",
        "All-gather step s: rank r sends chunk (r + 1 - s) mod P onward; the "
        "receiver copies instead of adding. Collect sends within a step before "
        "applying them, or a rank reads a chunk its neighbour already "
        "overwrote this step.",
    ],
    solution=(
        "def ring_allreduce(xs):\n"
        "    P = len(xs)\n"
        "    if P == 1:\n"
        "        return {'results': [xs[0].clone()], 'sends_per_rank': [0]}\n"
        "    D = xs[0].shape[0]\n"
        "    C = D // P\n"
        "    bufs = [x.clone() for x in xs]\n"
        "    sends = [0] * P\n"
        "    for step in range(P - 1):\n"
        "        moves = []\n"
        "        for r in range(P):\n"
        "            idx = (r - step) % P\n"
        "            moves.append(((r + 1) % P, idx, bufs[r][idx * C:(idx + 1) * C].clone()))\n"
        "            sends[r] += 1\n"
        "        for dst, idx, data in moves:\n"
        "            bufs[dst][idx * C:(idx + 1) * C] += data\n"
        "    for step in range(P - 1):\n"
        "        moves = []\n"
        "        for r in range(P):\n"
        "            idx = (r + 1 - step) % P\n"
        "            moves.append(((r + 1) % P, idx, bufs[r][idx * C:(idx + 1) * C].clone()))\n"
        "            sends[r] += 1\n"
        "        for dst, idx, data in moves:\n"
        "            bufs[dst][idx * C:(idx + 1) * C] = data\n"
        "    return {'results': bufs, 'sends_per_rank': sends}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Gather-to-root and broadcast: correct sums, but the root sends P(P-1) "
        "chunks while a ring rank sends 2(P-1) — the whole point of the ring "
        "is that no link is the bottleneck.",
        "Applying sends eagerly inside a step, so a rank forwards a chunk its "
        "neighbour already modified in the same step.",
        "Adding during the all-gather phase, which double-counts everything "
        "the reduce-scatter already summed.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    P, D = 4, 12
    xs = [torch.randn(D) for _ in range(P)]
    total = sum(xs)
    o = fn([x.clone() for x in xs])
    check("every rank ends with the exact sum",
          lambda: all(close(r, total, 1e-5) for r in o["results"]))
    check("every rank sends exactly 2(P-1) chunks",
          lambda: o["sends_per_rank"] == [2 * (P - 1)] * P)
    check("total traffic is 2P(P-1) chunk-sends",
          lambda: sum(o["sends_per_rank"]) == 2 * P * (P - 1))
    def other_p():
        xs2 = [torch.randn(6) for _ in range(2)]
        o2 = fn([x.clone() for x in xs2])
        return all(close(r, xs2[0] + xs2[1], 1e-5) for r in o2["results"]) \\
               and o2["sends_per_rank"] == [2, 2]
    check("works for P = 2", other_p)
    check("P = 1 sends nothing",
          lambda: fn([torch.ones(4)])["sends_per_rank"] == [0])
    def inputs_untouched():
        before = [x.clone() for x in xs]
        fn(xs)
        return all(close(a, b) for a, b in zip(xs, before))
    check("input tensors are not mutated", inputs_untouched)
''',
),

task(
    id="tensor-parallel-mlp",
    title="Tensor-parallel MLP (Megatron-style)",
    chapter=CH,
    section="8.1 What each parallelism axis costs — tensor parallelism",
    level=2,
    entry="tp_mlp",
    statement=(
        "Split an MLP across P ranks the way Megatron does: the first weight "
        "COLUMN-wise (each rank computes a slice of the hidden layer), the "
        "second ROW-wise (each rank consumes its own slice), so the "
        "nonlinearity applies locally and ONE all-reduce at the end — the sum "
        "over ranks — restores the exact result. The split direction is the "
        "entire exam: with the first matrix split the other way, the "
        "nonlinearity would need the summed pre-activations, GELU of a "
        "partial sum is not a partial GELU, and the answer is silently wrong."
    ),
    shapes="x (B, D) · W1 (D, H) · W2 (H, D2) · P dividing H  ->  (B, D2)",
    stub=("def tp_mlp(x, W1, W2, P):\n"
          "    # column-split W1, row-split W2, sum the per-rank partials\n    pass\n"),
    hints=[
        "Rank r takes W1[:, r*h:(r+1)*h] and W2[r*h:(r+1)*h, :] with h = H // P.",
        "Per rank: F.gelu(x @ W1_r) @ W2_r — the activation never crosses "
        "ranks.",
        "The final sum of partials IS the all-reduce; nothing else is "
        "communicated.",
    ],
    solution=(
        "def tp_mlp(x, W1, W2, P):\n"
        "    H = W1.shape[1]\n"
        "    h = H // P\n"
        "    out = None\n"
        "    for r in range(P):\n"
        "        part = F.gelu(x @ W1[:, r * h:(r + 1) * h]) @ W2[r * h:(r + 1) * h, :]\n"
        "        out = part if out is None else out + part\n"
        "    return out\n"
    ),
    solution_np=(
        "def tp_mlp(x, W1, W2, P):\n"
        "    _erf = np.vectorize(math.erf)\n"
        "    def gelu(v):\n"
        "        return 0.5 * v * (1 + _erf(v / math.sqrt(2)))\n"
        "    H = W1.shape[1]\n"
        "    h = H // P\n"
        "    out = None\n"
        "    for r in range(P):\n"
        "        part = gelu(x @ W1[:, r * h:(r + 1) * h]) @ W2[r * h:(r + 1) * h, :]\n"
        "        out = part if out is None else out + part\n"
        "    return out\n"
    ),
    traps=[
        "Summing the pre-activations before the GELU — the nonlinearity of a "
        "sum is not the sum of nonlinearities, and this is exactly why the "
        "first matrix must be split by columns.",
        "Splitting both matrices the same way, which needs a second "
        "communication in the middle of the block.",
        "Averaging the partials instead of summing — the row-split second "
        "matmul already distributes the sum.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    B, D, H, D2 = 3, 5, 8, 4
    x = torch.randn(B, D)
    W1 = torch.randn(D, H)
    W2 = torch.randn(H, D2)
    full = F.gelu(x @ W1) @ W2
    check("P = 2 equals the unsharded MLP exactly", lambda: close(fn(x, W1, W2, 2), full, 1e-5))
    check("P = 4 equals it too", lambda: close(fn(x, W1, W2, 4), full, 1e-5))
    check("P = 1 is the plain MLP", lambda: close(fn(x, W1, W2, 1), full, 1e-6))
    check("P = 8 (one hidden unit per rank) still exact",
          lambda: close(fn(x, W1, W2, 8), full, 1e-5))
    check("output shape", lambda: shape(fn(x, W1, W2, 2)) == (B, D2))
''',
),

task(
    id="fsdp-forward",
    title="Sharded parameters: an FSDP forward pass",
    chapter=CH,
    section="8.1 What each parallelism axis costs — parameter sharding (ZeRO-3/FSDP)",
    level=3,
    entry="fsdp_forward",
    statement=(
        "Simulate a ZeRO-3 / FSDP forward on one rank's view: every layer's "
        "weight lives row-sharded across P ranks, and to compute a layer you "
        "all-gather ITS shards, use the full weight, then free it before "
        "touching the next layer. Return the output (ReLU between layers, "
        "none after the last) and the peak parameter count resident at once: "
        "all of this rank's shards, plus the (P-1)/P of the single largest "
        "layer that gathering temporarily adds. Gathering everything up front "
        "computes the same numbers while holding the whole model — the "
        "memory accounting is what distinguishes FSDP from a plain forward."
    ),
    shapes=("x (B, D) · shards list per layer of P tensors (rows/P, cols)"
            "  ->  dict 'out' (B, ...), 'peak_params' int"),
    stub=("def fsdp_forward(x, shards):\n"
          "    # gather one layer, compute, free; track peak resident params\n    pass\n"),
    hints=[
        "Full weight of layer l: torch.cat(shards[l], dim=0); apply as "
        "x @ W.T with ReLU between layers.",
        "Always resident: sum over layers of (layer size / P). Gathering "
        "layer l adds its OTHER ranks' share, size_l * (P-1)/P, until freed.",
        "peak_params = own_total + max over layers of that gathered extra.",
    ],
    solution=(
        "def fsdp_forward(x, shards):\n"
        "    P = len(shards[0])\n"
        "    own = sum(sum(s.numel() for s in layer) // P for layer in shards)\n"
        "    peak_extra = 0\n"
        "    h = x\n"
        "    for li, layer in enumerate(shards):\n"
        "        W = torch.cat(layer, dim=0)\n"
        "        extra = W.numel() - W.numel() // P\n"
        "        peak_extra = max(peak_extra, extra)\n"
        "        h = h @ W.T\n"
        "        if li < len(shards) - 1:\n"
        "            h = torch.relu(h)\n"
        "    return {'out': h, 'peak_params': own + peak_extra}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Gathering every layer before the forward — same output, peak equal to "
        "the whole model, which is the thing FSDP exists to avoid.",
        "Counting the gathered layer twice: this rank's own shard of it is "
        "already in the resident total.",
        "Concatenating shards along the wrong axis, which transposes the "
        "layer and only fails when the matrices are non-square.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    P = 4
    sizes = [(8, 6), (12, 8), (4, 12)]        # (out, in) per layer, out % P == 0
    Ws = [torch.randn(o, i) for o, i in sizes]
    shards = [[W[r * (W.shape[0] // P):(r + 1) * (W.shape[0] // P)] for r in range(P)]
              for W in Ws]
    x = torch.randn(3, 6)
    def full():
        h = x
        for li, W in enumerate(Ws):
            h = h @ W.T
            if li < len(Ws) - 1:
                h = torch.relu(h)
        return h
    o = fn(x, [list(s) for s in shards])
    check("output equals the unsharded forward", lambda: close(o["out"], full(), 1e-5))
    def peak_formula():
        own = sum(W.numel() // P for W in Ws)
        extra = max(W.numel() - W.numel() // P for W in Ws)
        return o["peak_params"] == own + extra
    check("peak = own shards + largest layer's gathered remainder", peak_formula)
    def peak_beats_full_model():
        return o["peak_params"] < sum(W.numel() for W in Ws)
    check("peak is genuinely below holding the whole model", peak_beats_full_model)
    def two_ranks():
        sh2 = [[W[:W.shape[0] // 2], W[W.shape[0] // 2:]] for W in Ws]
        o2 = fn(x, sh2)
        own = sum(W.numel() // 2 for W in Ws)
        extra = max(W.numel() // 2 for W in Ws)
        return close(o2["out"], full(), 1e-5) and o2["peak_params"] == own + extra
    check("P = 2 output and accounting both hold", two_ranks)
''',
),

task(
    id="pipeline-schedule",
    title="A GPipe schedule and its bubble",
    chapter=CH,
    section="8.1 What each parallelism axis costs — pipeline parallelism",
    level=2,
    entry="pipeline_schedule",
    statement=(
        "Lay out the forward schedule of a p-stage pipeline over m "
        "microbatches, one time unit per (stage, microbatch): stage s starts "
        "microbatch j at time s + j. Return the timeline, the finish time "
        "m + p - 1, and the bubble fraction (p-1)/(m+p-1) — the share of the "
        "schedule each stage spends idle. The timeline must be a real "
        "schedule: no stage runs two microbatches at once, and no microbatch "
        "reaches stage s+1 before stage s finished it. The formulas are "
        "checked AGAINST the timeline, not copied from it."
    ),
    shapes=("p, m int  ->  dict 'timeline' list of (stage, micro, start),"
            " 'finish' int, 'bubble' float"),
    stub=("def pipeline_schedule(p, m):\n"
          "    # start(s, j) = s + j; finish m+p-1; bubble (p-1)/(m+p-1)\n    pass\n"),
    hints=[
        "Two nested loops emit the (s, j, s + j) triples.",
        "Finish is the last start plus one unit: (p-1) + (m-1) + 1.",
        "Per stage, m busy slots in a span of m + p - 1: idle fraction "
        "(p-1)/(m+p-1).",
    ],
    solution=(
        "def pipeline_schedule(p, m):\n"
        "    timeline = [(s, j, s + j) for s in range(p) for j in range(m)]\n"
        "    return {'timeline': timeline,\n"
        "            'finish': p + m - 1,\n"
        "            'bubble': (p - 1) / (m + p - 1)}\n"
    ),
    solution_np=(
        "def pipeline_schedule(p, m):\n"
        "    timeline = [(s, j, s + j) for s in range(p) for j in range(m)]\n"
        "    return {'timeline': timeline,\n"
        "            'finish': p + m - 1,\n"
        "            'bubble': (p - 1) / (m + p - 1)}\n"
    ),
    traps=[
        "The sequential schedule — finish one microbatch through all stages "
        "before starting the next — which finishes at m*p and has no pipeline "
        "in it at all.",
        "Bubble computed as (p-1)/m, the approximation that only holds when "
        "m >> p.",
        "A timeline where stage s+1 starts a microbatch in the same slot "
        "stage s is still processing it.",
    ],
    tests='''
def checks(fn, check):
    o = fn(4, 8)
    tl = {(s, j): t for s, j, t in o["timeline"]}
    check("every (stage, microbatch) pair is scheduled once",
          lambda: len(tl) == 32 and len(o["timeline"]) == 32)
    check("no stage runs two microbatches at once",
          lambda: all(len({tl[(s, j)] for j in range(8)}) == 8 for s in range(4)))
    check("a microbatch only advances after the previous stage finished it",
          lambda: all(tl[(s + 1, j)] >= tl[(s, j)] + 1
                      for s in range(3) for j in range(8)))
    check("finish equals the last start plus one",
          lambda: o["finish"] == max(t for _, _, t in o["timeline"]) + 1)
    check("finish is m + p - 1", lambda: o["finish"] == 11)
    check("bubble is (p-1)/(m+p-1)", lambda: abs(o["bubble"] - 3 / 11) < 1e-12)
    check("m = 1 degenerates to no pipelining: bubble (p-1)/p",
          lambda: abs(fn(4, 1)["bubble"] - 3 / 4) < 1e-12)
    check("many microbatches shrink the bubble",
          lambda: fn(4, 64)["bubble"] < fn(4, 8)["bubble"])
''',
),

task(
    id="ring-attention-combine",
    title="Combine attention across sequence shards",
    chapter=CH,
    section="8.1 What each parallelism axis costs — context parallelism (ring attention)",
    level=3,
    entry="combine_attention",
    statement=(
        "Context parallelism splits the KEYS AND VALUES across ranks; each "
        "rank computes attention against its shard and reports three "
        "statistics per query: the shard's row-max m_i, its normaliser "
        "l_i = sum of exp(s - m_i), and its local output o_i. Merge them into "
        "the EXACT full-sequence attention: rescale every shard's weight by "
        "exp(m_i - m) against the global max m, and mix the local outputs by "
        "l_i exp(m_i - m) / L. This is the online-softmax identity turned "
        "into a distributed reduction — a plain average of shard outputs "
        "ignores how much probability mass each shard actually held."
    ),
    shapes=("ms, ls lists of (B, L) · os list of (B, L, Dh)"
            "  ->  (B, L, Dh), equal to full attention"),
    stub=("def combine_attention(ms, ls, os):\n"
          "    # global max, rescale each shard's mass, mix the outputs\n    pass\n"),
    hints=[
        "m = elementwise max over the shards' m_i.",
        "w_i = l_i * exp(m_i - m); L = sum of w_i.",
        "out = sum of (w_i / L)[..., None] * o_i.",
    ],
    solution=(
        "def combine_attention(ms, ls, os):\n"
        "    m = ms[0]\n"
        "    for mi in ms[1:]:\n"
        "        m = torch.maximum(m, mi)\n"
        "    ws = [li * torch.exp(mi - m) for mi, li in zip(ms, ls)]\n"
        "    L = ws[0].clone()\n"
        "    for w in ws[1:]:\n"
        "        L = L + w\n"
        "    out = torch.zeros_like(os[0])\n"
        "    for w, o in zip(ws, os):\n"
        "        out = out + (w / L).unsqueeze(-1) * o\n"
        "    return out\n"
    ),
    frameworks=["torch"],
    traps=[
        "Averaging the shard outputs uniformly, which is only right when every "
        "shard happens to hold identical probability mass.",
        "Weighting by l_i without the exp(m_i - m) rescale — the shard maxima "
        "differ, so the normalisers live on different scales.",
        "Taking the global max over the wrong axis, which mixes queries.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    B, L, S, Dh = 2, 4, 12, 8
    q = torch.randn(B, L, Dh)
    k = torch.randn(B, S, Dh)
    v = torch.randn(B, S, Dh)
    scale = 1 / math.sqrt(Dh)
    full = torch.softmax(q @ k.mT * scale, -1) @ v

    def shard_stats(n_shards):
        ms, ls, os = [], [], []
        step = S // n_shards
        for i in range(n_shards):
            ks = k[:, i * step:(i + 1) * step]
            vs = v[:, i * step:(i + 1) * step]
            s = q @ ks.mT * scale
            m = s.max(-1).values
            e = torch.exp(s - m.unsqueeze(-1))
            l = e.sum(-1)
            o = (e / l.unsqueeze(-1)) @ vs
            ms.append(m); ls.append(l); os.append(o)
        return ms, ls, os

    check("two shards combine to the exact full attention",
          lambda: close(fn(*shard_stats(2)), full, 1e-5))
    check("four shards combine to it too",
          lambda: close(fn(*shard_stats(4)), full, 1e-5))
    check("a single shard passes through unchanged",
          lambda: close(fn(*shard_stats(1)), full, 1e-5))
    def stable_at_large_scores():
        q2, k2 = q * 30, k * 30
        full2 = torch.softmax(q2 @ k2.mT * scale, -1) @ v
        ms, ls, os = [], [], []
        for i in range(2):
            ks = k2[:, i * 6:(i + 1) * 6]
            vs = v[:, i * 6:(i + 1) * 6]
            s = q2 @ ks.mT * scale
            m = s.max(-1).values
            e = torch.exp(s - m.unsqueeze(-1))
            l = e.sum(-1)
            ms.append(m); ls.append(l); os.append((e / l.unsqueeze(-1)) @ vs)
        return close(fn(ms, ls, os), full2, 1e-4)
    check("stays exact when shard maxima differ wildly", stable_at_large_scores)
    check("output shape", lambda: shape(fn(*shard_stats(2))) == (B, L, Dh))
''',
),

]
