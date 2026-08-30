"""torch-api — do you actually know what these functions do?

Where the other two volumes ask you to build a construct, this one asks whether
you know the semantics of the tools: which operations copy, how broadcasting
resolves, what gather wants from its index, where autograd stops.
"""
from .schema import task

BOOK = "torch-api"
C_SHAPE = "Shapes, views, and memory"
C_BCAST = "Broadcasting"
C_INDEX = "Indexing, gather, scatter"
C_EINSUM = "einsum and matmul"
C_JOIN = "Joining, splitting, windowing"
C_GRAD = "Autograd and dtypes"

TASKS = [

task(
    id="broadcast-shape",
    title="Resolve a broadcast",
    book=BOOK, chapter=C_BCAST,
    section="Broadcasting",
    level=1,
    entry="broadcast_shape",
    statement=(
        "Implement the broadcasting rule itself: given two shapes, return the "
        "shape of their elementwise combination, or None if they are "
        "incompatible. Align from the right; each aligned pair must be equal or "
        "contain a 1, and the output takes the larger. Missing leading axes count "
        "as 1."
    ),
    shapes="a tuple · b tuple  ->  tuple or None",
    stub=("def broadcast_shape(a, b):\n"
          "    # -> resulting shape, or None if they cannot broadcast\n    pass\n"),
    hints=[
        "Right-align by reversing both shapes and zipping with a fill of 1.",
        "A pair (x, y) is compatible when x == y or either is 1; the result is "
        "max(x, y).",
        "Remember to reverse the result back at the end.",
    ],
    solution=(
        "def broadcast_shape(a, b):\n"
        "    out = []\n"
        "    ra, rb = list(a)[::-1], list(b)[::-1]\n"
        "    for i in range(max(len(ra), len(rb))):\n"
        "        x = ra[i] if i < len(ra) else 1\n"
        "        y = rb[i] if i < len(rb) else 1\n"
        "        if x != y and x != 1 and y != 1:\n"
        "            return None\n"
        "        out.append(max(x, y))\n"
        "    return tuple(out[::-1])\n"
    ),
    solution_np=(
        "def broadcast_shape(a, b):\n"
        "    out = []\n"
        "    ra, rb = list(a)[::-1], list(b)[::-1]\n"
        "    for i in range(max(len(ra), len(rb))):\n"
        "        x = ra[i] if i < len(ra) else 1\n"
        "        y = rb[i] if i < len(rb) else 1\n"
        "        if x != y and x != 1 and y != 1:\n"
        "            return None\n"
        "        out.append(max(x, y))\n"
        "    return tuple(out[::-1])\n"
    ),
    traps=[
        "Aligning from the left, which is what makes people expect (N,) and (N,1) "
        "to combine into (N,).",
        "Treating a 0-length axis as broadcastable.",
        "Returning the longer shape verbatim instead of taking the max per axis.",
    ],
    tests='''
def checks(fn, check):
    check("classic case", lambda: fn((5, 1, 4), (3, 1)) == (5, 3, 4))
    check("equal shapes pass through", lambda: fn((2, 3), (2, 3)) == (2, 3))
    check("scalars broadcast to anything", lambda: fn((), (4, 5)) == (4, 5))
    check("incompatible returns None", lambda: fn((3, 4), (4, 3)) is None)
    check("the (N,) vs (N,1) surprise", lambda: fn((5,), (5, 1)) == (5, 5))
    check("matches torch.broadcast_shapes",
          lambda: fn((8, 1, 6, 1), (7, 1, 5)) == tuple(torch.broadcast_shapes((8,1,6,1), (7,1,5))))
    check("leading axes are filled with 1", lambda: fn((4,), (3, 4)) == (3, 4))
''',
),

task(
    id="pairwise-sqdist",
    title="Pairwise squared distances",
    book=BOOK, chapter=C_BCAST,
    section="Broadcasting",
    level=2,
    entry="sqdist",
    statement=(
        "Return the matrix of squared Euclidean distances between every row of A "
        "and every row of B, using the expansion "
        "‖x-y‖² = ‖x‖² + ‖y‖² - 2x·y rather than materialising the (N, M, D) "
        "difference. Clamp at zero: the expansion is exact in real arithmetic but "
        "can go slightly negative in floating point, and a negative distance "
        "becomes NaN the moment anything takes its square root."
    ),
    shapes="A (N, D) · B (M, D)  ->  (N, M) non-negative",
    stub="def sqdist(A, B):\n    # -> (N, M) squared distances\n    pass\n",
    hints=[
        "‖x‖² per row is (A*A).sum(-1); keep it as a column with [:, None].",
        "The cross term is a single matrix product A @ B.T.",
        "Clamp the result at min=0 before returning.",
    ],
    solution=(
        "def sqdist(A, B):\n"
        "    a2 = (A * A).sum(-1)[:, None]\n"
        "    b2 = (B * B).sum(-1)[None, :]\n"
        "    return (a2 + b2 - 2 * A @ B.T).clamp(min=0)\n"
    ),
    solution_np=(
        "def sqdist(A, B):\n"
        "    a2 = (A * A).sum(-1)[:, None]\n"
        "    b2 = (B * B).sum(-1)[None, :]\n"
        "    return np.maximum(a2 + b2 - 2 * A @ B.T, 0)\n"
    ),
    traps=[
        "Skipping the clamp, so a point compared with itself yields a tiny "
        "negative number.",
        "Building the (N, M, D) broadcast difference, which is the memory blow-up "
        "the expansion avoids.",
        "Transposing the wrong operand and getting (M, N).",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(5, 3); B = torch.randn(4, 3)
    check("shape", lambda: shape(fn(A, B)) == (5, 4))
    check("matches the direct broadcast form",
          lambda: close(fn(A, B), ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 1e-4))
    check("matches torch.cdist squared",
          lambda: close(fn(A, B), torch.cdist(A, B) ** 2, 1e-3))
    check("self-distance is zero on the diagonal",
          lambda: close(torch.diagonal(fn(A, A)), torch.zeros(5), 1e-4))
    check("never negative", lambda: bool((fn(A, A) >= 0).all()))
    check("survives a sqrt without NaN",
          lambda: bool(torch.isfinite(fn(A, A).sqrt()).all()))
''',
),

task(
    id="batched-gather",
    title="Gather per row",
    book=BOOK, chapter=C_INDEX,
    section="Indexing, gather, scatter",
    level=1,
    entry="pick",
    statement=(
        "Given a matrix and one column index per row, return the selected value "
        "from each row, without a Python loop. This is the operation hiding inside "
        "cross-entropy, and the thing to know is that gather wants an index of the "
        "same rank as the source."
    ),
    shapes="x (B, C) · idx (B,) int64  ->  (B,)",
    stub="def pick(x, idx):\n    # x (B, C), idx (B,) -> (B,)\n    pass\n",
    hints=[
        "x[idx] selects whole rows — that is not what is wanted.",
        "torch.gather(input, dim, index) needs index.ndim == input.ndim and "
        "returns a tensor shaped like index.",
        "Promote idx to (B, 1) with idx[:, None], gather along dim=1, squeeze.",
    ],
    solution=(
        "def pick(x, idx):\n"
        "    return x.gather(1, idx[:, None]).squeeze(1)\n"
    ),
    solution_np=(
        "def pick(x, idx):\n"
        "    return np.take_along_axis(x, idx[:, None], axis=1).squeeze(1)\n"
    ),
    traps=[
        "Using x[idx], which indexes rows.",
        "Forgetting to squeeze, leaving a (B, 1) that broadcasts badly later.",
        "Passing an int32 index — gather requires int64.",
    ],
    tests='''
def checks(fn, check):
    x = torch.arange(12.).reshape(3, 4)
    check("matches the explicit loop",
          lambda: close(fn(x, torch.tensor([0, 2, 1])), torch.tensor([0., 6., 9.])))
    check("returns rank 1", lambda: shape(fn(torch.randn(5, 7), torch.zeros(5, dtype=torch.long))) == (5,))
    check("works for B=1", lambda: shape(fn(torch.randn(1, 4), torch.tensor([3]))) == (1,))
    check("equals a per-row list comprehension",
          lambda: close(fn(x, torch.tensor([3, 1, 2])),
                        torch.stack([x[i, j] for i, j in enumerate([3, 1, 2])])))
    check("selecting column 0 everywhere is the first column",
          lambda: close(fn(x, torch.zeros(3, dtype=torch.long)), x[:, 0]))
''',
),

task(
    id="segment-sum",
    title="Sum by segment id",
    book=BOOK, chapter=C_INDEX,
    section="Indexing, gather, scatter",
    level=2,
    entry="segment_sum",
    statement=(
        "Sum rows of X into buckets given by an integer id per row, returning one "
        "row per bucket. Duplicate ids must accumulate, not overwrite — which is "
        "the entire difference between scatter_ and scatter_add_, and the reason a "
        "naive index-assignment silently drops all but the last contribution."
    ),
    shapes="X (N, D) · ids (N,) int64 in [0, K) · K int  ->  (K, D)",
    stub="def segment_sum(X, ids, K):\n    # -> (K, D), rows summed by bucket\n    pass\n",
    hints=[
        "Allocate the (K, D) output as zeros first.",
        "index_add_(0, ids, X) accumulates rows at the given indices.",
        "Equivalently scatter_add_ with the ids expanded to X's shape — but "
        "index_add_ is the direct tool here.",
    ],
    solution=(
        "def segment_sum(X, ids, K):\n"
        "    out = torch.zeros(K, X.shape[1], dtype=X.dtype)\n"
        "    out.index_add_(0, ids, X)\n"
        "    return out\n"
    ),
    solution_np=(
        "def segment_sum(X, ids, K):\n"
        "    out = np.zeros((K, X.shape[1]), dtype=X.dtype)\n"
        "    np.add.at(out, ids, X)\n"
        "    return out\n"
    ),
    traps=[
        "Using out[ids] = X, which keeps only the last row per bucket.",
        "Forgetting that an id with no rows must produce a zero row, not a gap.",
        "Looping over K, which is the thing being tested against.",
    ],
    tests='''
def checks(fn, check):
    X = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    ids = torch.tensor([0, 1, 0, 2])
    out = fn(X, ids, 3)
    check("shape", lambda: shape(out) == (3, 2))
    check("duplicates accumulate", lambda: close(out[0], torch.tensor([4., 4.])))
    check("single membership passes through", lambda: close(out[1], torch.tensor([2., 2.])))
    check("empty bucket is zero", lambda: close(fn(X, ids, 5)[4], torch.zeros(2)))
    check("total is preserved", lambda: close(out.sum(), X.sum(), 1e-5))
    check("all-same id sums everything",
          lambda: close(fn(X, torch.zeros(4, dtype=torch.long), 1)[0], X.sum(0)))
''',
),

task(
    id="masked-softmax",
    title="Softmax with a mask",
    book=BOOK, chapter=C_INDEX,
    section="Indexing, gather, scatter",
    level=2,
    entry="masked_softmax",
    statement=(
        "Softmax over the last axis while ignoring masked positions, which must "
        "receive exactly zero weight. A row that is entirely masked must return "
        "zeros rather than NaN — that row occurs in any padded batch, and -inf "
        "masking turns it into NaN that spreads through everything downstream."
    ),
    shapes="x (..., N) · mask (..., N) bool, True = keep  ->  (..., N)",
    stub=("def masked_softmax(x, mask):\n"
          "    # masked positions get exactly 0; an all-masked row gives zeros\n    pass\n"),
    hints=[
        "Fill the masked positions with a very negative but finite value before "
        "the softmax — torch.finfo(x.dtype).min.",
        "After the softmax, force the masked positions to exactly 0; the finite "
        "fill leaves them merely tiny.",
        "For an all-masked row, the softmax gives a uniform row, which the "
        "post-multiply by the mask then zeroes.",
    ],
    solution=(
        "def masked_softmax(x, mask):\n"
        "    x = x.masked_fill(~mask, torch.finfo(x.dtype).min)\n"
        "    out = torch.softmax(x, -1)\n"
        "    return out * mask.to(out.dtype)\n"
    ),
    solution_np=(
        "def masked_softmax(x, mask):\n"
        "    x = np.where(mask, x, np.finfo(x.dtype).min)\n"
        "    e = np.exp(x - x.max(-1, keepdims=True))\n"
        "    out = e / e.sum(-1, keepdims=True)\n"
        "    return out * mask.astype(out.dtype)\n"
    ),
    traps=[
        "Filling with -inf, which NaNs an all-masked row.",
        "Skipping the post-multiply, leaving masked positions at a tiny non-zero "
        "weight that leaks information.",
        "Zeroing before the softmax rather than after — exp(0) is 1, not 0.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(3, 5)
    m = torch.tensor([[1,1,1,0,0],[1,0,1,0,1],[0,0,0,0,0]]).bool()
    out = fn(x, m)
    check("masked positions are exactly zero", lambda: close(out[~m], torch.zeros(int((~m).sum()))))
    check("kept rows sum to 1", lambda: close(out[0].sum(), torch.tensor(1.0), 1e-5))
    check("all-masked row is all zeros and finite",
          lambda: close(out[2], torch.zeros(5)) and bool(torch.isfinite(out[2]).all()))
    check("no NaN anywhere", lambda: bool(torch.isfinite(out).all()))
    check("matches plain softmax when nothing is masked",
          lambda: close(fn(x, torch.ones_like(m)), torch.softmax(x, -1), 1e-5))
    def relative_weights_preserved():
        full = torch.softmax(x[0, :3], -1)
        return close(out[0, :3], full, 1e-4)
    check("kept weights match a softmax over the kept entries alone",
          relative_weights_preserved)
''',
),

task(
    id="einsum-toolkit",
    title="Five operations, one notation",
    book=BOOK, chapter=C_EINSUM,
    section="einsum and matmul",
    level=2,
    entry="einsum_ops",
    statement=(
        "Return a dict of five results computed with torch.einsum only: the trace "
        "of M, the outer product of u and v, the batched matrix product of A and "
        "B, the row sums of M, and the diagonal of M. The rule is one sentence — "
        "indices in the output are kept, indices absent from it are summed — and "
        "these five exercise every part of it."
    ),
    shapes=("M (n, n) · u (p,) · v (q,) · A (b, i, k) · B (b, k, j)"
            "  ->  dict 'trace', 'outer', 'bmm', 'rowsum', 'diag'"),
    stub=("def einsum_ops(M, u, v, A, B):\n"
          "    # every value must come from torch.einsum\n"
          "    return {'trace': ..., 'outer': ..., 'bmm': ...,\n"
          "            'rowsum': ..., 'diag': ...}\n"),
    hints=[
        "A repeated index within one operand selects the diagonal: 'ii'.",
        "Dropping an index from the output sums over it; keeping it does not.",
        "trace 'ii->', outer 'i,j->ij', bmm 'bik,bkj->bij', rowsum 'ij->i', "
        "diag 'ii->i'.",
    ],
    solution=(
        "def einsum_ops(M, u, v, A, B):\n"
        "    return {\n"
        "        'trace': torch.einsum('ii->', M),\n"
        "        'outer': torch.einsum('i,j->ij', u, v),\n"
        "        'bmm': torch.einsum('bik,bkj->bij', A, B),\n"
        "        'rowsum': torch.einsum('ij->i', M),\n"
        "        'diag': torch.einsum('ii->i', M),\n"
        "    }\n"
    ),
    solution_np=(
        "def einsum_ops(M, u, v, A, B):\n"
        "    return {\n"
        "        'trace': np.einsum('ii->', M),\n"
        "        'outer': np.einsum('i,j->ij', u, v),\n"
        "        'bmm': np.einsum('bik,bkj->bij', A, B),\n"
        "        'rowsum': np.einsum('ij->i', M),\n"
        "        'diag': np.einsum('ii->i', M),\n"
        "    }\n"
    ),
    traps=[
        "Writing 'ii->i' for the trace, which returns the diagonal instead of its "
        "sum.",
        "Summing the wrong axis for rowsum — 'ij->i' keeps rows, 'ij->j' keeps "
        "columns.",
        "Forgetting the batch index in bmm, which contracts across the batch.",
    ],
    tests='''
def checks(fn, check):
    M = torch.randn(4, 4); u = torch.randn(3); v = torch.randn(5)
    A = torch.randn(2, 3, 6); B = torch.randn(2, 6, 4)
    o = fn(M, u, v, A, B)
    check("trace", lambda: close(o["trace"], torch.trace(M), 1e-5))
    check("outer", lambda: close(o["outer"], torch.outer(u, v), 1e-5))
    check("batched matmul", lambda: close(o["bmm"], A @ B, 1e-4))
    check("row sums", lambda: close(o["rowsum"], M.sum(-1), 1e-5))
    check("diagonal", lambda: close(o["diag"], torch.diagonal(M), 1e-6))
    check("trace is a scalar, not the diagonal", lambda: o["trace"].ndim == 0)
''',
),

task(
    id="sliding-windows",
    title="Sliding windows without copying",
    book=BOOK, chapter=C_JOIN,
    section="Joining, splitting, windowing",
    level=3,
    entry="windows",
    statement=(
        "Return all overlapping windows of a 1-D sequence as a 2-D tensor, using "
        "unfold rather than a loop or a stack of slices. unfold is a pure "
        "re-striding, so it allocates nothing — which is what makes it the right "
        "tool for n-gram features or local attention over a long sequence."
    ),
    shapes="x (N,) · size int · step int  ->  ((N - size)//step + 1, size)",
    stub="def windows(x, size, step=1):\n    # -> stacked sliding windows\n    pass\n",
    hints=[
        "Tensor.unfold(dimension, size, step) returns a view with a new trailing "
        "axis of length `size`.",
        "For a 1-D input, unfold(0, size, step) is the whole answer.",
        "The number of windows is (N - size) // step + 1.",
    ],
    solution=(
        "def windows(x, size, step=1):\n"
        "    return x.unfold(0, size, step)\n"
    ),
    traps=[
        "Building the result with a Python loop and torch.stack, which copies.",
        "Off-by-one in the window count — the last window must fit entirely.",
        "Assuming the result is contiguous; it is a strided view, so a later "
        ".view() will refuse.",
    ],
    frameworks=["torch"],
    tests='''
def checks(fn, check):
    x = torch.arange(10.)
    w = fn(x, 3, 1)
    check("shape", lambda: shape(w) == (8, 3))
    check("first window", lambda: close(w[0], torch.tensor([0., 1., 2.])))
    check("last window", lambda: close(w[-1], torch.tensor([7., 8., 9.])))
    check("stride 2 halves the count", lambda: shape(fn(x, 3, 2)) == (4, 3))
    check("stride 2 skips correctly", lambda: close(fn(x, 3, 2)[1], torch.tensor([2., 3., 4.])))
    check("a window equal to the length gives one row", lambda: shape(fn(x, 10, 1)) == (1, 10))
    check("shares storage with the input (no copy)",
          lambda: fn(x, 3, 1).data_ptr() == x.data_ptr())
''',
),

task(
    id="moving-average",
    title="Moving average by prefix sums",
    book=BOOK, chapter=C_JOIN,
    section="Joining, splitting, windowing",
    level=2,
    entry="moving_average",
    statement=(
        "Return the mean of every window of length k, in O(N) rather than O(N·k), "
        "using a cumulative sum. The window sum is a difference of two prefix "
        "sums; the only real work is padding the prefix array with a leading zero "
        "so the first window has something to subtract."
    ),
    shapes="x (N,) · k int  ->  (N - k + 1,)",
    stub="def moving_average(x, k):\n    # -> mean of each length-k window\n    pass\n",
    hints=[
        "Build c = cumsum(x) prefixed by a single 0, giving length N+1.",
        "The sum of the window starting at i is c[i+k] - c[i].",
        "Divide by k. Slicing c handles every window at once.",
    ],
    solution=(
        "def moving_average(x, k):\n"
        "    c = torch.cat([torch.zeros(1, dtype=x.dtype), x.cumsum(0)])\n"
        "    return (c[k:] - c[:-k]) / k\n"
    ),
    solution_np=(
        "def moving_average(x, k):\n"
        "    c = np.concatenate([np.zeros(1, dtype=x.dtype), np.cumsum(x)])\n"
        "    return (c[k:] - c[:-k]) / k\n"
    ),
    traps=[
        "Forgetting the leading zero, which breaks the first window and shifts "
        "everything by one.",
        "Returning N values instead of N-k+1 — no padding was asked for.",
        "Accumulating in float32 over a very long sequence, where cumsum loses "
        "precision; float64 is safer for that case.",
    ],
    tests='''
def checks(fn, check):
    x = torch.arange(1., 11.)
    out = fn(x, 3)
    check("length is N-k+1", lambda: shape(out) == (8,))
    check("first window", lambda: close(out[0], torch.tensor(2.0), 1e-5))
    check("last window", lambda: close(out[-1], torch.tensor(9.0), 1e-5))
    check("k=1 is the identity", lambda: close(fn(x, 1), x, 1e-5))
    check("k=N is the overall mean", lambda: close(fn(x, 10), x.mean()[None], 1e-5))
    check("matches an explicit loop",
          lambda: close(out, torch.tensor([float(x[i:i+3].mean()) for i in range(8)]), 1e-5))
    check("constant input returns the constant",
          lambda: close(fn(torch.full((6,), 7.), 3), torch.full((4,), 7.), 1e-5))
''',
),

task(
    id="make-batch",
    title="Pad a batch and build its mask",
    book=BOOK, chapter=C_JOIN,
    section="Joining, splitting, windowing",
    level=2,
    entry="make_batch",
    statement=(
        "Given a list of variable-length sequences, return a padded batch and the "
        "boolean mask marking real positions. The mask comes from comparing a "
        "position index against each length — a broadcast between a row vector and "
        "a column vector, not a loop."
    ),
    shapes="seqs list of (L_i, D)  ->  (batch (B, L_max, D), mask (B, L_max) bool)",
    stub=("def make_batch(seqs):\n"
          "    # -> (padded batch, bool mask where True = real)\n    pass\n"),
    hints=[
        "torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True) does the padding.",
        "Collect the lengths, then compare torch.arange(L_max)[None, :] against "
        "lengths[:, None].",
        "Return both, batch first.",
    ],
    solution=(
        "def make_batch(seqs):\n"
        "    batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)\n"
        "    lens = torch.tensor([s.shape[0] for s in seqs])\n"
        "    mask = torch.arange(batch.shape[1])[None, :] < lens[:, None]\n"
        "    return batch, mask\n"
    ),
    frameworks=["torch"],
    traps=[
        "Forgetting batch_first, which returns (L, B, D) and silently transposes "
        "everything downstream.",
        "Building the mask with <= instead of <, which marks one padding position "
        "as real.",
        "Assuming every sequence has the same feature width — pad_sequence "
        "requires it.",
    ],
    tests='''
def checks(fn, check):
    seqs = [torch.ones(3, 2), torch.ones(5, 2) * 2, torch.ones(2, 2) * 3]
    b, m = fn(seqs)
    check("batch shape", lambda: shape(b) == (3, 5, 2))
    check("mask shape", lambda: shape(m) == (3, 5))
    check("mask counts the true lengths", lambda: m.sum(-1).tolist() == [3, 5, 2])
    check("padding is zero", lambda: close(b[0, 3:], torch.zeros(2, 2)))
    check("content is preserved", lambda: close(b[1], torch.ones(5, 2) * 2))
    check("mask is boolean", lambda: m.dtype == torch.bool)
    check("masked mean recovers the per-sequence mean",
          lambda: close((b * m.unsqueeze(-1)).sum(1) / m.sum(-1, keepdim=True),
                        torch.tensor([[1., 1.], [2., 2.], [3., 3.]]), 1e-5))
''',
),

task(
    id="detach-semantics",
    title="Where the gradient stops",
    book=BOOK, chapter=C_GRAD,
    section="Autograd and dtypes",
    level=2,
    entry="split_grad",
    statement=(
        "Given x, return a dict with three values: `full` = (x*x).sum(), "
        "`stopped` = (x * x.detach()).sum(), and `ratio` = the gradient of "
        "`stopped` divided by the gradient of `full`, elementwise. Detaching one "
        "factor halves the gradient — the product rule contributes one term "
        "instead of two, which is exactly the mechanism behind the "
        "straight-through estimator and every stop-gradient trick."
    ),
    shapes="x (N,) requires_grad  ->  dict 'full', 'stopped', 'ratio' (N,)",
    stub=("def split_grad(x):\n"
          "    # -> {'full': scalar, 'stopped': scalar, 'ratio': (N,)}\n    pass\n"),
    hints=[
        "d/dx of (x*x).sum() is 2x; of (x * x.detach()).sum() it is x.",
        "Use torch.autograd.grad(out, x) to get a gradient without touching .grad.",
        "The ratio is therefore 0.5 everywhere x is non-zero.",
    ],
    solution=(
        "def split_grad(x):\n"
        "    full = (x * x).sum()\n"
        "    stopped = (x * x.detach()).sum()\n"
        "    g_full = torch.autograd.grad(full, x, retain_graph=True)[0]\n"
        "    g_stop = torch.autograd.grad(stopped, x, retain_graph=True)[0]\n"
        "    return {'full': full, 'stopped': stopped, 'ratio': g_stop / g_full}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Calling .backward() twice without retain_graph, which raises on the "
        "second call.",
        "Expecting the two forward values to differ — they are numerically "
        "identical; only the gradients differ.",
        "Reading x.grad after two backward passes, which accumulates them.",
    ],
    tests='''
def checks(fn, check):
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    o = fn(x)
    check("forward values are identical", lambda: close(o["full"], o["stopped"], 1e-6))
    check("detaching halves the gradient",
          lambda: close(o["ratio"], torch.full((3,), 0.5), 1e-5))
    check("ratio shape", lambda: shape(o["ratio"]) == (3,))
    check("full value is the sum of squares", lambda: close(o["full"], torch.tensor(14.0), 1e-5))
    def works_on_other_inputs():
        y = torch.tensor([4., -2.], requires_grad=True)
        return close(fn(y)["ratio"], torch.full((2,), 0.5), 1e-5)
    check("holds for other inputs, including negatives", works_on_other_inputs)
''',
),

task(
    id="dtype-promotion",
    title="Predict the result dtype",
    book=BOOK, chapter=C_GRAD,
    section="Autograd and dtypes",
    level=1,
    entry="result_dtype",
    statement=(
        "Return the dtype of a + b without computing the full sum — use "
        "torch.promote_types. Knowing this matters because the default float "
        "differs between the libraries: torch.tensor([1.0]) is float32 and "
        "np.array([1.0]) is float64, so converting a NumPy array into a model "
        "silently doubles memory unless you cast."
    ),
    shapes="a tensor · b tensor  ->  torch.dtype",
    stub="def result_dtype(a, b):\n    # -> the dtype that a + b would have\n    pass\n",
    hints=[
        "torch.promote_types(dt1, dt2) answers this directly.",
        "It takes dtypes, not tensors — pass a.dtype and b.dtype.",
        "Integer plus float promotes to the float type; two different float "
        "widths promote to the wider.",
    ],
    solution=(
        "def result_dtype(a, b):\n"
        "    return torch.promote_types(a.dtype, b.dtype)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Computing a + b and reading its dtype, which allocates the whole result.",
        "Assuming int + float stays integer.",
        "Expecting float16 + float32 to be float16 — it promotes upward, not down.",
    ],
    tests='''
def checks(fn, check):
    check("float32 + float32", lambda: fn(torch.zeros(1), torch.zeros(1)) == torch.float32)
    check("int64 + float32 promotes to float32",
          lambda: fn(torch.zeros(1, dtype=torch.long), torch.zeros(1)) == torch.float32)
    check("float16 + float32 promotes upward",
          lambda: fn(torch.zeros(1, dtype=torch.float16), torch.zeros(1)) == torch.float32)
    check("float64 wins over float32",
          lambda: fn(torch.zeros(1, dtype=torch.float64), torch.zeros(1)) == torch.float64)
    check("int32 + int64", lambda: fn(torch.zeros(1, dtype=torch.int32),
                                      torch.zeros(1, dtype=torch.long)) == torch.int64)
    check("agrees with an actual addition",
          lambda: fn(torch.zeros(1, dtype=torch.float16), torch.zeros(1, dtype=torch.float64))
                  == (torch.zeros(1, dtype=torch.float16) + torch.zeros(1, dtype=torch.float64)).dtype)
''',
),

task(
    id="contiguous-view",
    title="Reshape after a permute",
    book=BOOK, chapter=C_SHAPE,
    section="Shapes, views, and memory",
    level=2,
    entry="merge_heads",
    statement=(
        "Take a (B, H, L, Dh) attention output and return (B, L, H·Dh), with the "
        "features of head 0 first. The transpose leaves a permuted layout that "
        "view refuses, so a copy is required — and whether it is required depends "
        "on how the tensor was produced, not on its shape."
    ),
    shapes="x (B, H, L, Dh)  ->  (B, L, H*Dh)",
    stub="def merge_heads(x):\n    # (B, H, L, Dh) -> (B, L, H*Dh)\n    pass\n",
    hints=[
        "Move the head axis next to the feature axis with transpose(1, 2).",
        "The result is not contiguous, so .view() will raise; call .contiguous() "
        "first, or use .reshape() which copies when it must.",
        "Order matters: transpose first, then merge the last two axes.",
    ],
    solution=(
        "def merge_heads(x):\n"
        "    B, H, L, Dh = x.shape\n"
        "    return x.transpose(1, 2).contiguous().view(B, L, H * Dh)\n"
    ),
    solution_np=(
        "def merge_heads(x):\n"
        "    B, H, L, Dh = x.shape\n"
        "    return np.ascontiguousarray(x.swapaxes(1, 2)).reshape(B, L, H * Dh)\n"
    ),
    traps=[
        "Calling .view() directly on the transposed tensor, which raises.",
        "Reshaping without transposing, which interleaves heads and features.",
        "Assuming .contiguous() is always a copy — it is a no-op when the tensor "
        "already is contiguous, which is why a round-trip test can hide the bug.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(2, 4, 5, 3)
    out = fn(x)
    check("shape", lambda: shape(out) == (2, 5, 12))
    check("head 0 lands in the first Dh features", lambda: close(out[:, :, :3], x[:, 0]))
    check("head 1 lands in the next Dh features", lambda: close(out[:, :, 3:6], x[:, 1]))
    check("inverts split_heads",
          lambda: close(fn(x.view(2, 4, 5, 3)), out, 1e-6))
    def round_trips():
        y = torch.randn(2, 5, 12)
        split = y.view(2, 5, 4, 3).transpose(1, 2)
        return close(fn(split), y, 1e-6)
    check("round-trips with a split", round_trips)
''',
),

]
