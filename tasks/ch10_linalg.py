"""Chapter 10 — Linear algebra and matrix calculus."""
from .schema import task

CH = "10 · Linear algebra and matrix calculus"

TASKS = [

task(
    id="low-rank",
    title="Best rank-k approximation",
    chapter=CH,
    section="10.3 The singular value decomposition",
    level=2,
    entry="low_rank",
    statement=(
        "Return the best rank-k approximation of A in Frobenius norm, by "
        "truncating its SVD to the k largest singular values. The Eckart–Young "
        "theorem says no rank-k matrix does better, which is why truncated SVD is "
        "the reference every compression scheme is measured against — and the "
        "reason a LoRA adapter of rank k can only ever capture so much."
    ),
    shapes="A (m, n) · k int  ->  (m, n) of rank k",
    stub="def low_rank(A, k):\n    # -> best rank-k approximation of A\n    pass\n",
    hints=[
        "Take a reduced SVD: A = U diag(S) Vᵀ.",
        "Keep the first k columns of U, the first k values of S, the first k rows "
        "of Vᵀ.",
        "Reassemble: U[:, :k] @ diag(S[:k]) @ Vh[:k, :].",
    ],
    solution=(
        "def low_rank(A, k):\n"
        "    U, S, Vh = torch.linalg.svd(A, full_matrices=False)\n"
        "    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]\n"
    ),
    solution_np=(
        "def low_rank(A, k):\n"
        "    U, S, Vh = np.linalg.svd(A, full_matrices=False)\n"
        "    return U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]\n"
    ),
    traps=[
        "Zeroing the smallest singular values but keeping full-size factors, which "
        "is the same matrix but wastes the compression.",
        "Assuming the singular values come back unsorted — they are descending.",
        "Using eigendecomposition instead of SVD on a non-symmetric matrix.",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(6, 5)
    Ak = fn(A, 2)
    check("shape is preserved", lambda: shape(Ak) == (6, 5))
    check("rank is k", lambda: int(torch.linalg.matrix_rank(Ak)) == 2)
    check("k = min(m, n) reconstructs A exactly", lambda: close(fn(A, 5), A, 1e-4))
    def is_optimal():
        err = (A - Ak).norm()
        S = torch.linalg.svdvals(A)
        return abs(float(err) - float((S[2:] ** 2).sum().sqrt())) < 1e-3
    check("error equals the tail of the spectrum (Eckart-Young)", is_optimal)
    check("beats a random rank-2 matrix",
          lambda: float((A - Ak).norm()) <
                  float((A - torch.randn(6, 2) @ torch.randn(2, 5)).norm()))
''',
),

task(
    id="power-iteration",
    title="Spectral norm by power iteration",
    chapter=CH,
    section="10.4 Matrix norms and duality",
    level=2,
    entry="spectral_norm",
    statement=(
        "Estimate the largest singular value of A by power iteration on AᵀA, "
        "without calling an SVD. Repeatedly map v to Aᵀ(Av) and renormalise; the "
        "iterate converges to the leading right singular vector, and the norm of "
        "Av converges to the largest singular value. This is how spectral "
        "normalisation is done inside a training loop, where a full SVD each step "
        "would be far too slow."
    ),
    shapes="A (m, n) · iters int  ->  scalar estimate of the largest singular value",
    stub=("def spectral_norm(A, iters=100):\n"
          "    # -> largest singular value, by power iteration\n    pass\n"),
    hints=[
        "Start from a random unit vector v of length n.",
        "Each step: u = A v, then v = Aᵀ u, then renormalise v to unit length.",
        "The estimate is ‖A v‖ with v the converged unit vector.",
    ],
    solution=(
        "def spectral_norm(A, iters=100):\n"
        "    v = torch.randn(A.shape[1])\n"
        "    v = v / v.norm()\n"
        "    for _ in range(iters):\n"
        "        v = A.T @ (A @ v)\n"
        "        v = v / (v.norm() + 1e-12)\n"
        "    return (A @ v).norm()\n"
    ),
    solution_np=(
        "def spectral_norm(A, iters=100):\n"
        "    v = np.random.randn(A.shape[1])\n"
        "    v = v / np.linalg.norm(v)\n"
        "    for _ in range(iters):\n"
        "        v = A.T @ (A @ v)\n"
        "        v = v / (np.linalg.norm(v) + 1e-12)\n"
        "    return np.linalg.norm(A @ v)\n"
    ),
    traps=[
        "Renormalising only at the end, so the iterate overflows or underflows.",
        "Returning ‖v‖ rather than ‖Av‖ — v is a unit vector by construction.",
        "Iterating on A rather than AᵀA, which finds an eigenvalue and only "
        "coincides with the singular value when A is symmetric positive definite.",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(6, 4)
    want = float(torch.linalg.matrix_norm(A, 2))
    check("matches torch's spectral norm", lambda: abs(float(fn(A, 200)) - want) < 1e-3)
    check("diagonal matrix gives its largest entry",
          lambda: abs(float(fn(torch.diag(torch.tensor([3., 1., 2.])), 200)) - 3.0) < 1e-4)
    check("identity has norm 1", lambda: abs(float(fn(torch.eye(5), 100)) - 1.0) < 1e-4)
    check("scales linearly", lambda: abs(float(fn(A * 4, 200)) - 4 * want) < 1e-2)
    W = torch.randn(3, 8)
    check("wide matrices work",
          lambda: abs(float(fn(W, 300)) - float(torch.linalg.matrix_norm(W, 2))) < 1e-3)
''',
),

task(
    id="kronecker",
    title="Kronecker product",
    chapter=CH,
    section="10.5 Kronecker products",
    level=2,
    entry="kron",
    statement=(
        "Build the Kronecker product of A (m×n) and B (p×q): the (mp)×(nq) block "
        "matrix whose (i,j) block is A[i,j]·B. Do it with broadcasting and a "
        "reshape rather than nested loops. This structure is what lets K-FAC and "
        "similar preconditioners store a curvature approximation as two small "
        "factors instead of one enormous matrix."
    ),
    shapes="A (m, n) · B (p, q)  ->  (m·p, n·q)",
    stub="def kron(A, B):\n    # -> (m*p, n*q) Kronecker product\n    pass\n",
    hints=[
        "Think of the result as a 4-D array indexed (i, k, j, l) = A[i,j]·B[k,l].",
        "Broadcast A[:, None, :, None] against B[None, :, None, :].",
        "Then reshape the 4-D result to (m·p, n·q) — the axis order above is "
        "already correct.",
    ],
    solution=(
        "def kron(A, B):\n"
        "    m, n = A.shape\n"
        "    p, q = B.shape\n"
        "    out = A[:, None, :, None] * B[None, :, None, :]\n"
        "    return out.reshape(m * p, n * q)\n"
    ),
    solution_np=(
        "def kron(A, B):\n"
        "    m, n = A.shape\n"
        "    p, q = B.shape\n"
        "    out = A[:, None, :, None] * B[None, :, None, :]\n"
        "    return out.reshape(m * p, n * q)\n"
    ),
    traps=[
        "Getting the axis order wrong, producing the blocks transposed — the shape "
        "is right and the values are scrambled.",
        "Writing nested Python loops, which is the thing being tested against.",
        "Assuming it commutes: A⊗B and B⊗A are permutation-similar, not equal.",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(3, 2); B = torch.randn(4, 5)
    check("matches torch.kron", lambda: close(fn(A, B), torch.kron(A, B), 1e-5))
    check("shape is (m*p, n*q)", lambda: shape(fn(A, B)) == (12, 10))
    check("identity kron identity is identity",
          lambda: close(fn(torch.eye(2), torch.eye(3)), torch.eye(6), 1e-6))
    check("the (0,0) block is A[0,0]*B",
          lambda: close(fn(A, B)[:4, :5], A[0, 0] * B, 1e-5))
    check("the (1,0) block is A[1,0]*B",
          lambda: close(fn(A, B)[4:8, :5], A[1, 0] * B, 1e-5))
    check("does not commute in general",
          lambda: not close(fn(A, B), torch.kron(B, A).reshape(12, 10), 1e-3))
''',
),

task(
    id="projection",
    title="Project onto a column space",
    chapter=CH,
    section="10.8 Projections and orthogonalisation",
    level=3,
    entry="project",
    statement=(
        "Project b onto the column space of A: the closest point in that subspace, "
        "P = A(AᵀA)⁻¹Aᵀ b. Equivalently it is the least-squares fit — the residual "
        "b - Pb is orthogonal to every column of A, which is the normal equation "
        "written geometrically."
    ),
    shapes="A (m, n) full column rank · b (m,)  ->  (m,) the projection of b",
    stub="def project(A, b):\n    # -> projection of b onto the column space of A\n    pass\n",
    hints=[
        "Solve the normal equations AᵀA x = Aᵀb for the coefficients x.",
        "The projection is A x.",
        "Prefer torch.linalg.lstsq or solve over forming an explicit inverse — "
        "AᵀA squares the condition number.",
    ],
    solution=(
        "def project(A, b):\n"
        "    x = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution\n"
        "    return (A @ x).squeeze(-1)\n"
    ),
    solution_np=(
        "def project(A, b):\n"
        "    x, *_ = np.linalg.lstsq(A, b, rcond=None)\n"
        "    return A @ x\n"
    ),
    traps=[
        "Forming (AᵀA)⁻¹ explicitly, which squares the condition number and loses "
        "precision on nearly collinear columns.",
        "Returning the coefficients x rather than the projected vector A x.",
        "Assuming A is square — it is generally tall.",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(6, 3); b = torch.randn(6)
    p = fn(A, b)
    check("output shape", lambda: shape(p) == (6,))
    check("residual is orthogonal to every column",
          lambda: close(A.T @ (b - p), torch.zeros(3), 1e-3))
    check("projection is idempotent", lambda: close(fn(A, p), p, 1e-4))
    check("a vector already in the span is unchanged",
          lambda: close(fn(A, A @ torch.tensor([1., 2., 3.])),
                        A @ torch.tensor([1., 2., 3.]), 1e-4))
    check("it is the closest point in the subspace",
          lambda: float((b - p).norm()) <=
                  float((b - A @ torch.tensor([0.3, -0.2, 0.5])).norm()) + 1e-5)
    check("projecting onto a full-rank square A returns b",
          lambda: close(fn(torch.eye(4), torch.tensor([1., 2., 3., 4.])),
                        torch.tensor([1., 2., 3., 4.]), 1e-4))
''',
),

]
