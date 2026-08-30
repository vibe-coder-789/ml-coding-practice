"""ml-basics — PCA and k-means, kernels and GPs, mixtures and EM, sequences."""
from .schema import task

BOOK = "ml-basics"
C_UNSUP = "PCA and k-means"
C_KERN = "Kernels, SVMs, and Gaussian processes"
C_EM = "Mixtures and EM"
C_SEQ = "Sequential models"
C_BAYES = "Bayesian machinery"

TASKS = [

task(
    id="pca",
    title="Principal components",
    book=BOOK, chapter=C_UNSUP,
    section="PCA and k-means",
    level=2,
    entry="pca",
    statement=(
        "Return the top-k principal directions of X and the variance each "
        "explains. Centre the data first — PCA without centring finds the "
        "directions of largest second moment, which is a different and usually "
        "wrong answer. Take the SVD of the centred matrix rather than forming the "
        "covariance, which squares the condition number."
    ),
    shapes="X (N, D) · k int  ->  dict 'components' (k, D), 'variance' (k,)",
    stub=("def pca(X, k):\n"
          "    # -> {'components': (k, D), 'variance': (k,)}\n    pass\n"),
    hints=[
        "Subtract the column means first.",
        "SVD the centred matrix: Xc = U diag(S) Vᵀ. The rows of Vᵀ are the "
        "principal directions, already ordered by decreasing S.",
        "The variance along component i is S_i² / (N - 1).",
    ],
    solution=(
        "def pca(X, k):\n"
        "    Xc = X - X.mean(0, keepdim=True)\n"
        "    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)\n"
        "    return {'components': Vh[:k], 'variance': (S[:k] ** 2) / (X.shape[0] - 1)}\n"
    ),
    solution_np=(
        "def pca(X, k):\n"
        "    Xc = X - X.mean(0, keepdims=True)\n"
        "    U, S, Vh = np.linalg.svd(Xc, full_matrices=False)\n"
        "    return {'components': Vh[:k], 'variance': (S[:k] ** 2) / (X.shape[0] - 1)}\n"
    ),
    traps=[
        "Skipping the centring step.",
        "Returning the columns of V rather than the rows of Vᵀ, transposing the "
        "components.",
        "Dividing by N instead of N-1 when reporting explained variance, which "
        "disagrees with the sample covariance.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    # data lying mostly along one direction
    t = torch.randn(200, 1)
    X = t @ torch.tensor([[3.0, 1.0]]) + 0.05 * torch.randn(200, 2)
    o = fn(X, 1)
    check("component shape", lambda: shape(o["components"]) == (1, 2))
    check("variance shape", lambda: shape(o["variance"]) == (1,))
    check("finds the dominant direction",
          lambda: abs(abs(float(o["components"][0] @
                    (torch.tensor([3.0, 1.0]) / torch.tensor([3.0, 1.0]).norm()))) - 1.0) < 0.02)
    check("components are unit norm", lambda: close(o["components"].norm(dim=-1), torch.ones(1), 1e-4))
    check("variance matches the sample covariance eigenvalue",
          lambda: abs(float(o["variance"][0]) -
                      float(torch.linalg.eigvalsh(torch.cov(X.T)).max())) < 1e-2)
    def centring_matters():
        shifted = X + 100.0
        return close(fn(shifted, 1)["variance"], o["variance"], 1e-2)
    check("result is invariant to a shift (data is centred)", centring_matters)
    check("orthogonal components for k=2",
          lambda: abs(float(fn(X, 2)["components"][0] @ fn(X, 2)["components"][1])) < 1e-4)
''',
),

task(
    id="kmeans-step",
    title="One Lloyd iteration",
    book=BOOK, chapter=C_UNSUP,
    section="PCA and k-means",
    level=2,
    entry="kmeans_step",
    statement=(
        "Perform one k-means iteration: assign every point to its nearest centre, "
        "then move each centre to the mean of the points assigned to it. Compute "
        "the distances without a Python loop. Handle an empty cluster by leaving "
        "its centre where it is, rather than producing NaN."
    ),
    shapes="X (N, D) · centres (K, D)  ->  (assignments (N,) int64, new_centres (K, D))",
    stub=("def kmeans_step(X, centres):\n"
          "    # -> (assignments, updated centres)\n    pass\n"),
    hints=[
        "Pairwise distances come from broadcasting: X[:, None, :] - centres[None, :, :].",
        "Assign with argmin over the centre axis; the squared distance suffices, "
        "no square root needed.",
        "For the update, average the points of each cluster; if a cluster is "
        "empty, keep the old centre.",
    ],
    solution=(
        "def kmeans_step(X, centres):\n"
        "    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)\n"
        "    assign = d2.argmin(-1)\n"
        "    new = centres.clone()\n"
        "    for k in range(centres.shape[0]):\n"
        "        m = assign == k\n"
        "        if bool(m.any()):\n"
        "            new[k] = X[m].mean(0)\n"
        "    return assign, new\n"
    ),
    solution_np=(
        "def kmeans_step(X, centres):\n"
        "    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)\n"
        "    assign = d2.argmin(-1)\n"
        "    new = centres.copy()\n"
        "    for k in range(centres.shape[0]):\n"
        "        m = assign == k\n"
        "        if m.any():\n"
        "            new[k] = X[m].mean(0)\n"
        "    return assign, new\n"
    ),
    traps=[
        "Producing NaN for an empty cluster by averaging nothing.",
        "Taking a square root that changes nothing about the argmin but costs time.",
        "Updating centres one at a time while assigning, which is a different "
        "algorithm (MacQueen's) with different fixed points.",
    ],
    tests='''
def checks(fn, check):
    X = torch.tensor([[0., 0.], [0.1, 0.], [5., 5.], [5.1, 5.]])
    c = torch.tensor([[0., 0.], [5., 5.]])
    a, nc = fn(X, c)
    check("assignment shape", lambda: shape(a) == (4,))
    check("centre shape", lambda: shape(nc) == (2, 2))
    check("points go to the nearer centre", lambda: a.tolist() == [0, 0, 1, 1])
    check("centres move to the cluster means",
          lambda: close(nc, torch.tensor([[0.05, 0.], [5.05, 5.]]), 1e-5))
    def empty_cluster():
        far = torch.tensor([[0., 0.], [0.1, 0.]])
        cc = torch.tensor([[0., 0.], [99., 99.]])
        _, out = fn(far, cc)
        return bool(torch.isfinite(out).all()) and close(out[1], torch.tensor([99., 99.]))
    check("an empty cluster keeps its centre and does not NaN", empty_cluster)
    def converged_is_fixed():
        a2, c2 = fn(X, nc)
        return close(c2, nc, 1e-5)
    check("a converged configuration is a fixed point", converged_is_fixed)
''',
),

task(
    id="rbf-kernel",
    title="RBF kernel matrix",
    book=BOOK, chapter=C_KERN,
    section="Kernels, SVMs, and Gaussian processes",
    level=1,
    entry="rbf",
    statement=(
        "Build the Gaussian kernel matrix k(x, y) = exp(-‖x-y‖²/(2·l²)) between "
        "two sets of points, without a Python loop. The squared-distance expansion "
        "‖x-y‖² = ‖x‖² + ‖y‖² - 2x·y turns this into one matrix product, which is "
        "the trick worth knowing — it is how every kernel method scales."
    ),
    shapes="A (N, D) · B (M, D) · length float  ->  (N, M)",
    stub="def rbf(A, B, length=1.0):\n    # -> (N, M) kernel matrix\n    pass\n",
    hints=[
        "Broadcasting A[:, None, :] - B[None, :, :] works but allocates N·M·D.",
        "The expansion ‖x‖² + ‖y‖² - 2x·y needs only an (N, M) matrix product.",
        "Clamp the squared distance at 0 before exponentiating — the expansion can "
        "go slightly negative in floating point.",
    ],
    solution=(
        "def rbf(A, B, length=1.0):\n"
        "    d2 = (A * A).sum(-1)[:, None] + (B * B).sum(-1)[None, :] - 2 * A @ B.T\n"
        "    d2 = d2.clamp(min=0)\n"
        "    return torch.exp(-d2 / (2 * length ** 2))\n"
    ),
    solution_np=(
        "def rbf(A, B, length=1.0):\n"
        "    d2 = (A * A).sum(-1)[:, None] + (B * B).sum(-1)[None, :] - 2 * A @ B.T\n"
        "    d2 = np.maximum(d2, 0)\n"
        "    return np.exp(-d2 / (2 * length ** 2))\n"
    ),
    traps=[
        "Letting a tiny negative squared distance through, which makes the kernel "
        "exceed 1.",
        "Forgetting the factor 2 in the denominator, which rescales the length "
        "scale by sqrt(2).",
        "Writing the nested loop the problem exists to avoid.",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(5, 3); B = torch.randn(4, 3)
    K = fn(A, B)
    check("shape", lambda: shape(K) == (5, 4))
    check("self-kernel is 1 on the diagonal",
          lambda: close(torch.diagonal(fn(A, A)), torch.ones(5), 1e-5))
    check("values lie in (0, 1]", lambda: bool(((K > 0) & (K <= 1 + 1e-6)).all()))
    check("symmetric when A is B", lambda: close(fn(A, A), fn(A, A).T, 1e-6))
    check("matches the direct broadcast form",
          lambda: close(K, torch.exp(-((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / 2), 1e-4))
    check("a longer length scale gives a flatter kernel",
          lambda: bool((fn(A, B, 5.0) >= K - 1e-6).all()))
    check("self-kernel is positive semidefinite",
          lambda: bool((torch.linalg.eigvalsh(fn(A, A) + 1e-6 * torch.eye(5)) > 0).all()))
''',
),

task(
    id="hinge-loss",
    title="SVM hinge loss and subgradient",
    book=BOOK, chapter=C_KERN,
    section="Kernels, SVMs, and Gaussian processes",
    level=2,
    entry="hinge",
    statement=(
        "Return the regularised hinge objective mean(max(0, 1 - y·(Xw))) + "
        "0.5·lam·‖w‖² and its subgradient. Labels are ±1, not 0/1. Only the "
        "examples inside the margin contribute — the rest have exactly zero "
        "gradient, which is what makes the solution depend on a sparse set of "
        "support vectors."
    ),
    shapes="X (N, D) · y (N,) in {-1,+1} · w (D,) · lam float  ->  dict 'loss', 'grad'",
    stub=("def hinge(X, y, w, lam=0.01):\n"
          "    # -> {'loss': scalar, 'grad': (D,)}\n    pass\n"),
    hints=[
        "The margin is m = y·(Xw); the per-example loss is max(0, 1-m).",
        "A subgradient of max(0, 1-m) with respect to w is -y·x when m < 1, and 0 "
        "otherwise.",
        "Add the regulariser's gradient, lam·w, and remember the data term is a "
        "mean.",
    ],
    solution=(
        "def hinge(X, y, w, lam=0.01):\n"
        "    m = y * (X @ w)\n"
        "    loss = torch.clamp(1 - m, min=0).mean() + 0.5 * lam * (w @ w)\n"
        "    active = (m < 1).to(X.dtype)\n"
        "    grad = -(X * (active * y)[:, None]).mean(0) + lam * w\n"
        "    return {'loss': loss, 'grad': grad}\n"
    ),
    solution_np=(
        "def hinge(X, y, w, lam=0.01):\n"
        "    m = y * (X @ w)\n"
        "    loss = np.maximum(1 - m, 0).mean() + 0.5 * lam * (w @ w)\n"
        "    active = (m < 1).astype(X.dtype)\n"
        "    grad = -(X * (active * y)[:, None]).mean(0) + lam * w\n"
        "    return {'loss': loss, 'grad': grad}\n"
    ),
    traps=[
        "Using 0/1 labels, which makes the margin meaningless for the negative class.",
        "Including correctly classified points beyond the margin in the gradient.",
        "Forgetting the regulariser's contribution to the gradient, or "
        "double-counting the 0.5.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    y = torch.sign(torch.randn(20))
    w = torch.randn(3)
    o = fn(X, y, w, 0.01)
    check("grad shape", lambda: shape(o["grad"]) == (3,))
    def matches_autograd():
        ww = w.clone().requires_grad_(True)
        (torch.clamp(1 - y * (X @ ww), min=0).mean() + 0.5 * 0.01 * (ww @ ww)).backward()
        return close(fn(X, y, w, 0.01)["grad"], ww.grad, 1e-5)
    check("subgradient matches autograd", matches_autograd)
    def wide_margin_is_zero():
        Xw = torch.eye(2); yw = torch.tensor([1., -1.])
        big = torch.tensor([100., -100.])
        return abs(float(fn(Xw, yw, big, 0.0)["loss"])) < 1e-6
    check("points outside the margin contribute nothing", wide_margin_is_zero)
    check("zero weights give loss 1 plus regulariser",
          lambda: abs(float(fn(X, y, torch.zeros(3), 0.0)["loss"]) - 1.0) < 1e-6)
    check("regulariser enters the gradient",
          lambda: not close(fn(X, y, w, 1.0)["grad"], fn(X, y, w, 0.0)["grad"], 1e-3))
''',
),

task(
    id="gp-posterior",
    title="Gaussian process posterior",
    book=BOOK, chapter=C_KERN,
    section="Kernels, SVMs, and Gaussian processes",
    level=3,
    entry="gp_posterior",
    statement=(
        "Given a kernel matrix on training inputs, the cross-kernel to test "
        "inputs, and the test prior variance, return the GP predictive mean and "
        "variance under Gaussian noise. mean = K*ᵀ(K + σ²I)⁻¹y and "
        "var = k** - K*ᵀ(K + σ²I)⁻¹K*. Solve through a Cholesky factor rather "
        "than inverting."
    ),
    shapes=("K (N,N) · Ks (N,M) · kss (M,) · y (N,) · noise float"
            "  ->  dict 'mean' (M,), 'var' (M,)"),
    stub=("def gp_posterior(K, Ks, kss, y, noise=1e-2):\n"
          "    # -> {'mean': (M,), 'var': (M,)}\n    pass\n"),
    hints=[
        "Form A = K + noise·I and factor it once: L = cholesky(A).",
        "mean = Ksᵀ · A⁻¹ y — solve, do not invert.",
        "var = kss - sum over the training axis of (L⁻¹Ks)², i.e. the diagonal of "
        "KsᵀA⁻¹Ks.",
    ],
    solution=(
        "def gp_posterior(K, Ks, kss, y, noise=1e-2):\n"
        "    N = K.shape[0]\n"
        "    A = K + noise * torch.eye(N, dtype=K.dtype)\n"
        "    L = torch.linalg.cholesky(A)\n"
        "    alpha = torch.cholesky_solve(y[:, None], L)\n"
        "    mean = (Ks.T @ alpha).squeeze(-1)\n"
        "    v = torch.linalg.solve_triangular(L, Ks, upper=False)\n"
        "    var = kss - (v ** 2).sum(0)\n"
        "    return {'mean': mean, 'var': var}\n"
    ),
    solution_np=(
        "def gp_posterior(K, Ks, kss, y, noise=1e-2):\n"
        "    N = K.shape[0]\n"
        "    A = K + noise * np.eye(N)\n"
        "    L = np.linalg.cholesky(A)\n"
        "    alpha = np.linalg.solve(A, y)\n"
        "    mean = Ks.T @ alpha\n"
        "    v = np.linalg.solve(L, Ks)\n"
        "    var = kss - (v ** 2).sum(0)\n"
        "    return {'mean': mean, 'var': var}\n"
    ),
    traps=[
        "Omitting the noise term, which makes the Cholesky fail on a kernel matrix "
        "that is only positive semidefinite.",
        "Returning the full posterior covariance when only its diagonal was asked "
        "for.",
        "Letting the variance go slightly negative through rounding and not "
        "noticing.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    Xtr = torch.linspace(-3, 3, 12)[:, None]
    Xte = torch.linspace(-3, 3, 7)[:, None]
    def rbf(A, B):
        d2 = (A * A).sum(-1)[:, None] + (B * B).sum(-1)[None, :] - 2 * A @ B.T
        return torch.exp(-d2.clamp(min=0) / 2)
    K, Ks, kss = rbf(Xtr, Xtr), rbf(Xtr, Xte), torch.ones(7)
    y = torch.sin(Xtr.squeeze(-1))
    o = fn(K, Ks, kss, y, 1e-4)
    check("mean shape", lambda: shape(o["mean"]) == (7,))
    check("var shape", lambda: shape(o["var"]) == (7,))
    check("variance is non-negative", lambda: bool((o["var"] > -1e-6).all()))
    check("variance is below the prior", lambda: bool((o["var"] <= kss + 1e-6).all()))
    check("interpolates the training function",
          lambda: close(fn(K, K, torch.ones(12), y, 1e-6)["mean"], y, 1e-2))
    check("variance is near zero at a training point",
          lambda: float(fn(K, K, torch.ones(12), y, 1e-6)["var"].max()) < 1e-2)
    check("more noise gives more posterior variance",
          lambda: float(fn(K, Ks, kss, y, 1.0)["var"].mean()) >
                  float(fn(K, Ks, kss, y, 1e-4)["var"].mean()))
''',
),

task(
    id="gmm-estep",
    title="GMM responsibilities (the E step)",
    book=BOOK, chapter=C_EM,
    section="Mixtures and EM",
    level=3,
    entry="e_step",
    statement=(
        "Compute the responsibilities of each Gaussian component for each point: "
        "the posterior probability that component k generated point n. Work in "
        "log space and normalise with log-sum-exp — the raw densities underflow "
        "for any point far from all components, which is exactly where EM "
        "otherwise produces NaN and the fit collapses."
    ),
    shapes=("X (N, D) · means (K, D) · variances (K,) isotropic · weights (K,)"
            "  ->  (N, K) responsibilities, rows summing to 1"),
    stub=("def e_step(X, means, variances, weights):\n"
          "    # -> (N, K) posterior responsibilities\n    pass\n"),
    hints=[
        "For an isotropic component, log N(x|mu, s²I) = "
        "-0.5·(D·log(2pi s²) + ‖x-mu‖²/s²).",
        "Add log weights to get the joint log probability, giving an (N, K) matrix.",
        "Normalise each row by subtracting its log-sum-exp, then exponentiate.",
    ],
    solution=(
        "def e_step(X, means, variances, weights):\n"
        "    D = X.shape[1]\n"
        "    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)\n"
        "    logp = -0.5 * (D * torch.log(2 * math.pi * variances)[None, :]\n"
        "                   + d2 / variances[None, :])\n"
        "    logj = logp + torch.log(weights)[None, :]\n"
        "    return torch.softmax(logj, dim=-1)\n"
    ),
    solution_np=(
        "def e_step(X, means, variances, weights):\n"
        "    D = X.shape[1]\n"
        "    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)\n"
        "    logp = -0.5 * (D * np.log(2 * np.pi * variances)[None, :]\n"
        "                   + d2 / variances[None, :])\n"
        "    logj = logp + np.log(weights)[None, :]\n"
        "    m = logj.max(-1, keepdims=True)\n"
        "    e = np.exp(logj - m)\n"
        "    return e / e.sum(-1, keepdims=True)\n"
    ),
    traps=[
        "Computing densities directly and dividing, which underflows to 0/0 for "
        "distant points.",
        "Forgetting the mixing weights, which turns the posterior into a likelihood "
        "comparison.",
        "Normalising over the wrong axis — responsibilities sum to 1 across "
        "components, not across points.",
    ],
    tests='''
def checks(fn, check):
    X = torch.tensor([[0., 0.], [10., 10.], [5., 5.]])
    means = torch.tensor([[0., 0.], [10., 10.]])
    var = torch.tensor([1., 1.])
    w = torch.tensor([0.5, 0.5])
    r = fn(X, means, var, w)
    check("shape", lambda: shape(r) == (3, 2))
    check("rows sum to 1", lambda: close(r.sum(-1), torch.ones(3), 1e-5))
    check("a point on a component is claimed by it", lambda: float(r[0, 0]) > 0.99)
    check("the other point likewise", lambda: float(r[1, 1]) > 0.99)
    check("an equidistant point splits evenly",
          lambda: abs(float(r[2, 0]) - 0.5) < 1e-4)
    def far_point_is_finite():
        far = torch.tensor([[1e3, 1e3]])
        out = fn(far, means, var, w)
        return bool(torch.isfinite(out).all()) and abs(float(out.sum()) - 1.0) < 1e-4
    check("a very distant point does not underflow to NaN", far_point_is_finite)
    check("mixing weights shift the posterior",
          lambda: float(fn(X, means, var, torch.tensor([0.9, 0.1]))[2, 0]) > 0.5)
''',
),

task(
    id="hmm-forward",
    title="HMM forward algorithm",
    book=BOOK, chapter=C_SEQ,
    section="Sequential models",
    level=3,
    entry="forward",
    statement=(
        "Return the log-likelihood of an observation sequence under a discrete "
        "HMM, by the forward recursion. Work in log space throughout: the "
        "unscaled alphas shrink geometrically and underflow to zero within a few "
        "dozen steps, which is why the textbook recursion is always implemented "
        "either scaled or in logs."
    ),
    shapes=("log_pi (K,) · log_A (K,K) · log_B (K,V) · obs (T,) int"
            "  ->  scalar log P(obs)"),
    stub=("def forward(log_pi, log_A, log_B, obs):\n"
          "    # -> scalar log-likelihood of the observation sequence\n    pass\n"),
    hints=[
        "Initialise alpha = log_pi + log_B[:, obs[0]].",
        "Each step: alpha_next[j] = logsumexp_i(alpha[i] + log_A[i,j]) + "
        "log_B[j, obs[t]].",
        "The answer is logsumexp of the final alpha. Use torch.logsumexp, never "
        "log(sum(exp(...))).",
    ],
    solution=(
        "def forward(log_pi, log_A, log_B, obs):\n"
        "    alpha = log_pi + log_B[:, obs[0]]\n"
        "    for t in range(1, len(obs)):\n"
        "        alpha = torch.logsumexp(alpha[:, None] + log_A, dim=0) + log_B[:, obs[t]]\n"
        "    return torch.logsumexp(alpha, dim=0)\n"
    ),
    solution_np=(
        "def forward(log_pi, log_A, log_B, obs):\n"
        "    from scipy.special import logsumexp as _lse\n"
        "    alpha = log_pi + log_B[:, obs[0]]\n"
        "    for t in range(1, len(obs)):\n"
        "        m = (alpha[:, None] + log_A).max(0)\n"
        "        alpha = m + np.log(np.exp(alpha[:, None] + log_A - m).sum(0)) + log_B[:, obs[t]]\n"
        "    m = alpha.max()\n"
        "    return m + np.log(np.exp(alpha - m).sum())\n"
    ),
    frameworks=["torch"],
    traps=[
        "Multiplying probabilities directly, which underflows on any realistic "
        "sequence length.",
        "Summing over the wrong axis in the transition step — the sum is over the "
        "source state i, not the destination j.",
        "Forgetting the emission term on the first step.",
    ],
    tests='''
def checks(fn, check):
    K, V = 2, 3
    log_pi = torch.log(torch.tensor([0.6, 0.4]))
    A = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    B = torch.tensor([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    log_A, log_B = A.log(), B.log()

    def brute(obs):
        total = 0.0
        import itertools
        for path in itertools.product(range(K), repeat=len(obs)):
            p = float(log_pi[path[0]].exp() * B[path[0], obs[0]])
            for t in range(1, len(obs)):
                p *= float(A[path[t-1], path[t]] * B[path[t], obs[t]])
            total += p
        return math.log(total)

    o1 = torch.tensor([0, 1, 2])
    check("matches brute-force enumeration (T=3)",
          lambda: abs(float(fn(log_pi, log_A, log_B, o1)) - brute([0, 1, 2])) < 1e-4)
    o2 = torch.tensor([2, 2, 0, 1, 1])
    check("matches brute force on a longer sequence",
          lambda: abs(float(fn(log_pi, log_A, log_B, o2)) - brute([2, 2, 0, 1, 1])) < 1e-4)
    check("single observation is the marginal",
          lambda: abs(float(fn(log_pi, log_A, log_B, torch.tensor([0]))) -
                      math.log(0.6 * 0.5 + 0.4 * 0.1)) < 1e-5)
    check("returns a scalar", lambda: fn(log_pi, log_A, log_B, o1).ndim == 0)
    def no_underflow():
        long_obs = torch.zeros(500, dtype=torch.long)
        v = fn(log_pi, log_A, log_B, long_obs)
        return bool(torch.isfinite(v)) and float(v) < -100
    check("survives a 500-step sequence without underflowing", no_underflow)
''',
),

task(
    id="bayes-linear",
    title="Bayesian linear regression posterior",
    book=BOOK, chapter=C_BAYES,
    section="Bayesian machinery",
    level=3,
    entry="posterior",
    statement=(
        "With a Gaussian prior N(0, alpha⁻¹I) on the weights and Gaussian noise of "
        "precision beta, the posterior over w is Gaussian with "
        "S⁻¹ = alpha·I + beta·XᵀX and mean = beta·S·Xᵀy. Return that mean and "
        "covariance. As alpha grows the mean shrinks toward zero — the prior and "
        "ridge regularisation are the same thing seen from two directions."
    ),
    shapes="X (N, D) · y (N,) · alpha float · beta float  ->  dict 'mean' (D,), 'cov' (D,D)",
    stub=("def posterior(X, y, alpha, beta):\n"
          "    # -> {'mean': (D,), 'cov': (D, D)}\n    pass\n"),
    hints=[
        "Build the precision first: S_inv = alpha·I + beta·XᵀX.",
        "The covariance is its inverse; invert once and reuse.",
        "mean = beta · S · Xᵀ y.",
    ],
    solution=(
        "def posterior(X, y, alpha, beta):\n"
        "    D = X.shape[1]\n"
        "    S_inv = alpha * torch.eye(D, dtype=X.dtype) + beta * (X.T @ X)\n"
        "    S = torch.linalg.inv(S_inv)\n"
        "    return {'mean': beta * (S @ (X.T @ y)), 'cov': S}\n"
    ),
    solution_np=(
        "def posterior(X, y, alpha, beta):\n"
        "    D = X.shape[1]\n"
        "    S_inv = alpha * np.eye(D) + beta * (X.T @ X)\n"
        "    S = np.linalg.inv(S_inv)\n"
        "    return {'mean': beta * (S @ (X.T @ y)), 'cov': S}\n"
    ),
    traps=[
        "Dropping beta from the mean, which mis-scales it whenever the noise "
        "precision is not 1.",
        "Returning the precision where the covariance was asked for.",
        "Assuming the posterior mean equals the least-squares solution — it does "
        "only in the limit alpha -> 0.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    X = torch.randn(25, 3)
    w_true = torch.tensor([1., -2., 0.5])
    y = X @ w_true
    o = fn(X, y, 1e-6, 1e6)
    check("mean shape", lambda: shape(o["mean"]) == (3,))
    check("cov shape", lambda: shape(o["cov"]) == (3, 3))
    check("weak prior recovers least squares",
          lambda: close(o["mean"], w_true, 1e-2))
    check("covariance is symmetric", lambda: close(o["cov"], o["cov"].T, 1e-6))
    check("covariance is positive definite",
          lambda: bool((torch.linalg.eigvalsh(o["cov"]) > 0).all()))
    check("a strong prior shrinks the mean",
          lambda: float(fn(X, y, 1e6, 1.0)["mean"].norm()) < 1e-2)
    check("equals ridge with lam = alpha/beta",
          lambda: close(fn(X, y, 2.0, 1.0)["mean"],
                        torch.linalg.solve(X.T @ X + 2.0 * torch.eye(3), X.T @ y), 1e-4))
    check("more data shrinks the posterior variance",
          lambda: float(torch.trace(fn(X, y, 1.0, 1.0)["cov"])) <
                  float(torch.trace(fn(X[:5], y[:5], 1.0, 1.0)["cov"])))
''',
),

]
