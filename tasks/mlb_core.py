"""ml-basics — probability, the Gaussian, regression, classification, networks."""
from .schema import task

BOOK = "ml-basics"
C_PROB = "Probability and estimation · The Gaussian"
C_REG = "Linear regression"
C_CLS = "Linear classification"
C_NN = "Neural networks"

TASKS = [

task(
    id="mle-gaussian",
    title="Maximum likelihood for a Gaussian",
    book=BOOK, chapter=C_PROB,
    section="Probability and estimation",
    level=1,
    entry="mle",
    statement=(
        "Return the maximum-likelihood mean and variance of a sample. The ML "
        "variance divides by N, not N-1 — it is biased low, and the correction to "
        "N-1 is a deliberate departure from maximum likelihood, not a refinement "
        "of it. Return the ML values."
    ),
    shapes="x (N,) float  ->  dict 'mean', 'var'",
    stub="def mle(x):\n    # -> {'mean': ..., 'var': ...}, both maximum likelihood\n    pass\n",
    hints=[
        "Setting the derivative of the log-likelihood to zero gives the sample mean.",
        "The ML variance is the mean squared deviation from that mean.",
        "Divide by N. Torch's .var() defaults to N-1, so pass unbiased=False.",
    ],
    solution=(
        "def mle(x):\n"
        "    mu = x.mean()\n"
        "    return {'mean': mu, 'var': ((x - mu) ** 2).mean()}\n"
    ),
    solution_np=(
        "def mle(x):\n"
        "    mu = x.mean()\n"
        "    return {'mean': mu, 'var': ((x - mu) ** 2).mean()}\n"
    ),
    traps=[
        "Dividing by N-1, which is the unbiased estimator, not the ML one.",
        "Using a fixed mean rather than the estimated one when forming the variance.",
        "Assuming ML is unbiased in general — for the Gaussian variance it is not.",
    ],
    tests='''
def checks(fn, check):
    x = torch.tensor([2., 4., 4., 4., 5., 5., 7., 9.])
    o = fn(x)
    check("mean is the sample mean", lambda: close(o["mean"], torch.tensor(5.0)))
    check("variance divides by N", lambda: close(o["var"], torch.tensor(4.0)))
    check("not the unbiased estimator",
          lambda: not close(o["var"], x.var(unbiased=True), 1e-4))
    check("constant data has zero variance",
          lambda: close(fn(torch.full((5,), 3.))["var"], torch.tensor(0.)))
    check("shift changes mean but not variance",
          lambda: close(fn(x + 10.)["var"], o["var"], 1e-4))
''',
),

task(
    id="gaussian-logpdf",
    title="Multivariate Gaussian log density",
    book=BOOK, chapter=C_PROB,
    section="The Gaussian",
    level=2,
    entry="logpdf",
    statement=(
        "Return the log density of a multivariate normal at each row of X, given "
        "a mean and a covariance. Work through a Cholesky factor rather than "
        "inverting: solving L z = (x-mu) gives the quadratic form as ‖z‖², and "
        "the log determinant is twice the sum of log-diagonal — both numerically "
        "stabler and cheaper than an explicit inverse."
    ),
    shapes="X (N, D) · mu (D,) · cov (D, D) SPD  ->  (N,) log densities",
    stub="def logpdf(X, mu, cov):\n    # -> (N,) log N(x | mu, cov)\n    pass\n",
    hints=[
        "log p = -0.5·(D·log 2pi + log|Σ| + (x-mu)ᵀ Σ⁻¹ (x-mu)).",
        "With Σ = L Lᵀ, solve L z = (x-mu); then the quadratic form is z·z.",
        "log|Σ| = 2·sum(log(diag(L))).",
    ],
    solution=(
        "def logpdf(X, mu, cov):\n"
        "    D = X.shape[-1]\n"
        "    L = torch.linalg.cholesky(cov)\n"
        "    diff = (X - mu).T\n"
        "    z = torch.linalg.solve_triangular(L, diff, upper=False)\n"
        "    quad = (z ** 2).sum(0)\n"
        "    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()\n"
        "    return -0.5 * (D * math.log(2 * math.pi) + logdet + quad)\n"
    ),
    solution_np=(
        "def logpdf(X, mu, cov):\n"
        "    D = X.shape[-1]\n"
        "    L = np.linalg.cholesky(cov)\n"
        "    z = np.linalg.solve(L, (X - mu).T)\n"
        "    quad = (z ** 2).sum(0)\n"
        "    logdet = 2.0 * np.log(np.diag(L)).sum()\n"
        "    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)\n"
    ),
    traps=[
        "Inverting the covariance explicitly, which is slower and loses precision "
        "on ill-conditioned matrices.",
        "Using log|L| instead of 2·log|L| for the log determinant.",
        "Dropping the D·log(2pi) normaliser, which cancels in a comparison but "
        "not in a likelihood.",
    ],
    tests='''
def checks(fn, check):
    X = torch.randn(5, 3)
    mu = torch.zeros(3)
    A = torch.randn(3, 3)
    cov = A @ A.T + 3 * torch.eye(3)
    want = torch.distributions.MultivariateNormal(mu, cov).log_prob(X)
    check("matches torch.distributions", lambda: close(fn(X, mu, cov), want, 1e-4))
    check("output shape", lambda: shape(fn(X, mu, cov)) == (5,))
    check("standard normal in 1-D",
          lambda: close(fn(torch.zeros(1, 1), torch.zeros(1), torch.eye(1)),
                        torch.tensor([-0.5 * math.log(2 * math.pi)]), 1e-5))
    check("density is highest at the mean",
          lambda: float(fn(mu[None], mu, cov)) > float(fn(mu[None] + 5.0, mu, cov)))
    check("exponentiates to a normalised density in 1-D",
          lambda: abs(float(torch.exp(fn(torch.linspace(-8, 8, 4001)[:, None],
                                         torch.zeros(1), torch.eye(1))).sum() * 0.004) - 1.0) < 1e-3)
''',
),

task(
    id="kl-gaussians",
    title="KL between two Gaussians",
    book=BOOK, chapter=C_PROB,
    section="The Gaussian · Information theory",
    level=2,
    entry="kl",
    statement=(
        "Return KL(N(mu0, S0) ‖ N(mu1, S1)) in closed form: "
        "0.5·(tr(S1⁻¹S0) + (mu1-mu0)ᵀS1⁻¹(mu1-mu0) - D + log(|S1|/|S0|)). This is "
        "the term that regularises a VAE's encoder toward the prior, and the "
        "closed form is why no sampling is needed to compute it."
    ),
    shapes="mu0 (D,) · S0 (D,D) · mu1 (D,) · S1 (D,D)  ->  scalar >= 0",
    stub="def kl(mu0, S0, mu1, S1):\n    # -> KL( N(mu0,S0) || N(mu1,S1) )\n    pass\n",
    hints=[
        "Four terms: a trace, a Mahalanobis distance, minus D, and a log-det ratio.",
        "Every inverse is against S1, the second distribution.",
        "torch.linalg.slogdet gives a stable log determinant.",
    ],
    solution=(
        "def kl(mu0, S0, mu1, S1):\n"
        "    D = mu0.shape[0]\n"
        "    S1i = torch.linalg.inv(S1)\n"
        "    diff = (mu1 - mu0)\n"
        "    tr = torch.trace(S1i @ S0)\n"
        "    quad = diff @ S1i @ diff\n"
        "    logdet = torch.linalg.slogdet(S1)[1] - torch.linalg.slogdet(S0)[1]\n"
        "    return 0.5 * (tr + quad - D + logdet)\n"
    ),
    solution_np=(
        "def kl(mu0, S0, mu1, S1):\n"
        "    D = mu0.shape[0]\n"
        "    S1i = np.linalg.inv(S1)\n"
        "    diff = mu1 - mu0\n"
        "    tr = np.trace(S1i @ S0)\n"
        "    quad = diff @ S1i @ diff\n"
        "    logdet = np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1]\n"
        "    return 0.5 * (tr + quad - D + logdet)\n"
    ),
    traps=[
        "Inverting S0 instead of S1 — KL is not symmetric, and the inverse belongs "
        "to the second argument.",
        "Getting the log-det ratio upside down.",
        "Forgetting the -D term, which is what makes KL(p‖p) exactly zero.",
    ],
    tests='''
def checks(fn, check):
    D = 3
    mu0, mu1 = torch.zeros(D), torch.ones(D)
    A = torch.randn(D, D); S0 = A @ A.T + 2 * torch.eye(D)
    B = torch.randn(D, D); S1 = B @ B.T + 2 * torch.eye(D)
    check("identical distributions give zero",
          lambda: abs(float(fn(mu0, S0, mu0, S0.clone()))) < 1e-4)
    check("non-negative", lambda: float(fn(mu0, S0, mu1, S1)) >= -1e-5)
    check("matches torch.distributions.kl_divergence",
          lambda: close(fn(mu0, S0, mu1, S1),
                        torch.distributions.kl_divergence(
                            torch.distributions.MultivariateNormal(mu0, S0),
                            torch.distributions.MultivariateNormal(mu1, S1)), 1e-4))
    check("asymmetric", lambda: not close(fn(mu0, S0, mu1, S1), fn(mu1, S1, mu0, S0), 1e-3))
    check("unit-variance shift is half the squared distance",
          lambda: abs(float(fn(torch.zeros(1), torch.eye(1),
                               torch.tensor([2.]), torch.eye(1))) - 2.0) < 1e-4)
''',
),

task(
    id="ols",
    title="Least squares by the normal equations",
    book=BOOK, chapter=C_REG,
    section="Linear regression",
    level=1,
    entry="ols",
    statement=(
        "Fit w minimising ‖Xw - y‖². The normal equations give XᵀX w = Xᵀy. Solve "
        "them — do not form (XᵀX)⁻¹ explicitly, because squaring the design "
        "matrix squares its condition number, and a least-squares solver works "
        "directly on X without that penalty."
    ),
    shapes="X (N, D) · y (N,)  ->  w (D,)",
    stub="def ols(X, y):\n    # -> least-squares coefficients w\n    pass\n",
    hints=[
        "The minimiser satisfies XᵀX w = Xᵀ y.",
        "Use a solver rather than an inverse.",
        "torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1) does it "
        "directly and stably.",
    ],
    solution=(
        "def ols(X, y):\n"
        "    return torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1)\n"
    ),
    solution_np=(
        "def ols(X, y):\n"
        "    w, *_ = np.linalg.lstsq(X, y, rcond=None)\n"
        "    return w\n"
    ),
    traps=[
        "Computing inv(X.T @ X) @ X.T @ y, which is numerically worse and slower.",
        "Forgetting that X must include an explicit bias column if an intercept "
        "is wanted — lstsq does not add one.",
        "Assuming a unique solution when X is rank-deficient.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    X = torch.randn(40, 3)
    w_true = torch.tensor([1.5, -2.0, 0.5])
    y = X @ w_true
    check("recovers the exact solution on noiseless data",
          lambda: close(fn(X, y), w_true, 1e-4))
    check("output shape", lambda: shape(fn(X, y)) == (3,))
    def residual_orthogonal():
        yn = y + 0.1 * torch.randn(40)
        return close(X.T @ (yn - X @ fn(X, yn)), torch.zeros(3), 1e-3)
    check("residual is orthogonal to the columns", residual_orthogonal)
    check("matches torch's lstsq",
          lambda: close(fn(X, y), torch.linalg.lstsq(X, y.unsqueeze(-1)).solution.squeeze(-1), 1e-5))
    def beats_a_perturbation():
        w = fn(X, y)
        return float((y - X @ w).norm()) <= float((y - X @ (w + 0.1)).norm())
    check("minimises the residual", beats_a_perturbation)
''',
),

task(
    id="ridge",
    title="Ridge regression",
    book=BOOK, chapter=C_REG,
    section="Linear regression",
    level=2,
    entry="ridge",
    statement=(
        "Fit w minimising ‖Xw - y‖² + lam·‖w‖², whose solution is "
        "(XᵀX + lam·I)⁻¹Xᵀy. The added lam·I makes the system invertible even when "
        "XᵀX is singular, which is the practical reason ridge is used on "
        "collinear or wide data — and why the solution shrinks toward zero as lam "
        "grows."
    ),
    shapes="X (N, D) · y (N,) · lam float  ->  w (D,)",
    stub="def ridge(X, y, lam):\n    # -> ridge coefficients\n    pass\n",
    hints=[
        "Form A = XᵀX + lam·I and b = Xᵀy, then solve A w = b.",
        "Do not penalise by lam·I of the wrong size — I is D×D.",
        "torch.linalg.solve(A, b) rather than an explicit inverse.",
    ],
    solution=(
        "def ridge(X, y, lam):\n"
        "    D = X.shape[1]\n"
        "    A = X.T @ X + lam * torch.eye(D, dtype=X.dtype)\n"
        "    return torch.linalg.solve(A, X.T @ y)\n"
    ),
    solution_np=(
        "def ridge(X, y, lam):\n"
        "    D = X.shape[1]\n"
        "    A = X.T @ X + lam * np.eye(D)\n"
        "    return np.linalg.solve(A, X.T @ y)\n"
    ),
    traps=[
        "Penalising the intercept along with the weights, which makes the fit "
        "depend on where y is centred.",
        "Adding lam to XᵀX elementwise rather than to its diagonal.",
        "Expecting lam = 0 to be well posed on rank-deficient X — it is not, "
        "which is the whole point.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    X = torch.randn(30, 4)
    w_true = torch.tensor([1., -1., 2., 0.5])
    y = X @ w_true
    check("lam -> 0 approaches least squares",
          lambda: close(fn(X, y, 1e-8), w_true, 1e-3))
    check("shrinks toward zero as lam grows",
          lambda: float(fn(X, y, 100.).norm()) < float(fn(X, y, 1.).norm()))
    check("very large lam drives w to zero",
          lambda: float(fn(X, y, 1e9).norm()) < 1e-3)
    check("output shape", lambda: shape(fn(X, y, 1.0)) == (4,))
    def handles_singular():
        Xs = torch.cat([X[:, :2], X[:, :2]], dim=1)   # rank 2, D = 4
        return bool(torch.isfinite(fn(Xs, y, 1.0)).all())
    check("stays finite on a rank-deficient design", handles_singular)
    check("matches the closed form",
          lambda: close(fn(X, y, 2.0),
                        torch.linalg.solve(X.T @ X + 2.0 * torch.eye(4), X.T @ y), 1e-5))
''',
),

task(
    id="logistic-gradient",
    title="Logistic regression gradient",
    book=BOOK, chapter=C_CLS,
    section="Linear classification",
    level=2,
    entry="logistic",
    statement=(
        "Return the mean negative log-likelihood of logistic regression and its "
        "gradient with respect to w. The gradient is Xᵀ(sigmoid(Xw) - y)/N — the "
        "same predicted-minus-target structure as softmax cross-entropy, and for "
        "the same reason. Compute the loss with a numerically stable form, not "
        "log(sigmoid(z))."
    ),
    shapes="X (N, D) · y (N,) in {0,1} · w (D,)  ->  dict 'loss' scalar, 'grad' (D,)",
    stub=("def logistic(X, y, w):\n"
          "    # -> {'loss': scalar mean NLL, 'grad': (D,)}\n    pass\n"),
    hints=[
        "z = Xw. The stable per-example loss is softplus(z) - y·z, which equals "
        "log(1+e^z) - y·z and never overflows.",
        "The gradient of that with respect to z is sigmoid(z) - y.",
        "Chain back through X: grad = Xᵀ(sigmoid(z) - y) / N.",
    ],
    solution=(
        "def logistic(X, y, w):\n"
        "    z = X @ w\n"
        "    loss = (F.softplus(z) - y * z).mean()\n"
        "    grad = X.T @ (torch.sigmoid(z) - y) / X.shape[0]\n"
        "    return {'loss': loss, 'grad': grad}\n"
    ),
    solution_np=(
        "def logistic(X, y, w):\n"
        "    z = X @ w\n"
        "    loss = (np.logaddexp(0.0, z) - y * z).mean()\n"
        "    grad = X.T @ (1.0 / (1.0 + np.exp(-z)) - y) / X.shape[0]\n"
        "    return {'loss': loss, 'grad': grad}\n"
    ),
    traps=[
        "Using log(sigmoid(z)), which underflows to -inf for confidently wrong "
        "predictions.",
        "Forgetting to divide the gradient by N when the loss is a mean.",
        "Transposing the wrong way and getting an (N,) gradient instead of (D,).",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    y = (torch.randn(20) > 0).float()
    w = torch.randn(3)
    o = fn(X, y, w)
    check("gradient shape", lambda: shape(o["grad"]) == (3,))
    check("loss matches BCE-with-logits",
          lambda: close(o["loss"], F.binary_cross_entropy_with_logits(X @ w, y), 1e-5))
    def matches_autograd():
        ww = w.clone().requires_grad_(True)
        F.binary_cross_entropy_with_logits(X @ ww, y).backward()
        return close(fn(X, y, w)["grad"], ww.grad, 1e-5)
    check("gradient matches autograd", matches_autograd)
    check("zero weights give log 2",
          lambda: close(fn(X, y, torch.zeros(3))["loss"], torch.tensor(math.log(2)), 1e-5))
    check("stable at extreme logits",
          lambda: bool(torch.isfinite(fn(X * 500, y, w)["loss"]).all()))
''',
),

task(
    id="mlp-backward",
    title="Backprop through a two-layer MLP",
    book=BOOK, chapter=C_NN,
    section="Neural networks",
    level=3,
    entry="mlp_backward",
    statement=(
        "Given a two-layer network h = relu(X W1 + b1), out = h W2 + b2 and the "
        "upstream gradient of a scalar loss with respect to out, return the "
        "gradients with respect to W1, b1, W2, b2. Write the chain rule by hand — "
        "this is the classic whiteboard exercise, and the ReLU's derivative is "
        "where most attempts go wrong."
    ),
    shapes=("X (N,D) · W1 (D,H) · b1 (H,) · W2 (H,C) · b2 (C,) · dout (N,C)"
            "  ->  dict 'W1', 'b1', 'W2', 'b2'"),
    stub=("def mlp_backward(X, W1, b1, W2, b2, dout):\n"
          "    # -> {'W1':…, 'b1':…, 'W2':…, 'b2':…}\n    pass\n"),
    hints=[
        "Recompute the forward pass first: z1 = XW1+b1, h = relu(z1).",
        "dW2 = hᵀ·dout, db2 = sum of dout over the batch, dh = dout·W2ᵀ.",
        "Through the ReLU: dz1 = dh · (z1 > 0). Then dW1 = Xᵀ·dz1 and "
        "db1 = sum of dz1 over the batch.",
    ],
    solution=(
        "def mlp_backward(X, W1, b1, W2, b2, dout):\n"
        "    z1 = X @ W1 + b1\n"
        "    h = torch.relu(z1)\n"
        "    dW2 = h.T @ dout\n"
        "    db2 = dout.sum(0)\n"
        "    dh = dout @ W2.T\n"
        "    dz1 = dh * (z1 > 0).to(dh.dtype)\n"
        "    dW1 = X.T @ dz1\n"
        "    db1 = dz1.sum(0)\n"
        "    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}\n"
    ),
    solution_np=(
        "def mlp_backward(X, W1, b1, W2, b2, dout):\n"
        "    z1 = X @ W1 + b1\n"
        "    h = np.maximum(z1, 0)\n"
        "    dW2 = h.T @ dout\n"
        "    db2 = dout.sum(0)\n"
        "    dh = dout @ W2.T\n"
        "    dz1 = dh * (z1 > 0)\n"
        "    dW1 = X.T @ dz1\n"
        "    db1 = dz1.sum(0)\n"
        "    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}\n"
    ),
    traps=[
        "Applying the ReLU mask to h rather than to z1 — equivalent here only "
        "because relu(z)>0 iff z>0, but wrong the moment the nonlinearity changes.",
        "Using W2 instead of W2ᵀ when propagating back to h.",
        "Averaging the bias gradients instead of summing, which silently rescales "
        "them by N.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    N, D, H, C = 6, 4, 5, 3
    X = torch.randn(N, D)
    W1 = torch.randn(D, H, requires_grad=True); b1 = torch.randn(H, requires_grad=True)
    W2 = torch.randn(H, C, requires_grad=True); b2 = torch.randn(C, requires_grad=True)
    dout = torch.randn(N, C)
    out = torch.relu(X @ W1 + b1) @ W2 + b2
    out.backward(dout)
    got = fn(X, W1.detach(), b1.detach(), W2.detach(), b2.detach(), dout)
    check("dW1 matches autograd", lambda: close(got["W1"], W1.grad, 1e-4))
    check("db1 matches autograd", lambda: close(got["b1"], b1.grad, 1e-4))
    check("dW2 matches autograd", lambda: close(got["W2"], W2.grad, 1e-4))
    check("db2 matches autograd", lambda: close(got["b2"], b2.grad, 1e-4))
    check("shapes are right",
          lambda: shape(got["W1"]) == (D, H) and shape(got["b1"]) == (H,)
                  and shape(got["W2"]) == (H, C) and shape(got["b2"]) == (C,))
    def relu_gate_applied():
        # with all-negative pre-activations every first-layer gradient vanishes
        g = fn(X, torch.zeros(D, H), torch.full((H,), -50.),
               W2.detach(), b2.detach(), dout)
        return close(g["W1"], torch.zeros(D, H), 1e-6)
    check("the ReLU gate blocks gradient where z1 <= 0", relu_gate_applied)
''',
),

task(
    id="reparameterise",
    title="The reparameterisation trick",
    book=BOOK, chapter="Variational inference and sampling",
    section="Variational inference and sampling",
    level=2,
    entry="reparameterise",
    statement=(
        "Sample z ~ N(mu, diag(exp(logvar))) in a way that keeps the gradient "
        "flowing to mu and logvar. Sampling directly from the distribution blocks "
        "the gradient — the trick is to draw the randomness from a fixed "
        "distribution and make z a differentiable function of it: "
        "z = mu + exp(0.5·logvar)·eps with eps ~ N(0, I)."
    ),
    shapes="mu (N, D) · logvar (N, D) · eps (N, D) drawn N(0,I)  ->  (N, D)",
    stub=("def reparameterise(mu, logvar, eps):\n"
          "    # -> z, differentiable in mu and logvar\n    pass\n"),
    hints=[
        "The standard deviation is exp(0.5·logvar), not exp(logvar).",
        "z = mu + sigma·eps. Nothing else is needed.",
        "Half the log-variance because sigma is the square root of the variance.",
    ],
    solution=(
        "def reparameterise(mu, logvar, eps):\n"
        "    return mu + torch.exp(0.5 * logvar) * eps\n"
    ),
    frameworks=["torch"],
    traps=[
        "Using exp(logvar) as the scale, which squares the intended standard "
        "deviation.",
        "Drawing eps inside the function, which is fine numerically but makes the "
        "function untestable and hides the fixed-noise idea.",
        "Parameterising sigma directly and letting it go negative — the log "
        "parameterisation exists to prevent that.",
    ],
    tests='''
def checks(fn, check):
    mu = torch.zeros(2, 3); lv = torch.zeros(2, 3); eps = torch.randn(2, 3)
    check("unit variance reduces to mu + eps", lambda: close(fn(mu, lv, eps), eps, 1e-6))
    check("scale is exp(logvar/2)",
          lambda: close(fn(mu, torch.full((2, 3), 2.0), torch.ones(2, 3)),
                        torch.full((2, 3), math.e), 1e-5))
    check("zero noise returns the mean",
          lambda: close(fn(torch.full((2, 3), 5.), lv, torch.zeros(2, 3)),
                        torch.full((2, 3), 5.), 1e-6))
    def grads_flow():
        m = torch.zeros(4, requires_grad=True)
        v = torch.zeros(4, requires_grad=True)
        fn(m, v, torch.ones(4)).sum().backward()
        return m.grad is not None and v.grad is not None and close(m.grad, torch.ones(4))
    check("gradients reach both mu and logvar", grads_flow)
    def empirical_moments():
        m = torch.full((20000, 1), 2.0); l = torch.full((20000, 1), math.log(9.0))
        z = fn(m, l, torch.randn(20000, 1))
        return abs(float(z.mean()) - 2.0) < 0.1 and abs(float(z.std()) - 3.0) < 0.1
    check("samples have the intended mean and standard deviation", empirical_moments)
''',
),

]
