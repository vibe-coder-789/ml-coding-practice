"""Additions to the ml-basics volume."""
from .schema import task

BOOK = "ml-basics"
C_LA = "Linear algebra"
C_PROB = "Probability and estimation · The Gaussian"
C_REG = "Linear regression"
C_CLS = "Linear classification"
C_NN = "Neural networks"
C_UNSUP = "PCA and k-means"
C_EM = "Mixtures and EM"
C_SEQ = "Sequential models"
C_VI = "Variational inference and sampling"

TASKS = [

task(
    id="gram-schmidt",
    title="Modified Gram–Schmidt",
    book=BOOK, chapter=C_LA,
    section="Linear algebra — projection and orthogonalisation",
    level=2,
    entry="gram_schmidt",
    statement=(
        "Orthonormalise the columns of A with MODIFIED Gram–Schmidt: subtract "
        "each new column's projection onto the already-orthogonalised vectors "
        "one at a time, updating as you go. The classical variant — computing "
        "every coefficient against the original columns first — is "
        "algebraically identical and numerically much worse: its loss of "
        "orthogonality grows with the square of the condition number, and the "
        "tests include a matrix ill-conditioned enough to expose it."
    ),
    shapes="A (m, n) float64, full column rank  ->  Q (m, n), Q^T Q = I, span(Q) = span(A)",
    stub=("def gram_schmidt(A):\n"
          "    # modified GS: project against the running Q, not the original A\n    pass\n"),
    hints=[
        "Work on a copy. For column j: subtract (q_i . a_j) q_i for every "
        "earlier i, updating a_j after each subtraction; then normalise.",
        "The 'modified' part is exactly that the inner products use the "
        "already-updated a_j, not the original one.",
        "Q^T A should come out upper triangular with a positive diagonal — a "
        "good self-check.",
    ],
    solution=(
        "def gram_schmidt(A):\n"
        "    Q = A.clone()\n"
        "    m, n = Q.shape\n"
        "    for j in range(n):\n"
        "        for i in range(j):\n"
        "            Q[:, j] = Q[:, j] - (Q[:, i] @ Q[:, j]) * Q[:, i]\n"
        "        Q[:, j] = Q[:, j] / Q[:, j].norm()\n"
        "    return Q\n"
    ),
    solution_np=(
        "def gram_schmidt(A):\n"
        "    Q = A.copy()\n"
        "    m, n = Q.shape\n"
        "    for j in range(n):\n"
        "        for i in range(j):\n"
        "            Q[:, j] = Q[:, j] - (Q[:, i] @ Q[:, j]) * Q[:, i]\n"
        "        Q[:, j] = Q[:, j] / np.linalg.norm(Q[:, j])\n"
        "    return Q\n"
    ),
    traps=[
        "Classical Gram–Schmidt — coefficients against the original columns — "
        "which loses orthogonality as cond(A)^2 times machine epsilon.",
        "Forgetting to normalise each column after the subtractions.",
        "Mutating the caller's matrix instead of a copy.",
    ],
    tests='''
def checks(fn, check):
    A = torch.randn(12, 5, dtype=torch.float64)
    Q = fn(A)
    check("columns are orthonormal",
          lambda: close(Q.T @ Q, torch.eye(5, dtype=torch.float64), 1e-10))
    def upper_triangular():
        R = Q.T @ A
        below = torch.tril(R, diagonal=-1)
        return float(below.abs().max()) < 1e-9 and bool((torch.diagonal(R) > 0).all())
    check("Q^T A is upper triangular with positive diagonal", upper_triangular)
    def matches_qr():
        Qr, _ = torch.linalg.qr(A)
        M = (Q.T @ Qr).abs()
        return close(M, torch.eye(5, dtype=torch.float64), 1e-8)
    check("spans agree with torch.linalg.qr (up to column signs)", matches_qr)
    def lauchli():
        # the textbook stress case: classical GS ends up with an orthogonality
        # error of about 0.5 here; modified GS stays below 1e-8
        eps = 1e-8
        L = torch.zeros(4, 3, dtype=torch.float64)
        L[0] = 1.0
        L[1, 0] = L[2, 1] = L[3, 2] = eps
        Qb = fn(L)
        err = float((Qb.T @ Qb - torch.eye(3, dtype=torch.float64)).abs().max())
        return err < 1e-6
    check("stays orthogonal on the Lauchli matrix (classical GS scores ~0.5)",
          lauchli)
    def not_mutated():
        B = torch.randn(6, 3, dtype=torch.float64)
        C = B.clone()
        fn(B)
        return close(B, C)
    check("input is not mutated", not_mutated)
''',
),

task(
    id="gaussian-conditioning",
    title="Condition a joint Gaussian",
    book=BOOK, chapter=C_PROB,
    section="The Gaussian — marginals and conditionals",
    level=3,
    entry="condition",
    statement=(
        "Given a joint Gaussian over D variables, condition on observing some of "
        "them. With a for the free indices and b the observed ones: the "
        "conditional mean is mu_a + S_ab S_bb^{-1} (x_b - mu_b) and the "
        "conditional covariance is S_aa - S_ab S_bb^{-1} S_ba. Return both, over "
        "the free indices in ascending order. Note the covariance does not "
        "depend on the observed VALUE — only the mean moves."
    ),
    shapes=("mu (D,) · S (D, D) SPD · obs_idx list[int] · obs_val (len(obs),)"
            "  ->  dict 'mean' (D-k,), 'cov' (D-k, D-k)"),
    stub=("def condition(mu, S, obs_idx, obs_val):\n"
          "    # -> {'mean': ..., 'cov': ...} over the remaining indices\n    pass\n"),
    hints=[
        "Split the index set: a = sorted free indices, b = observed. Slice mu "
        "and S into the four blocks with fancy indexing.",
        "Solve S_bb X = (x_b - mu_b) and S_bb Y = S_ba rather than forming an "
        "inverse.",
        "mean = mu_a + S_ab X;  cov = S_aa - S_ab Y.",
    ],
    solution=(
        "def condition(mu, S, obs_idx, obs_val):\n"
        "    D = mu.shape[0]\n"
        "    b = list(obs_idx)\n"
        "    a = [i for i in range(D) if i not in b]\n"
        "    ia = torch.tensor(a)\n"
        "    ib = torch.tensor(b)\n"
        "    S_aa = S[ia][:, ia]\n"
        "    S_ab = S[ia][:, ib]\n"
        "    S_bb = S[ib][:, ib]\n"
        "    diff = obs_val - mu[ib]\n"
        "    X = torch.linalg.solve(S_bb, diff)\n"
        "    Y = torch.linalg.solve(S_bb, S_ab.T)\n"
        "    return {'mean': mu[ia] + S_ab @ X, 'cov': S_aa - S_ab @ Y}\n"
    ),
    solution_np=(
        "def condition(mu, S, obs_idx, obs_val):\n"
        "    D = mu.shape[0]\n"
        "    b = list(int(i) for i in obs_idx)\n"
        "    a = [i for i in range(D) if i not in b]\n"
        "    S_aa = S[np.ix_(a, a)]\n"
        "    S_ab = S[np.ix_(a, b)]\n"
        "    S_bb = S[np.ix_(b, b)]\n"
        "    diff = obs_val - mu[b]\n"
        "    X = np.linalg.solve(S_bb, diff)\n"
        "    Y = np.linalg.solve(S_bb, S_ab.T)\n"
        "    return {'mean': mu[a] + S_ab @ X, 'cov': S_aa - S_ab @ Y}\n"
    ),
    traps=[
        "Inverting S_aa instead of S_bb — the inverse always belongs to the "
        "observed block.",
        "Adding the correction to the covariance instead of subtracting: "
        "observing can only reduce uncertainty.",
        "Returning the free dimensions in observation order rather than "
        "ascending index order.",
    ],
    tests='''
def checks(fn, check):
    # 2-D hand case: unit variances, correlation rho, observe x2 = v
    rho, v = 0.6, 1.5
    mu2 = torch.zeros(2)
    S2 = torch.tensor([[1.0, rho], [rho, 1.0]])
    o = fn(mu2, S2, [1], torch.tensor([v]))
    check("2-D hand case: mean is rho*x", lambda: close(o["mean"], torch.tensor([rho * v]), 1e-6))
    check("2-D hand case: var is 1-rho^2", lambda: close(o["cov"], torch.tensor([[1 - rho**2]]), 1e-6))

    torch.manual_seed(0)
    D = 4
    Amat = torch.randn(D, D)
    S = Amat @ Amat.T + D * torch.eye(D)
    mu = torch.randn(D)

    def independent_unchanged():
        Sd = S.clone()
        Sd[0, 1:] = 0; Sd[1:, 0] = 0            # variable 0 independent of the rest
        r = fn(mu, Sd, [2], torch.tensor([0.7]))
        return close(r["mean"][0], mu[0], 1e-5) and close(r["cov"][0, 0], Sd[0, 0], 1e-5)
    check("conditioning on an independent variable changes nothing", independent_unchanged)

    def sequential_equals_joint():
        joint = fn(mu, S, [2, 3], torch.tensor([0.5, -1.0]))
        step1 = fn(mu, S, [3], torch.tensor([-1.0]))       # remaining dims: 0,1,2
        step2 = fn(step1["mean"], step1["cov"], [2], torch.tensor([0.5]))
        return close(step2["mean"], joint["mean"], 1e-5) and close(step2["cov"], joint["cov"], 1e-5)
    check("conditioning sequentially equals conditioning jointly", sequential_equals_joint)

    def value_free_cov():
        c1 = fn(mu, S, [1], torch.tensor([10.0]))["cov"]
        c2 = fn(mu, S, [1], torch.tensor([-10.0]))["cov"]
        return close(c1, c2, 1e-6)
    check("the conditional covariance ignores the observed value", value_free_cov)
    check("covariance is symmetric PSD",
          lambda: bool((torch.linalg.eigvalsh(fn(mu, S, [0], torch.tensor([1.0]))["cov"])
                        > -1e-8).all()))
''',
),

task(
    id="importance-sampling",
    title="Self-normalised importance sampling",
    book=BOOK, chapter=C_PROB,
    section="Probability and estimation — Monte Carlo",
    level=2,
    entry="is_mean",
    statement=(
        "Estimate E_target[f] from samples drawn from a PROPOSAL distribution: "
        "weight each sample by w = target(x)/proposal(x) and take the weighted "
        "average, normalising by the sum of weights. Self-normalisation is the "
        "point — it makes the estimator work when the target density is known "
        "only up to a constant, which is the usual situation. Work with "
        "log-densities and subtract the max before exponentiating, or the "
        "weights underflow to zero the moment the log-densities are shifted."
    ),
    shapes="f callable · target_logpdf, proposal_logpdf callables · xs (N,)  ->  scalar",
    stub=("def is_mean(f, target_logpdf, proposal_logpdf, xs):\n"
          "    # sum(w f) / sum(w), stably in log space\n    pass\n"),
    hints=[
        "logw = target_logpdf(xs) - proposal_logpdf(xs).",
        "Subtract logw.max() before exp — the normalisation cancels the shift.",
        "Return (w * f(xs)).sum() / w.sum().",
    ],
    solution=(
        "def is_mean(f, target_logpdf, proposal_logpdf, xs):\n"
        "    logw = target_logpdf(xs) - proposal_logpdf(xs)\n"
        "    w = torch.exp(logw - logw.max())\n"
        "    return (w * f(xs)).sum() / w.sum()\n"
    ),
    frameworks=["torch"],
    traps=[
        "Plain (unnormalised) importance sampling, mean(w f) — correct only when "
        "both densities are normalised, and it breaks the moment a constant is "
        "added to the target log-density.",
        "Exponentiating raw log-weights, which underflow together and give 0/0.",
        "Forgetting that the weights must use densities of the SAME variable — "
        "no Jacobians are hiding here, but only because both are on x.",
    ],
    tests='''
def checks(fn, check):
    def norm_logpdf(mu, sigma):
        return lambda x: -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma) \
                         - 0.5 * math.log(2 * math.pi)
    torch.manual_seed(0)
    xs = torch.randn(40000) * 2.0                     # proposal N(0, 2)
    prop = norm_logpdf(0.0, 2.0)
    targ = norm_logpdf(3.0, 1.0)

    check("estimates the target mean",
          lambda: abs(float(fn(lambda x: x, targ, prop, xs)) - 3.0) < 0.15)
    check("estimates the target second moment",
          lambda: abs(float(fn(lambda x: x * x, targ, prop, xs)) - 10.0) < 0.7)
    check("proposal == target reduces to the plain sample mean",
          lambda: close(fn(lambda x: x, prop, prop, xs), xs.mean(), 1e-6))
    def unnormalised_target():
        shifted = lambda x: targ(x) + 7.0             # unnormalised log-density
        a = float(fn(lambda x: x, targ, prop, xs))
        b = float(fn(lambda x: x, shifted, prop, xs))
        return abs(a - b) < 1e-6
    check("invariant to an additive constant in the target log-density",
          unnormalised_target)
    def extreme_shift():
        far = lambda x: targ(x) - 5000.0
        v = float(fn(lambda x: x, far, prop, xs))
        return v == v and abs(v - 3.0) < 0.15          # finite, still correct
    check("survives log-densities shifted by -5000 (max-subtraction)", extreme_shift)
''',
),

task(
    id="kfold-split",
    title="k-fold cross-validation splits",
    book=BOOK, chapter=C_REG,
    section="Linear regression — model selection",
    level=1,
    entry="kfold",
    statement=(
        "Produce the k train/validation index splits for n examples: the "
        "validation folds partition range(n) — every index appears in exactly "
        "one validation fold — fold sizes differ by at most one, and each train "
        "set is exactly the complement of its validation fold. Deterministic, "
        "no shuffling: fold boundaries are contiguous, with the first n mod k "
        "folds one element longer."
    ),
    shapes="n, k int  ->  list of k pairs (train list[int], val list[int])",
    stub=("def kfold(n, k):\n"
          "    # -> [(train, val), ...], val folds partition range(n)\n    pass\n"),
    hints=[
        "Fold sizes: n // k, with the first n % k folds getting one extra.",
        "Walk a start pointer; val = indices[start:start+size].",
        "train is everything else, order preserved.",
    ],
    solution=(
        "def kfold(n, k):\n"
        "    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]\n"
        "    out, start = [], 0\n"
        "    for s in sizes:\n"
        "        val = list(range(start, start + s))\n"
        "        train = list(range(0, start)) + list(range(start + s, n))\n"
        "        out.append((train, val))\n"
        "        start += s\n"
        "    return out\n"
    ),
    solution_np=(
        "def kfold(n, k):\n"
        "    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]\n"
        "    out, start = [], 0\n"
        "    for s in sizes:\n"
        "        val = list(range(start, start + s))\n"
        "        train = list(range(0, start)) + list(range(start + s, n))\n"
        "        out.append((train, val))\n"
        "        start += s\n"
        "    return out\n"
    ),
    traps=[
        "Dropping the remainder — n=10, k=3 must produce folds of 4, 3, 3 that "
        "cover everything, not three folds of 3.",
        "Leakage: an index appearing in both the train and validation sets of "
        "the same split.",
        "Validation folds that overlap between splits, so some examples are "
        "validated twice and others never.",
    ],
    tests='''
def checks(fn, check):
    out = fn(10, 3)
    check("k splits", lambda: len(out) == 3)
    check("val folds partition range(n)",
          lambda: sorted(i for _, v in out for i in v) == list(range(10)))
    check("val folds are disjoint",
          lambda: len(set(i for _, v in out for i in v)) == 10)
    check("sizes differ by at most one",
          lambda: sorted(len(v) for _, v in out) == [3, 3, 4])
    check("no leakage: train and val are disjoint in every split",
          lambda: all(not (set(t) & set(v)) for t, v in out))
    check("train is the exact complement",
          lambda: all(sorted(list(t) + list(v)) == list(range(10)) for t, v in out))
    check("n divisible by k gives equal folds",
          lambda: all(len(v) == 4 for _, v in fn(12, 3)))
''',
),

task(
    id="naive-bayes",
    title="Gaussian naive Bayes",
    book=BOOK, chapter=C_CLS,
    section="Linear classification — generative classifiers",
    level=2,
    entry="gnb_predict",
    statement=(
        "Fit a Gaussian naive Bayes classifier and predict: per class, estimate "
        "a prior and a per-feature mean and variance; classify by the largest "
        "log prior + sum of per-feature Gaussian log-densities. Work in log "
        "space from the start — with many features or large scales the product "
        "of densities underflows to zero for EVERY class, and an argmax over "
        "zeros returns class 0 with full confidence."
    ),
    shapes="Xtr (N, D) · ytr (N,) int in [0, C) · Xte (M, D)  ->  (M,) int64",
    stub=("def gnb_predict(Xtr, ytr, Xte):\n"
          "    # log prior + sum of per-feature Gaussian log-densities; argmax\n    pass\n"),
    hints=[
        "Per class c: prior N_c/N, mean and variance per feature over the class's "
        "rows (add a small floor like 1e-9 to the variance).",
        "log N(x; m, v) = -0.5 * (log(2 pi v) + (x - m)^2 / v), summed over "
        "features.",
        "Score every test row against every class and take the argmax.",
    ],
    solution=(
        "def gnb_predict(Xtr, ytr, Xte):\n"
        "    C = int(ytr.max()) + 1\n"
        "    scores = []\n"
        "    for c in range(C):\n"
        "        Xc = Xtr[ytr == c]\n"
        "        prior = math.log(Xc.shape[0] / Xtr.shape[0])\n"
        "        m = Xc.mean(0)\n"
        "        v = Xc.var(0, unbiased=False) + 1e-9\n"
        "        ll = -0.5 * (torch.log(2 * math.pi * v) + (Xte - m) ** 2 / v)\n"
        "        scores.append(prior + ll.sum(-1))\n"
        "    return torch.stack(scores, -1).argmax(-1)\n"
    ),
    solution_np=(
        "def gnb_predict(Xtr, ytr, Xte):\n"
        "    C = int(ytr.max()) + 1\n"
        "    scores = []\n"
        "    for c in range(C):\n"
        "        Xc = Xtr[ytr == c]\n"
        "        prior = math.log(Xc.shape[0] / Xtr.shape[0])\n"
        "        m = Xc.mean(0)\n"
        "        v = Xc.var(0) + 1e-9\n"
        "        ll = -0.5 * (np.log(2 * np.pi * v) + (Xte - m) ** 2 / v)\n"
        "        scores.append(prior + ll.sum(-1))\n"
        "    return np.stack(scores, -1).argmax(-1)\n"
    ),
    traps=[
        "Multiplying probabilities instead of adding log-densities — every "
        "class underflows to zero and the argmax silently returns class 0.",
        "Ignoring the class priors, which decides every point near the boundary "
        "when the classes are imbalanced.",
        "A zero variance on a constant feature dividing the whole score by zero.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    # 1-D hand case: classes at 0 and 4, unit-ish variance, equal priors
    Xtr = torch.cat([torch.randn(200, 1), torch.randn(200, 1) + 4.0])
    ytr = torch.cat([torch.zeros(200, dtype=torch.long),
                     torch.ones(200, dtype=torch.long)])
    check("boundary sits near the midpoint",
          lambda: fn(Xtr, ytr, torch.tensor([[1.5], [2.5]])).tolist() == [0, 1])
    check("far points are certain",
          lambda: fn(Xtr, ytr, torch.tensor([[-3.0], [7.0]])).tolist() == [0, 1])
    def log_space_survives_scale():
        # 40 features scaled so per-class densities underflow float in prob space:
        # the correct class is 1, a prob-space argmax returns 0
        Xa = torch.randn(100, 40) * 100.0
        Xb = torch.randn(100, 40) * 100.0 + 4000.0
        X = torch.cat([Xa, Xb]); y = torch.cat([torch.zeros(100, dtype=torch.long),
                                                torch.ones(100, dtype=torch.long)])
        test = torch.full((1, 40), 3900.0)
        return int(fn(X, y, test)[0]) == 1
    check("survives scales where probability space underflows", log_space_survives_scale)
    def priors_matter():
        # class 1 has 9x the examples; the exact midpoint should go to class 1
        Xtr2 = torch.cat([torch.randn(40, 1), torch.randn(360, 1) + 4.0])
        ytr2 = torch.cat([torch.zeros(40, dtype=torch.long),
                          torch.ones(360, dtype=torch.long)])
        return int(fn(Xtr2, ytr2, torch.tensor([[2.0]]))[0]) == 1
    check("priors decide the boundary under class imbalance", priors_matter)
    check("output shape and dtype",
          lambda: shape(fn(Xtr, ytr, torch.randn(7, 1))) == (7,))
''',
),

task(
    id="inverted-dropout",
    title="Inverted dropout",
    book=BOOK, chapter=C_NN,
    section="Neural networks — SGD and regularisation",
    level=1,
    entry="dropout",
    statement=(
        "Implement inverted dropout: in training, zero each element "
        "independently with probability p and scale the survivors by 1/(1-p); "
        "in evaluation, return the input untouched. The 1/(1-p) is the "
        "'inverted' part — it keeps the layer's expected output equal to its "
        "input, so evaluation needs no compensation. Forgetting it trains a "
        "network whose activations shrink by (1-p) the moment dropout switches "
        "off."
    ),
    shapes="x (…) · p in [0, 1) · training bool  ->  same shape",
    stub=("def dropout(x, p, training=True):\n"
          "    # zero with prob p, scale survivors by 1/(1-p); eval = identity\n    pass\n"),
    hints=[
        "mask = (torch.rand_like(x) >= p) — keep with probability 1-p.",
        "Return x * mask / (1 - p) in training.",
        "Evaluation returns x itself, exactly.",
    ],
    solution=(
        "def dropout(x, p, training=True):\n"
        "    if not training or p == 0:\n"
        "        return x\n"
        "    mask = (torch.rand_like(x) >= p).to(x.dtype)\n"
        "    return x * mask / (1 - p)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Forgetting the 1/(1-p) rescale, so train-time and eval-time activations "
        "live on different scales.",
        "Scaling at evaluation instead — that is the original (non-inverted) "
        "formulation, and this task asks for the inverted one.",
        "Using the same mask across the batch instead of sampling per element.",
    ],
    tests='''
def checks(fn, check):
    x = torch.randn(200, 50) + 3.0
    check("evaluation is the identity, exactly",
          lambda: close(fn(x, 0.5, training=False), x))
    check("p = 0 is the identity", lambda: close(fn(x, 0.0, True), x))
    def survivors_scaled_exactly():
        out = fn(x, 0.25, True)
        kept = out != 0
        return close(out[kept], x[kept] / 0.75, 1e-5)
    check("surviving entries equal x/(1-p) exactly", survivors_scaled_exactly)
    def drop_fraction():
        out = fn(torch.ones(100000), 0.3, True)
        return abs(float((out == 0).float().mean()) - 0.3) < 0.01
    check("about p of the entries are zeroed", drop_fraction)
    def mean_preserved():
        outs = torch.stack([fn(x, 0.5, True) for _ in range(40)]).mean(0)
        return close(outs.mean(), x.mean(), 0.05)
    check("expectation is preserved", mean_preserved)
    def grads_flow():
        xx = torch.randn(50, requires_grad=True)
        fn(xx, 0.5, True).sum().backward()
        g = xx.grad
        ok_vals = ((g - 2.0).abs() < 1e-5) | (g.abs() < 1e-6)
        return bool(ok_vals.all()) and float(g.abs().sum()) > 0
    check("gradient is 1/(1-p) on survivors and 0 on dropped", grads_flow)
''',
),

task(
    id="gmm-mstep",
    title="GMM M-step",
    book=BOOK, chapter=C_EM,
    section="Mixtures and EM — why EM works",
    level=2,
    entry="m_step",
    statement=(
        "Complete EM for the isotropic Gaussian mixture: given responsibilities "
        "from the E-step, re-estimate weights, means, and per-component "
        "isotropic variances. Everything is a responsibility-weighted average: "
        "N_k = sum of column k, weight = N_k/N, mean = weighted average of the "
        "points, variance = weighted mean squared distance divided by N_k * D "
        "(the D because one scalar variance covers D coordinates). Iterated with "
        "the E-step, the data log-likelihood must never decrease — that is the "
        "EM guarantee, and the tests check it."
    ),
    shapes=("X (N, D) · resp (N, K) rows sum to 1"
            "  ->  dict 'weights' (K,), 'means' (K, D), 'variances' (K,)"),
    stub=("def m_step(X, resp):\n"
          "    # responsibility-weighted weights, means, isotropic variances\n    pass\n"),
    hints=[
        "N_k = resp.sum(0); weights = N_k / N.",
        "means = resp.T @ X / N_k[:, None].",
        "variances: weight each squared distance ||x_n - mu_k||^2 by r_nk, sum, "
        "divide by N_k * D.",
    ],
    solution=(
        "def m_step(X, resp):\n"
        "    N, D = X.shape\n"
        "    Nk = resp.sum(0)\n"
        "    weights = Nk / N\n"
        "    means = resp.T @ X / Nk[:, None]\n"
        "    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)\n"
        "    variances = (resp * d2).sum(0) / (Nk * D)\n"
        "    return {'weights': weights, 'means': means, 'variances': variances}\n"
    ),
    solution_np=(
        "def m_step(X, resp):\n"
        "    N, D = X.shape\n"
        "    Nk = resp.sum(0)\n"
        "    weights = Nk / N\n"
        "    means = resp.T @ X / Nk[:, None]\n"
        "    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)\n"
        "    variances = (resp * d2).sum(0) / (Nk * D)\n"
        "    return {'weights': weights, 'means': means, 'variances': variances}\n"
    ),
    traps=[
        "Dividing the variance by N_k instead of N_k * D, which inflates it by "
        "the dimensionality.",
        "Using the OLD means when computing the variances — the M-step variance "
        "is measured around the NEW means.",
        "Hardening the responsibilities to their argmax, which is k-means, not "
        "EM, and breaks the monotone-likelihood guarantee.",
    ],
    extra=(
        "def _e_ref(X, means, variances, weights):\n"
        "    D = X.shape[1]\n"
        "    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)\n"
        "    logp = -0.5 * (D * torch.log(2 * math.pi * variances)[None, :]\n"
        "                   + d2 / variances[None, :])\n"
        "    return torch.softmax(logp + torch.log(weights)[None, :], dim=-1)\n"
        "\n"
        "def _loglik(X, means, variances, weights):\n"
        "    D = X.shape[1]\n"
        "    d2 = ((X[:, None, :] - means[None, :, :]) ** 2).sum(-1)\n"
        "    logp = -0.5 * (D * torch.log(2 * math.pi * variances)[None, :]\n"
        "                   + d2 / variances[None, :])\n"
        "    return float(torch.logsumexp(logp + torch.log(weights)[None, :],\n"
        "                                 dim=-1).sum())\n"
    ),
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    X = torch.cat([torch.randn(60, 2) * 0.5,
                   torch.randn(60, 2) * 0.5 + torch.tensor([5.0, 0.0])])
    def hard():
        r = torch.zeros(120, 2)
        r[:60, 0] = 1.0; r[60:, 1] = 1.0
        return r

    o = fn(X, hard())
    check("hard responsibilities give per-cluster sample means",
          lambda: close(o["means"][0], X[:60].mean(0), 1e-5)
                  and close(o["means"][1], X[60:].mean(0), 1e-5))
    def hard_variance():
        want = ((X[:60] - X[:60].mean(0)) ** 2).sum() / (60 * 2)
        return close(o["variances"][0], want, 1e-5)
    check("hard variance is the mean squared distance over N_k * D", hard_variance)
    check("weights sum to 1", lambda: close(o["weights"].sum(), torch.tensor(1.0), 1e-6))
    check("hard weights are the cluster fractions",
          lambda: close(o["weights"], torch.tensor([0.5, 0.5]), 1e-6))

    def em_is_monotone():
        means = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        variances = torch.tensor([4.0, 4.0])
        weights = torch.tensor([0.5, 0.5])
        prev = _loglik(X, means, variances, weights)
        for _ in range(12):
            r = _e_ref(X, means, variances, weights)
            p = fn(X, r)
            means, variances, weights = p["means"], p["variances"], p["weights"]
            cur = _loglik(X, means, variances, weights)
            if cur < prev - 1e-6:
                return False
            prev = cur
        return True
    check("iterated with the E-step, log-likelihood never decreases", em_is_monotone)

    def recovers_separated():
        means = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        variances = torch.tensor([1.0, 1.0])
        weights = torch.tensor([0.5, 0.5])
        for _ in range(30):
            r = _e_ref(X, means, variances, weights)
            p = fn(X, r)
            means, variances, weights = p["means"], p["variances"], p["weights"]
        centers = means[means[:, 0].argsort()]
        return close(centers[0], torch.tensor([0.0, 0.0]), 0.3) \
               and close(centers[1], torch.tensor([5.0, 0.0]), 0.3)
    check("EM recovers well-separated clusters", recovers_separated)
''',
),

task(
    id="viterbi",
    title="Viterbi decoding",
    book=BOOK, chapter=C_SEQ,
    section="Sequential models — hidden Markov models",
    level=3,
    entry="viterbi",
    statement=(
        "Find the single most probable STATE PATH of an HMM given the "
        "observations, in log space, with back-pointers. This is the forward "
        "algorithm with the sum replaced by a max — and the distinction matters: "
        "the sequence of individually-most-likely states can be an impossible "
        "path (it can use zero-probability transitions), which is exactly what "
        "the tests construct. Return the path and its log-probability."
    ),
    shapes=("log_pi (K,) · log_A (K, K) · log_B (K, V) · obs (T,) int"
            "  ->  (path (T,) int64, logprob float)"),
    stub=("def viterbi(log_pi, log_A, log_B, obs):\n"
          "    # max-product forward pass + back-pointers\n    pass\n"),
    hints=[
        "delta[0] = log_pi + log_B[:, obs[0]]; then delta_next[j] = "
        "max_i(delta[i] + log_A[i, j]) + log_B[j, obs[t]].",
        "Store the argmax i at every (t, j) — those are the back-pointers.",
        "Start from the argmax of the final delta and walk the pointers "
        "backwards.",
    ],
    solution=(
        "def viterbi(log_pi, log_A, log_B, obs):\n"
        "    T = len(obs)\n"
        "    K = log_pi.shape[0]\n"
        "    delta = log_pi + log_B[:, obs[0]]\n"
        "    ptr = torch.zeros(T, K, dtype=torch.long)\n"
        "    for t in range(1, T):\n"
        "        cand = delta[:, None] + log_A\n"
        "        best, arg = cand.max(dim=0)\n"
        "        ptr[t] = arg\n"
        "        delta = best + log_B[:, obs[t]]\n"
        "    path = torch.zeros(T, dtype=torch.long)\n"
        "    path[-1] = delta.argmax()\n"
        "    for t in range(T - 1, 0, -1):\n"
        "        path[t - 1] = ptr[t, path[t]]\n"
        "    return path, float(delta.max())\n"
    ),
    frameworks=["torch"],
    traps=[
        "Summing where the max belongs — that is the forward algorithm, and its "
        "per-step argmax can pick a path with an impossible transition.",
        "Choosing each state independently by its emission likelihood, which "
        "ignores the transition structure entirely.",
        "Walking the back-pointers with an off-by-one, which shifts the whole "
        "path.",
    ],
    extra=(
        "def _brute(log_pi, log_A, log_B, obs):\n"
        "    import itertools\n"
        "    K, T = log_pi.shape[0], len(obs)\n"
        "    best, best_path = -1e30, None\n"
        "    for path in itertools.product(range(K), repeat=T):\n"
        "        s = float(log_pi[path[0]] + log_B[path[0], obs[0]])\n"
        "        for t in range(1, T):\n"
        "            s += float(log_A[path[t - 1], path[t]] + log_B[path[t], obs[t]])\n"
        "        if s > best:\n"
        "            best, best_path = s, path\n"
        "    return torch.tensor(best_path), best\n"
    ),
    tests='''
def checks(fn, check):
    log_pi = torch.log(torch.tensor([0.6, 0.4]))
    A = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
    Bm = torch.log(torch.tensor([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]))
    obs = torch.tensor([0, 2, 1, 2, 0])
    def matches_brute():
        path, score = fn(log_pi, A, Bm, obs)
        bp, bs = _brute(log_pi, A, Bm, obs)
        return close(path, bp) and abs(score - bs) < 1e-5
    check("matches brute-force enumeration", matches_brute)
    def second_model():
        pi2 = torch.log(torch.tensor([0.2, 0.8]))
        obs2 = torch.tensor([2, 2, 0, 1])
        path, score = fn(pi2, A, Bm, obs2)
        bp, bs = _brute(pi2, A, Bm, obs2)
        return close(path, bp) and abs(score - bs) < 1e-5
    check("matches brute force on a second model", second_model)
    def forbidden_transition():
        # emissions pull towards alternating states, but switching is forbidden:
        # per-step emission argmax alternates and is an IMPOSSIBLE path
        A2 = torch.log(torch.tensor([[1.0, 1e-12], [1e-12, 1.0]]))
        B2 = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
        pi2 = torch.log(torch.tensor([0.7, 0.3]))      # asymmetric: staying in 0 wins
        obs2 = torch.tensor([0, 1, 0, 1])
        path, score = fn(pi2, A2, B2, obs2)
        bp, bs = _brute(pi2, A2, B2, obs2)
        return close(path, bp) and abs(score - bs) < 1e-4 \
               and bool((path == path[0]).all())
    check("respects forbidden transitions (per-step argmax would not)",
          forbidden_transition)
    check("T = 1 is the posterior argmax",
          lambda: int(fn(log_pi, A, Bm, torch.tensor([2]))[0][0])
                  == int((log_pi + Bm[:, 2]).argmax()))
    def score_is_real():
        path, score = fn(log_pi, A, Bm, obs)
        s = float(log_pi[path[0]] + Bm[path[0], obs[0]])
        for t in range(1, len(obs)):
            s += float(A[path[t - 1], path[t]] + Bm[path[t], obs[t]])
        return abs(s - score) < 1e-5
    check("the returned score is the returned path's score", score_is_real)
''',
),

task(
    id="kalman-1d",
    title="A one-dimensional Kalman filter",
    book=BOOK, chapter=C_SEQ,
    section="Sequential models — linear dynamical systems",
    level=3,
    entry="kalman_1d",
    statement=(
        "Filter a scalar linear-Gaussian state space model: x_t = a x_{t-1} + w "
        "(process noise q), y_t = c x_t + v (observation noise r). Per step: "
        "predict (mu, P) through the dynamics, form the gain "
        "K = P c / (c^2 P + r), then update. Return the filtered means and "
        "variances after each observation. The static special case a=1, q=0, "
        "c=1 is exact Bayesian inference for a Gaussian mean, and the tests hold "
        "you to its closed form."
    ),
    shapes=("ys (T,) · a, c, q, r, mu0, p0 float"
            "  ->  dict 'means' (T,), 'vars' (T,)"),
    stub=("def kalman_1d(ys, a, c, q, r, mu0, p0):\n"
          "    # predict, gain, update — in that order, per observation\n    pass\n"),
    hints=[
        "Predict: mu_pred = a * mu; P_pred = a^2 * P + q.",
        "Gain: K = P_pred * c / (c^2 * P_pred + r).",
        "Update: mu = mu_pred + K * (y - c * mu_pred); P = (1 - K c) * P_pred.",
    ],
    solution=(
        "def kalman_1d(ys, a, c, q, r, mu0, p0):\n"
        "    mu, P = mu0, p0\n"
        "    means, variances = [], []\n"
        "    for y in ys:\n"
        "        mu_pred = a * mu\n"
        "        P_pred = a * a * P + q\n"
        "        K = P_pred * c / (c * c * P_pred + r)\n"
        "        mu = mu_pred + K * (float(y) - c * mu_pred)\n"
        "        P = (1 - K * c) * P_pred\n"
        "        means.append(mu)\n"
        "        variances.append(P)\n"
        "    return {'means': torch.tensor(means), 'vars': torch.tensor(variances)}\n"
    ),
    solution_np=(
        "def kalman_1d(ys, a, c, q, r, mu0, p0):\n"
        "    mu, P = mu0, p0\n"
        "    means, variances = [], []\n"
        "    for y in ys:\n"
        "        mu_pred = a * mu\n"
        "        P_pred = a * a * P + q\n"
        "        K = P_pred * c / (c * c * P_pred + r)\n"
        "        mu = mu_pred + K * (float(y) - c * mu_pred)\n"
        "        P = (1 - K * c) * P_pred\n"
        "        means.append(mu)\n"
        "        variances.append(P)\n"
        "    return {'means': np.array(means), 'vars': np.array(variances)}\n"
    ),
    traps=[
        "Updating before predicting, which shifts every estimate one dynamics "
        "step out of phase.",
        "Building the gain from the pre-prediction P instead of P_pred.",
        "Forgetting q, so the filter grows overconfident and stops tracking a "
        "moving state.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    ys = torch.tensor([1.0, 2.0, 0.5, 1.5, 1.2])

    def static_closed_form():
        # a=1, q=0, c=1: exact Bayes for a Gaussian mean.
        # After t obs: var_t = 1/(1/p0 + t/r), mean_t = var_t*(mu0/p0 + sum(y)/r)
        o = fn(ys, 1.0, 1.0, 0.0, 2.0, 0.0, 10.0)
        for t in range(1, len(ys) + 1):
            var_t = 1.0 / (1.0 / 10.0 + t / 2.0)
            mean_t = var_t * (0.0 / 10.0 + float(ys[:t].sum()) / 2.0)
            if abs(float(o["vars"][t - 1]) - var_t) > 1e-6:
                return False
            if abs(float(o["means"][t - 1]) - mean_t) > 1e-6:
                return False
        return True
    check("static case matches exact Bayesian updating, step by step",
          static_closed_form)

    def variance_decreases_static():
        o = fn(ys, 1.0, 1.0, 0.0, 2.0, 0.0, 10.0)
        v = o["vars"]
        return bool((v[1:] < v[:-1]).all())
    check("static variance is monotone decreasing", variance_decreases_static)

    def huge_r_ignores_obs():
        o = fn(ys, 0.9, 1.0, 0.0, 1e12, 5.0, 0.01)
        want = torch.tensor([5.0 * 0.9 ** t for t in range(1, len(ys) + 1)])
        return close(o["means"], want, 1e-3)
    check("r -> infinity: the filter just propagates the prior", huge_r_ignores_obs)

    def tiny_r_tracks_obs():
        o = fn(ys, 1.0, 2.0, 1.0, 1e-9, 0.0, 1.0)
        return close(o["means"], ys / 2.0, 1e-3)
    check("r -> 0: the estimate is y/c", tiny_r_tracks_obs)

    check("shapes", lambda: shape(fn(ys, 1.0, 1.0, 0.1, 1.0, 0.0, 1.0)["means"]) == (5,))
    def q_keeps_uncertainty():
        o = fn(ys, 1.0, 1.0, 0.5, 2.0, 0.0, 10.0)
        return float(o["vars"][-1]) > 0.2
    check("process noise keeps the variance from collapsing", q_keeps_uncertainty)
''',
),

task(
    id="metropolis-hastings",
    title="Random-walk Metropolis–Hastings",
    book=BOOK, chapter=C_VI,
    section="Variational inference and sampling — MCMC",
    level=2,
    entry="mh_sample",
    statement=(
        "Sample from an unnormalised log-density with random-walk "
        "Metropolis–Hastings: propose x' = x + step * noise, accept with "
        "probability min(1, p(x')/p(x)) — in log space, accept when "
        "log u < logp(x') - logp(x) — otherwise keep x. Discard a burn-in "
        "prefix. The acceptance test is the whole algorithm: comparing u to the "
        "log-ratio directly (without the log on u) never accepts a downhill "
        "move, and the chain freezes at the mode."
    ),
    shapes="logpdf callable · n, burn int · step, x0 float  ->  (n,) samples",
    stub=("def mh_sample(logpdf, n, step=1.0, x0=0.0, burn=1000):\n"
          "    # accept iff log(u) < logp(x') - logp(x)\n    pass\n"),
    hints=[
        "One noise draw and one uniform draw per iteration; run burn + n "
        "iterations and keep the last n states.",
        "The comparison is log(torch.rand(())) < logp_new - logp_old.",
        "A rejected proposal REPEATS the current state — do not skip it.",
    ],
    solution=(
        "def mh_sample(logpdf, n, step=1.0, x0=0.0, burn=1000):\n"
        "    x = torch.tensor(float(x0))\n"
        "    lp = logpdf(x)\n"
        "    out = []\n"
        "    for i in range(burn + n):\n"
        "        prop = x + step * torch.randn(())\n"
        "        lp_new = logpdf(prop)\n"
        "        if float(torch.log(torch.rand(()))) < float(lp_new - lp):\n"
        "            x, lp = prop, lp_new\n"
        "        if i >= burn:\n"
        "            out.append(float(x))\n"
        "    return torch.tensor(out)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Comparing u (not log u) to the log-ratio: downhill moves are never "
        "accepted and the chain collapses onto the mode with near-zero variance.",
        "Dropping rejected iterations instead of repeating the state, which "
        "biases the chain towards high-density regions.",
        "Evaluating an unnormalised density as if it needed normalising — the "
        "ratio cancels the constant, which is the reason MH exists.",
    ],
    tests='''
def checks(fn, check):
    def gauss(mu, sigma):
        return lambda x: -0.5 * ((x - mu) / sigma) ** 2
    s = fn(gauss(2.0, 1.5), 20000, step=2.0, x0=0.0, burn=2000)
    check("sample mean matches the target", lambda: abs(float(s.mean()) - 2.0) < 0.1)
    check("sample std matches the target", lambda: abs(float(s.std()) - 1.5) < 0.12)
    check("downhill moves happen (the chain is not frozen at the mode)",
          lambda: len(set(s.tolist())) > len(s) // 3)
    def respects_support():
        # exponential(1): log p = -x for x >= 0, -inf below
        target = lambda x: -x if float(x) >= 0 else torch.tensor(float('-inf'))
        e = fn(target, 15000, step=1.0, x0=1.0, burn=2000)
        return bool((e >= 0).all()) and abs(float(e.mean()) - 1.0) < 0.12
    check("rejection keeps the chain inside the support", respects_support)
    def unnormalised_ok():
        shifted = lambda x: gauss(2.0, 1.5)(x) + 123.0
        t = fn(shifted, 8000, step=2.0, x0=0.0, burn=1000)
        return abs(float(t.mean()) - 2.0) < 0.15
    check("an additive constant in the log-density changes nothing", unnormalised_ok)
    check("returns exactly n samples", lambda: shape(fn(gauss(0., 1.), 500, 1.0, 0.0, 100)) == (500,))
''',
),

task(
    id="kmeans-pp",
    title="k-means++ seeding",
    book=BOOK, chapter=C_UNSUP,
    section="PCA and k-means",
    level=2,
    entry="kmeans_pp",
    statement=(
        "Choose k initial centres by the k-means++ rule: the first uniformly at "
        "random from the data, each subsequent one sampled with probability "
        "proportional to D(x)^2 — the squared distance to the nearest centre "
        "already chosen. On well-separated clusters this puts one seed per "
        "cluster almost every time, where uniform seeding collides constantly; "
        "the tests measure exactly that difference."
    ),
    shapes="X (N, D) · k int  ->  (k, D), each row a row of X",
    stub=("def kmeans_pp(X, k):\n"
          "    # first centre uniform; then sample proportional to D(x)^2\n    pass\n"),
    hints=[
        "Track the running squared distance to the nearest chosen centre; "
        "update it with a minimum after each pick.",
        "torch.multinomial(d2, 1) samples an index proportional to d2.",
        "Centres are data points — return rows of X, not averages.",
    ],
    solution=(
        "def kmeans_pp(X, k):\n"
        "    N = X.shape[0]\n"
        "    first = int(torch.randint(N, (1,)))\n"
        "    centres = [X[first]]\n"
        "    d2 = ((X - centres[0]) ** 2).sum(-1)\n"
        "    for _ in range(k - 1):\n"
        "        idx = int(torch.multinomial(d2, 1))\n"
        "        centres.append(X[idx])\n"
        "        d2 = torch.minimum(d2, ((X - X[idx]) ** 2).sum(-1))\n"
        "    return torch.stack(centres)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Sampling every centre uniformly, which lands two seeds in one cluster "
        "about as often as not.",
        "Weighting by D(x) instead of D(x)^2 — the ++ guarantee is proved for "
        "the square.",
        "Forgetting to update the distances after each new centre, so later "
        "picks still measure distance to the first one only.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    blob = lambda cx, cy: torch.randn(20, 2) * 0.05 + torch.tensor([cx, cy])
    X = torch.cat([blob(0., 0.), blob(10., 0.), blob(0., 10.)])

    def rows_of_X():
        C = fn(X, 3)
        return all(bool((X == C[i]).all(-1).any()) for i in range(3))
    check("centres are actual data points", rows_of_X)
    check("returns k centres", lambda: shape(fn(X, 3)) == (3, 2))
    check("k = 1 works", lambda: shape(fn(X, 1)) == (1, 2))

    def covers_clusters():
        hits = 0
        trials = 100
        for _ in range(trials):
            C = fn(X, 3)
            labels = {int(((C[i] - torch.tensor([[0., 0.], [10., 0.], [0., 10.]]))
                          ** 2).sum(-1).argmin()) for i in range(3)}
            hits += int(len(labels) == 3)
        return hits / trials
    rate = covers_clusters()
    check("one seed per cluster at least 80% of the time (uniform manages ~22%)",
          lambda: rate >= 0.8)
    def distinct():
        C = fn(X, 3)
        return not close(C[0], C[1], 1e-9) and not close(C[1], C[2], 1e-9)
    check("centres are distinct on distinct data", distinct)
''',
),

]
