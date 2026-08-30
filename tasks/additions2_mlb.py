"""Batch A additions to ml-basics: trees, ensembles, metrics, streaming, NN blocks."""
from .schema import task

BOOK = "ml-basics"
C_TREE = "Trees and ensembles"
C_CLS = "Linear classification"
C_MET = "Metrics and evaluation"
C_PROB = "Probability and estimation · The Gaussian"
C_NN = "Neural networks"
C_VI = "Variational inference and sampling"

TASKS = [

task(
    id="gini-split",
    title="Best Gini split for a decision stump",
    book=BOOK, chapter=C_TREE,
    section="Decision trees — impurity and splitting",
    level=2,
    entry="best_split",
    statement=(
        "Find the best threshold split of one feature for binary labels: for "
        "every candidate threshold (midpoints between consecutive sorted unique "
        "values), split into left (x <= t) and right, and score by the "
        "SIZE-WEIGHTED average of the children's Gini impurities, "
        "G = 1 - p0^2 - p1^2. Return the best threshold and its weighted "
        "impurity. Forgetting the size weighting is the classic bug — it lets a "
        "tiny pure child outvote a huge impure one."
    ),
    shapes="x (N,) float · y (N,) in {0,1}  ->  dict 'threshold' float, 'impurity' float",
    stub=("def best_split(x, y):\n"
          "    # weighted Gini over all midpoint thresholds\n    pass\n"),
    hints=[
        "Candidates: midpoints between consecutive sorted UNIQUE values of x.",
        "Gini of a subset: 1 - p0^2 - p1^2 over its label fractions.",
        "Score = (n_left * G_left + n_right * G_right) / N; take the minimum.",
    ],
    solution=(
        "def best_split(x, y):\n"
        "    xs = torch.unique(x).sort().values\n"
        "    best_t, best_g = None, float('inf')\n"
        "    def gini(labels):\n"
        "        if labels.numel() == 0:\n"
        "            return 0.0\n"
        "        p1 = float(labels.float().mean())\n"
        "        return 1.0 - p1 ** 2 - (1 - p1) ** 2\n"
        "    for i in range(len(xs) - 1):\n"
        "        t = float((xs[i] + xs[i + 1]) / 2)\n"
        "        left = y[x <= t]\n"
        "        right = y[x > t]\n"
        "        g = (left.numel() * gini(left) + right.numel() * gini(right)) / y.numel()\n"
        "        if g < best_g:\n"
        "            best_t, best_g = t, g\n"
        "    return {'threshold': best_t, 'impurity': best_g}\n"
    ),
    solution_np=(
        "def best_split(x, y):\n"
        "    xs = np.sort(np.unique(x))\n"
        "    best_t, best_g = None, float('inf')\n"
        "    def gini(labels):\n"
        "        if labels.size == 0:\n"
        "            return 0.0\n"
        "        p1 = float(labels.mean())\n"
        "        return 1.0 - p1 ** 2 - (1 - p1) ** 2\n"
        "    for i in range(len(xs) - 1):\n"
        "        t = float((xs[i] + xs[i + 1]) / 2)\n"
        "        left, right = y[x <= t], y[x > t]\n"
        "        g = (left.size * gini(left) + right.size * gini(right)) / y.size\n"
        "        if g < best_g:\n"
        "            best_t, best_g = t, g\n"
        "    return {'threshold': best_t, 'impurity': best_g}\n"
    ),
    traps=[
        "Averaging the children's impurities without weighting by size.",
        "Splitting AT data values instead of between them, which makes the "
        "boundary depend on which side gets the tie.",
        "Using misclassification error instead of Gini — flatter, and often "
        "unable to distinguish splits Gini separates.",
    ],
    tests='''
def checks(fn, check):
    # perfectly separable: threshold between 2 and 3, impurity 0
    x1 = torch.tensor([1., 2., 3., 4.])
    y1 = torch.tensor([0, 0, 1, 1])
    o1 = fn(x1, y1)
    check("separable data gives impurity 0", lambda: abs(o1["impurity"]) < 1e-9)
    check("separable threshold sits between the classes",
          lambda: 2.0 < o1["threshold"] < 3.0)
    # hand case: x=[1,2,3,4,5,6], y=[0,0,0,1,0,1]
    # split at 3.5: left {0,0,0} G=0 (3), right {1,0,1} G=4/9 (3)
    # weighted = (3*0 + 3*4/9)/6 = 2/9 = 0.2222
    x2 = torch.tensor([1., 2., 3., 4., 5., 6.])
    y2 = torch.tensor([0, 0, 0, 1, 0, 1])
    o2 = fn(x2, y2)
    # 1e-6, not tighter: the torch path computes label means in float32
    check("hand-computed weighted impurity", lambda: abs(o2["impurity"] - 2 / 9) < 1e-6)
    check("hand-computed threshold", lambda: 3.0 < o2["threshold"] < 4.0)
    def weighting_matters():
        # tiny pure left {0} vs huge mixed right: at t=1.5 weighted G is
        # (1*0 + 5*0.48)/6 = 0.4; unweighted average would be 0.24 and would
        # wrongly prefer this split over 3.5's unweighted 0.222... vs weighted 0.222
        # here the correct best remains 3.5; the unweighted trap picks 1.5
        return 3.0 < fn(x2, y2)["threshold"] < 4.0
    check("size weighting picks the right split", weighting_matters)
    check("pure input gives impurity 0 anywhere",
          lambda: abs(fn(torch.tensor([1., 2., 3.]), torch.tensor([1, 1, 1]))["impurity"]) < 1e-9)
''',
),

task(
    id="knn",
    title="k-nearest neighbours",
    book=BOOK, chapter=C_CLS,
    section="Classification — nearest neighbours",
    level=1,
    entry="knn_predict",
    statement=(
        "Classify each test point by majority vote among its k nearest training "
        "points in Euclidean distance. Use an odd k to dodge ties. The two "
        "things this checks: that you take the k SMALLEST distances (sorting the "
        "wrong way is a real bug that still returns labels), and that k=1 on the "
        "training set itself reproduces the training labels exactly."
    ),
    shapes="Xtr (N, D) · ytr (N,) int · Xte (M, D) · k odd int  ->  (M,) int64",
    stub=("def knn_predict(Xtr, ytr, Xte, k=3):\n"
          "    # majority vote among the k nearest by Euclidean distance\n    pass\n"),
    hints=[
        "Pairwise squared distances via the (a-b)^2 expansion or cdist; no loop "
        "over test points needed.",
        "torch.topk with largest=False gives the k smallest distances.",
        "Vote with a mode over the gathered labels.",
    ],
    solution=(
        "def knn_predict(Xtr, ytr, Xte, k=3):\n"
        "    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)\n"
        "    idx = d2.topk(k, dim=-1, largest=False).indices\n"
        "    votes = ytr[idx]\n"
        "    return votes.mode(dim=-1).values\n"
    ),
    solution_np=(
        "def knn_predict(Xtr, ytr, Xte, k=3):\n"
        "    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)\n"
        "    idx = np.argsort(d2, axis=-1)[:, :k]\n"
        "    votes = ytr[idx]\n"
        "    out = []\n"
        "    for row in votes:\n"
        "        vals, counts = np.unique(row, return_counts=True)\n"
        "        out.append(vals[counts.argmax()])\n"
        "    return np.array(out)\n"
    ),
    traps=[
        "Taking the k LARGEST distances — the code runs, the accuracy is "
        "garbage, and nothing raises.",
        "Voting over distances instead of labels.",
        "Even k with no tie-break rule, which makes predictions depend on sort "
        "stability.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    Xtr = torch.cat([torch.randn(30, 2) * 0.3,
                     torch.randn(30, 2) * 0.3 + torch.tensor([4.0, 0.0])])
    ytr = torch.cat([torch.zeros(30, dtype=torch.long),
                     torch.ones(30, dtype=torch.long)])
    check("k=1 on the training set reproduces the labels",
          lambda: bool((fn(Xtr, ytr, Xtr, 1) == ytr).all()))
    check("clear points classify correctly",
          lambda: fn(Xtr, ytr, torch.tensor([[0., 0.], [4., 0.]]), 5).tolist() == [0, 1])
    def hand_vote():
        X = torch.tensor([[0.], [1.], [2.], [10.]])
        y = torch.tensor([0, 0, 1, 1])
        # query 1.6: nearest 3 are x=2 (d .4), x=1 (d .6), x=0 (d 1.6) -> labels 1,0,0 -> 0
        return int(fn(X, y, torch.tensor([[1.6]]), 3)[0]) == 0
    check("hand-worked 3-NN vote", hand_vote)
    def uses_smallest():
        X = torch.tensor([[0.], [0.1], [100.]])
        y = torch.tensor([1, 1, 0])
        return int(fn(X, y, torch.tensor([[0.05]]), 1)[0]) == 1
    check("uses the NEAREST neighbour, not the farthest", uses_smallest)
    check("output shape", lambda: shape(fn(Xtr, ytr, torch.randn(7, 2), 3)) == (7,))
''',
),

task(
    id="adaboost-round",
    title="One AdaBoost round",
    book=BOOK, chapter=C_TREE,
    section="Combining models — committees and boosting",
    level=2,
    entry="adaboost_round",
    statement=(
        "Given example weights, predictions of a weak learner, and true labels "
        "(all in ±1), perform one AdaBoost round: weighted error "
        "e = sum of weights on the mistakes, learner coefficient "
        "alpha = 0.5·ln((1-e)/e), then reweight w <- w·exp(-alpha·y·h) and "
        "renormalise to sum 1. After the update, the mistakes carry exactly half "
        "the total weight — that invariant is the algorithm, and the tests check "
        "it directly."
    ),
    shapes="w (N,) sums to 1 · h (N,) in {-1,+1} · y (N,) in {-1,+1}  ->  dict 'alpha', 'weights'",
    stub=("def adaboost_round(w, h, y):\n"
          "    # -> {'alpha': float, 'weights': (N,) summing to 1}\n    pass\n"),
    hints=[
        "The weighted error uses the CURRENT weights, not counts.",
        "alpha = 0.5 * log((1 - e) / e).",
        "New weight w_i * exp(-alpha * y_i * h_i), then divide by the sum.",
    ],
    solution=(
        "def adaboost_round(w, h, y):\n"
        "    e = float(w[(h != y)].sum())\n"
        "    alpha = 0.5 * math.log((1 - e) / e)\n"
        "    new = w * torch.exp(-alpha * y.float() * h.float())\n"
        "    return {'alpha': alpha, 'weights': new / new.sum()}\n"
    ),
    solution_np=(
        "def adaboost_round(w, h, y):\n"
        "    e = float(w[(h != y)].sum())\n"
        "    alpha = 0.5 * math.log((1 - e) / e)\n"
        "    new = w * np.exp(-alpha * y * h)\n"
        "    return {'alpha': alpha, 'weights': new / new.sum()}\n"
    ),
    traps=[
        "Forgetting to renormalise, so the weights stop being a distribution "
        "and every later error estimate is wrong.",
        "Counting mistakes instead of summing their weights.",
        "Flipping the sign in the exponent, which DOWN-weights the mistakes.",
    ],
    tests='''
def checks(fn, check):
    w = torch.full((4,), 0.25)
    y = torch.tensor([1., 1., -1., -1.])
    h = torch.tensor([1., -1., -1., -1.])       # one mistake: index 1, e = 0.25
    o = fn(w, h, y)
    check("hand-computed alpha", lambda: abs(o["alpha"] - 0.5 * math.log(3)) < 1e-9)
    check("weights renormalise to 1",
          lambda: abs(float(o["weights"].sum()) - 1.0) < 1e-6)
    check("the mistake carries exactly half the new weight",
          lambda: abs(float(o["weights"][1]) - 0.5) < 1e-6)
    check("mistake weight rises, correct weights fall",
          lambda: float(o["weights"][1]) > 0.25 > float(o["weights"][0]))
    def error_half_gives_zero():
        y2 = torch.tensor([1., 1., -1., -1.])
        h2 = torch.tensor([1., -1., 1., -1.])   # e = 0.5
        return abs(fn(w, h2, y2)["alpha"]) < 1e-9
    check("error 1/2 gives alpha 0", error_half_gives_zero)
    def invariant():
        torch.manual_seed(0)
        ww = torch.rand(10); ww = ww / ww.sum()
        yy = torch.sign(torch.randn(10)); hh = torch.sign(torch.randn(10))
        if bool((hh == yy).all()) or bool((hh != yy).all()):
            return True
        out = fn(ww, hh, yy)
        return abs(float(out["weights"][(hh != yy)].sum()) - 0.5) < 1e-5
    check("post-update, mistakes always hold half the mass", invariant)
''',
),

task(
    id="roc-auc",
    title="ROC AUC, two ways",
    book=BOOK, chapter=C_MET,
    section="Metrics — ranking quality",
    level=2,
    entry="auc",
    statement=(
        "Compute the area under the ROC curve from raw scores and binary "
        "labels, by sweeping thresholds and integrating with the trapezoid "
        "rule. AUC has a second identity — it equals the probability that a "
        "random positive outscores a random negative, with ties counting half — "
        "and the tests verify your curve integration against that pairwise "
        "formula computed independently. Handle tied scores: all examples with "
        "an equal score enter the curve together."
    ),
    shapes="scores (N,) float · labels (N,) in {0,1}  ->  float in [0, 1]",
    stub=("def auc(scores, labels):\n"
          "    # threshold sweep + trapezoid; ties move together\n    pass\n"),
    hints=[
        "Sort by score descending; walk the unique score values, accumulating "
        "TPR = TP/P and FPR = FP/N at each threshold.",
        "Append the (0,0) start and (1,1) end, then trapezoid over FPR.",
        "Processing tied scores as one group is what makes the trapezoid agree "
        "with the half-credit pairwise convention.",
    ],
    solution=(
        "def auc(scores, labels):\n"
        "    order = torch.argsort(scores, descending=True)\n"
        "    s = scores[order]\n"
        "    l = labels[order].float()\n"
        "    P = float(l.sum())\n"
        "    Nn = float((1 - l).sum())\n"
        "    tpr, fpr = [0.0], [0.0]\n"
        "    tp = fp = 0.0\n"
        "    i = 0\n"
        "    n = len(s)\n"
        "    while i < n:\n"
        "        j = i\n"
        "        while j < n and float(s[j]) == float(s[i]):\n"
        "            j += 1\n"
        "        tp += float(l[i:j].sum())\n"
        "        fp += float((1 - l[i:j]).sum())\n"
        "        tpr.append(tp / P)\n"
        "        fpr.append(fp / Nn)\n"
        "        i = j\n"
        "    area = 0.0\n"
        "    for k in range(1, len(tpr)):\n"
        "        area += (fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1]) / 2\n"
        "    return area\n"
    ),
    solution_np=(
        "def auc(scores, labels):\n"
        "    order = np.argsort(-scores)\n"
        "    s = scores[order]\n"
        "    l = labels[order].astype(float)\n"
        "    P = float(l.sum())\n"
        "    Nn = float((1 - l).sum())\n"
        "    tpr, fpr = [0.0], [0.0]\n"
        "    tp = fp = 0.0\n"
        "    i, n = 0, len(s)\n"
        "    while i < n:\n"
        "        j = i\n"
        "        while j < n and s[j] == s[i]:\n"
        "            j += 1\n"
        "        tp += float(l[i:j].sum())\n"
        "        fp += float((1 - l[i:j]).sum())\n"
        "        tpr.append(tp / P)\n"
        "        fpr.append(fp / Nn)\n"
        "        i = j\n"
        "    area = 0.0\n"
        "    for k in range(1, len(tpr)):\n"
        "        area += (fpr[k] - fpr[k - 1]) * (tpr[k] + tpr[k - 1]) / 2\n"
        "    return area\n"
    ),
    traps=[
        "Thresholding each example separately when scores tie — the curve takes "
        "a staircase detour and the area disagrees with the pairwise identity.",
        "Using rectangles instead of trapezoids, which breaks exactly and only "
        "on tied groups.",
        "Computing accuracy at a fixed 0.5 threshold — AUC is a ranking metric "
        "and never touches a threshold on the score scale.",
    ],
    tests='''
def checks(fn, check):
    def pairwise(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        wins = ties = 0
        for p in pos:
            for q in neg:
                if float(p) > float(q):
                    wins += 1
                elif float(p) == float(q):
                    ties += 1
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    perfect_s = torch.tensor([0.9, 0.8, 0.2, 0.1])
    perfect_l = torch.tensor([1, 1, 0, 0])
    check("perfect ranking gives 1", lambda: abs(fn(perfect_s, perfect_l) - 1.0) < 1e-9)
    check("inverted ranking gives 0",
          lambda: abs(fn(perfect_s, 1 - perfect_l)) < 1e-9)
    torch.manual_seed(0)
    s = torch.randn(40)
    l = (torch.rand(40) < 0.4).long()
    check("matches the pairwise-concordance identity",
          lambda: abs(fn(s, l) - pairwise(s, l)) < 1e-9)
    def with_ties():
        st = torch.tensor([0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
        lt = torch.tensor([1, 1, 0, 1, 0, 0])
        return abs(fn(st, lt) - pairwise(st, lt)) < 1e-9
    check("agrees with half-credit on tied scores", with_ties)
    check("all scores equal gives exactly 0.5",
          lambda: abs(fn(torch.full((6,), 0.3), torch.tensor([1, 0, 1, 0, 1, 0])) - 0.5) < 1e-9)
''',
),

task(
    id="welford",
    title="Welford's online mean and variance",
    book=BOOK, chapter=C_PROB,
    section="Probability and estimation — streaming statistics",
    level=2,
    entry="online_stats",
    statement=(
        "Compute the mean and (sample) variance of a stream in one pass with "
        "Welford's update: delta = x - mean; mean += delta/n; "
        "M2 += delta * (x - mean_new). The one-pass textbook alternative — "
        "accumulate sum(x) and sum(x^2), then E[x^2] - E[x]^2 — is "
        "catastrophically cancellative: shift the data to mean 1e8 and it "
        "returns garbage or a negative variance, which the tests check."
    ),
    shapes="xs (N,) float64  ->  dict 'mean' float, 'var' float (sample, /(n-1))",
    stub=("def online_stats(xs):\n"
          "    # one pass, Welford's recurrence; never store the data\n    pass\n"),
    hints=[
        "Keep three scalars: n, mean, M2.",
        "delta = x - mean BEFORE the mean update; delta2 = x - mean AFTER; "
        "M2 += delta * delta2.",
        "Sample variance is M2 / (n - 1).",
    ],
    solution=(
        "def online_stats(xs):\n"
        "    n = 0\n"
        "    mean = 0.0\n"
        "    M2 = 0.0\n"
        "    for x in xs:\n"
        "        x = float(x)\n"
        "        n += 1\n"
        "        delta = x - mean\n"
        "        mean += delta / n\n"
        "        M2 += delta * (x - mean)\n"
        "    return {'mean': mean, 'var': M2 / (n - 1)}\n"
    ),
    solution_np=(
        "def online_stats(xs):\n"
        "    n = 0\n"
        "    mean = 0.0\n"
        "    M2 = 0.0\n"
        "    for x in xs:\n"
        "        x = float(x)\n"
        "        n += 1\n"
        "        delta = x - mean\n"
        "        mean += delta / n\n"
        "        M2 += delta * (x - mean)\n"
        "    return {'mean': mean, 'var': M2 / (n - 1)}\n"
    ),
    traps=[
        "sum(x^2)/n - mean^2 in one pass — two enormous nearly-equal numbers "
        "subtracted; shifted data returns nonsense or a negative variance.",
        "Using delta twice instead of delta * delta2, which biases M2.",
        "Dividing by n when the sample variance was asked for.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    xs = torch.randn(2000, dtype=torch.float64)
    o = fn(xs)
    check("mean matches the batch mean", lambda: abs(o["mean"] - float(xs.mean())) < 1e-10)
    check("variance matches the batch sample variance",
          lambda: abs(o["var"] - float(xs.var(unbiased=True))) < 1e-10)
    def shifted_survives():
        big = xs + 1e8
        ob = fn(big)
        return abs(ob["var"] - float(xs.var(unbiased=True))) < 1e-4 and ob["var"] > 0
    check("survives data shifted to mean 1e8 (kills sum-of-squares)", shifted_survives)
    check("constant stream has zero variance",
          lambda: abs(fn(torch.full((50,), 7.0, dtype=torch.float64))["var"]) < 1e-12)
    check("two points: var is half the squared gap",
          lambda: abs(fn(torch.tensor([1.0, 3.0], dtype=torch.float64))["var"] - 2.0) < 1e-12)
''',
),

task(
    id="reservoir-sampling",
    title="Reservoir sampling",
    book=BOOK, chapter=C_PROB,
    section="Probability and estimation — Monte Carlo",
    level=2,
    entry="reservoir",
    statement=(
        "Draw a uniform sample of k items from a stream of unknown length in "
        "one pass and O(k) memory: keep the first k, then replace a random "
        "reservoir slot with item i (1-indexed) with probability k/i. The "
        "correctness claim is exact — every item ends up in the reservoir with "
        "probability exactly k/n — and the tests measure it over thousands of "
        "seeded runs."
    ),
    shapes="stream (N,) · k int  ->  (k,) items from the stream",
    stub=("def reservoir(stream, k):\n"
          "    # keep first k; item i replaces a random slot with prob k/i\n    pass\n"),
    hints=[
        "For item i > k: draw j uniform in [0, i); if j < k, overwrite slot j.",
        "One uniform integer per item does both decisions at once.",
        "torch.randint(0, i, (1,)) — i here is the 1-indexed position.",
    ],
    solution=(
        "def reservoir(stream, k):\n"
        "    res = [stream[i] for i in range(k)]\n"
        "    for i in range(k, len(stream)):\n"
        "        j = int(torch.randint(0, i + 1, (1,)))\n"
        "        if j < k:\n"
        "            res[j] = stream[i]\n"
        "    return torch.stack([torch.as_tensor(r) for r in res])\n"
    ),
    frameworks=["torch"],
    traps=[
        "Replacing with probability k/n or a constant instead of k/i — the "
        "early items end up over- or under-represented.",
        "Always replacing (probability 1), which biases the reservoir hard "
        "toward the tail of the stream.",
        "Sampling the slot and the accept decision with two draws that are not "
        "independent of each other.",
    ],
    tests='''
def checks(fn, check):
    stream = torch.arange(10)
    check("n == k returns the whole stream",
          lambda: sorted(fn(stream[:4], 4).tolist()) == [0, 1, 2, 3])
    check("returns k items", lambda: shape(fn(stream, 3)) == (3,))
    check("items come from the stream",
          lambda: all(int(v) in range(10) for v in fn(stream, 3)))
    def uniform_inclusion():
        counts = torch.zeros(10)
        trials = 4000
        for _ in range(trials):
            for v in fn(stream, 3):
                counts[int(v)] += 1
        probs = counts / trials
        return bool(((probs - 0.3).abs() < 0.03).all())
    check("every item is included with probability k/n = 0.3 (4000 runs)",
          uniform_inclusion)
    def distinct():
        out = fn(stream, 5)
        return len(set(out.tolist())) == 5
    check("no duplicates", distinct)
''',
),

task(
    id="grad-check",
    title="Numerical gradient checking",
    book=BOOK, chapter=C_NN,
    section="Neural networks — backpropagation",
    level=2,
    entry="numerical_grad",
    statement=(
        "Estimate the gradient of a scalar function by CENTRAL differences: "
        "perturb each coordinate by ±eps and take (f(x+e) - f(x-e)) / (2 eps). "
        "This is how hand-written backward passes get debugged. The central "
        "form matters: its error is O(eps^2) against the forward difference's "
        "O(eps), and the tests hold you to a tolerance only the central form "
        "meets."
    ),
    shapes="f callable (D,) -> scalar · x (D,) · eps float  ->  (D,)",
    stub=("def numerical_grad(f, x, eps=1e-5):\n"
          "    # central differences, one coordinate at a time\n    pass\n"),
    hints=[
        "Loop over coordinates; perturb a CLONE of x in place, evaluate, "
        "restore.",
        "g[i] = (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps).",
        "Do not touch autograd — the whole point is an independent estimate.",
    ],
    solution=(
        "def numerical_grad(f, x, eps=1e-5):\n"
        "    g = torch.zeros_like(x)\n"
        "    for i in range(x.numel()):\n"
        "        xp = x.clone(); xp.view(-1)[i] += eps\n"
        "        xm = x.clone(); xm.view(-1)[i] -= eps\n"
        "        g.view(-1)[i] = (float(f(xp)) - float(f(xm))) / (2 * eps)\n"
        "    return g\n"
    ),
    frameworks=["torch"],
    traps=[
        "The forward difference (f(x+e) - f(x)) / e — an order of accuracy "
        "worse, and it fails the tolerance here.",
        "Perturbing x itself without restoring it, so later coordinates see a "
        "corrupted point.",
        "eps too small: below ~1e-8 in float64 the subtraction's rounding noise "
        "dominates the signal.",
    ],
    tests='''
def checks(fn, check):
    x = torch.tensor([0.7, -1.2, 2.0], dtype=torch.float64)
    def poly(v):
        return (v ** 3).sum() + 2 * v[0] * v[1]
    want = torch.tensor([3 * 0.7 ** 2 + 2 * (-1.2),
                         3 * 1.2 ** 2 + 2 * 0.7,
                         3 * 4.0], dtype=torch.float64)
    check("matches the hand gradient of a polynomial (tight tolerance)",
          lambda: close(fn(poly, x), want, 1e-7))
    def vs_autograd():
        A = torch.randn(3, 3, dtype=torch.float64)
        Q = A @ A.T + torch.eye(3, dtype=torch.float64)
        def quad(v):
            return 0.5 * v @ Q @ v
        xx = x.clone().requires_grad_(True)
        quad(xx).backward()
        return close(fn(quad, x), xx.grad, 1e-7)
    check("matches autograd on a quadratic form", vs_autograd)
    check("gradient of sin at 0 is cos(0) = 1",
          lambda: abs(float(fn(lambda v: torch.sin(v).sum(),
                               torch.zeros(1, dtype=torch.float64))[0]) - 1.0) < 1e-8)
    check("output shape", lambda: shape(fn(poly, x)) == (3,))
    def does_not_mutate():
        xc = x.clone()
        fn(poly, x)
        return close(x, xc)
    check("input is not mutated", does_not_mutate)
''',
),

task(
    id="vae-elbo",
    title="The VAE loss (negative ELBO)",
    book=BOOK, chapter=C_VI,
    section="Variational inference — mean-field and the ELBO",
    level=2,
    entry="vae_loss",
    statement=(
        "Assemble the VAE training loss: binary cross-entropy reconstruction "
        "plus the closed-form KL from the diagonal-Gaussian encoder to the "
        "standard-normal prior, KL = -0.5 * sum(1 + logvar - mu^2 - "
        "exp(logvar)), both summed over dimensions and averaged over the "
        "batch. Return the pieces separately as well — inspecting their balance "
        "is how VAE training gets debugged."
    ),
    shapes=("x, x_recon (B, D) in [0,1] · mu, logvar (B, Z)"
            "  ->  dict 'recon', 'kl', 'loss' scalars"),
    stub=("def vae_loss(x, x_recon, mu, logvar):\n"
          "    # -> {'recon': ..., 'kl': ..., 'loss': recon + kl}\n    pass\n"),
    hints=[
        "recon = BCE(x_recon, x) summed over pixels, averaged over the batch.",
        "KL per example: -0.5 * sum over latent dims of (1 + logvar - mu^2 - "
        "exp(logvar)); average over the batch.",
        "The total is their sum — no extra weighting unless a beta is asked "
        "for.",
    ],
    solution=(
        "def vae_loss(x, x_recon, mu, logvar):\n"
        "    recon = F.binary_cross_entropy(x_recon, x, reduction='none').sum(-1).mean()\n"
        "    kl = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)).mean()\n"
        "    return {'recon': recon, 'kl': kl, 'loss': recon + kl}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Averaging over latent dimensions instead of summing — it silently "
        "rescales the KL term by 1/Z and unbalances the loss.",
        "A sign error in the KL, which rewards the encoder for drifting from "
        "the prior.",
        "Using MSE where the decoder is Bernoulli — a modelling mismatch that "
        "still trains, badly.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    B, D, Z = 4, 6, 3
    x = torch.rand(B, D)
    xr = torch.rand(B, D).clamp(1e-4, 1 - 1e-4)
    mu = torch.randn(B, Z)
    lv = torch.randn(B, Z)
    o = fn(x, xr, mu, lv)
    def kl_matches_torch():
        q = torch.distributions.Normal(mu, (0.5 * lv).exp())
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        want = torch.distributions.kl_divergence(q, p).sum(-1).mean()
        return close(o["kl"], want, 1e-5)
    check("KL matches torch.distributions in closed form", kl_matches_torch)
    check("recon matches summed BCE",
          lambda: close(o["recon"],
                        F.binary_cross_entropy(xr, x, reduction='none').sum(-1).mean(), 1e-6))
    check("loss = recon + kl", lambda: close(o["loss"], o["recon"] + o["kl"], 1e-6))
    check("standard-normal encoder has zero KL",
          lambda: abs(float(fn(x, xr, torch.zeros(B, Z), torch.zeros(B, Z))["kl"])) < 1e-7)
    check("KL is non-negative", lambda: float(o["kl"]) >= -1e-6)
''',
),

task(
    id="conv2d",
    title="Conv2d from scratch",
    book=BOOK, chapter=C_NN,
    section="Neural networks — convolution",
    level=3,
    entry="conv2d",
    statement=(
        "Implement a 2-D convolution with stride and zero padding, matching "
        "F.conv2d. Deep-learning 'convolution' is cross-correlation — the "
        "kernel is NOT flipped — and implementing the flipped textbook version "
        "produces confidently wrong numbers against every framework. unfold "
        "turns the whole thing into one matrix multiply."
    ),
    shapes=("x (B, Cin, H, W) · w (Cout, Cin, kh, kw) · b (Cout,) · stride, padding int"
            "  ->  (B, Cout, H', W')"),
    stub=("def conv2d(x, w, b, stride=1, padding=0):\n"
          "    # cross-correlation, matching F.conv2d\n    pass\n"),
    hints=[
        "F.unfold(x, (kh, kw), stride=stride, padding=padding) gives "
        "(B, Cin*kh*kw, L) patches.",
        "Flatten the kernel to (Cout, Cin*kh*kw) and matmul.",
        "H' = (H + 2p - kh)//stride + 1; fold the L axis back to (H', W') and "
        "add the bias per output channel.",
    ],
    solution=(
        "def conv2d(x, w, b, stride=1, padding=0):\n"
        "    B, Cin, H, W = x.shape\n"
        "    Cout, _, kh, kw = w.shape\n"
        "    cols = F.unfold(x, (kh, kw), stride=stride, padding=padding)\n"
        "    out = w.view(Cout, -1) @ cols + b.view(1, Cout, 1)\n"
        "    Ho = (H + 2 * padding - kh) // stride + 1\n"
        "    Wo = (W + 2 * padding - kw) // stride + 1\n"
        "    return out.view(B, Cout, Ho, Wo)\n"
    ),
    frameworks=["torch"],
    traps=[
        "Flipping the kernel — true convolution, wrong against every DL "
        "framework.",
        "Getting the output size formula wrong with stride and padding "
        "together.",
        "Adding the bias per patch instead of per output channel.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    w = torch.randn(5, 3, 3, 3)
    b = torch.randn(5)
    check("matches F.conv2d, stride 1 no padding",
          lambda: close(fn(x, w, b), F.conv2d(x, w, b), 1e-4))
    check("matches with padding 1",
          lambda: close(fn(x, w, b, 1, 1), F.conv2d(x, w, b, padding=1), 1e-4))
    check("matches with stride 2 and padding 1",
          lambda: close(fn(x, w, b, 2, 1), F.conv2d(x, w, b, stride=2, padding=1), 1e-4))
    check("1x1 identity kernel returns the channel",
          lambda: close(fn(x[:, :1], torch.ones(1, 1, 1, 1), torch.zeros(1)),
                        x[:, :1], 1e-5))
    def not_flipped():
        # an asymmetric kernel distinguishes correlation from convolution
        k = torch.zeros(1, 1, 1, 2); k[0, 0, 0, 0] = 1.0
        got = fn(x[:1, :1], k, torch.zeros(1))
        return close(got, F.conv2d(x[:1, :1], k, torch.zeros(1)), 1e-5)
    check("kernel is NOT flipped (cross-correlation)", not_flipped)
    check("output shape formula", lambda: shape(fn(x, w, b, 2, 1)) == (2, 5, 4, 4))
''',
),

task(
    id="batchnorm2d",
    title="BatchNorm2d, training mode",
    book=BOOK, chapter=C_NN,
    section="Neural networks — normalisation over the batch",
    level=3,
    entry="batchnorm2d",
    statement=(
        "One training-mode BatchNorm2d step: normalise each channel by the "
        "batch statistics computed over (B, H, W), apply the affine, and update "
        "the running statistics with momentum. The asymmetry that everyone "
        "misses: normalisation uses the BIASED variance, but the running "
        "variance is updated with the UNBIASED one (Bessel's correction) — "
        "that is what nn.BatchNorm2d does, and the tests compare against it "
        "exactly."
    ),
    shapes=("x (B, C, H, W) · gamma, beta, run_mean, run_var (C,) · momentum, eps"
            "  ->  dict 'out' (B,C,H,W), 'run_mean' (C,), 'run_var' (C,)"),
    stub=("def batchnorm2d(x, gamma, beta, run_mean, run_var, momentum=0.1, eps=1e-5):\n"
          "    # normalise with biased var; update running stats with unbiased\n    pass\n"),
    hints=[
        "Per channel: mean and var over dims (0, 2, 3).",
        "out = gamma * (x - mean) / sqrt(var_biased + eps) + beta, with (C,) "
        "stats broadcast as (1, C, 1, 1).",
        "run_stat_new = (1 - momentum) * run_stat + momentum * batch_stat, "
        "where the variance fed in is the UNBIASED one (n/(n-1) * biased).",
    ],
    solution=(
        "def batchnorm2d(x, gamma, beta, run_mean, run_var, momentum=0.1, eps=1e-5):\n"
        "    dims = (0, 2, 3)\n"
        "    mean = x.mean(dims)\n"
        "    var_b = x.var(dims, unbiased=False)\n"
        "    n = x.numel() / x.shape[1]\n"
        "    var_u = var_b * n / (n - 1)\n"
        "    xhat = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var_b.view(1, -1, 1, 1) + eps)\n"
        "    out = gamma.view(1, -1, 1, 1) * xhat + beta.view(1, -1, 1, 1)\n"
        "    return {'out': out,\n"
        "            'run_mean': (1 - momentum) * run_mean + momentum * mean,\n"
        "            'run_var': (1 - momentum) * run_var + momentum * var_u}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Updating the running variance with the biased estimate — off by "
        "n/(n-1), invisible until evaluation.",
        "Normalising per (B, C) instead of per channel over (B, H, W).",
        "Writing the momentum convention backwards; in torch, momentum weights "
        "the NEW statistic.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    B, C, H, W = 4, 3, 5, 5
    x = torch.randn(B, C, H, W) * 2 + 1
    gamma, beta = torch.randn(C), torch.randn(C)
    rm, rv = torch.zeros(C), torch.ones(C)

    def oracle():
        m = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5)
        with torch.no_grad():
            m.weight.copy_(gamma); m.bias.copy_(beta)
            m.running_mean.copy_(rm); m.running_var.copy_(rv)
        m.train()
        out = m(x)
        return out, m.running_mean.clone(), m.running_var.clone()

    o = fn(x, gamma, beta, rm.clone(), rv.clone())
    want_out, want_rm, want_rv = oracle()
    check("output matches nn.BatchNorm2d in training mode",
          lambda: close(o["out"], want_out, 1e-4))
    check("running mean matches", lambda: close(o["run_mean"], want_rm, 1e-5))
    check("running var uses Bessel's correction (matches torch)",
          lambda: close(o["run_var"], want_rv, 1e-5))
    check("normalised output has ~zero channel means under identity affine",
          lambda: close(fn(x, torch.ones(C), torch.zeros(C), rm.clone(), rv.clone())["out"]
                        .mean(dim=(0, 2, 3)), torch.zeros(C), 1e-5))
    check("inputs' running stats are not mutated in place",
          lambda: (fn(x, gamma, beta, rm, rv), close(rm, torch.zeros(C)))[-1])
    def constant_channel_finite():
        xc = x.clone()
        xc[:, 0] = 3.0                          # zero-variance channel
        out = fn(xc, gamma, beta, rm.clone(), rv.clone())["out"]
        return bool(torch.isfinite(out).all())
    check("a zero-variance channel stays finite (eps is doing its job)",
          constant_channel_finite)
''',
),

task(
    id="lstm-cell",
    title="An LSTM cell from scratch",
    book=BOOK, chapter=C_NN,
    section="Neural networks — recurrent cells",
    level=3,
    entry="lstm_cell",
    statement=(
        "One LSTM step, matching nn.LSTMCell exactly: gates = W_ih x + b_ih + "
        "W_hh h + b_hh, split into four chunks in torch's order (input, "
        "forget, cell candidate, output); i, f, o pass through sigmoid, g "
        "through tanh; c' = f*c + i*g and h' = o * tanh(c'). The gate ORDER is "
        "the whole exam — any permutation still runs and returns "
        "plausible-looking states."
    ),
    shapes=("x (B, D) · h, c (B, H) · W_ih (4H, D) · W_hh (4H, H) · b_ih, b_hh (4H,)"
            "  ->  (h' (B, H), c' (B, H))"),
    stub=("def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):\n"
          "    # torch gate order: i, f, g, o\n    pass\n"),
    hints=[
        "gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh, shape (B, 4H).",
        "chunk(4, dim=1) in the order i, f, g, o.",
        "c' = sigmoid(f)*c + sigmoid(i)*tanh(g); h' = sigmoid(o)*tanh(c').",
    ],
    solution=(
        "def lstm_cell(x, h, c, W_ih, W_hh, b_ih, b_hh):\n"
        "    gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh\n"
        "    i, f, g, o = gates.chunk(4, dim=1)\n"
        "    c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)\n"
        "    h_new = torch.sigmoid(o) * torch.tanh(c_new)\n"
        "    return h_new, c_new\n"
    ),
    frameworks=["torch"],
    traps=[
        "Permuting the gate order — i, f, g, o is torch's convention, and any "
        "other order silently mismatches every pretrained weight.",
        "tanh on the gates or sigmoid on the candidate.",
        "Computing h' from the OLD cell state instead of the new one.",
    ],
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    B, D, H = 3, 4, 5
    x = torch.randn(B, D)
    h = torch.randn(B, H)
    c = torch.randn(B, H)
    cell = torch.nn.LSTMCell(D, H)
    h2, c2 = fn(x, h, c, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
    want_h, want_c = cell(x, (h, c))
    check("hidden state matches nn.LSTMCell", lambda: close(h2, want_h, 1e-5))
    check("cell state matches nn.LSTMCell", lambda: close(c2, want_c, 1e-5))
    check("shapes", lambda: shape(h2) == (B, H) and shape(c2) == (B, H))
    def saturated_forget_preserves_cell():
        big = torch.zeros(4 * H); big[H:2 * H] = 50.0          # forget gate wide open
        hn, cn = fn(torch.zeros(1, D), torch.zeros(1, H), torch.ones(1, H),
                    torch.zeros(4 * H, D), torch.zeros(4 * H, H), big, torch.zeros(4 * H))
        return close(cn, torch.ones(1, H), 1e-4)
    check("a saturated forget gate carries the cell state through",
          saturated_forget_preserves_cell)
    check("h' is bounded by tanh", lambda: bool((h2.abs() <= 1.0 + 1e-6).all()))
''',
),

]
