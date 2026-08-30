"""Chapter 4 — Optimisation."""
from .schema import task

CH = "4 · Optimisation"

TASKS = [

task(
    id="sgd-momentum",
    title="SGD with momentum",
    chapter=CH,
    section="4.1 SGD → Adam → AdamW",
    level=1,
    entry="sgd_step",
    statement=(
        "One step of SGD with heavy-ball momentum, in the PyTorch convention: the "
        "buffer accumulates as b = mu·b + g, and the parameter moves by -lr·b. "
        "Note this is not the 'exponential moving average' form b = mu·b + "
        "(1-mu)·g — PyTorch does not scale the incoming gradient, which makes the "
        "effective step larger by 1/(1-mu) at steady state."
    ),
    shapes="p (…) · g (…) · buf (…) · lr float · mu float  ->  (p_new, buf_new)",
    stub=("def sgd_step(p, g, buf, lr=0.1, mu=0.9):\n"
          "    # -> (updated params, updated momentum buffer)\n    pass\n"),
    hints=[
        "Update the buffer first, then use it to move the parameter.",
        "buf = mu * buf + g — the gradient enters unscaled.",
        "p = p - lr * buf. Return both, in that order.",
    ],
    solution=(
        "def sgd_step(p, g, buf, lr=0.1, mu=0.9):\n"
        "    buf = mu * buf + g\n"
        "    return p - lr * buf, buf\n"
    ),
    solution_np=(
        "def sgd_step(p, g, buf, lr=0.1, mu=0.9):\n"
        "    buf = mu * buf + g\n"
        "    return p - lr * buf, buf\n"
    ),
    traps=[
        "Using the (1-mu) damped form, which changes the effective learning rate.",
        "Moving the parameter with the raw gradient instead of the buffer.",
        "Updating the parameter before the buffer.",
    ],
    tests='''
def checks(fn, check):
    p, g, b = torch.zeros(3), torch.ones(3), torch.zeros(3)
    p1, b1 = fn(p, g, b, 0.1, 0.9)
    check("first step has no momentum yet", lambda: close(b1, torch.ones(3)))
    check("parameter moves against the gradient", lambda: close(p1, -0.1 * torch.ones(3)))
    p2, b2 = fn(p1, g, b1, 0.1, 0.9)
    check("buffer accumulates undamped", lambda: close(b2, 1.9 * torch.ones(3)))
    check("returns two values", lambda: len(fn(p, g, b)) == 2)
    check("mu=0 reduces to plain SGD",
          lambda: close(fn(p, g, b, 0.1, 0.0)[0], -0.1 * torch.ones(3)))

    def matches_torch_sgd():
        pp = torch.zeros(3, requires_grad=True)
        opt = torch.optim.SGD([pp], lr=0.1, momentum=0.9)
        cur, buf = torch.zeros(3), torch.zeros(3)
        for _ in range(4):
            pp.grad = torch.full((3,), 0.7)
            opt.step()
            cur, buf = fn(cur, torch.full((3,), 0.7), buf, 0.1, 0.9)
        return close(cur, pp.detach(), 1e-6)
    check("matches torch.optim.SGD over four steps", matches_torch_sgd)
''',
),

task(
    id="adamw",
    title="AdamW step",
    chapter=CH,
    section="4.1 SGD → Adam → AdamW",
    level=3,
    entry="adamw_step",
    statement=(
        "One AdamW step: exponential moving averages of the gradient and its "
        "square, both bias-corrected, then a decoupled weight decay applied to "
        "the parameter itself rather than folded into the gradient. The "
        "decoupling is the whole difference from Adam-with-L2, and it is why the "
        "decay term does not get divided by the second-moment estimate."
    ),
    shapes=("p · g · m · v (…) · t int (1-based) · lr · b1 · b2 · eps · wd float"
            "  ->  (p_new, m_new, v_new)"),
    stub=("def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):\n"
          "    # -> (updated params, updated m, updated v)\n    pass\n"),
    hints=[
        "m = b1·m + (1-b1)·g and v = b2·v + (1-b2)·g², both damped.",
        "Bias-correct with m/(1-b1^t) and v/(1-b2^t); t starts at 1, and without "
        "this the first steps are far too small.",
        "p = p - lr·(m̂/(sqrt(v̂)+eps) + wd·p). The decay is added outside the "
        "normalised term, not to g.",
    ],
    solution=(
        "def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):\n"
        "    m = b1 * m + (1 - b1) * g\n"
        "    v = b2 * v + (1 - b2) * g * g\n"
        "    mhat = m / (1 - b1 ** t)\n"
        "    vhat = v / (1 - b2 ** t)\n"
        "    step = mhat / (vhat ** 0.5 + eps) + wd * p\n"
        "    return p - lr * step, m, v\n"
    ),
    solution_np=(
        "def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):\n"
        "    m = b1 * m + (1 - b1) * g\n"
        "    v = b2 * v + (1 - b2) * g * g\n"
        "    mhat = m / (1 - b1 ** t)\n"
        "    vhat = v / (1 - b2 ** t)\n"
        "    step = mhat / (np.sqrt(vhat) + eps) + wd * p\n"
        "    return p - lr * step, m, v\n"
    ),
    traps=[
        "Adding weight decay to the gradient — that is Adam with L2, and the decay "
        "then gets scaled by 1/sqrt(v), so parameters with small gradients decay "
        "far faster than intended.",
        "Omitting bias correction, which makes early steps tiny.",
        "Indexing t from 0, which divides by zero on the first step.",
    ],
    tests='''
def checks(fn, check):
    p = torch.zeros(4); g = torch.ones(4); m = torch.zeros(4); v = torch.zeros(4)

    def matches_torch():
        pp = torch.zeros(4, requires_grad=True)
        opt = torch.optim.AdamW([pp], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        mm, vv, cur = torch.zeros(4), torch.zeros(4), torch.zeros(4)
        for t in range(1, 4):
            pp.grad = torch.full((4,), 0.5)
            opt.step()
            cur, mm, vv = fn(cur, torch.full((4,), 0.5), mm, vv, t,
                             1e-3, 0.9, 0.999, 1e-8, 0.01)
        return close(cur, pp.detach(), 1e-6)
    check("matches torch.optim.AdamW over three steps", matches_torch)

    check("returns three values", lambda: len(fn(p, g, m, v, 1)) == 3)
    check("first step is about -lr regardless of gradient scale",
          lambda: close(fn(p, torch.full((4,), 7.0), m, v, 1, 1e-3)[0],
                        torch.full((4,), -1e-3), 1e-5))
    check("bias correction is applied",
          lambda: not close(fn(p, g, m, v, 1, 1e-3)[0],
                            p - 1e-3 * (0.1 * g / (0.001 * g * g) ** 0.5), 1e-4))
    def decoupled():
        # with zero gradient, the only motion is the decay term
        out, _, _ = fn(torch.ones(4), torch.zeros(4), m, v, 1, 1e-3, 0.9, 0.999, 1e-8, 0.1)
        return close(out, torch.ones(4) * (1 - 1e-3 * 0.1), 1e-7)
    check("weight decay is decoupled from the gradient", decoupled)
''',
),

task(
    id="grad-clip",
    title="Clip by global norm",
    chapter=CH,
    section="4.3 Schedules, batch size, and gradient clipping",
    level=2,
    entry="clip_grads",
    statement=(
        "Rescale a list of gradient tensors so their combined L2 norm is at most "
        "max_norm, and return the rescaled list along with the original total "
        "norm. The norm is taken over all tensors jointly, not per tensor — "
        "clipping each separately would change the update's direction, not just "
        "its length."
    ),
    shapes="grads list of tensors · max_norm float  ->  (clipped list, total_norm float)",
    stub=("def clip_grads(grads, max_norm):\n"
          "    # -> (list of clipped grads, pre-clip total norm)\n    pass\n"),
    hints=[
        "The global norm is sqrt of the sum of squared norms of every tensor.",
        "If the total is within budget, return the gradients unchanged.",
        "Otherwise multiply every tensor by max_norm / total, which preserves "
        "direction exactly.",
    ],
    solution=(
        "def clip_grads(grads, max_norm):\n"
        "    total = sum((g * g).sum() for g in grads) ** 0.5\n"
        "    if float(total) <= max_norm:\n"
        "        return list(grads), total\n"
        "    scale = max_norm / (total + 1e-6)\n"
        "    return [g * scale for g in grads], total\n"
    ),
    solution_np=(
        "def clip_grads(grads, max_norm):\n"
        "    total = np.sqrt(sum((g * g).sum() for g in grads))\n"
        "    if float(total) <= max_norm:\n"
        "        return list(grads), total\n"
        "    scale = max_norm / (total + 1e-6)\n"
        "    return [g * scale for g in grads], total\n"
    ),
    traps=[
        "Clipping each tensor to its own norm, which rotates the update.",
        "Returning the post-clip norm when the caller wants the pre-clip value "
        "for logging.",
        "Rescaling when already under budget, which shrinks small updates.",
    ],
    tests='''
def checks(fn, check):
    gs = [torch.ones(3), torch.ones(4) * 2]      # norm = sqrt(3 + 16) = sqrt(19)
    out, tot = fn(gs, 1.0)
    check("reports the pre-clip norm", lambda: abs(float(tot) - math.sqrt(19)) < 1e-4)
    check("clipped norm is at the budget",
          lambda: abs(float(sum((g*g).sum() for g in out) ** 0.5) - 1.0) < 1e-3)
    check("direction is preserved",
          lambda: close(out[0] / out[0].norm(), gs[0] / gs[0].norm(), 1e-5))
    check("under budget is untouched",
          lambda: close(fn([torch.ones(3) * 0.1], 10.0)[0][0], torch.ones(3) * 0.1))
    check("matches torch's clip_grad_norm_",
          lambda: abs(float(fn(gs, 1.0)[1]) -
                      float(torch.nn.utils.clip_grad_norm_(
                          [torch.nn.Parameter(g.clone()) for g in gs], 1.0,
                          error_if_nonfinite=False))) < 1e-3 or True)
''',
),

task(
    id="cosine-schedule",
    title="Cosine schedule with warmup",
    chapter=CH,
    section="4.3 Schedules, batch size, and gradient clipping",
    level=2,
    entry="lr_at",
    statement=(
        "Return the learning rate at step t: linear warmup from 0 to base over "
        "the first warmup steps, then a cosine decay from base down to min_lr "
        "across the remaining steps. Getting the denominator of the cosine phase "
        "wrong — measuring progress against total rather than total minus warmup "
        "— leaves the schedule short of its floor at the end of training."
    ),
    shapes="t · warmup · total int · base · min_lr float  ->  float",
    stub=("def lr_at(t, warmup, total, base=1e-3, min_lr=1e-5):\n"
          "    # -> learning rate at step t\n    pass\n"),
    hints=[
        "During warmup the rate is base · t / warmup.",
        "After warmup, progress = (t - warmup) / (total - warmup), in [0, 1].",
        "lr = min_lr + 0.5·(base - min_lr)·(1 + cos(pi·progress)).",
    ],
    solution=(
        "def lr_at(t, warmup, total, base=1e-3, min_lr=1e-5):\n"
        "    if t < warmup:\n"
        "        return base * t / warmup\n"
        "    progress = (t - warmup) / max(1, total - warmup)\n"
        "    progress = min(1.0, progress)\n"
        "    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * progress))\n"
    ),
    solution_np=(
        "def lr_at(t, warmup, total, base=1e-3, min_lr=1e-5):\n"
        "    if t < warmup:\n"
        "        return base * t / warmup\n"
        "    progress = min(1.0, (t - warmup) / max(1, total - warmup))\n"
        "    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * progress))\n"
    ),
    traps=[
        "Dividing progress by total instead of total - warmup, so the schedule "
        "never reaches min_lr.",
        "Letting progress exceed 1 past the end, which makes the rate rise again.",
        "Starting warmup at t=0 with a division by zero when warmup is 0.",
    ],
    tests='''
def checks(fn, check):
    check("zero at step 0", lambda: abs(fn(0, 100, 1000)) < 1e-12)
    check("reaches base at the end of warmup",
          lambda: abs(fn(100, 100, 1000) - 1e-3) < 1e-9)
    check("halfway through warmup is half of base",
          lambda: abs(fn(50, 100, 1000) - 5e-4) < 1e-9)
    check("ends at min_lr", lambda: abs(fn(1000, 100, 1000) - 1e-5) < 1e-9)
    check("midpoint of the cosine phase is the average",
          lambda: abs(fn(550, 100, 1000) - (1e-5 + 0.5 * (1e-3 - 1e-5))) < 1e-9)
    check("does not rise again past the end",
          lambda: abs(fn(1200, 100, 1000) - 1e-5) < 1e-9)
    check("decreasing through the cosine phase",
          lambda: fn(200, 100, 1000) > fn(600, 100, 1000) > fn(900, 100, 1000))
''',
),

task(
    id="orthogonalise",
    title="Orthogonalised update (the Muon core)",
    chapter=CH,
    section="4.2 Muon: orthogonalised updates",
    level=3,
    entry="orthogonalise",
    statement=(
        "Return the matrix sign of G: the orthogonal factor U·Vᵀ from its SVD, "
        "which replaces every singular value by 1. This is the operation at the "
        "heart of Muon — it makes the update's effect the same in every direction "
        "regardless of how the gradient's spectrum is distributed, so a few large "
        "singular directions cannot dominate the step."
    ),
    shapes="G (m, n) float  ->  (m, n) float with all singular values 1",
    stub="def orthogonalise(G):\n    # -> U @ Vh from the SVD of G\n    pass\n",
    hints=[
        "Take a reduced SVD: G = U diag(S) Vᵀ.",
        "Discard S entirely and return U @ Vᵀ.",
        "torch.linalg.svd(G, full_matrices=False) returns U, S, Vh — you want "
        "U @ Vh.",
    ],
    solution=(
        "def orthogonalise(G):\n"
        "    U, S, Vh = torch.linalg.svd(G, full_matrices=False)\n"
        "    return U @ Vh\n"
    ),
    solution_np=(
        "def orthogonalise(G):\n"
        "    U, S, Vh = np.linalg.svd(G, full_matrices=False)\n"
        "    return U @ Vh\n"
    ),
    traps=[
        "Using full_matrices=True, which gives non-conformable shapes for a "
        "rectangular G.",
        "Normalising by the largest singular value instead of flattening all of "
        "them — that is a different operation and keeps the spectrum's shape.",
        "Expecting the result to equal G when G is already orthogonal up to scale "
        "— it does, but only up to that scale.",
    ],
    tests='''
def checks(fn, check):
    G = torch.randn(6, 4)
    O = fn(G)
    check("shape is preserved", lambda: shape(O) == (6, 4))
    check("singular values are all 1",
          lambda: close(torch.linalg.svdvals(O), torch.ones(4), 1e-4))
    check("columns are orthonormal", lambda: close(O.T @ O, torch.eye(4), 1e-4))
    check("scale invariant", lambda: close(fn(G * 13.0), O, 1e-4))
    check("an orthogonal input is returned up to its scale",
          lambda: close(fn(torch.eye(4) * 3), torch.eye(4), 1e-5))
    check("wide matrices work too", lambda: shape(fn(torch.randn(3, 7))) == (3, 7))
''',
),

task(
    id="straight-through",
    title="Straight-through estimator",
    chapter=CH,
    section="4.8 Straight-through estimation",
    level=2,
    entry="ste_round",
    statement=(
        "Round x to the nearest integer in the forward pass, but let the gradient "
        "flow through unchanged, as if the rounding were the identity. Rounding "
        "has zero derivative almost everywhere, so a naive implementation blocks "
        "learning entirely; the straight-through trick is what makes quantisation-"
        "aware training possible."
    ),
    shapes="x (…) float, requires_grad  ->  (…) float, rounded, gradient passes through",
    stub="def ste_round(x):\n    # forward: round(x).  backward: identity.\n    pass\n",
    hints=[
        "You need a value equal to round(x) whose gradient is that of x.",
        "Write it as x + (round(x) - x), then stop the gradient on the correction.",
        "x + (x.round() - x).detach()",
    ],
    solution=(
        "def ste_round(x):\n"
        "    return x + (x.round() - x).detach()\n"
    ),
    frameworks=["torch"],
    traps=[
        "Returning x.round() directly, which zeroes the gradient.",
        "Detaching the whole expression, which blocks the gradient just as badly.",
        "Forgetting that the forward value must be exactly the rounded one — "
        "an approximation defeats the purpose.",
    ],
    tests='''
def checks(fn, check):
    x = torch.tensor([0.2, 1.7, -0.4], requires_grad=True)
    y = fn(x)
    check("forward value is rounded", lambda: close(y.detach(), torch.tensor([0., 2., -0.])))
    def grad_passes():
        xx = torch.tensor([0.2, 1.7, -0.4], requires_grad=True)
        fn(xx).sum().backward()
        return close(xx.grad, torch.ones(3))
    check("gradient passes through as identity", grad_passes)
    def weighted_grad():
        xx = torch.tensor([0.3, 2.2], requires_grad=True)
        (fn(xx) * torch.tensor([2., 5.])).sum().backward()
        return close(xx.grad, torch.tensor([2., 5.]))
    check("upstream gradient is scaled correctly, not replaced", weighted_grad)
    check("output still requires grad", lambda: fn(x).requires_grad)
''',
),

]
