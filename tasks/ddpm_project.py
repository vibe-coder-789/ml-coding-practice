"""build-ddpm — train a diffusion model, step by step.

A miniature DDPM on 2-D toy data: schedule -> forward process -> time
embedding -> denoiser -> loss -> training -> sampling -> the full pipeline
learning a two-mode distribution. Steps compose but are checked in isolation
against the reference stack below. Thresholds are calibrated, not guessed:
the reference trains to ~0.23 in 2.5s on CPU (predict-zero baseline: 1.0), and
an analytic optimal denoiser for Gaussian data anchors the sampler's output
distribution in closed form.
"""
from .schema import task

BOOK = "build-ddpm"
CH = "Train a diffusion model"

REF = '''
import torch.nn as nn

_T = 50


def _schedule(T=_T, beta_start=1e-4, beta_end=0.25):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    return {"betas": betas, "alphas": alphas,
            "abar": torch.cumprod(alphas, dim=0)}


def _q_sample(x0, t, abar, eps):
    a = abar[t].unsqueeze(-1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps


def _time_emb(t, dim=32):
    i = torch.arange(dim // 2, dtype=torch.float32)
    w = 10000.0 ** (-2 * i / dim)
    ang = t.float()[:, None] * w[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class _Denoiser(nn.Module):
    def __init__(self, tdim=32, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x, temb):
        return self.net(torch.cat([x, temb], dim=-1))


def _loss(model, x0, sched, t, eps):
    x_t = _q_sample(x0, t, sched["abar"], eps)
    return ((model(x_t, _time_emb(t)) - eps) ** 2).mean()


def _two_gaussians(n):
    m = torch.where(torch.rand(n, 1) < 0.5, -2.0, 2.0)
    return torch.cat([m, torch.zeros(n, 1)], dim=1) + 0.3 * torch.randn(n, 2)


def _train(model, data_fn, sched, steps=2500, lr=2e-3, batch=256):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x0 = data_fn(batch)
        t = torch.randint(0, len(sched["betas"]), (batch,))
        eps = torch.randn_like(x0)
        loss = _loss(model, x0, sched, t, eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def _sample(denoise, sched, n):
    betas, alphas, abar = sched["betas"], sched["alphas"], sched["abar"]
    T = len(betas)
    x = torch.randn(n, 2)
    for t in range(T - 1, -1, -1):
        with torch.no_grad():
            eps_hat = denoise(x, t)
        mu = (x - betas[t] / torch.sqrt(1 - abar[t]) * eps_hat) / torch.sqrt(alphas[t])
        if t > 0:
            btilde = (1 - abar[t - 1]) / (1 - abar[t]) * betas[t]
            x = mu + torch.sqrt(btilde) * torch.randn_like(x)
        else:
            x = mu
    return x


def _oracle_denoise(sched, m=1.5, s=0.6):
    """The closed-form optimal denoiser for x0 ~ N(m, s^2 I)."""
    abar = sched["abar"]
    def d(x, t):
        ab = abar[t]
        return torch.sqrt(1 - ab) * (x - torch.sqrt(ab) * m) / (ab * s * s + 1 - ab)
    return d
'''

TASKS = [

task(
    id="ddpm-schedule",
    title="Step 1 · The noise schedule",
    book=BOOK, chapter=CH, section="Step 1 · Schedule",
    level=1,
    entry="make_schedule",
    statement=(
        "Build the linear DDPM schedule: betas evenly spaced from beta_start to "
        "beta_end over T steps, alphas = 1 - betas, and abar the RUNNING PRODUCT "
        "of the alphas. Everything downstream — the forward closed form, the "
        "posterior variance, the sampler — reads these three tensors, and the "
        "single most damaging mistake is a cumulative SUM where the cumulative "
        "product belongs: shapes agree, training even limps along, and nothing "
        "works."
    ),
    shapes="T int · beta_start, beta_end float  ->  dict 'betas', 'alphas', 'abar', each (T,)",
    stub=("def make_schedule(T=50, beta_start=1e-4, beta_end=0.25):\n"
          "    # -> {'betas': ..., 'alphas': ..., 'abar': ...}\n    pass\n"),
    hints=[
        "torch.linspace for the betas.",
        "abar_t = prod_{s<=t} alpha_s — torch.cumprod, dim=0.",
        "abar must start at 1 - beta_start and decrease monotonically.",
    ],
    solution=(
        "def make_schedule(T=50, beta_start=1e-4, beta_end=0.25):\n"
        "    betas = torch.linspace(beta_start, beta_end, T)\n"
        "    alphas = 1 - betas\n"
        "    return {'betas': betas, 'alphas': alphas,\n"
        "            'abar': torch.cumprod(alphas, dim=0)}\n"
    ),
    solution_np=(
        "def make_schedule(T=50, beta_start=1e-4, beta_end=0.25):\n"
        "    betas = np.linspace(beta_start, beta_end, T)\n"
        "    alphas = 1 - betas\n"
        "    return {'betas': betas, 'alphas': alphas,\n"
        "            'abar': np.cumprod(alphas)}\n"
    ),
    traps=[
        "cumsum instead of cumprod for abar — right shape, wrong process.",
        "abar indexed from alpha_1 instead of alpha_0, an off-by-one that "
        "shifts every noise level downstream.",
        "Betas outside (0, 1), which makes some alpha negative and the square "
        "roots downstream NaN.",
    ],
    tests='''
def checks(fn, check):
    o = fn(50, 1e-4, 0.25)
    check("betas are the linspace endpoints",
          lambda: abs(float(o["betas"][0]) - 1e-4) < 1e-9
                  and abs(float(o["betas"][-1]) - 0.25) < 1e-7)
    check("alphas = 1 - betas", lambda: close(o["alphas"], 1 - o["betas"], 1e-7))
    def abar_is_product():
        prod = 1.0
        for i in (0, 3, 17, 49):
            prod = float(torch.prod(torch.as_tensor(o["alphas"][:i + 1])))
            if abs(float(o["abar"][i]) - prod) > 1e-6:
                return False
        return True
    check("abar is the running PRODUCT of the alphas", abar_is_product)
    check("abar starts at 1 - beta_start",
          lambda: abs(float(o["abar"][0]) - (1 - 1e-4)) < 1e-7)
    check("abar decreases monotonically",
          lambda: bool((torch.as_tensor(o["abar"])[1:] <
                        torch.as_tensor(o["abar"])[:-1]).all()))
    check("shapes", lambda: shape(o["betas"]) == (50,) and shape(o["abar"]) == (50,))
''',
),

task(
    id="ddpm-q-sample",
    title="Step 2 · The forward process",
    book=BOOK, chapter=CH, section="Step 2 · Forward",
    level=1,
    entry="q_sample",
    statement=(
        "Jump to any noise level in one step: x_t = sqrt(abar_t) x_0 + "
        "sqrt(1 - abar_t) eps, with a per-sample t. This closed form is why "
        "diffusion training is cheap — every (x_0, t, eps) triple is a training "
        "example without simulating the chain. The coefficients are a "
        "signal/noise split whose squares sum to one; both identities in the "
        "tests fail the moment they are swapped."
    ),
    shapes="x0 (B, D) · t (B,) int64 · abar (T,) · eps (B, D)  ->  (B, D)",
    stub=("def q_sample(x0, t, abar, eps):\n"
          "    # sqrt(abar_t) x0 + sqrt(1 - abar_t) eps\n    pass\n"),
    hints=[
        "Gather abar at each sample's own t; unsqueeze to broadcast over "
        "features.",
        "Signal gets sqrt(abar), noise gets sqrt(1 - abar).",
        "The noise is an argument so the tests can invert the formula exactly.",
    ],
    solution=(
        "def q_sample(x0, t, abar, eps):\n"
        "    a = abar[t].unsqueeze(-1)\n"
        "    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * eps\n"
    ),
    solution_np=(
        "def q_sample(x0, t, abar, eps):\n"
        "    a = abar[t][:, None]\n"
        "    return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps\n"
    ),
    traps=[
        "Swapping the two coefficients — the marginal still looks Gaussian and "
        "training learns the wrong thing silently.",
        "Using alpha_t where the CUMULATIVE abar_t belongs.",
        "One shared t for the whole batch when each sample carries its own.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    sched = _schedule()
    B = 6
    x0 = torch.randn(B, 2)
    eps = torch.randn(B, 2)
    t = torch.randint(0, _T, (B,))
    xt = fn(x0, t, sched["abar"], eps)
    check("matches the reference forward process",
          lambda: close(xt, _q_sample(x0, t, sched["abar"], eps), 1e-6))
    def eps_recoverable():
        a = sched["abar"][t].unsqueeze(-1)
        return close((xt - torch.sqrt(a) * x0) / torch.sqrt(1 - a), eps, 1e-5)
    check("the injected noise inverts out exactly", eps_recoverable)
    def x0_recoverable():
        a = sched["abar"][t].unsqueeze(-1)
        return close((xt - torch.sqrt(1 - a) * eps) / torch.sqrt(a), x0, 1e-5)
    check("x0 inverts out exactly", x0_recoverable)
    check("per-sample t is honoured",
          lambda: not close(fn(x0, torch.zeros(B, dtype=torch.long), sched["abar"], eps),
                            fn(x0, torch.full((B,), _T - 1), sched["abar"], eps), 1e-3))
    check("output shape", lambda: shape(xt) == (B, 2))
''',
),

task(
    id="ddpm-time-emb",
    title="Step 3 · Sinusoidal time embedding",
    book=BOOK, chapter=CH, section="Step 3 · Time embedding",
    level=2,
    entry="time_embedding",
    statement=(
        "The denoiser needs to know the noise level, and an integer will not "
        "do: embed t as [sin(t w_0..w_{d/2-1}) | cos(t w_0..w_{d/2-1})] — the "
        "sin half CONCATENATED before the cos half, with frequencies "
        "w_i = 10000^(-2i/d). This project uses the half-split layout (like "
        "Llama's RoPE, unlike the interleaved original), and the reference "
        "denoiser was trained against it, so the layout is part of the "
        "contract."
    ),
    shapes="t (B,) int64 · dim even int  ->  (B, dim), [sin half | cos half]",
    stub=("def time_embedding(t, dim=32):\n"
          "    # first dim/2 columns sin, last dim/2 columns cos\n    pass\n"),
    hints=[
        "Angles: outer product of t with the dim/2 frequencies.",
        "torch.cat([sin, cos], dim=-1) — concatenate, do not interleave.",
        "t = 0 must give [0]*dim/2 + [1]*dim/2 exactly.",
    ],
    solution=(
        "def time_embedding(t, dim=32):\n"
        "    i = torch.arange(dim // 2, dtype=torch.float32)\n"
        "    w = 10000.0 ** (-2 * i / dim)\n"
        "    ang = t.float()[:, None] * w[None, :]\n"
        "    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)\n"
    ),
    solution_np=(
        "def time_embedding(t, dim=32):\n"
        "    i = np.arange(dim // 2, dtype=np.float64)\n"
        "    w = 10000.0 ** (-2 * i / dim)\n"
        "    ang = t.astype(np.float64)[:, None] * w[None, :]\n"
        "    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)\n"
    ),
    traps=[
        "Interleaving sin and cos — the drill volume's OTHER convention; both "
        "are valid encodings, but this project's denoiser expects the "
        "half-split, and mixing layouts silently degrades everything trained "
        "against one of them.",
        "Frequencies with exponent i/d instead of 2i/d.",
        "Integer angles: forgetting to cast t to float before the outer "
        "product.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    t = torch.tensor([0, 1, 3, 49])
    e = fn(t, 32)
    check("matches the reference embedding", lambda: close(e, _time_emb(t, 32), 1e-5))
    check("t = 0 is [zeros | ones]",
          lambda: close(e[0], torch.cat([torch.zeros(16), torch.ones(16)]), 1e-6))
    check("first sin column advances at frequency 1",
          lambda: abs(float(e[2, 0]) - math.sin(3.0)) < 1e-5)
    check("cos half sits in the SECOND half (layout contract)",
          lambda: abs(float(e[2, 16]) - math.cos(3.0)) < 1e-5)
    check("bounded by 1", lambda: bool((e.abs() <= 1 + 1e-6).all()))
    check("distinct times embed distinctly",
          lambda: not close(e[1], e[2], 1e-3))
''',
),

task(
    id="ddpm-denoiser",
    title="Step 4 · The denoiser network",
    book=BOOK, chapter=CH, section="Step 4 · Denoiser",
    level=2,
    entry="Denoiser",
    statement=(
        "A small MLP that predicts the noise: concatenate the 2-D point with "
        "its time embedding and map through Linear-SiLU-Linear-SiLU-Linear to "
        "2 outputs. The checks are behavioural, not name-based: the output "
        "must genuinely depend on BOTH inputs (a denoiser that ignores t "
        "predicts one average noise for all fifty noise levels and cannot "
        "learn the schedule), must be nonlinear in x, and every parameter "
        "must receive gradient."
    ),
    shapes="__init__(tdim=32, hidden=96) · forward(x (B,2), temb (B,tdim)) -> (B,2)",
    stub=("class Denoiser(nn.Module):\n"
          "    def __init__(self, tdim=32, hidden=96):\n"
          "        super().__init__()\n"
          "        # Linear(2+tdim, hidden), SiLU, Linear(hidden, hidden), SiLU,\n"
          "        # Linear(hidden, 2)\n"
          "\n"
          "    def forward(self, x, temb):\n"
          "        pass\n"),
    hints=[
        "torch.cat([x, temb], dim=-1) is the network input, width 2 + tdim.",
        "nn.Sequential keeps it to three lines.",
        "SiLU (not ReLU) matches the reference stack the later steps train.",
    ],
    solution=(
        "class Denoiser(nn.Module):\n"
        "    def __init__(self, tdim=32, hidden=96):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(\n"
        "            nn.Linear(2 + tdim, hidden), nn.SiLU(),\n"
        "            nn.Linear(hidden, hidden), nn.SiLU(),\n"
        "            nn.Linear(hidden, 2))\n"
        "\n"
        "    def forward(self, x, temb):\n"
        "        return self.net(torch.cat([x, temb], dim=-1))\n"
    ),
    frameworks=["torch"],
    traps=[
        "Ignoring the time embedding — the model then predicts one noise for "
        "every noise level and the sampler cannot work.",
        "No nonlinearity, which collapses the whole network to a single "
        "linear map.",
        "Adding the embedding to x instead of concatenating: the shapes only "
        "coincide when tdim == 2.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    m = fn(32, 96)
    x = torch.randn(5, 2)
    te = _time_emb(torch.randint(0, _T, (5,)), 32)
    check("output shape", lambda: shape(m(x, te)) == (5, 2))
    check("output depends on x",
          lambda: not close(m(x, te), m(x + 1.0, te), 1e-4))
    check("output depends on the time embedding",
          lambda: not close(m(x, te), m(x, _time_emb(torch.zeros(5, dtype=torch.long), 32)), 1e-4))
    def nonlinear_in_x():
        out = m(torch.zeros(1, 2), te[:1]) + m(2 * x[:1], te[:1]) - 2 * m(x[:1], te[:1])
        return float(out.abs().max()) > 1e-4
    check("nonlinear in x (a single Linear cannot pass)", nonlinear_in_x)
    def all_params_get_gradient():
        m2 = fn(32, 96)
        m2(x, te).sum().backward()
        return all(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in m2.parameters())
    check("every parameter receives gradient", all_params_get_gradient)
    check("deterministic", lambda: close(m(x, te), m(x, te)))
''',
),

task(
    id="ddpm-loss",
    title="Step 5 · The noise-prediction loss",
    book=BOOK, chapter=CH, section="Step 5 · Loss",
    level=2,
    entry="ddpm_loss",
    statement=(
        "The whole training objective in four lines: noise x_0 to x_t with the "
        "given eps and per-sample t, embed t, and return the MSE between the "
        "model's prediction and THE NOISE. Predicting eps, not x_0, is the "
        "parameterisation everything downstream assumes — regress on x_0 here "
        "and the sampler's arithmetic in step 7 is wrong for your model. An "
        "oracle that returns the true eps must score exactly zero."
    ),
    shapes="model · x0 (B,2) · sched dict · t (B,) · eps (B,2)  ->  scalar",
    stub=("def ddpm_loss(model, x0, sched, t, eps):\n"
          "    # MSE( model(x_t, temb), eps )\n    pass\n"),
    hints=[
        "x_t comes from the step-2 closed form (available as _q_sample).",
        "The time embedding is _time_emb(t).",
        "The target is eps — the noise that was mixed in, nothing else.",
    ],
    solution=(
        "def ddpm_loss(model, x0, sched, t, eps):\n"
        "    x_t = _q_sample(x0, t, sched['abar'], eps)\n"
        "    return ((model(x_t, _time_emb(t)) - eps) ** 2).mean()\n"
    ),
    frameworks=["torch"],
    traps=[
        "Regressing on x0 instead of eps — trains fine, and step 7's sampler "
        "formulas no longer describe your model.",
        "Sampling fresh noise inside the loss instead of using the eps that "
        "built x_t, which makes the target independent of the input.",
        "Mean over the batch but sum over dimensions (or vice versa) — the "
        "thresholds downstream assume a plain mean.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    sched = _schedule()
    B = 8
    x0 = torch.randn(B, 2)
    t = torch.randint(0, _T, (B,))
    eps = torch.randn(B, 2)

    class Oracle(nn.Module):
        def forward(self, x, temb):
            return eps
    check("an oracle returning the true noise scores exactly zero",
          lambda: abs(float(fn(Oracle(), x0, sched, t, eps))) < 1e-10)

    class Passthrough(nn.Module):
        def forward(self, x, temb):
            return x
    def matches_manual():
        x_t = _q_sample(x0, t, sched["abar"], eps)
        want = ((x_t - eps) ** 2).mean()
        return close(fn(Passthrough(), x0, sched, t, eps), want, 1e-6)
    check("composes q_sample + embedding + MSE exactly", matches_manual)

    class Zero(nn.Module):
        def forward(self, x, temb):
            return torch.zeros_like(x)
    check("the predict-zero baseline scores about 1.0",
          lambda: abs(float(fn(Zero(), x0, sched, t, eps)) -
                      float((eps ** 2).mean())) < 1e-6)

    def grads_flow():
        m = _Denoiser()
        fn(m, x0, sched, t, eps).backward()
        return all(p.grad is not None for p in m.parameters())
    check("gradients reach the model", grads_flow)
    check("returns a scalar", lambda: fn(_Denoiser(), x0, sched, t, eps).ndim == 0)
''',
),

task(
    id="ddpm-train",
    title="Step 6 · Prove it learns",
    book=BOOK, chapter=CH, section="Step 6 · Training",
    level=3,
    entry="train_ddpm",
    statement=(
        "The training loop: per step, draw a data batch, per-sample t, fresh "
        "noise, compute the noise-prediction loss, and take an AdamW step. "
        "Return the losses. The thresholds are calibrated against the "
        "reference on this exact data: it starts near 0.55, and 2500 steps "
        "land it near 0.23 — comfortably below the 1.0 a model that predicts "
        "zero scores, which is the baseline your loop must beat to prove "
        "anything trained at all."
    ),
    shapes=("model · data_fn(n)->(n,2) · sched · steps, lr, batch"
            "  ->  list of float losses"),
    stub=("def train_ddpm(model, data_fn, sched, steps=2500, lr=2e-3, batch=256):\n"
          "    # AdamW; per step: batch, per-sample t, fresh eps, loss, step\n    pass\n"),
    hints=[
        "One optimiser built OUTSIDE the loop.",
        "t = torch.randint(0, T, (batch,)) and eps = torch.randn — fresh every "
        "step.",
        "zero_grad before backward; append float(loss) per step.",
    ],
    solution=(
        "def train_ddpm(model, data_fn, sched, steps=2500, lr=2e-3, batch=256):\n"
        "    opt = torch.optim.AdamW(model.parameters(), lr=lr)\n"
        "    losses = []\n"
        "    for _ in range(steps):\n"
        "        x0 = data_fn(batch)\n"
        "        t = torch.randint(0, len(sched['betas']), (batch,))\n"
        "        eps = torch.randn_like(x0)\n"
        "        loss = _loss(model, x0, sched, t, eps)\n"
        "        opt.zero_grad()\n"
        "        loss.backward()\n"
        "        opt.step()\n"
        "        losses.append(float(loss.detach()))\n"
        "    return losses\n"
    ),
    frameworks=["torch"],
    traps=[
        "Missing zero_grad, so gradients accumulate and the loss curve blows "
        "up instead of falling.",
        "Rebuilding the optimiser inside the loop, which throws Adam's "
        "moments away every step and barely learns.",
        "Sampling one t for the whole batch, which starves most noise levels "
        "and stalls well above the threshold.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    sched = _schedule()
    m = _Denoiser()
    losses = fn(m, _two_gaussians, sched, 2500, 2e-3, 256)
    check("one loss per step", lambda: len(losses) == 2500)
    check("all finite", lambda: all(v == v and abs(v) < 1e4 for v in losses))
    first = sum(losses[:50]) / 50
    last = sum(losses[-200:]) / 200
    check("starts in the untrained band (calibrated ~0.55)",
          lambda: 0.3 < first < 1.6)
    check("converges below 0.35 (reference reaches ~0.23)", lambda: last < 0.35)
    check("beats the predict-zero baseline decisively", lambda: last < 0.5)
    check("loss falls by at least a third", lambda: last < 0.67 * first)
    def params_moved():
        m2 = _Denoiser()
        before = [p.detach().clone() for p in m2.parameters()]
        fn(m2, _two_gaussians, sched, 5, 2e-3, 64)
        return any(not close(a, b, 1e-9) for a, b in zip(before, m2.parameters()))
    check("parameters actually change", params_moved)
''',
),

task(
    id="ddpm-sample",
    title="Step 7 · The sampling loop",
    book=BOOK, chapter=CH, section="Step 7 · Sampling",
    level=3,
    entry="ddpm_sample",
    statement=(
        "Ancestral sampling, T steps from pure noise: at each t, ask the "
        "denoiser for eps_hat, form the posterior mean, and add noise scaled "
        "by the POSTERIOR variance btilde — none at t = 0. The anchor is "
        "analytic: for Gaussian data the optimal denoiser has a closed form, "
        "and run through a correct sampler it must reproduce that Gaussian's "
        "mean and spread. A sampler that starts from zeros, uses beta for "
        "btilde, or keeps noising at t = 0 lands measurably off that target."
    ),
    shapes="denoise(x (n,2), t int) -> (n,2) · sched · n int  ->  (n, 2)",
    stub=("def ddpm_sample(denoise, sched, n):\n"
          "    # x ~ randn; for t = T-1 .. 0: posterior mean, + sqrt(btilde) z\n    pass\n"),
    hints=[
        "Start from torch.randn(n, 2) — the prior the forward process ends at.",
        "mu = (x - beta_t/sqrt(1-abar_t) * eps_hat) / sqrt(alpha_t).",
        "btilde_t = (1 - abar_{t-1})/(1 - abar_t) * beta_t; abar_{-1} is 1, "
        "and t = 0 adds no noise at all.",
    ],
    solution=(
        "def ddpm_sample(denoise, sched, n):\n"
        "    betas, alphas, abar = sched['betas'], sched['alphas'], sched['abar']\n"
        "    T = len(betas)\n"
        "    x = torch.randn(n, 2)\n"
        "    for t in range(T - 1, -1, -1):\n"
        "        with torch.no_grad():\n"
        "            eps_hat = denoise(x, t)\n"
        "        mu = (x - betas[t] / torch.sqrt(1 - abar[t]) * eps_hat) / torch.sqrt(alphas[t])\n"
        "        if t > 0:\n"
        "            btilde = (1 - abar[t - 1]) / (1 - abar[t]) * betas[t]\n"
        "            x = mu + torch.sqrt(btilde) * torch.randn_like(x)\n"
        "        else:\n"
        "            x = mu\n"
        "    return x\n"
    ),
    frameworks=["torch"],
    traps=[
        "Starting the chain from zeros instead of the Gaussian prior — the "
        "sample spread collapses below the target's.",
        "Noise scaled by beta_t instead of the posterior btilde_t, which "
        "over-noises the low-t steps where the two differ most.",
        "Adding noise at t = 0, so the final samples never sharpen onto the "
        "distribution.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    sched = _schedule()
    oracle = _oracle_denoise(sched, m=1.5, s=0.6)
    xs = fn(oracle, sched, 4000)
    check("output shape", lambda: shape(xs) == (4000, 2))
    check("all finite", lambda: bool(torch.isfinite(xs).all()))
    # calibrated: reference sampler measures mean ~1.49-1.52, std ~0.54
    check("oracle denoiser reproduces the target mean (1.5 ± 0.12)",
          lambda: bool(((xs.mean(0) - 1.5).abs() < 0.12).all()))
    check("oracle denoiser reproduces the target spread (0.6 ± 0.12)",
          lambda: bool(((xs.std(0) - 0.6).abs() < 0.12).all()))
    def zero_denoiser_variance_recursion():
        # with eps_hat = 0 the chain is linear-Gaussian, so its variance obeys
        # an exact recursion computable independently of the implementation:
        # v <- v / alpha_t + btilde_t, from v = 1 at the prior.
        v = 1.0
        for t in range(_T - 1, -1, -1):
            v = v / float(sched["alphas"][t])
            if t > 0:
                v += float((1 - sched["abar"][t - 1]) / (1 - sched["abar"][t])
                           * sched["betas"][t])
        z = fn(lambda x, t: torch.zeros_like(x), sched, 4000)
        want = math.sqrt(v)
        return abs(float(z.std()) - want) / want < 0.15 and abs(float(z.mean())) < 0.1 * want
    check("zero-denoiser spread matches the exact variance recursion",
          zero_denoiser_variance_recursion)
    def point_mass_stays_sharp():
        # a near-point-mass target isolates the low-t noise scale, which is
        # where beta and the posterior btilde differ most: the reference
        # measures std ~0.010 here, a beta-variance sampler ~0.058
        tight = _oracle_denoise(sched, m=0.0, s=0.02)
        z = fn(tight, sched, 4000)
        return float(z.std()) < 0.03
    check("a near-point-mass target stays sharp (btilde, not beta)",
          point_mass_stays_sharp)
''',
),

task(
    id="ddpm-pipeline",
    title="Step 8 · Learn two modes, end to end",
    book=BOOK, chapter=CH, section="Step 8 · The pipeline",
    level=3,
    entry="two_mode_pipeline",
    statement=(
        "Put it together: build the schedule, train a fresh denoiser on the "
        "two-Gaussian data (modes at x = -2 and x = +2), and sample 2000 "
        "points. Return the samples and losses. The final exam is the "
        "distribution itself — both modes present in fair proportion, mode "
        "distance right, no mass parked between them — because the classic "
        "end-to-end failure is a sampler that collapses onto the data mean, "
        "which every per-step check can miss."
    ),
    shapes="train_steps int  ->  dict 'samples' (2000, 2), 'losses' list",
    stub=("def two_mode_pipeline(train_steps=2500):\n"
          "    # schedule -> train _Denoiser on _two_gaussians -> sample 2000\n    pass\n"),
    hints=[
        "The reference stack is available: _schedule, _Denoiser, "
        "_two_gaussians, _loss, and your own or the reference train/sample.",
        "Sampling needs a denoise(x, t) callable — wrap the trained model "
        "with its time embedding: lambda x, t: model(x, _time_emb(...)).",
        "2500 steps at lr 2e-3, batch 256 is the calibrated recipe.",
    ],
    solution=(
        "def two_mode_pipeline(train_steps=2500):\n"
        "    sched = _schedule()\n"
        "    model = _Denoiser()\n"
        "    losses = _train(model, _two_gaussians, sched, train_steps)\n"
        "    def denoise(x, t):\n"
        "        return model(x, _time_emb(torch.full((x.shape[0],), t)))\n"
        "    return {'samples': _sample(denoise, sched, 2000), 'losses': losses}\n"
    ),
    frameworks=["torch"],
    traps=[
        "Sampling from the untrained model — the output is a blob with no "
        "modes, and only the distribution checks see it.",
        "Forgetting the time embedding in the sampling wrapper, so the "
        "denoiser sees garbage for t.",
        "Training on one fixed batch instead of fresh draws, which memorises "
        "256 points instead of the distribution.",
    ],
    extra=REF,
    tests='''
def checks(fn, check):
    torch.manual_seed(0)
    out = fn(2500)
    xs = out["samples"]
    check("2000 samples of dimension 2", lambda: shape(xs) == (2000, 2))
    check("training converged below 0.35",
          lambda: sum(out["losses"][-200:]) / 200 < 0.35)
    left = float((xs[:, 0] < -1).float().mean())
    right = float((xs[:, 0] > 1).float().mean())
    check("both modes are populated (>= 25% each; reference: ~50/50)",
          lambda: left > 0.25 and right > 0.25)
    check("little mass between the modes (a mean-collapse fails here)",
          lambda: float(((xs[:, 0].abs()) < 1).float().mean()) < 0.2)
    check("mode distance is right (|x| mean ~1.9, in [1.5, 2.4])",
          lambda: 1.5 < float(xs[:, 0].abs().mean()) < 2.4)
    check("the y-coordinate stays tight (std < 0.6; reference ~0.3)",
          lambda: float(xs[:, 1].std()) < 0.6)
''',
),

]
