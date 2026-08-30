# ML Coding Practice

LeetCode-style problems for the constructs in *The Mathematics of Large Language
Models*. Pick a problem, write the function, run it against real tests — the
server executes your code in a subprocess against PyTorch, so a pass is a pass,
not a self-assessment.

```bash
./.venv/bin/python server.py         # http://127.0.0.1:8000, opens a browser
./.venv/bin/python selftest.py       # every reference solution vs its own checks
./.venv/bin/python selftest.py gpt   # one volume (substring match on the id)
./.venv/bin/python make_examples.py  # regenerate the worked examples
./.venv/bin/python audit.py          # adversarial audit: wrong answers must FAIL
```

If the venv is missing: `python3 -m venv .venv && ./.venv/bin/pip install torch numpy`.

## What's here

141 problems: three drill volumes plus a **Projects** section. The volumes
started as mirrors of the two source books but are not limited to them —
off-book topics (diffusion, LoRA, BPE) live in whichever chapter fits, marked
"(off-book)" in their section line. Projects are multi-step builds whose tasks compose in order: "Build GPT-2,
Step by Step" and "Train a Diffusion Model, Step by Step".

| Volume | n | Covers |
|---|---|---|
| **The Mathematics of LLMs** | 71 | ten chapters: the loss, the transformer (now incl. linear attention and MLA), accounting, optimisation (incl. Kahan summation), scaling laws, alignment and RL (incl. GAE and distillation), inference (incl. beam search, MCTS, and pass@k), infrastructure — incl. local SGD and five distributed drills where sharded must equal unsharded exactly (ring all-reduce with per-rank send counts, Megatron tensor parallelism, an FSDP forward with memory accounting, a GPipe schedule and its bubble, ring-attention shard combination) — data (incl. BPE), LoRA, linear algebra |
| **The Mathematics of ML: The Basics** | 40 | MLE and the Gaussian, Gaussian conditioning, Gram–Schmidt, importance sampling, OLS/ridge, k-fold CV, logistic gradients, naive Bayes, backprop by hand, dropout, PCA, k-means and k-means++, kernels, GPs, GMM E- and M-steps, HMM forward and Viterbi, a Kalman filter, Metropolis–Hastings, Bayesian linear regression, reparameterisation — plus decision stumps, k-NN, AdaBoost, ROC/AUC both ways, Welford, reservoir sampling, gradient checking, the VAE loss, Conv2d/BatchNorm2d/LSTM cells against their torch oracles, and the diffusion forward/reverse steps |
| **Do You Know What These Functions Do?** | 12 | the torch API itself: broadcasting rules, gather and scatter_add, masked softmax, einsum, unfold, cumsum tricks, dtype promotion, where autograd stops, when `.contiguous()` is required |
| **Train a Diffusion Model** *(project)* | 8 | schedule → forward process → time embedding → denoiser → loss → train (2500 real steps in ~3s) → sample → learn a two-mode distribution end to end |
| **Build GPT-2, Step by Step** *(project)* | 10 | embeddings → attention → MLP → block → model → initialisation → shifted loss → training step → overfit one batch → generate |

22 Easy · 68 Medium · 38 Hard. Every problem carries progressive hints, a list of
what the tests actually probe for, and a reference solution behind a disclosure.

### Projects: Train a Diffusion Model

Eight steps to a working DDPM on 2-D toy data, calibrated so real training fits
inside a check (2500 steps ≈ 3s, converging to ~0.23 against a predict-zero
baseline of 1.0). Two analytic anchors do the heavy lifting: for Gaussian data
the OPTIMAL denoiser has a closed form, so the sampling loop is checked by
whether it reproduces a known distribution — including a near-point-mass target
that separates the posterior variance btilde from beta by a factor of six — and
with a zero denoiser the chain is linear-Gaussian, so its spread must match an
exact variance recursion computed independently.

### Projects: Build GPT-2

Projects sit in their own sidebar section, below the drills. Adding one is a
module of ordered steps plus one line in `PROJECTS` (`tasks/__init__.py`).
These ten steps compose: step 4 builds the block step 5 stacks, and step 9 trains
the model step 5 defined. Each is still checked in isolation against a reference
stack, so a wrong answer at step 2 does not block step 8. The last three steps do
real work — the training step must actually move the parameters, step 9 must drive
the loss below 0.3 on a memorisable batch, and step 10 must generate in range and
crop a prompt longer than the context.

Step 6 exists because of a measurement: with PyTorch's default initialisation this
model starts at a loss of **23.2**, where log(vocab) is **4.17**. GPT-2's N(0, 0.02)
brings it to **4.185**. The nineteen extra nats are the network shouting confident
nonsense, and the check requires you to remove them.

## PyTorch or NumPy

Solve in either. Checks are written once against PyTorch tensors; when a
submission is NumPy the driver converts tensors to arrays on the way in and
whatever comes back on the way out, so both are held to the same tests. 90 of the 128 problems accept both. The rest are PyTorch-only because their checks
reach for autograd, `nn.Module`, or a torch-specific API — every one of the ten
GPT-2 steps, plus a handful elsewhere that call `.backward()`.

## Adding problems

The authoring contract — fields, the quality bar for checks, the five failure
classes the audit actually found, and the gates a new task must pass — lives in
**AUTHORING.md**. It is written to be handed to an agent when task creation
goes dynamic: a generated task must ship the same contract (pinned statement,
reference, anchored checks, a trap implementation) and pass `selftest.py <id>`
and `audit.py <id>` before entering the bank.

## Two directions of correctness

`selftest.py` proves every reference solution passes its own checks. `audit.py`
proves the other direction: it attacks every problem with an identity function,
with calls to the oracle the statement forbids, and with the reference solution
carrying one classic bug injected (dropped `keepdim`, wrong variance divisor,
`-inf` masking, the wrong shift, sum for mean, ...). Any attack that comes back
Accepted is a hole in that problem's checks. The audit found ten on first run —
including a `rope` check that accepted the identity function because it compared
`fn` against itself — seven were fixed, and the three survivors are behaviourally
equivalent mutants (a causal row can never be fully masked, so `-inf` and
`finfo.min` cannot be distinguished there).

Problems whose statement forbids an oracle now enforce it: submissions to
`stable-softmax` containing `torch.softmax`, to `power-iteration` containing an
SVD, and so on, are rejected before running (the `BANNED` table in
`tasks/__init__.py`, checked with comments stripped).

## The editor

The code pane is CodeMirror 5, vendored locally under `vendor/` (no CDN, works
offline): Python syntax highlighting themed by the app's own tokens, line
numbers, native undo, auto-indent after `:`, bracket matching and auto-close,
Tab/Shift-Tab block indent, Cmd-/ comment toggle, Cmd-Enter to run. A plain
textarea takes over automatically if the vendor files go missing. An earlier
hand-rolled overlay editor was reverted — this replaces it with a battle-tested
engine instead. `http://127.0.0.1:8000/cmtest.html` runs 13 behaviour
assertions against the live app, ending with an end-to-end: code typed into the
editor is submitted and comes back Accepted.

## Test cases, not just verdicts

A failing check reports the case, not only its name: the input your function was
called with, what was expected, and what you returned.

```
✕ correct on the overflow case
    INPUT     tensor(1, 3) [[1000. 1001. 1002.]]
    EXPECTED  tensor(1, 3) [[0.09 0.2447 0.6652]]
    YOURS     tensor(1, 3) [[nan nan nan]]
```

Nothing in a task spells those out. The harness wraps your function to record
its first call, and `close()` records both sides of the comparison, so every
existing check gained this for free. Property checks that make no comparison —
"no NaN anywhere" — still show the input and your output.

Each problem also opens with worked **Examples**, real input/output pairs
captured by running the reference solution under the same recorder
(`make_examples.py` → `examples.json`). They cannot drift from the tests because
they are produced by them. 77 of 86 problems have them; the rest build their
object outside the checks, so there is nothing to capture.

## Structure

```
server.py     local HTTP server: problem list, /api/run, progress
runner.py     executes one submission in a subprocess, 30s cap
selftest.py   holds every reference solution to its own checks
app.html      the interface
tasks/
  schema.py       task fields, the harness preamble, the NumPy adapter
  ch1_loss.py                chapter 1
  ch2_transformer.py         chapter 2
  ch3_accounting.py          chapter 3
  ch4_optimisation.py        chapter 4
  ch56_scaling_alignment.py  chapters 5 and 6
  ch79_inference_data.py     chapters 7 and 9
  ch8_infra.py               chapter 8
  ch10_linalg.py             chapter 10
  mlb_core.py                ml-basics: probability, regression, networks
  mlb_models.py              ml-basics: PCA, kernels, EM, sequences
  torch_api.py               the torch API volume
  gpt2_project.py            the GPT-2 build, in order
make_examples.py  regenerates examples.json from the reference solutions
examples.json     worked input/output pairs shown in each problem
AUTHORING.md      the contract for adding problems (also the spec for dynamic generation)
audit.py          adversarial audit: identity, oracle cheats, mutants (--coverage lists trap gaps)
vendor/           CodeMirror 5, pinned and served locally
cmtest.html       13 editor assertions against the live app (open via the server)
progress.json     written as you solve; delete it to reset
```

Adding a problem means appending one `task(...)` to a chapter module and running
`selftest.py`. A task only advertises a backend once a reference solution exists
for it, so the bank cannot claim support it has not tested.

## Safety

`runner.py` executes arbitrary Python on this machine — that is the point, and
it is why `server.py` binds `127.0.0.1` and never `0.0.0.0`. Do not expose it to
a network you do not control. Submissions run in a subprocess with a wall-clock
timeout, so an infinite loop costs one process rather than the server.
