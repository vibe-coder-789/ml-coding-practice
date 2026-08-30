"""The problem bank, organised by llm-math chapter."""
from . import (ch1_loss, ch2_transformer, ch3_accounting, ch4_optimisation,
               ch56_scaling_alignment, ch79_inference_data, ch8_infra,
               ch10_linalg, mlb_core, mlb_models, torch_api, gpt2_project,
               additions_llm, additions_mlb, additions2_llm,
               additions2_mlb, beyond, ddpm_project, additions_infra)
from .schema import DRIVER, IMPORTS, PREAMBLE, task  # noqa: F401

MODULES = [ch1_loss, ch2_transformer, ch3_accounting, ch4_optimisation,
           ch56_scaling_alignment, ch79_inference_data, ch8_infra, ch10_linalg,
           mlb_core, mlb_models, torch_api, gpt2_project,
           additions_llm, additions_mlb, additions2_llm, additions2_mlb,
           beyond, ddpm_project, additions_infra]

TASKS = [t for m in MODULES for t in m.TASKS]
BY_ID = {t["id"]: t for t in TASKS}
BOOKS = []
CHAPTERS = []
for t in TASKS:
    if t["book"] not in BOOKS:
        BOOKS.append(t["book"])
    if (t["book"], t["chapter"]) not in CHAPTERS:
        CHAPTERS.append((t["book"], t["chapter"]))

BOOK_TITLES = {
    "llm-math": "The Mathematics of Large Language Models",
    "ml-basics": "The Mathematics of Machine Learning: The Basics",
    "torch-api": "Do You Know What These Functions Do?",
    "build-gpt2": "Build GPT-2, Step by Step",
    "build-ddpm": "Train a Diffusion Model, Step by Step",
}

DIFFICULTY = {1: "Easy", 2: "Medium", 3: "Hard"}

# Books listed here render under a "Projects" section in the sidebar: multi-step
# builds whose tasks compose in order, as opposed to standalone drills. Adding a
# project = a new module of ordered steps + its BOOK_TITLES entry + a line here.
PROJECTS = ["build-gpt2", "build-ddpm"]

# Submissions may not call the oracle their checks compare against, nor anything
# the statement explicitly forbids. Checked as substrings of the submission with
# comments stripped (see runner.py); the reference solutions must stay clean of
# their own bans — selftest covers that by construction.
BANNED = {
    "stable-softmax":   ["torch.softmax", "F.softmax", "log_softmax"],
    "log-softmax":      ["F.log_softmax", "torch.log_softmax"],
    "cross-entropy":    ["F.cross_entropy", "CrossEntropyLoss", "nll_loss"],
    "softmax-jacobian": ["autograd"],
    "sdpa":             ["scaled_dot_product_attention"],
    "causal-mask":      ["scaled_dot_product_attention"],
    "layer-norm":       ["F.layer_norm", "nn.LayerNorm"],
    "rms-norm":         ["F.rms_norm", "nn.RMSNorm"],
    "online-softmax":   ["torch.softmax", "F.softmax"],
    "power-iteration":  ["svd", "matrix_norm", "eig"],
    "kronecker":        ["torch.kron", "np.kron"],
    "pairwise-sqdist":  ["cdist"],
    "gaussian-logpdf":  ["distributions"],
    "kl-gaussians":     ["kl_divergence", "distributions"],
    "mlp-backward":     [".backward(", "autograd"],
    "adamw":            ["torch.optim"],
    "sgd-momentum":     ["torch.optim"],
    "grad-clip":        ["clip_grad_norm"],
    "kahan-summation":  ["float64", ".double()", "np.longdouble"],
}
_unknown = [t for t in BANNED if t not in BY_ID]
if _unknown:
    raise ValueError(f"BANNED names unknown task ids: {_unknown}")
for _tid, _b in BANNED.items():
    BY_ID[_tid]["banned"] = _b

_dupes = [i for i in BY_ID if [t["id"] for t in TASKS].count(i) > 1]
if _dupes:
    raise ValueError(f"duplicate task ids: {_dupes}")


def public(t):
    """The task as the browser sees it — no solution, no test source."""
    return {
        "id": t["id"],
        "title": t["title"],
        "book": t["book"],
        "chapter": t["chapter"],
        "section": t["section"],
        "level": t["level"],
        "difficulty": DIFFICULTY[t["level"]],
        "entry": t["entry"],
        "statement": t["statement"],
        "shapes": t["shapes"],
        "stub": t["stub"],
        "hints": t["hints"],
        "traps": t["traps"],
        "frameworks": t["frameworks"],
        "n_checks": t["tests"].count("check("),
    }


def build_program(t, code, framework):
    """Assemble the exact source that will be executed for a submission."""
    return "".join([
        PREAMBLE,
        "\n", t["extra"], "\n",
        "\n# ---- submission ----\n", code, "\n",
        "\n# ---- checks ----\n", t["tests"], "\n",
        DRIVER % {"entry": t["entry"], "framework": framework},
    ])


def reference(t, framework):
    return t["solution_np"] if framework == "numpy" else t["solution"]
