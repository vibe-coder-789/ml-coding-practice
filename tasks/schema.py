"""Task schema and the harness every submission is executed against.

A task is a plain dict built by `task(...)`. Fields:

  id          stable slug — used in URLs and downloaded filenames
  title       short imperative name
  book        which volume the concept comes from — "llm-math" or "ml-basics"
  chapter     the chapter within that book
  section     the specific section it formalises
  level       1 warm-up · 2 core · 3 hard
  entry       the function name the candidate must define
  statement   what to implement, in prose
  shapes      the shape / type contract, one line
  stub        starting code placed in the editor
  hints       progressive hints, revealed one at a time
  solution    reference implementation in PyTorch
  solution_np optional reference implementation in NumPy
  traps       what an interviewer actually probes for
  frameworks  which backends the checks accept — default both. A task whose
              checks call .backward() is torch-only, because autograd has no
              NumPy equivalent.
  extra       optional helper source injected before the candidate's code
  tests       source defining `checks(fn, check)`; each `check(name, thunk)`
              records one pass/fail, and an exception inside a thunk is caught
              and reported as that check failing

FRAMEWORKS
----------
Checks are written once, against PyTorch tensors. When the submission is NumPy,
the driver wraps the candidate's function: torch tensors are converted to arrays
on the way in, and whatever comes back — array, scalar, tuple, list or dict — is
converted to torch on the way out. The same checks then apply unchanged, so a
NumPy answer is held to exactly the same standard as a PyTorch one.

Every reference solution is executed against its own checks by
`python3 -m tasks.selftest`, in each framework it claims to support, so the bank
cannot ship a reference that fails its own tests.
"""

PREAMBLE = '''
import json, math, sys
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)


class _Rec:
    """Per-check recorder, so a failure can show input / expected / actual."""
    args = None      # the arguments your function was called with
    ret = None       # what your function returned
    got = None       # the left side of the comparison that failed
    want = None      # the right side of it
    has_ret = False
    has_got = False
    has_want = False

    @classmethod
    def reset(cls):
        cls.args = cls.ret = cls.got = cls.want = None
        cls.has_ret = cls.has_got = cls.has_want = False


def _fmt(v, budget=260):
    """Compact, readable rendering of a value for the results panel."""
    try:
        if isinstance(v, torch.Tensor):
            body = np.array2string(v.detach().cpu().numpy(), precision=4,
                                   threshold=12, edgeitems=3, suppress_small=True)
            head = "tensor" + str(tuple(v.shape))
            return (head + " " + " ".join(body.split()))[:budget]
        if isinstance(v, np.ndarray):
            body = np.array2string(v, precision=4, threshold=12, edgeitems=3,
                                   suppress_small=True)
            return ("array" + str(v.shape) + " " + " ".join(body.split()))[:budget]
        if isinstance(v, (np.floating, np.integer)):
            return repr(v.item())[:budget]
        if isinstance(v, float):
            return f"{v:.6g}"
        if isinstance(v, dict):
            return ("{" + ", ".join(f"{k}: {_fmt(x, 90)}" for k, x in v.items())
                    + "}")[:budget]
        if isinstance(v, (list, tuple)):
            inner = ", ".join(_fmt(x, 90) for x in v[:6])
            more = ", …" if len(v) > 6 else ""
            return (("(" if isinstance(v, tuple) else "[") + inner + more
                    + (")" if isinstance(v, tuple) else "]"))[:budget]
        return repr(v)[:budget]
    except Exception:
        return "<unprintable>"


def _fmt_args(args, kw):
    parts = [_fmt(a, 110) for a in args]
    parts += [f"{k}={_fmt(v, 110)}" for k, v in kw.items()]
    return ", ".join(parts)[:320]


def close(a, b, tol=1e-5):
    """Tolerant comparison: accepts tensors, arrays, scalars, lists.

    Records both sides so a failing check can show expected vs actual without
    the task having to spell them out.
    """
    if not _Rec.has_want:                 # keep the first comparison of a check
        _Rec.got, _Rec.want = a, b
        _Rec.has_got = _Rec.has_want = True
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    return bool(np.allclose(np.asarray(a, dtype=np.float64),
                            np.asarray(b, dtype=np.float64),
                            atol=tol, rtol=tol))


def shape(x):
    return tuple(np.asarray(x).shape) if not isinstance(x, torch.Tensor) else tuple(x.shape)


def _to_torch(v):
    if isinstance(v, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(v))
    if isinstance(v, np.generic):
        return torch.tensor(v.item())
    if isinstance(v, tuple):
        return tuple(_to_torch(x) for x in v)
    if isinstance(v, list):
        return [_to_torch(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_torch(x) for k, x in v.items()}
    return v


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, tuple):
        return tuple(_to_numpy(x) for x in v)
    if isinstance(v, list):
        return [_to_numpy(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_numpy(x) for k, x in v.items()}
    return v


def _adapt(fn, framework):
    """Record every call, and let a NumPy submission face torch-based checks."""

    def wrapped(*args, **kw):
        if _Rec.args is None:
            _Rec.args = _fmt_args(args, kw)
        if framework == "numpy":
            out = _to_torch(fn(*tuple(_to_numpy(a) for a in args),
                               **{k: _to_numpy(v) for k, v in kw.items()}))
        else:
            out = fn(*args, **kw)
        if not _Rec.has_ret:
            _Rec.ret, _Rec.has_ret = out, True
        return out
    return wrapped
'''

DRIVER = '''
_results = []


def _check(name, thunk):
    _Rec.reset()
    try:
        ok = bool(thunk())
        err = ""
    except Exception as exc:
        ok, err = False, f"{type(exc).__name__}: {exc}"[:300]
    row = {"name": name, "ok": ok, "error": err}
    import os as _os
    if not ok or _os.environ.get("PRACTICE_CAPTURE"):
        if _Rec.args is not None:
            row["given"] = _Rec.args
        got = _fmt(_Rec.got) if _Rec.has_got else None
        ret = _fmt(_Rec.ret) if _Rec.has_ret else None
        if _Rec.has_want:
            row["want"] = _fmt(_Rec.want)
        if got is not None:
            row["got"] = got
        # when the check compared something DERIVED from your output (a softmax,
        # a shape, a norm), the raw return value is the more useful thing to see
        if ret is not None and ret != got:
            row["ret"] = ret
    _results.append(row)


try:
    _raw = globals()[%(entry)r]
except KeyError:
    print("__RESULTS__" + json.dumps({
        "fatal": "no function named %(entry)r was defined"}))
    sys.exit(0)

torch.manual_seed(0)
checks(_adapt(_raw, %(framework)r), _check)
print("__RESULTS__" + json.dumps({"checks": _results}))
'''

IMPORTS = {
    "torch": "import torch\nimport torch.nn.functional as F\nimport math\n\n",
    "numpy": "import numpy as np\nimport math\n\n",
}


def task(**kw):
    required = ("id", "title", "chapter", "section", "level", "entry",
                "statement", "shapes", "stub", "hints", "solution", "traps",
                "tests")
    kw.setdefault("book", "llm-math")
    missing = [k for k in required if k not in kw]
    if missing:
        raise ValueError(f"task {kw.get('id')!r} missing {missing}")
    kw.setdefault("extra", "")
    kw.setdefault("frameworks", ["torch", "numpy"])
    kw.setdefault("solution_np", "")
    kw.setdefault("banned", [])
    if "numpy" in kw["frameworks"] and not kw["solution_np"]:
        # no NumPy reference written yet — do not advertise a backend the bank
        # cannot self-test
        kw["frameworks"] = [f for f in kw["frameworks"] if f != "numpy"]
    return kw
