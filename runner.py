"""Execute a submission against a task's checks, in a separate process.

The submitted code is written to a temp file and run by a fresh interpreter with
a wall-clock timeout, so an infinite loop or a crash costs one subprocess rather
than the server. Results come back over stdout as a single JSON line prefixed
with __RESULTS__, which keeps any print() the candidate wrote from being
mistaken for the result.

This runs arbitrary Python on this machine, by design — it is a local practice
tool and the code being executed is the one the user just typed. The server it
serves is bound to 127.0.0.1 for the same reason. Do not expose either to a
network you do not control.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import tasks

TIMEOUT = 30
MARKER = "__RESULTS__"


def run(task_id, code, framework="torch", capture=False):
    t = tasks.BY_ID.get(task_id)
    if t is None:
        return {"error": f"unknown task {task_id!r}"}
    if framework not in t["frameworks"]:
        return {"error": f"{t['title']} does not accept {framework} submissions "
                         f"(accepts: {', '.join(t['frameworks'])})"}

    stripped = re.sub(r"#[^\n]*", "", code)
    for banned in t.get("banned", []):
        if banned in stripped:
            return {"error": f"this problem asks you to implement it yourself — "
                             f"{banned!r} is off-limits here"}

    program = tasks.build_program(t, code, framework)

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                      prefix=f"submit_{task_id.replace('-', '_')}_")
    try:
        tmp.write(program)
        tmp.close()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
        if capture:
            env["PRACTICE_CAPTURE"] = "1"
        try:
            proc = subprocess.run([sys.executable, tmp.name],
                                  capture_output=True, text=True,
                                  timeout=TIMEOUT, env=env,
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {TIMEOUT}s — an infinite loop, or "
                             f"a shape mistake that made the tensors enormous",
                    "stdout": "", "stderr": ""}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    stdout, result_line = [], None
    for line in proc.stdout.splitlines():
        if line.startswith(MARKER):
            result_line = line[len(MARKER):]
        else:
            stdout.append(line)

    printed = "\n".join(stdout)[-4000:]

    if result_line is None:
        # the program died before reporting — surface the traceback verbatim
        return {"error": "your code raised before any check could run",
                "stdout": printed, "stderr": (proc.stderr or "")[-4000:]}

    payload = json.loads(result_line)
    if "fatal" in payload:
        return {"error": payload["fatal"], "stdout": printed,
                "stderr": (proc.stderr or "")[-4000:]}

    checks = payload["checks"]
    passed = sum(1 for c in checks if c["ok"])
    return {
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "accepted": passed == len(checks) and len(checks) > 0,
        "stdout": printed,
        "stderr": (proc.stderr or "")[-4000:],
    }
