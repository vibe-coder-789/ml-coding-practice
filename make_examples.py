#!/usr/bin/env python3
"""Precompute worked examples for every problem.

Runs each reference solution with capture on and keeps the first few passing
checks that recorded both an input and an output. Those become the "Examples"
shown in the problem description — real input/output pairs produced by running
the reference, so they cannot drift from the tests.

    ./.venv/bin/python make_examples.py     # writes examples.json
"""
import json
import sys

import runner
import tasks

MAX_PER_TASK = 3


def main():
    out, missing = {}, []
    for i, t in enumerate(tasks.TASKS, 1):
        fw = t["frameworks"][0]
        sol = tasks.reference(t, fw)
        r = runner.run(t["id"], sol, fw, capture=True)
        if r.get("error"):
            missing.append((t["id"], r["error"][:60]))
            continue
        rows, seen = [], set()
        for c in r["checks"]:
            if not c["ok"] or "given" not in c:
                continue
            # "got" is the left side of whatever the check compared, which is
            # often derived (a softmax, a norm, a shape). An example should show
            # what the function actually returned, so prefer "ret".
            shown = c.get("ret") or c.get("got")   # not `out` — that is the result dict
            if shown is None:
                continue
            key = (c["given"], shown)
            if key in seen:          # several checks often probe the same call
                continue
            seen.add(key)
            rows.append({"name": c["name"], "given": c["given"], "got": shown})
            if len(rows) >= MAX_PER_TASK:
                break
        if rows:
            out[t["id"]] = rows
        else:
            missing.append((t["id"], "no capturable case"))
        sys.stderr.write(f"\r  {i}/{len(tasks.TASKS)} {t['id']:22s}")
    sys.stderr.write("\n")

    json.dump(out, open("examples.json", "w"), indent=1, sort_keys=True)
    print(f"  examples.json: {len(out)}/{len(tasks.TASKS)} problems have examples")
    for tid, why in missing:
        print(f"    none for {tid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
