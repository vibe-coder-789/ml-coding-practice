"""Run every reference solution against its own checks, in every framework it
claims to support. A bank that fails this must not ship."""
import sys
import runner
import tasks

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    fails, total = [], 0
    for t in tasks.TASKS:
        if only and only not in t["id"]:
            continue
        for fw in t["frameworks"]:
            total += 1
            sol = tasks.reference(t, fw)
            if not sol:
                fails.append((t["id"], fw, "no reference solution"))
                print(f"  MISSING  {t['id']:18s} {fw}")
                continue
            r = runner.run(t["id"], sol, fw)
            if r.get("error"):
                fails.append((t["id"], fw, r["error"]))
                print(f"  ERROR    {t['id']:18s} {fw:6s} {r['error'][:80]}")
                if r.get("stderr"):
                    print("           " + r["stderr"].strip().splitlines()[-1][:100])
                continue
            bad = [c for c in r["checks"] if not c["ok"]]
            mark = "ok      " if r["accepted"] else "FAIL    "
            print(f"  {mark} {t['id']:18s} {fw:6s} {r['passed']}/{r['total']}")
            for c in bad:
                fails.append((t["id"], fw, c["name"]))
                print(f"           - {c['name']}  {c['error'][:90]}")
    print(f"\n{total - len(set((f[0], f[1]) for f in fails))}/{total} "
          f"reference solutions pass their own checks")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
