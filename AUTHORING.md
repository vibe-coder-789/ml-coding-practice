# Adding a problem

The contract for new tasks — written for a person, and deliberately precise
enough to hand to an agent when task creation goes dynamic. Every rule below
exists because its violation shipped once and had to be found by audit.

## The fields

A task is one `task(...)` call in a chapter module under `tasks/` (schema and
field docs in `tasks/schema.py`). New volume → new module + entry in
`tasks/__init__.py` (`MODULES`, `BOOK_TITLES`). Slugs are permanent: they key
progress, examples, and URLs.

Volumes are not limited to their source books. A topic neither book covers goes
into whichever existing chapter fits it best, with "(off-book)" appended to its
section line — do not mint a separate catch-all volume for it (one existed
briefly and was folded back in).

## Projects

A project is a multi-step build (its book id listed in `PROJECTS` in
`tasks/__init__.py`), rendered in the sidebar's Projects section. The rules that
made the GPT-2 project work, distilled for the next one:

- Steps are ordered and compose, but every step is CHECKED IN ISOLATION against
  a reference stack injected via `extra` — a wrong step 2 must not block step 8.
- Title steps "Step N · ..." and keep one chapter per project.
- The final steps must do real work, pinned to exact values, not directions:
  the GPT-2 training step's loss must equal the correctly shifted cross-entropy
  to 1e-4, and the overfit run must reach a loss the model can only reach by
  actually learning ("loss goes down" also holds for copying bugs).
- Weight-copying checks must state their naming contract in the problem and
  load with strict=False, or correct solutions get rejected over an attribute
  name.

- `statement` must pin a single contract. If two readings can both pass, the
  problem is broken even when the solution is right — `temperature` shipped
  accepting both "return logits" and "return probabilities" until a user hit it.
  State the return convention explicitly.
- `solution` (and `solution_np` where NumPy is meaningful) — a task only
  advertises a backend it has a reference for; `selftest.py` runs both.
- `traps` are claims, and claims get tested (see below).
- `banned`: if the statement forbids an oracle ("do not call torch.softmax"),
  add the substrings to `BANNED` in `tasks/__init__.py`, or the prohibition is
  decoration. The reference must not trip its own ban.

## The quality bar for checks

Anchor correctness outside the task. In order of preference:

1. **An independent oracle** — `F.cross_entropy`, `torch.optim.AdamW` run
   step-for-step, `torch.kron`, autograd for a hand-derived gradient, brute
   force (HMM forward vs full path enumeration), a statistical test
   (speculative decoding vs 20k empirical samples), or a literature value
   (Chinchilla reproduces the paper's 5.76e23 → ~70B/1.4T point).
2. **Hand-computed fixed cases** with values worked out on paper.
3. A check that merely restates the solution's formula proves nothing — it is
   self-consistency, acceptable only when 1–2 are impossible, and then the
   formula needs a written review.

Five failure classes the first audit actually found — test against each:

- **Self-referential checks.** `rope` accepted the identity function because its
  property check used `fn` on both sides. At least one check must compare
  against values the submission cannot influence.
- **Degenerate test data.** PPO's sum-vs-mean bug survived because the test
  advantages summed to zero; RLOO's dropped-`keepdim` survived because every
  check used batch size 1. Use data where the bug changes the answer; vary the
  batch dimension.
- **Happy-path-only edge cases.** Cross-entropy's stability check put the large
  logit on the target class — the one place the naive version doesn't overflow.
  Put the stress where the wrong implementation breaks, not where it survives.
- **Proxy objectives.** "Loss goes down" accepted the wrong-way shift, because
  copying a periodic corpus also reaches zero loss. Pin exact values
  (`losses[0]` must equal the correctly shifted cross-entropy) rather than
  directions.
- **Untested claims.** Every `traps` entry is a testable assertion. If no check
  would reject an implementation of the trap, either add the check or delete
  the trap.

Checks may not falsely reject: naming used by weight-copying tests must be
stated in the problem (`strict=False` on `load_state_dict`), and anything a
correct-but-differently-written solution might do must still pass.

## The gates

```bash
./.venv/bin/python selftest.py <id>     # reference passes its own checks
./.venv/bin/python audit.py <id>        # identity + traps + mutants all FAIL
./.venv/bin/python audit.py --coverage  # tasks with no written trap
./.venv/bin/python make_examples.py     # refresh worked examples
# restart server.py — it imports tasks at startup
```

A new task ships with **at least one trap implementation** in `TRAPS`
(`audit.py`) — the wrong solution a real person would write, asserted rejected.
Generic mutants are a safety net, not coverage. If a mutant is behaviourally
equivalent (a causal row can never be fully masked, so `-inf` ≡ `finfo.min`
there), whitelist it in `EQUIVALENT` with the reasoning in the comment.

## Dynamic task creation (planned)

When tasks are generated from a user-supplied concept, the generator must emit
the full contract — statement with a pinned return convention, reference
solution(s), checks meeting the bar above, and at least one trap
implementation — and the pipeline gates on `selftest.py <id>` and
`audit.py <id>` both passing before the task enters the bank. A generated task
that cannot state an oracle or a hand-computed anchor for itself should be
rejected, not shipped: an unanchored check pair is exactly how a generator's
own misconception becomes an "Accepted" wrong answer. Until then, the paste-in
path is: append the `task(...)`, add the trap, run the gates.
