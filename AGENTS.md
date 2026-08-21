# Agent instructions

Read `STATUS.md` before substantial work and update it before every handoff or pull request.

## STATUS.md policy

`STATUS.md` is a compact current-state ledger, not an append-only journal.

- Keep it to a few hundred lines at most; target under 200 and hard-cap it at 300 lines.
- Compact or rewrite stale sections instead of appending history indefinitely.
- Record the current milestone, implemented capabilities, validation commands/results, active blockers, cross-repository pins/contracts, and the next concrete work.
- Distinguish verified behavior from code that exists but has not run successfully.
- Remove completed transient tasks once their durable result is represented by code, tests, documentation, commits, or PRs.
- Put historical narrative in Git history, PR descriptions, ADRs, or dedicated design documents rather than growing `STATUS.md`.
- When a change affects another scientific-stack repository, record the exact dependency/contract state needed for reproducibility.

Do not weaken tests, semantic gates, evidence requirements, or ownership boundaries merely to make a branch green.

## Compiler boundaries

- Keep one `.res` source and semantic model. Do not add alternate ASTs or mirror Quantitas types.
- Scientia may depend on Quantitas, Resolvent, and Malleus. Resolvent and Malleus must never
  depend on Scientia.
- Keep meshes, concrete elements/DOFs, global operators, runtime state, and solver policy out of
  this repository.
- Mathematical transformations need explicit provenance and independent validation; removed code
  is available in Git history but is not an oracle.
