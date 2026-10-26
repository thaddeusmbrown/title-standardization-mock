---
name: verify
description: Verify a finished change to this repository end to end, before it is committed or merged. Stages new files, runs the gate, runs the program in the mode the change touches and checks what no test checks. Use when a change is functionally complete, or when asked to verify or confirm that a change works.
argument-hint: "[<files> | <rev> | <from>..<to>] [-- <paths>]"
---

# Verify a change

`make check` already holds most of what "it works" means here: both suite runs, two default
runs compared byte for byte, the frozen table on Python 3.12 and a replay of the committed
recording with no miss. This recipe adds what the gate cannot see. Nothing in it costs money:
never run `record` or `--llm anthropic` to verify.

**The change** is the one the caller names: a list of files, which is how `/ship` passes the
files its commits touch, or `git diff` arguments (a revision, a range, paths after `--`). With
nothing named, it is the branch and the working tree:
`git diff $(git merge-base main HEAD)` and the untracked files. A list of files is read as
their diff in the range the caller used, or in that default when it names none. Steps 3 and 5
read the change, whether it is committed or not. The default is wider than `code-review`'s on
purpose: this recipe gates a merge.

1. **Stage new files first,** each by its explicit path. The hygiene tests list files with
   `git ls-files`, so an untracked file is scanned by nothing.
2. **Run the gate:** `.claude/check.sh` from the repository root. It refuses an untracked file
   or another run in the checkout, then runs `make check`. On a red gate, fix it and run
   again. `make format` applies the ruff fixes.
3. **Run the program in the mode the change touches,** and read what it prints and writes. A
   passing suite is not that observation.
   `.claude/skills/run-title-standardization-mock/smoke.sh <folder outside the repository>`
   runs every free mode and keeps each run under the folder: `default.txt`, then `default/`,
   `replay/` and the rest. For a setting it does not cover, use that skill's "One mode by
   hand". A `FAIL` on replay with "replay miss" means the change altered what a model is
   asked: CLAUDE.md, "Paid calls and the recording", says what follows. A change to the
   architecture page (`titlestd/explorer.py`, `static/app.*`) is read with that skill's "The
   architecture page".
4. **If the printed table changed,** follow the `refresh-frozen-output` skill, then go back to
   step 2.
5. **Check what no test checks:**
   - Figures repeated in prose: the `verify-prose-figures` skill.
   - Names: read the diff once for any real company, client or person (design 2.8).
   - A decision the record did not settle is written in the module's docstring, under
     "Decided here".
   - A change to a shared file (rule 2 of `docs/build-decisions.md`) sits in a commit of its
     own.
6. **Report** what ran and what was seen: the gate's result, the runs of step 3 with the
   lines that show the change, and anything still open. Before a merge the gate is
   `make check-all`, which `/merge-branch` runs.
