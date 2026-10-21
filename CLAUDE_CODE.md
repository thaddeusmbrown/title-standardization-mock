# How Claude Code is set up for this repository

The README says that Claude Code ran the build of this mock end to end. This note says what
this repository carries for Claude Code and how the build was driven. It describes the setup
as of 2026-10-21.

## In short

- **One placement rule, the owner's own.** Facts go in `CLAUDE.md`, procedures in skills,
  guidance for one part of the tree in `.claude/rules/`, and guarantees in hooks and
  `.claude/check.sh`. A fact or a procedure is advice to the model. A hook or a check is
  enforced.
- **This repository:** a `CLAUDE.md` of facts, one rules file, a check script that the
  Stop-hook gate runs, a merge gate, two project hooks and five skills, all under version
  control.
- **The build:** planned with the owner, every open question settled in
  `docs/build-decisions.md`, then built as multi-agent workflows behind one merge gate.
- **The owner's own setup** for every project (settings, hooks, skills and standing
  instructions) lives in a private configuration repository and is not described here.

## 1. This repository: the project level

A clone carries its own configuration, placed by the same rule.

| File | What it holds |
|---|---|
| `CLAUDE.md` | Facts every session loads: the gate commands, what is never run without the owner's word (the paid recording), what is generated and never edited by hand, the hygiene rules, and the branch and commit conventions. |
| `.claude/rules/package.md` | The rules for the package, loaded only when Claude Code reads a file they concern. |
| `.claude/merge-gate.sh` | What `/land` and `/promote` run on a branch and on its merge: `make check-all`, the gate and then the suite on Python 3.13 and 3.14. |
| `.claude/check.sh` | What "green" means for the Stop-hook gate: `make check`, the gate CI runs. It refuses first while another `make` or `pytest` run is going in the checkout, and while a file that is not ignored is untracked, because the hygiene tests scan only the files git tracks. |
| `.claude/settings.json` | Deny rules for the plain spellings of the pushes and `gh` calls that the PR flow of owner decision 11 never makes: another remote, main, a force, a delete, a tag, and every `gh` command but the flow's. No pattern can name everything but the flow's own commands, so the rules are a floor and `no-push.py` decides. Two hooks: `no-push.py`, below, and `.claude/hooks/ruff-format.sh`, which formats a Python file after each edit with the checkout's locked ruff. |
| `.claude/hooks/no-push.py` | Reads each shell command and refuses every push and every `gh` call except the PR flow's: the push of `build/<topic>` to the `claude` remote and the draft pull request into main that the user-level `/pr-open` makes, and the read-only `gh` calls of `/pr-watch` and `/pr-setup`, each only word for word and alone on the command line. It finds the spellings the deny rules miss, such as `git -C <dir> push`. `tests/unit/test_no_push_hook.py` holds the spellings, each command of the flow and near misses of each, and checks that no deny rule stops the flow. |
| `.claude/skills/` | Five skills, below. Merging is left to the user-level `/land` and `/promote`. |

| Skill | What it does |
|---|---|
| `code-review` | The review link of `/ship`, in place of the bundled one: a fresh subagent reports every finding against the design, the decisions record, the package rules and `CLAUDE.md`, then a second pass ranks and filters them. |
| `verify` | The recipe `/ship` follows: the gate, then the program run in the mode the change touches, then what no test checks. |
| `run-title-standardization-mock` | How to run and drive the program. Its `smoke.sh` runs every mode that costs nothing, each into a folder outside the repository, and checks exit statuses and output. |
| `refresh-frozen-output` | What to do when a change alters what the default run prints. |
| `verify-prose-figures` | Checks the run's figures that prose repeats and no test reads. |

`.gitignore` keeps only the per-user files out: `.claude/settings.local.json` and
`.claude/worktrees/`.

## 2. How the build was run

**Set-up, with the owner present.** The first prompt asked for a plan and for every open
question before the build. Agents each read the design for a different concern and raised
questions. These were merged and sorted: most could be settled from the design or by a safe
default, and the rest went to the owner with options, among them how to merge, whether to
make a live run and how the mock produces leftover titles. The answers went into
`docs/build-decisions.md` on the first branch, before any component was built.

**The build.** The work was cut into stages and run as workflows. After the shared scaffold
branch, one workflow script ran one pipeline for each branch:

1. A builder agent works in a git worktree of its own.
2. Two reviewers read the result at the same time, one for conformance to the design and one
   for the tests and correctness.
3. A fixer agent acts on what they found, unless it is only small things.

**The merge gate, one script for every branch.** Rebase on main. Run `make check-all` on the
branch. Merge with `--no-ff` in a separate integration worktree. Run `make check-all` on the
merge result. Only then move main forward. That is more than Q34 and rule 7 of the decisions
record ask, which is `make check` at both points. The user-level `/land` and `/promote` now
run the same steps, with `.claude/merge-gate.sh` as the gate.

**The configuration during the build.** It lived in the session, not in the repository:
there was no `CLAUDE.md`, and git ignored `.claude/`. A per-user permission file allowed the
routine commands and denied `git push` and `gh`, and a pre-push hook in `.git/hooks` refused
every push for the whole run. The lead session and the subagents ran on Claude Opus 5.5. The
owner made the API key for the one live run outside the session, and it never entered the
conversation.

**The review, with the owner back.** The owner decided against a second live recording, so
the failed quotes of the live run stay as a finding (`docs/live-run-observation.md`), and
asked for a short README (owner decision 10). Read-only reviewers then read main, and a last
branch fixed most of what they found. Before the push the owner had the history rewritten
twice: once to replace the merge commits' long hand-off reports with short messages, and once
in a clean-up that also put a commit straight onto main. The owner wrote the provenance
paragraph of the README and pushed.

## 3. After the build

The owner's runbook for a new project was then applied in two sessions. The first began with
Claude Code's `/init` and added `CLAUDE.md`, the rules file, the check script, the shared
settings with the push guard and the first three skills, and moved the design to `DESIGN.md`
at the root, where `/design` looks for it. The second ran `/project-tune`,
`/run-skill-generator` and `/ship`. It added the code-review, prose-figures and run skills,
the check script's refusal of untracked files and the push guard's tests, and the fixes
`/ship` found.

A third session, on 2026-10-18, replaced the project's merge skill. `/init` had made it one
only a person could start, following its own advice for skills with side effects. So every
build stopped at each branch. Branches now land in a build branch through the user-level
`/land`, and main moves only through `/promote`, which the owner starts.

## Check it

```sh
git log --first-parent --oneline main
make check
.claude/skills/run-title-standardization-mock/smoke.sh
```
