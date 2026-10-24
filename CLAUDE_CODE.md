# How Claude Code is set up for this repository

The README says that Claude Code ran the build of this mock end to end. This note says what
this repository carries for Claude Code and how the build was driven. It describes the setup
as of 2026-10-23.

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
| `.claude/settings.json` | Deny rules, a floor for when the hook does not run, for the plain spellings of what the PR flow of owner decision 11 never does. For `git push`: the command alone, or `git push claude` with no refspec; any option, such as a force, a delete, `--tags` or `--all`; `origin`, a URL or a path as the remote; main as a source or a destination; a refspec starting with `+`, `:`, `HEAD` or `refs/`; any word holding `@`; and a tag as a destination or as `tag <name>`. For `gh`: `gh` or `gh pr` alone or with an option before its command; the command groups the flow never uses, and the other commands of `gh pr`, `gh run` and `gh repo`, each by its name or a prefix of it; in the read groups the flow uses, `-R` or `--repo` in `gh pr view`, `gh pr checks`, `gh run list` and `gh run view`, and `-w` or `--web` in `gh pr view`, `gh pr checks`, `gh run view` and `gh repo view`; the body, comments or reviews of a pull request read with `gh pr view`; a run log read with `gh run view --log`; and in `gh api` a method, a field, an input file, a header or another host, `graphql`, and the pulls, issues and comments endpoints. Every other push or `gh` call is the hook's alone, among them another branch on `claude`, a tag by name, `gh pr view <n>` without `--json`, the options of `gh pr create` (its title is free text, so no pattern can scope them), another repository named as an argument (`gh repo view`, `gh repo set-default`, the paths of `gh api`) and the log endpoints of `gh api`. Each rule is the only one to meet one of the test's plain spellings, so none is dead or covered by another. No rule ends in `:*` while holding another `*`, since Claude Code reads that as a literal prefix. Claude Code's rule check warns about the two rules spelled with `:**` that they match as wildcards with a literal colon, which is what they are for (its text as read from the 2.1.310 binary on 2026-10-23). Two hooks: `no-push.py`, below, and `.claude/hooks/ruff-format.sh`, which formats a Python file after each edit with the checkout's locked ruff. |
| `.claude/hooks/no-push.py` | Reads each shell command and, as a guard against a slip, refuses every push and `gh` call it can read on the command line but the PR flow's own commands. Those pass only alone on the line and word for word as the user-level skills spell them: the push of `build/<topic>` to the `claude` remote; `gh pr create` with exactly `--draft`, `--head build/<topic>`, `--base main`, `--title` and `--body-file` with a `pr-body` file under the coding tool's per-user temporary folder; and the `gh` calls of `/pr-watch` and `/pr-setup`: reads, but for `/pr-setup`'s `gh repo set-default`, which writes the checkout's git config. A title may be quoted as bash quotes it. In single quotes it holds any text but a control or format character. In double quotes it holds no `$`, backtick or backslash. Outside quotes a backslash makes the next character text. A title made only of shell operators, or holding text that Claude Code's Bash tool would rewrite (`>nul`), is refused. The title, the body file, a PR number and a run id are taken on the skill's word. It finds the spellings the deny rules miss, such as `git -C <dir> push`, and `gh` after another word and gh's own options (`uv run gh -R <repo> pr merge`). It does not see a push from a script, an alias or a variable, or one that git or `gh` configuration shapes (`push.followTags` adds tags; `GH_REPO` or `set-default` picks the repository). `tests/unit/test_no_push_hook.py` holds the spellings, each command of the flow and a near miss of each and of each word. It checks the hook's wiring and, on a model of Claude Code 2.1.310's rule matcher, that no deny rule stops the flow and that each deny rule is the only one to meet some spelling it lists. |
| `.claude/skills/` | Five skills, below. Landing is the user-level `/land`'s; main moves by the owner's `/promote` until the PR flow of owner decision 11 is set up, and then through pull requests. |

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
build stopped at each branch. Since then, branches land in a build branch through the
user-level `/land`, and main has moved only through `/promote`, which the owner starts.
Owner decision 11, of 2026-10-19, sends builds to main as pull requests instead, once the
owner has set that flow up: the user-level `/pr-open` pushes a build and opens a draft pull
request, `/pr-watch` tends it, and the owner merges it on GitHub.

## Check it

```sh
git log --first-parent --oneline main
make check
.claude/skills/run-title-standardization-mock/smoke.sh
```
