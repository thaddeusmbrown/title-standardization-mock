# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

An offline mock of a job-title standardization design, on invented data. `DESIGN.md` is the
design and is firm. `docs/build-decisions.md` holds the points the record settled: owner
decisions 1 to 11, Q01 to Q40, S01 to S16 and nine rules for every branch. Where an entry
settles a point, build it as written and cite the entry. Do not re-decide it. A point the
record leaves open is settled in its module's docstring, under "Decided here" (rule 3).

## Commands

- `make check` is the gate, and what CI runs: ruff, then the suite in two environments. A green
  `uv run pytest` is not a green gate. `make format` applies the ruff fixes.
- Run ruff and pytest through `make` or `uv run` (one test:
  `uv run pytest <path>::<test_name>`), never `uvx ruff` or another install: a different ruff
  formats differently from the gate.
- One gate, test run or program run into `out/` per checkout at a time. `make check`
  re-syncs `.venv` exactly, and the suite and the default run rewrite `out/`. A second run
  needs its own worktree under `.worktrees/`, or a program run its own `--out-dir`.
- `.claude/check.sh` runs the gate, after refusing a concurrent run or an untracked file (its
  message says what to do). The repository wires no Stop hook; a user-level one that calls
  check.sh, where one is installed, holds a turn while it is red. Before asking the
  owner about a change the gate rejects, take the change out of the tree: `git stash` keeps
  it, once any new file is staged.

## Paid calls and the recording

- Never run `python3 -m titlestd record`, or any run with `--llm anthropic`, unless the owner
  asks for it in this session. Both call a paid API and replace the committed recording when
  they finish. The owner declined a second recording (`docs/live-run-observation.md`).
- `make check` replays the committed recording, and the requests of the run must equal the
  recorded ones exactly. So everything that shapes a request is frozen: `data/`, the default
  settings, the prompts, tools, schemas and `MAX_TOKENS` in `titlestd/recording.py`, and every
  step that runs before the last model call, which is the golden-set pass at the close. That
  takes in, among others, prepare's cleaning and masking, the catalog and reference lists, the
  embedder, the matcher, the look-up, the rendering in `fallback.py` and `agent.py`, the
  rules, the agent, every seeded draw and the simulated people. Only the generic-role step,
  the scoring of that pass, the release files and the report come after it, and writing the
  state file never shapes a request. A change that alters a request fails the replay tests, and only a
  paid recording repairs it: ask the owner. Do not deselect the `replay` marker, and never
  edit or add a file under `recordings/`.

## Generated files

- `data/*.csv` come from `python3 -m titlestd generate-data`, and
  `tests/fixtures/expected/default_run.txt` from a real default run on Python 3.12. Neither is
  edited by hand. `tests/fixtures/mini/` is hand-written: never point `generate-data` at it.
- If the default run prints anything different, the frozen file, the README's `text` block and
  every copy of its figures in prose change in the same commit. The `refresh-frozen-output`
  skill has the steps, and `verify-prose-figures` checks the copies in prose.
- `README.md` is parsed by tests and stays short (owner decision 10): only `sh` and `text`
  fences, one `text` block, and every command in it must be one the program or the Makefile
  takes. Detail goes under `docs/`.

## Code rules

- The rules for the package, no new dependency and no third-party import at module level
  among them, are in `.claude/rules/package.md`. They load by themselves when a file their
  `paths:` list names is read. Read that file first when planning or reviewing a change to
  the package.

## Hygiene

- No real company, client or person in code, data, docs, branch names or commit messages
  (design 2.8). Write "the owner". Technology vendors and the coding tool may be named.
- Every tracked file is scanned, this one and `.claude/` included: no email address outside
  `example.com`, no key-shaped string, no symbolic link, no `.env` file. So never paste the
  commit trailer lines or an SSH remote address into a file.

## Git

- Every change goes on a branch cut from main and named `feat/<topic>`, `docs/<topic>` or
  `chore/<topic>`. A build of several features first cuts `build/<topic>` from main, and its
  features branch from that instead.
- A finished branch lands in its build branch with the user-level `land` skill (pass
  `--prune` for one built in its own worktree). It runs `.claude/merge-gate.sh`, which is
  `make check-all` (the gate, then the suite on Python 3.13 and 3.14), on a `--no-ff` merge,
  and on the branch too when the merge's tree differs from the branch's (when it is the same,
  one run covers both), and moves the build branch only to a merge that passed. That is more
  than rule 7 and Q34 ask, which is `make check` on the merge result.
- Finish a build on its build branch and stop there. Builds reach main as pull requests
  (owner decision 11) once the owner has pushed main, run `/pr-setup` and imported the
  rulesets it writes. Until then, main moves as before: the owner runs `/promote <branch>`
  and pushes main, and a lone `feat/`, `docs/` or `chore/` branch goes the same way. Where
  the user-level `/land`'s report names `/promote` as the next step, that holds until then.
- The PR flow runs only from a checkout whose `.claude/` lets it through. The hook and deny
  rules in force are those of the checkout the session started in, and a checkout of an
  older branch keeps the old rules, which refuse every push and `gh` call.
- Once the flow is set up, a build's first `/pr-open build/<topic>` runs when the owner asks
  for it. It runs its preflight over what the push would publish, pushes the build to the
  `claude` remote and opens a draft pull request into main. `/pr-watch build/<topic>` then
  tends CI and review: each fix is a branch cut from the build's tip, landed with `/land` and
  pushed with `/pr-open` again, which the watch runs itself. On this public repository every
  push, the first and each fix's, and the pull request wait for the owner's approval at the
  ask. A lone branch is landed first in a `build/<topic>` of its own. The owner marks the
  pull request ready, merges it on GitHub and pulls. Once the default-branch ruleset is
  active, main is not promoted locally, because it refuses a push of main.
- If the land cannot fast-forward a branch that is checked out because an untracked file is
  in the way, move that file aside with one `mv` per literal path and land again.
- Claude pushes only `build/<topic>`, only to `claude` and only through `/pr-open`, and runs
  `gh` only as `/pr-open`, `/pr-watch` and `/pr-setup` do. Never push to `origin`, never push
  main or a tag, never force, never merge, ready, edit, review, comment on or close a pull
  request, never resolve a thread, and never re-run or cancel a workflow.
  `.claude/hooks/no-push.py` refuses the other pushes and `gh` calls it can read on the
  command line, as a guard against a slip. Never amend, rebase or otherwise rewrite main.
  `land.sh` rebases a branch only while it is unpushed.
- The message of a merge that `/land` or `/promote` makes goes in a file outside the
  repository, passed with `--message-file`. Its subject is git's own:
  `Merge branch '<branch>'`, ending ` into <build>` when the target is a build branch. Then:
  - a paragraph on what the branch adds;
  - where it applies, `Decisions worth knowing (the rest are in the module docstrings):` and
    a dash list;
  - any shared file the branch changed (rule 2) and any fix to a module merged earlier
    (rule 9), each with the subject of the commit that did it;
  - anything left to the owner;
  - a `Reviewed for ...` line only if a review of the branch really ran;
  - a blank line, then the co-author trailer line, and no session trailer (owner decision 7).

  A pull request merged on GitHub carries its own title and body instead.
  `git log --merges main` shows the earlier messages.
- Commit subject: imperative, capitalized, no `type:` prefix, no trailing period, at most 68
  characters. The body says what was wrong and what the commit does.
- Rule 2 of `docs/build-decisions.md` freezes the shared files it names. Change one only when
  it cannot be avoided, in a commit of its own with only what must change with it (its tests,
  the frozen output), and name it in the merge message.
- Stage by explicit path, never with `git add -A`.
