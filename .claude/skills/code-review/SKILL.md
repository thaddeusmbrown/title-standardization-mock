---
name: code-review
description: Review a change to this repository for bugs and for departures from the design, the decisions record and the rules for the package, in a fresh subagent that reports every finding. Takes the change as git diff arguments, which is how /ship passes it; with none, the working change, else the branch. Use when asked to review a change, and as the first link of /ship.
argument-hint: "[<rev> | <from>..<to> | <from>...<to>] [-- <paths>] [guidance]"
---

# Review a change

## The change under review

- Arguments that start with a revision, a range or `--` are `git diff` arguments, and that
  diff is the change. A lone revision means from it, exclusive, through the working tree, as
  `git diff <rev>` reads it: that is how `/ship` passes the start of its commits. `C^!` is
  commit C alone (`git show C` for a root commit). `A...B` is B's commits since it left A;
  `A..B` compares the two ends. Paths after `--` narrow any of them. When the diff reaches the working tree, add the untracked
  files under the same paths (`git ls-files --others --exclude-standard`).
- Paths alone narrow the default below. Any other text is guidance for this run, such as a
  focus, on top of the default.
- With no revision: a changed tree gives `git diff HEAD` and the untracked files. A clean tree
  gives the branch since main, `git diff main...HEAD`. On main with a clean tree, the commits
  not yet on its upstream (`git diff @{upstream}...HEAD`), else the last commit
  (`git show HEAD`).

## The review pass

Spawn one fresh subagent with the Agent tool, `subagent_type: general-purpose`. Where the tool
takes `run_in_background`, pass `false`: a reviewer in the background ends the turn, and the
Stop gate then runs in the middle of the review. Where it does not, end the turn saying that
the review is running, and do the second pass when its report arrives. Give it the diff
commands from above, any guidance, and this brief:

- Report every finding, with no severity filter: a bug, a doubtful line, a missing test, a
  departure from a lens below. Ranking and filtering come after you. For each: `file:line`,
  what is wrong, a concrete input or state that shows it, and the rule or entry it breaks.
- Read the whole of a touched file where the diff alone does not show the effect.
- The lenses of this repository:
  - `DESIGN.md` is the design and is firm. A change that departs from it is a finding.
  - `docs/build-decisions.md` settles owner decisions 1 to 10, Q01 to Q40 and S01 to S16,
    and holds the rules for every branch. A change that re-decides an entry is a finding:
    cite it.
  - `.claude/rules/package.md`: the rules for the package.
  - Every section of `CLAUDE.md`, which you have loaded: a change that breaks one is a
    finding. A change to anything that shapes a model request breaks the replay ("Paid calls
    and the recording"): say so in the finding.
  - A note (`PROMPT.md`, `CLAUDE_CODE.md`, `docs/`) or a file under `.claude/`: each claim of
    fact, command and path in it checked against the repository.
- Read only. Do not edit, stage, commit, run `make` or the suite, or run `record` or
  `--llm anthropic`. A program run that helps check a figure uses
  `"$(uv python find 3.12)" -S -E -m titlestd`, which leaves `.venv` alone, and goes to an
  `--out-dir` (and for `generate-data` a `--data-dir`) inside a scratch folder of your own
  outside the repository, never into `out/` or `data/`, one run at a time. Delete that folder
  by its literal path before you return. Return the findings as text and write no file.

## Second pass: rank and filter

Check each finding against the code. Mark it confirmed (shown in the code), plausible (it may
hold: say what would settle it) or rejected (say why in one line). Rank the confirmed and
plausible ones most severe first: a wrong output or a broken replay, then a departure from the
design, the record or a rule, then the rest.

## Report

The diff commands that were reviewed, the ranked findings with `file:line`, then the rejected
ones, one line each, so that nothing is dropped unseen. Fix nothing unless asked: under
`/ship` the chain fixes findings in its follow-up commit.
