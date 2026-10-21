---
name: refresh-frozen-output
description: Bring every copy of the default run's output back in line after a change alters what `python3 -m titlestd run` prints, or after the mock data changes. Covers the frozen table (regenerated on Python 3.12), the README's text block and measured figures, and the copies no test checks. Use whenever a change moves a number, a row, a label or a line of the printed table, or changes data/ or the three knobs at the top of titlestd/mockdata.py.
---

# Refresh the frozen output

What the default run prints lives in three places that tests compare byte for byte: the run
itself, `tests/fixtures/expected/default_run.txt` and the one `text` block of `README.md`.
More copies of its numbers sit in prose that no test reads. All of them change in the same
commit as the code.

## First: does the recording still replay?

Run `.claude/skills/run-title-standardization-mock/smoke.sh` from the repository root. A `FAIL`
on its replay line with "replay miss" means the change altered what a model is asked:
CLAUDE.md, "Paid calls and the recording", says what that takes. Ask the owner before going
further, and take the change out of the tree first if the turn has to end (`git stash` keeps
it, once any new file is staged): a red gate holds a turn open.

Exit status 0 is not proof that the recording still fits. A change that makes the run ask
fewer requests leaves recorded answers unasked, and only the replay tests in `make check`
catch that.

## A. The printed output changed

1. Regenerate the frozen file from a real run on Python 3.12, from the repository root:

   ```sh
   "$(uv python find 3.12)" -S -E -m titlestd run > tests/fixtures/expected/default_run.txt
   ```

   Not with plain `python3`, which may be a later version (the run skill's Gotchas), and never
   by hand. A run that fails leaves the file empty: check that it still starts with
   `Job-title standardization mock:` and read `git diff` on it. Every changed line should be
   one the change explains.
2. In `README.md`, replace the content of the one `text` block with the content of that file,
   byte for byte. Copy from the file, not from a terminal.
3. Bring the copies in prose in line, which no test reads: follow `verify-prose-figures` and
   fix each mismatch it reports. It lists them, the README's notes on reading the table among
   them, and measures the sentence on seeds 1 to 10.
4. Run `.claude/check.sh`. Only a run on Python 3.12 compares the frozen file: on 3.13 and 3.14
   that test is skipped, so a stale file passes there.
5. Commit the code, the frozen file, the README and the notes together.

## B. The mock data changed

Ask the owner before starting. New data makes every replay miss, which means a paid recording.
Q23 settles when the data may change: a tuning round is made only when the default run misses
the bar of Q23, by shaping the generator, never by aiming at a status, and at most three rounds
are made. The bar held at the build, so no round was made. Say which part of the bar the run
misses when asking.

1. Change only the three knobs at the top of `titlestd/mockdata.py`: `WORDING_WEIGHTS`,
   `DESCRIPTION_WEIGHTS` and `FLAGGED_ROLE_IDS`. They stay the first three constants of the
   module. The generator's seed and the sizes below them are fixed by the record.
2. From the repository root: `python3 -m titlestd generate-data`. Never edit a CSV in `data/`
   by hand, and never pass `--data-dir tests/fixtures/mini`: that set is hand-written and has
   no generator.
3. Correct the two docstrings that say no tuning round was made, in `titlestd/mockdata.py` and
   `titlestd/pipeline.py`.
4. Do all of part A: the default run now prints other numbers. Its one commit also holds
   `titlestd/mockdata.py` and the changed `data/*.csv`, so that no commit has a stale frozen
   file, and the new recording under `recordings/`, which only the owner's paid record command
   makes. Until that recording is in, the replay tests in `make check` fail and the gate stays
   red. Keep anything else out of that commit.
