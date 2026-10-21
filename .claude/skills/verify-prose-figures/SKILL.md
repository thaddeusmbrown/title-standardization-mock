---
name: verify-prose-figures
description: Check that every figure of the default run quoted in prose matches what the run prints now. Covers the README's notes on reading the table, the offline columns of docs/live-run-observation.md, any other note that quotes the run (PROMPT.md, CLAUDE_CODE.md, docs/), the docstring of tests/unit/test_hygiene_readme.py and the run-shape paragraph of titlestd/pipeline.py, which no test reads. Use after a change that could move what `python3 -m titlestd run` prints, before a merge, or when asked whether the prose still matches the run.
---

# Check the run figures quoted in prose

The gate holds the frozen file and the README's `text` block to the run. Nothing holds the
figures that prose repeats from it. Done means every quoted figure was compared with the run
of now, and each one that differs is reported and fixed.

1. **Get the run:** `.claude/skills/run-title-standardization-mock/smoke.sh <folder>`, with a
   folder outside the repository. A `FAIL` on "default run equals the frozen output" means the
   frozen output is stale: follow `refresh-frozen-output` first. The printed run is
   `<folder>/default.txt`. Count the queue's undecided titles, and its titles by reason:

   ```sh
   python3 -c "import csv; print(sum(r['decision'] == 'could not decide' for r in csv.DictReader(open('<folder>/default/review_queue.csv', encoding='utf-8'))))"
   python3 -c "import collections, csv; rows = list(csv.DictReader(open('<folder>/default/review_queue.csv', encoding='utf-8'))); print(len(rows), sorted(collections.Counter(reason for r in rows for reason in r['reasons'].split('; ')).items()))"
   ```

2. **Compare each quote** with that run:
   - `README.md`, the list under "Three things to keep in mind when reading the table of a
     run": the models-agree row (titles, wrong of audited, status), the agent-settled row
     (titles, labeled in full), the undecided titles of the queue against the review-queue
     count in the footer line `Sent to a person`, and the left-on-legacy-title row (titles,
     employees).
   - The sentence measured on `--seed` 1 to 10, in the README and in the docstring of
     `tests/unit/test_hygiene_readme.py`. When the change could move a seeded run, measure it
     again and count the statuses in the last column:

     ```sh
     for seed in 1 2 3 4 5 6 7 8 9 10; do "$(uv python find 3.12)" -S -E -m titlestd run --seed "$seed" --out-dir <folder>/seeds | grep '^models agree'; done
     ```
   - `docs/live-run-observation.md`: the offline columns of both tables, the second against
     the count by reason. The live columns come from the recorded run.
   - `titlestd/pipeline.py`, the run-shape paragraph that ends the module docstring: what it
     says of the seven rows, the models-agree audit and the agent-settled path.
   - Any other note that quotes the run (`PROMPT.md`, `CLAUDE_CODE.md`, `docs/`): search it
     for the figures the run prints now, each row's titles and employees and the footer's
     counts.
3. **Report** each mismatch as `file:line`, the quoted figure and the run's figure. Then fix
   the prose: in the commit of the change that moved the figure while that commit is still to
   be made, else in a commit of its own. A statement that was measured keeps saying so. The
   frozen file and the README's `text` block are not prose: `refresh-frozen-output` changes
   them, from a real run.
