---
name: run-title-standardization-mock
description: Run, start, smoke-test and drive the titlestd job-title standardization mock, from the command line or from Python. Covers the default offline run, replay of the recorded answers, a run with other settings, the local embedder, regenerating the mock data, and calling run_pipeline or a component directly. Use when asked to run, start, launch, try or demo the mock, to see a change working in the real program rather than in the tests, or to drive the pipeline from a script. Never for record or --llm anthropic, which call a paid API.
---

# Run the mock

A batch command line, `python3 -m titlestd`, with no server and no window. An agent drives it
with `.claude/skills/run-title-standardization-mock/smoke.sh`, which runs every mode that costs
nothing into a folder outside the repository and checks exit statuses and output. For a change
to one module, call the module or `run_pipeline` directly (below). Paths are relative to the
repository root. Never run `record` or `--llm anthropic`: both call a paid API and replace the
committed recording, and that is the owner's call alone.

## Prerequisites

Python 3.12, uv and make. Nothing else is installed for the default run: it uses the standard
library. Checked with uv 0.11.8 on 2026-10-18. The frozen output is compared on
3.12 only, so check that uv can name one:

```sh
uv python find 3.12
```

## Run (agent path)

```sh
.claude/skills/run-title-standardization-mock/smoke.sh
```

It takes about 3 seconds and prints one `PASS`, `FAIL` or `SKIP` line per check: the help,
the default run (compared with `tests/fixtures/expected/default_run.txt` on 3.12), replay, a
replay miss (exit 1), a bad setting (exit 2), another seed, `generate-data` against `data/`
byte for byte, and that `out/` did not change. It exits 1 if any check failed. Without a
folder it removes its runs at the end. Give it a folder outside the repository to keep them
for reading (about 6 MB, 8 MB with the local embedder, named on the first line; remove it by
its path when done), and `SMOKE_LOCAL=1` to add the local embedder:

```sh
SMOKE_LOCAL=1 .claude/skills/run-title-standardization-mock/smoke.sh "$(mktemp -d)"
```

One mode by hand, into its own folder, which the last line it prints names (about 2 MB a run;
remove it by its path when done):

```sh
dir=$(mktemp -d)
"$(uv python find 3.12)" -S -E -m titlestd run --llm replay --out-dir "$dir/replay"
uv run --locked --isolated --extra local python -m titlestd run --embedder local --out-dir "$dir/local"
```

`python3 -m titlestd run --help` lists every setting.

## Direct invocation

From the repository root, a script on standard input imports the checkout's `titlestd`:

```sh
"$(uv python find 3.12)" -S -E - "$(mktemp -d)" <<'EOF'
import sys
from collections import Counter
from pathlib import Path

from titlestd import datafiles, prepare
from titlestd.pipeline import run_pipeline
from titlestd.settings import load_settings

data = datafiles.read_inputs(Path("data"))
print(prepare.clean_title("Sr. Acct Mgr - NYC", data.abbreviations))
result = run_pipeline(load_settings(overrides={"out_dir": Path(sys.argv[1])}))
print(sorted(Counter(d.row.value for d in result.decisions).items()))
print([(a.path.value, a.audited, a.wrong) for a in result.audits])
EOF
```

It prints `CleanedTitle(text='senior account manager', has_person_name=False)`, then the
titles per table row, then each audited path with its audited and wrong counts: the same
figures as the printed table.

## Run (human path)

```sh
python3 -m titlestd run
```

It prints the table in about a second and rewrites the files in `out/`.

## Test

```sh
make check
```

It takes about a minute, one run per checkout at a time (CLAUDE.md, Commands). The smoke script
runs no make target.

## Gotchas

- **Plain `python3` may not be 3.12** (`python3 --version` says), and the frozen table is
  compared on 3.12 only (Q38). `"$(uv python find 3.12)" -S -E` names an interpreter without syncing
  `.venv`, and `-S -E` keeps the run on the standard library, as in a clean clone. `uv run`
  would sync `.venv`.
- **`--out-dir` changes the last line** (`Files written to the folder ...`). Drop the last
  line on both sides before comparing with the frozen file.
- **Replay holds only for the settings it was recorded with.** `--seed 2`, or a setting that
  shapes a request such as `--undecided-rate 0.5`, gives exit 1 and a replay miss. The CSV
  files are removed, but `state.sqlite` stays in the folder. Only a paid recording repairs a
  miss.
- **Exit statuses:** 0 when the run finished, 1 when it stopped (a replay miss), 2 for a bad
  setting, in which case the output folder is not created.
- **Keep agent runs out of `out/`.** A run rewrites its folder's five files, the suite rewrites
  `out/`, and `refresh-frozen-output` counts the default run from `out/`.
- **`generate-data` writes into `data/` by default.** Pass `--data-dir` to a scratch folder.
  Never point it at `tests/fixtures/mini`, which is written by hand.
- **The local embedder needs `--isolated`,** which keeps model2vec out of `.venv` as the gate
  synced it. Its model comes from the Hugging Face cache. The README says a first run
  downloads about 30 MB; that was not exercised when this was checked, as the cache was
  already warm.

## Troubleshooting

- **`titlestd: replay miss at step choose ... Its manifest differs from the settings in force:
  seed: recorded 20260101, now 2.`** Run replay with the recorded settings: the defaults.
- **`titlestd: setting llm must be one of offline, replay, anthropic; got 'bogus'`** (exit 2).
  `python3 -m titlestd run --help` lists the values.
- **`FAIL  default run differs from tests/fixtures/expected/default_run.txt`.** What the run
  prints has changed. If the change meant to do that, follow `refresh-frozen-output`.
  Otherwise, find out why.
