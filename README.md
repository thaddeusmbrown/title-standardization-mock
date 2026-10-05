# Job-title standardization mock

I translated my work to a docs/design.md file with Claude.  The
architecture is my own design, with my selection of models and
replay strategy.  The architecture PNGs are my design work as well to help
inform the final agnostic/Microsoft designs. Claude Code ran the end-to-end 
build, which is why there are many commits in a short overnight period, 
with light code and ML result checks from me in the morning.

Below this line is the LLM-generated README for setup.

_______________________________________________________________________________

A small offline mock of a job-title standardization design: every employee's free-text job
title is mapped onto a role from an approved catalog, which gives the employee a standard Short
Title and Long Title. It runs on made-up data. There is no real employer and there are no real
people in it.

## Quick start

You need Python 3.12 or later and nothing else: the default run uses the standard library only.
uv is needed only for the tests and for the two optional modes that install a library.

From a clean clone, in the repository root:

```sh
python3 -m titlestd run
```

It runs set-up, Wave 1, Wave 2 and the close on the mock data in `data/`, takes about a second
and prints:

```text
Job-title standardization mock: set-up, Wave 1, Wave 2 and the close

                      titles  employees  share of employees  audited  wrong  status
--------------------  ------  ---------  ------------------  -------  -----  -----------
look-up hit                8         26                1.7%                  Approved
approved in Wave 1        40        604               40.3%                  Approved
models agree             252        669               44.6%       60      1  Confirmed
agent-settled             34         79                5.3%       34      7  all labeled
review queue              56        105                7.0%                  Approved
generic family role        4          6                0.4%                  Provisional
left on legacy title       6         11                0.7%
--------------------  ------  ---------  ------------------  -------  -----
total                    400       1500              100.0%

Coverage by status, by employee: Approved 986 (65.7%), Confirmed 497 (33.1%), Provisional 6 (0.4%), left on legacy title 11 (0.7%)
Sent to a person: 106 of 400 titles (26.5%): 40 on the Wave 1 list, 66 in the review queue. Blind labels besides: 94 in the audits, 40 on the golden set.
Golden-set accuracy, by the SME's labels:
  Wave 1: ML first choice 32/40, LLM pick 34/40 (3 with no role named)
  Wave 2: ML first choice 29/40, LLM pick 34/40 (3 with no role named)
  close: ML first choice 30/40, LLM pick 34/40 (3 with no role named)
Exceptions list: 11 employees left on a legacy title; 20 employees with no grade given a title with no level word.
LLM and agent answers: stand-in. Embedder: offline.
People were simulated from the truth file: every decision a person made in this run is a simulated one.
Files written to the folder out: load.csv, reverse.csv, exceptions.csv, review_queue.csv, state.sqlite.
```

Every run writes these files afresh under `out/`, which git ignores:

- `load.csv`: one row per employee who gets a new title, with the role, the titles and the status.
- `reverse.csv`: the legacy title of each of those employees, to undo the load.
- `exceptions.csv`: employees left on a legacy title, and employees with no grade, with the reason.
- `review_queue.csv`: every title in the review queue, with what the models proposed and the decision.
- `state.sqlite`: the state of the run, with every score, model answer and decision.

## Optional modes

Run these from the repository root as well.

```sh
# Replay the committed recording of the hosted model's answers: nothing to install, no key.
python3 -m titlestd run --llm replay

# Use the local embedding model: the first run downloads three pinned files, about 30 MB.
uv run --extra local python -m titlestd run --embedder local
```

- Replay works only with the data, seed and embedder the recording was made with: any other
  request is a miss, which stops the run. The recording is under `recordings/`.
- The local model's files go into your Hugging Face cache, and later runs need no network. No
  PyTorch is installed.

Recording again calls the hosted model, a paid API, and the new recording replaces the old one:

```sh
uv run --env-file ~/titlestd.env --extra live python -m titlestd record --llm-model claude-haiku-4-5
```

The key comes only from the environment variable `ANTHROPIC_API_KEY`. Above, uv sets it from a
file outside version control that holds the line `ANTHROPIC_API_KEY=` followed by your key;
exporting the variable in the shell and leaving `--env-file` out works too.

## Running the tests

```sh
make check
```

It needs uv and make, and runs ruff, then the pytest suite with the `live` extra installed and
again with no extra. `make check-all` adds the suite on Python 3.13 and 3.14.

## Architecture at a glance

One batch program, one SQLite file and plain CSV files. `titlestd/pipeline.py` runs the flow of
section 1.6 of the design as set-up, Wave 1, Wave 2 and the close:

| Step | Module | In the mock |
|---|---|---|
| 1. Prepare | `titlestd/prepare.py` | Cleans each title, masks names, groups to one row per title and family. |
| 2. Look-up | `titlestd/lookup.py` | An exact match on cleaned title and family takes its role with no model. |
| 3. ML match | `titlestd/matcher.py` | Nearest neighbours on embeddings: ten roles, a first choice, a margin. |
| 4. LLM fallback | `titlestd/fallback.py` | The LLM sees the masked record and the ten roles; it answers with a number and a quote. |
| 5. Rules | `titlestd/rules.py` | Code checks every proposal; a failed check sends it to the agent or a person. |
| 6. Agent | `titlestd/agent.py` | A plain loop of at most four searches, then one answer the rules check again. |
| 7. People | `titlestd/people.py`, `titlestd/close.py` | Simulated HR SMEs and reviewers decide the rest; leftovers take a generic family role under a cap. |
| 8. Random audit | `titlestd/audit.py` | A blind sample of 60 titles per machine path sets the status of the path. |
| 9. Release | `titlestd/release.py` | Writes the load file, the reverse file, the exceptions list and the review queue. |
| 10. Learning | `titlestd/lookup.py` | Titles a person decided join the look-up table and the examples. Nothing is trained. |

- **The model switch.** The three model calls (embed, choose, review) sit behind
  `titlestd/switch.py`, set with `--embedder`, `--llm` and `--llm-model`. A model only picks a
  catalog role by number; masking, numbering and the quote check stay in shared code.
- **The settings.** Every setting but `--llm-model` has a default; `python3 -m titlestd run --help`
  lists them.
- **The state file.** `titlestd/store.py` writes `out/state.sqlite` as the run proceeds. It is
  rebuilt on every run, so no run depends on an earlier one.
- **The output files.** The four CSV files are written under `out/` only once every decision is
  made, and `titlestd/report.py` prints the table. Nothing is written to an HR system.

## Models and tools used

| Part | What is used |
|---|---|
| Embedder, default | Built in: hashed character n-grams (`titlestd/offline_embedder.py`). |
| LLM and agent, default | Deterministic stand-ins that decide by word overlap (`titlestd/standin.py`). |
| Embedder, optional | model2vec with the model `minishlab/potion-base-8M` (extra `local`). |
| LLM and agent, optional | The hosted model `claude-haiku-4-5` through its Python SDK (extra `live`). |
| Tooling | Standard-library Python, SQLite, uv, pytest and ruff. |

Three things to keep in mind when reading the table of a run:

- **Offline accuracy shows the flow, not model quality.** The default answers come from
  stand-ins, and the people are simulated.
- **At 60 titles per path the audit has little power.** A path that is truly 95% right is
  confirmed less than half the time, so a status can change with the seed. On the default seed
  the models-agree path (252 titles) is Confirmed with 1 wrong of 60; with `--seed` 1 to 10 it is
  Confirmed seven times and Provisional three. The agent-settled path holds 34 titles, so it is
  labeled in full and gets no status.
- **A few employees stay on a legacy title on purpose.** The simulated reviewer is set to leave
  one queue title in ten undecided: 10 of 66 on the default seed. An undecided title that
  touches a compliance-flagged role may not take a generic family role, so it stays on its
  legacy title and is listed in `exceptions.csv`: 6 titles, 11 employees on the default seed. In
  a real delivery a person decides each of these before sign-off.

## Where to read more

- `docs/design.md`: the design, and what the mock builds from it.
- `docs/build-decisions.md`: every point the design left open and how it was settled.
- `docs/live-run-observation.md`: what the one live run showed about the quote check.
