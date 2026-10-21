---
paths:
  - "titlestd/**"
  - "tests/**"
  - "pyproject.toml"
  - "docs/build-decisions.md"
---

# Rules for the package

Tests hold most of these; where none does, the entry of the record is named.

- `titlestd/` is flat. No module imports a third-party library at module level, not even
  inside `try`/`except ImportError`: an optional library is imported in the function that
  uses it.
- No new dependency: tests pin the dependency lists of `pyproject.toml`.
- Only `titlestd/seeding.py` imports `random`, and nothing calls `hash()`. Sort a set before
  it reaches an output, and put no clock time in the printed table, the output files or the
  state: two runs must give the same bytes (Q06). A recording's dates are decided (Q29, Q40).
- Only `titlestd/people.py` reads the truth file, and no other module may hold that file's
  name in any string, a docstring included. Only `pipeline.py` and `mockdata.py` import
  `people` or use its truth functions and types.
- A command-line flag is a field of `Settings` in `titlestd/settings.py`, never an argparse
  call (Q38, S13; no test holds this one). `settings.py` and `tests/unit/test_settings.py`
  are shared files of rule 2, and the defaults there are the S03 table, so a new setting is
  the owner's call. It needs its `HELP` line, its entries in `tests/unit/test_settings.py`
  and, unless it shapes a prompt, its name in `_NOT_IN_A_MANIFEST` in `titlestd/recording.py`
  and in that module's docstring, which no test checks.
- A component module's docstring keeps its `Owner:` and `Governed by:` lines (rule 5). The
  owner line names the branch that built the module, not the branch that edits it.
  `tests/unit/test_modules.py` holds how each line starts.
- A test that pins a vocabulary, a column list, a default, a count or an import list is a
  recorded decision. Do not edit the pin to make a change pass: find the entry in
  `docs/build-decisions.md`, and ask the owner if the change would re-decide it. When the
  owner re-decides it, the entry in the record changes too.
- A decision the record leaves open goes in the docstring of the module it concerns, under a
  "Decided here" heading, and is reported for the merge message (rule 3; no test holds the
  heading). It is not added to `docs/build-decisions.md`.
