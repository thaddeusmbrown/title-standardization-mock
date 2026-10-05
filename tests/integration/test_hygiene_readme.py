"""The quick start of the README, run as the README gives it.

Owner: docs/readme-hygiene (stage 5).
Governed by: docs/design.md 2.1 (the standard library by default), 2.6 (one
command prints one table and writes the files); docs/build-decisions.md owner
decisions 9 (run from a fresh clone with nothing installed) and 10 (a short
README), Q38 (the output files).

The first command of the README must be the one of owner decision 9, and no
other command is ever started from the README's text. It is started in a
process of its own, from the repository root, on a bare interpreter: no
site-packages, which is "nothing installed". Two folders are added to it, so
that the test is quick and writes nothing into the repository: the mini data
set and a temporary output folder. The run must print the table with the rows
the README shows and leave exactly the files the README lists.

That the default run on the mock data prints the very numbers the README
shows is held in two steps: tests/unit/test_hygiene_readme.py compares the
README with the frozen table, and tests/integration/test_pipeline_default_run.py
compares the frozen table with the run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from titlestd import report
from titlestd.contracts import TableRow

QUICK_START = "python3 -m titlestd run"  # owner decision 9
# A row of the printed table starts with its name; the counts follow.
TABLE_ROW = re.compile(r"^([a-zA-Z][a-zA-Z0-9 -]*?) {2,}\d", re.MULTILINE)


@pytest.fixture(scope="module")
def readme(repo_root: Path) -> str:
    return (repo_root / "README.md").read_text(encoding="utf-8")


def test_the_quick_start_runs_with_nothing_installed_and_writes_the_files_the_readme_lists(
    readme, bare_cli, mini_dir: Path, tmp_path: Path
):
    assert re.search(r"^```sh\n(.+)\n```$", readme, re.MULTILINE)[1] == QUICK_START
    out = tmp_path / "out"

    run = bare_cli("run", "--data-dir", str(mini_dir), "--out-dir", str(out))

    assert run.returncode == 0, run.stderr
    listed = set(re.findall(r"^- `([a-z_]+\.(?:csv|sqlite))`", readme, re.MULTILINE))
    assert {file.name for file in out.iterdir()} == listed
    shown = re.search(r"^```text\n(.*?)^```$", readme, re.MULTILINE | re.DOTALL)[1]
    rows = [*(row.value for row in TableRow), report.TOTAL]  # the seven rows of 2.6
    assert TABLE_ROW.findall(shown) == rows
    assert TABLE_ROW.findall(run.stdout) == rows
