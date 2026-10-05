"""The state file: a local SQLite database, recreated on every run.

Owner: feat/pipeline-report-cli (stage 4).
Governed by: docs/design.md 2.1 ("a local SQLite file or plain files for
state"); docs/build-decisions.md Q06, Q38.

The state file is out/state.sqlite (settings.out_dir / STATE_FILE), made with
the standard library's sqlite3. It is deleted and recreated at the start of
every run and loaded with only the seed look-up, so a run never depends on an
earlier one.

Components do not read or write the database: they exchange the plain types of
titlestd.contracts in memory, and the pipeline writes the state here as the
run proceeds (title rows, the look-up table and examples, every model answer
with its label, people's decisions with their simulated mark, the golden set,
the audits and the final decisions). Tests of the eight section 2.7 rules read
it back. The schema belongs to this module and is not a contract for any
other.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from titlestd.contracts import RunResult
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.store is built by feat/pipeline-report-cli"

STATE_FILE = "state.sqlite"


def create_store(path: Path) -> sqlite3.Connection:
    """Delete any file at path and create the empty schema."""
    raise NotImplementedError(_NOT_BUILT)


def save_run(connection: sqlite3.Connection, result: RunResult, settings: Settings) -> None:
    """Write a finished run's state, with the settings that were in force."""
    raise NotImplementedError(_NOT_BUILT)
