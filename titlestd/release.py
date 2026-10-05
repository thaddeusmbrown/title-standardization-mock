"""Release: the load file, the reverse file, the exceptions list, the review queue.

Owner: feat/close-release-files (stage 2).
Governed by: docs/design.md 1.6 step 9, 2.4 ("Missing data": no grade), 2.6
(what a run writes); docs/build-decisions.md Q05 (files and columns), Q16 (the
review-queue file), Q27 (the simulated mark), Q40 (the answer label).
Section 2.7 tests: no title outside the catalog can reach the load file;
applying the reverse file to the load file restores every legacy title.

All files are written with titlestd.datafiles.write_table (UTF-8 CSV, header
row) into settings.out_dir, rows sorted by employee_id (load, reverse,
exceptions) or by title key (review queue). No output holds the wall-clock
time: two runs give the same bytes (Q06).

load.csv has one row per employee who receives a new title; reverse.csv has
the same employees with the job title as received, byte for byte;
exceptions.csv lists every employee left on a legacy title (released = 0) and
every employee with no grade who received a new title (released = 1, reason
"no grade"), one row per employee. load.csv plus the released = 0 exceptions
cover every employee exactly once (Q05).

The writer refuses any title that titlestd.catalog.is_catalog_title rejects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from titlestd.contracts import Catalog, Decision, Employee, QueueEntry, TitleRow

_NOT_BUILT = "titlestd.release is built by feat/close-release-files"

LOAD_FILE = "load.csv"
LOAD_COLUMNS = (
    "employee_id",
    "role_id",
    "long_title",
    "short_title",
    "level",  # blank when the employee has no grade
    "status",  # Approved, Confirmed or Provisional
    "path",  # the table row that released the title
    "error_rate",  # wrong/audited of the path; only for Provisional from an audited path
    "no_grade",  # 1 or 0
    "decided_by",  # the persona on a person's decision; LOOKUP_DECIDED_BY on a look-up hit
    "simulated",  # 1 on every simulated person's decision, otherwise 0
)
REVERSE_FILE = "reverse.csv"
REVERSE_COLUMNS = ("employee_id", "legacy_job_title")
EXCEPTIONS_FILE = "exceptions.csv"
EXCEPTIONS_COLUMNS = ("employee_id", "legacy_job_title", "job_family_code", "reason", "released")
REVIEW_QUEUE_FILE = "review_queue.csv"
REVIEW_QUEUE_COLUMNS = (
    "cleaned_title",
    "job_family_code",
    "headcount",
    "reasons",  # contracts.Reason values joined by "; "
    "ml_first_choice",  # role_id, or blank
    "llm_pick",  # role_id, or blank
    "agent_recommendation",  # role_id, or blank
    "answer_source",  # the answer label: stand-in, live, or recorded <date>, <model>
    "decision",  # the person's role_id, or a contracts.HumanAnswer value
    "decided_by",
    "simulated",  # 1 or 0
    "outcome",  # the table row the title ended in
)


def write_release_files(
    out_dir: Path,
    decisions: Sequence[Decision],
    rows: Sequence[TitleRow],
    employees: Sequence[Employee],
    catalog: Catalog,
) -> None:
    """Write load.csv, reverse.csv and exceptions.csv.

    decisions holds exactly one Decision per title row. Level and titles are
    applied per employee from the grade (1.4). Raises TitlestdError rather than
    write a role or title that is not in the catalog.
    """
    raise NotImplementedError(_NOT_BUILT)


def write_review_queue(out_dir: Path, entries: Sequence[QueueEntry]) -> None:
    """Write review_queue.csv: one row per title that entered the review queue,
    as the queue stood after Wave 2, with what happened to it at the close (Q16)."""
    raise NotImplementedError(_NOT_BUILT)


def apply_reverse(
    load_rows: Sequence[Mapping[str, str]], reverse_rows: Sequence[Mapping[str, str]]
) -> dict[str, str]:
    """Apply the reverse file to the load file: employee_id to restored title.

    Raises TitlestdError when the two files do not hold the same employee ids.
    """
    raise NotImplementedError(_NOT_BUILT)
