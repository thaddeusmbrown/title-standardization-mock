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

Decided here, where the record is silent:
- Nothing is written unless everything may be: the rows of the three files
  are built and checked first, and any refusal is a ReleaseError raised before
  a file is opened. A file an earlier run left in the folder is not touched.
  The load file is written last, so it never stands without its reverse file.
- The guard of 2.7 has two halves. A decision whose role the catalog does not
  hold is refused. Then the rows are read as they are about to be written:
  the Long and Short Title of each, as the strings the file will hold, must
  pass is_catalog_title.
- What else is refused. Decisions, title rows and employees that do not fit
  each other: a title row without exactly one decision, a decision for no
  title row, an employee in no title row or in two, a row naming an employee
  the employees file does not hold, an employee listed twice among the
  employees (the reader of the employees file refuses that too; the writer
  does not count on its caller having used it). And a decision the files
  cannot state truthfully: one left on its legacy title that has a role, a
  status or no reason (the reason "no grade" belongs to an employee, never to
  a title); one released without a role or a status, or with a reason;
  Approved with nobody named (1.9); an error rate on anything but a
  Provisional title of a machine path (Q05).
- The error rate is checked both ways. A Provisional decision whose source is
  the machine and which has no error rate is refused as well: 1.9 releases
  such a path "with its measured error rate", so a blank there would be a
  rate somebody forgot. Every such decision has one to carry, a stopped path
  that a setting releases as Provisional included (2.4, Q21): a path that is
  sampled has 60 audited titles, and a smaller one is labeled in full and
  holds no Provisional title. A generic family role is Provisional with no
  rate, because code assigned it and no path was audited for it.
- "No grade" is no usable grade, as titlestd.catalog reads it: a missing
  grade, or one HR's table does not list. Either gives the title with no level
  word, a blank level, no_grade = 1 and a line on the exceptions list.
- error_rate is written with four decimal places: 0.0833 for 5 wrong of 60.
- job_family_code on the exceptions list is the employee's code as received,
  like the legacy title beside it. In the review queue it is the title key's.
- Rows are sorted by employee_id as text, the order a title row lists its
  employees in.
- The review queue: reasons keep the order the entry gives them; an entry
  with no decision has a blank decision and decided_by and simulated = 0; a
  title entered twice, or a decision for a role that names none, is refused.
- A title a person marked "no suitable role" is written with the reason "no
  suitable role, waiting for catalog owner" (Q03), whether or not the entry
  already lists it: that reason is how the file says whom the title waits
  for, since decided_by names the person who made the mark.
- apply_reverse reads only employee_id and legacy_job_title, answers in the
  order of the load rows, and also refuses an employee listed twice in either
  file.
- The released = 1 lines of exceptions.csv are the employees
  RunResult.no_grade_employees counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from titlestd import datafiles
from titlestd.catalog import get_role, is_catalog_title, standard_title
from titlestd.contracts import (
    Catalog,
    Decision,
    DecisionSource,
    Employee,
    ExceptionReason,
    HumanAnswer,
    QueueEntry,
    Reason,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
    TitlestdError,
)

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


class ReleaseError(TitlestdError):
    """The release is refused: a role or title outside the catalog, decisions
    that do not fit the title rows and employees, or files that do not fit
    each other."""


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
    # The title rows and the employees are each read twice. Read once here, so
    # that an iterator cannot pass the checks and then give files with no line.
    decisions, rows, employees = tuple(decisions), tuple(rows), tuple(employees)
    decision_of = _decision_of_each_employee(decisions, rows, employees)
    load, reverse, exceptions = [], [], []
    for employee in sorted(employees, key=lambda employee: employee.employee_id):
        decision = decision_of[employee.employee_id]
        if decision.row is TableRow.LEGACY:
            exceptions.append(_exception(employee, decision.legacy_reason, released=False))
            continue
        role = get_role(catalog, decision.role_id)
        if role is None:
            raise ReleaseError(
                f"employee {employee.employee_id}: role {decision.role_id} is not in the "
                "catalog; nothing was written"
            )
        title = standard_title(catalog, role, employee.grade)
        load.append(
            {
                "employee_id": employee.employee_id,
                "role_id": role.role_id,
                "long_title": title.long_title,
                "short_title": title.short_title,
                "level": title.level or "",
                "status": decision.status.value,
                "path": decision.row.value,
                "error_rate": "" if decision.error_rate is None else f"{decision.error_rate:.4f}",
                "no_grade": _flag(title.level is None),
                "decided_by": decision.decided_by,
                "simulated": _flag(decision.simulated),
            }
        )
        reverse.append(
            {"employee_id": employee.employee_id, "legacy_job_title": employee.job_title}
        )
        if title.level is None:
            exceptions.append(_exception(employee, ExceptionReason.NO_GRADE, released=True))
    _refuse_titles_outside_the_catalog(load, catalog)

    out_dir = Path(out_dir)
    datafiles.write_table(out_dir / REVERSE_FILE, REVERSE_COLUMNS, reverse)
    datafiles.write_table(out_dir / EXCEPTIONS_FILE, EXCEPTIONS_COLUMNS, exceptions)
    datafiles.write_table(out_dir / LOAD_FILE, LOAD_COLUMNS, load)


def write_review_queue(out_dir: Path, entries: Sequence[QueueEntry]) -> None:
    """Write review_queue.csv: one row per title that entered the review queue,
    as the queue stood after Wave 2, with what happened to it at the close (Q16)."""
    lines = {}
    for entry in entries:
        if entry.row.key in lines:
            raise ReleaseError(f"{_title(entry.row.key)} is in the review queue twice")
        lines[entry.row.key] = _queue_line(entry)
    datafiles.write_table(
        Path(out_dir) / REVIEW_QUEUE_FILE,
        REVIEW_QUEUE_COLUMNS,
        (lines[key] for key in sorted(lines)),
    )


def apply_reverse(
    load_rows: Sequence[Mapping[str, str]], reverse_rows: Sequence[Mapping[str, str]]
) -> dict[str, str]:
    """Apply the reverse file to the load file: employee_id to restored title.

    Raises TitlestdError when the two files do not hold the same employee ids.
    """
    legacy: dict[str, str] = {}
    for row in reverse_rows:
        if row["employee_id"] in legacy:
            raise ReleaseError(f"{REVERSE_FILE}: employee {row['employee_id']} is listed twice")
        legacy[row["employee_id"]] = row["legacy_job_title"]
    restored: dict[str, str] = {}
    for row in load_rows:
        employee_id = row["employee_id"]
        if employee_id in restored:
            raise ReleaseError(f"{LOAD_FILE}: employee {employee_id} is listed twice")
        if employee_id not in legacy:
            raise ReleaseError(
                f"{REVERSE_FILE} has no legacy title for employee {employee_id} of {LOAD_FILE}"
            )
        restored[employee_id] = legacy[employee_id]
    unloaded = sorted(set(legacy) - set(restored))
    if unloaded:
        raise ReleaseError(
            f"{REVERSE_FILE} lists employees that {LOAD_FILE} does not: {_some(unloaded)}"
        )
    return restored


def _decision_of_each_employee(
    decisions: Sequence[Decision], rows: Sequence[TitleRow], employees: Sequence[Employee]
) -> dict[str, Decision]:
    """employee_id to the decision of the employee's title row, once the three
    arguments are known to fit each other and every decision can be stated."""
    by_key = {}
    for decision in decisions:
        if decision.key in by_key:
            raise ReleaseError(f"{_title(decision.key)} has two decisions")
        _refuse_a_decision_the_files_cannot_state(decision)
        by_key[decision.key] = decision
    decision_of = {}
    for row in rows:
        if row.key not in by_key:
            raise ReleaseError(f"{_title(row.key)} has no decision")
        for employee_id in row.employee_ids:
            if employee_id in decision_of:
                raise ReleaseError(f"employee {employee_id} is in two title rows")
            decision_of[employee_id] = by_key[row.key]
    unknown = sorted(set(by_key) - {row.key for row in rows})
    if unknown:
        raise ReleaseError(f"{_title(unknown[0])} has a decision and is not a title row")
    held = [employee.employee_id for employee in employees]
    differ = sorted(set(held) ^ set(decision_of))
    if differ:
        raise ReleaseError(
            f"the title rows and the employees file do not hold the same employees: {_some(differ)}"
        )
    if len(set(held)) != len(held):
        raise ReleaseError("the employees file lists an employee twice")
    return decision_of


def _refuse_a_decision_the_files_cannot_state(decision: Decision) -> None:
    released = decision.row is not TableRow.LEGACY
    by_a_path = decision.status is Status.PROVISIONAL and decision.source is DecisionSource.MACHINE
    faults = (
        (released and not (decision.role_id and decision.status), "it has no role or no status"),
        (released and decision.legacy_reason is not None, "it has a reason to stay on legacy"),
        (
            not released and (decision.role_id is not None or decision.status is not None),
            "it stays on its legacy title and has a role or a status",
        ),
        (
            not released and decision.legacy_reason in (None, ExceptionReason.NO_GRADE),
            "it stays on its legacy title and gives no reason a title can have",
        ),
        (
            decision.status is Status.APPROVED and not decision.decided_by.strip(),
            "it is Approved and names nobody who decided it",
        ),
        (
            decision.error_rate is not None and not by_a_path,
            "it has an error rate and is not a Provisional title of a machine path",
        ),
        (
            by_a_path and decision.error_rate is None,
            "it is a Provisional title of a machine path and has no error rate",
        ),
    )
    for found, what in faults:
        if found:
            raise ReleaseError(f"the decision for {_title(decision.key)} cannot be written: {what}")


def _refuse_titles_outside_the_catalog(load: Sequence[Mapping[str, str]], catalog: Catalog) -> None:
    """The guard of 2.7, on the rows as they are about to be written."""
    passed = set()
    for line in load:
        titles = (line["long_title"], line["short_title"])
        if titles not in passed and not is_catalog_title(catalog, *titles):
            raise ReleaseError(
                f"employee {line['employee_id']}: the titles {titles[0]!r} and {titles[1]!r} "
                "are not a Long and Short Title of the catalog; nothing was written"
            )
        passed.add(titles)


def _queue_line(entry: QueueEntry) -> dict[str, str]:
    key, decision = entry.row.key, entry.decision
    reasons = list(entry.reasons)
    answer = ""
    if decision is not None:
        role_answer = decision.answer is HumanAnswer.ROLE
        answer = decision.role_id if role_answer else decision.answer.value
        if not answer:
            raise ReleaseError(f"the decision on {_title(key)} is a role and names none")
        marked = decision.answer is HumanAnswer.NO_SUITABLE_ROLE
        if marked and Reason.NO_SUITABLE_ROLE not in reasons:
            reasons.append(Reason.NO_SUITABLE_ROLE)  # whom the title waits for (Q03)
    return {
        "cleaned_title": key.cleaned_title,
        "job_family_code": key.family_code,
        "headcount": str(entry.row.headcount),
        "reasons": "; ".join(reason.value for reason in reasons),
        "ml_first_choice": entry.ml_first_choice or "",
        "llm_pick": entry.llm_pick or "",
        "agent_recommendation": entry.agent_recommendation or "",
        "answer_source": entry.answer_source,
        "decision": answer,
        "decided_by": decision.decided_by if decision is not None else "",
        "simulated": _flag(decision is not None and decision.simulated),
        "outcome": entry.outcome.value,
    }


def _exception(employee: Employee, reason: ExceptionReason, *, released: bool) -> dict[str, str]:
    return {
        "employee_id": employee.employee_id,
        "legacy_job_title": employee.job_title,
        "job_family_code": employee.job_family_code,
        "reason": reason.value,
        "released": _flag(released),
    }


def _flag(value: bool) -> str:
    return "1" if value else "0"


def _title(key: TitleKey) -> str:
    return f"title {key.cleaned_title!r} in family {key.family_code!r}"


def _some(employee_ids: Sequence[str]) -> str:
    """The first few of a sorted list of ids, for a message."""
    more = f" and {len(employee_ids) - 5} more" if len(employee_ids) > 5 else ""
    return ", ".join(employee_ids[:5]) + more
