"""The release files of a miniature run on the mini data set.

The run uses the real readers, prepare step, catalog, simulated people and
close. The models and the audit are built beside this branch, so two title
rows are handed a machine path's outcome by hand (one Confirmed, one
Provisional with an error rate) and no model is asked anything: the seed
look-up hits take their role, an SME approves the Wave 1 list, a reviewer
decides the rest of the queue or cannot, and the close deals with the
leftovers. Then the four files are written and read back.

The section 2.7 tests on the files: no title outside the catalog reaches the
load file, and applying the reverse file restores every legacy title. With
them, the file checks of Q05.
"""

from __future__ import annotations

import csv
import dataclasses
import shutil
from pathlib import Path
from typing import NamedTuple

import pytest

from titlestd import catalog, close, datafiles, people, prepare, release
from titlestd.contracts import (
    LABEL_STAND_IN,
    LOOKUP_DECIDED_BY,
    Decision,
    DecisionSource,
    ExceptionReason,
    HumanAnswer,
    InputData,
    Leftover,
    QueueEntry,
    Reason,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
)
from titlestd.release import ReleaseError
from titlestd.settings import Settings

FLAGGED = frozenset({TitleKey("FIN", "internal auditor")})  # the title of flagged role FIN-04
# Stand-ins for the two machine paths: (table row, role, status, error rate).
MACHINE = {
    TitleKey("FIN", "senior financial analyst"): (
        TableRow.MODELS_AGREE,
        "FIN-01",
        Status.CONFIRMED,
        None,
    ),
    TitleKey("SAL", "sales operations analyst"): (
        TableRow.AGENT_SETTLED,
        "SAL-03",
        Status.PROVISIONAL,
        6 / 60,
    ),
}
STATUSES = {status.value for status in Status}
RELEASED_ROWS = {row.value for row in TableRow if row is not TableRow.LEGACY}
LEGACY_REASONS = {
    reason.value for reason in ExceptionReason if reason is not ExceptionReason.NO_GRADE
}


class Run(NamedTuple):
    inputs: InputData
    rows: tuple[TitleRow, ...]
    decisions: tuple[Decision, ...]
    queue: tuple[QueueEntry, ...]
    settings: Settings


def _run(data_dir: Path, out_dir: Path, **overrides) -> Run:
    """A run without models, from the data files to the four output files."""
    values = {"sme_error_rate": 0.0, "undecided_rate": 0.85, "generic_cap_share": 0.1}
    settings = Settings(data_dir=data_dir, out_dir=out_dir, wave1_size=4, **{**values, **overrides})
    inputs = datafiles.read_inputs(settings.data_dir)
    rows = prepare.prepare(
        inputs.employees, inputs.abbreviations, catalog.family_codes(inputs.catalog)
    )
    simulated = people.load_people(settings, inputs.catalog)
    seed = {
        prepare.title_key(row.job_title, row.job_family_code, inputs.abbreviations): row.role_id
        for row in inputs.seed_lookup
    }
    by_headcount = sorted(rows, key=lambda row: (-row.headcount, row.key))
    wave1 = {row.key for row in by_headcount[: settings.wave1_size]}

    decisions, leftovers, asked = [], [], {}
    for row in rows:
        if row.key in seed:  # Q13: the look-up runs first
            decisions.append(
                Decision(
                    row.key,
                    TableRow.LOOKUP,
                    DecisionSource.LOOKUP,
                    seed[row.key],
                    Status.APPROVED,
                    LOOKUP_DECIDED_BY,
                )
            )
        elif row.key in MACHINE:
            table_row, role_id, status, rate = MACHINE[row.key]
            decisions.append(
                Decision(
                    row.key, table_row, DecisionSource.MACHINE, role_id, status, error_rate=rate
                )
            )
        else:
            on_list = row.key in wave1
            human = simulated.wave1_approval(row) if on_list else simulated.queue_decision(row)
            if not on_list:
                asked[row.key] = human
            if human.answer is HumanAnswer.ROLE:
                decisions.append(
                    Decision(
                        row.key,
                        TableRow.WAVE1 if on_list else TableRow.REVIEW_QUEUE,
                        DecisionSource.PERSON,
                        human.role_id,
                        Status.APPROVED,
                        human.decided_by,
                        human.simulated,
                    )
                )
            else:
                leftovers.append(
                    Leftover(
                        row,
                        on_wave1_list=on_list,
                        compliance_fired=row.key in FLAGGED,
                        no_suitable_role=human.answer is HumanAnswer.NO_SUITABLE_ROLE,
                    )
                )
    decisions += close.close_leftovers(leftovers, len(inputs.employees), inputs.catalog, settings)
    decisions.sort(key=lambda decision: decision.key)

    outcome = {decision.key: decision.row for decision in decisions}
    queue = tuple(
        QueueEntry(
            row=row,
            reasons=(Reason.NO_USABLE_FAMILY,) if not row.family_usable else (Reason.NO_MATCH,),
            ml_first_choice=None,
            llm_pick=None,
            agent_recommendation=None,
            answer_source=LABEL_STAND_IN if row.model_eligible else "",
            decision=asked[row.key],
            outcome=outcome[row.key],
        )
        for row in rows
        if row.key in asked
    )
    release.write_release_files(settings.out_dir, decisions, rows, inputs.employees, inputs.catalog)
    release.write_review_queue(settings.out_dir, queue)
    return Run(inputs, rows, tuple(decisions), queue, settings)


def _files(out_dir: Path):
    """The load, reverse and exceptions rows, read back with the reader of the data files."""
    return (
        datafiles.read_table(out_dir / release.LOAD_FILE, release.LOAD_COLUMNS),
        datafiles.read_table(out_dir / release.REVERSE_FILE, release.REVERSE_COLUMNS),
        datafiles.read_table(out_dir / release.EXCEPTIONS_FILE, release.EXCEPTIONS_COLUMNS),
    )


def _check_the_files(run: Run) -> None:
    """Every file check of Q05, and the two section 2.7 tests, on one run's files."""
    mini = run.inputs.catalog
    load, reverse, exceptions = _files(run.settings.out_dir)
    employees = {employee.employee_id: employee for employee in run.inputs.employees}
    decision_of = {
        employee_id: decision
        for row in run.rows
        for decision in run.decisions
        if decision.key == row.key
        for employee_id in row.employee_ids
    }
    assert set(decision_of) == set(employees)

    # Sorted by employee_id; each employee at most once in a file.
    for lines in (load, reverse, exceptions):
        ids = [line["employee_id"] for line in lines]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)

    # load.csv plus the released = 0 exceptions cover every employee exactly once.
    loaded = [line["employee_id"] for line in load]
    unreleased = [line["employee_id"] for line in exceptions if line["released"] == "0"]
    assert sorted(loaded + unreleased) == sorted(employees)
    assert set(loaded).isdisjoint(unreleased)
    assert unreleased == sorted(
        employee_id for employee_id, d in decision_of.items() if d.row is TableRow.LEGACY
    )

    # 2.7: no role or title outside the catalog is in the load file.
    for line in load:
        decision, employee = decision_of[line["employee_id"]], employees[line["employee_id"]]
        role = catalog.get_role(mini, line["role_id"])
        assert role is not None and role.role_id == decision.role_id
        assert catalog.is_catalog_title(mini, line["long_title"], line["short_title"])
        # Level and titles are the employee's own, from the grade (1.4).
        title = catalog.standard_title(mini, role, employee.grade)
        assert (line["long_title"], line["short_title"]) == (title.long_title, title.short_title)
        assert line["level"] == (catalog.level_for_grade(mini, employee.grade) or "")
        assert line["no_grade"] == ("1" if employee.grade is None else "0")
        assert (line["level"] == "") is (line["no_grade"] == "1")
        # Status, path, who decided, the simulated mark, the error rate.
        assert line["status"] == decision.status.value and line["status"] in STATUSES
        assert line["path"] == decision.row.value and line["path"] in RELEASED_ROWS
        assert bool(line["decided_by"]) is (line["status"] == "Approved")
        assert line["simulated"] == ("1" if decision.source is DecisionSource.PERSON else "0")
        if line["path"] == TableRow.LOOKUP.value:
            assert (line["decided_by"], line["simulated"]) == (LOOKUP_DECIDED_BY, "0")
        # The rate is on every Provisional row of a machine path and on no other (1.9, Q05).
        on_a_path = line["path"] in ("models agree", "agent-settled")
        assert bool(line["error_rate"]) is (line["status"] == "Provisional" and on_a_path)
        if line["error_rate"]:
            assert line["error_rate"] == f"{decision.error_rate:.4f}"
        if line["path"] == TableRow.GENERIC.value:
            assert role.is_generic and line["status"] == "Provisional" and not line["error_rate"]
        else:
            assert not role.is_generic

    # 2.7: applying the reverse file to the load file restores every legacy title.
    assert [line["employee_id"] for line in reverse] == loaded
    restored = release.apply_reverse(load, reverse)
    assert restored == {employee_id: employees[employee_id].job_title for employee_id in loaded}

    # The exceptions list: who stays on a legacy title and why, and who has no grade.
    for line in exceptions:
        employee = employees[line["employee_id"]]
        assert line["legacy_job_title"] == employee.job_title
        assert line["job_family_code"] == employee.job_family_code
        if line["released"] == "0":
            assert line["reason"] == decision_of[employee.employee_id].legacy_reason.value
            assert line["reason"] in LEGACY_REASONS
        else:
            assert (line["released"], line["reason"]) == ("1", "no grade")
    assert [line["employee_id"] for line in exceptions if line["released"] == "1"] == [
        line["employee_id"] for line in load if line["no_grade"] == "1"
    ]


@pytest.fixture(scope="module")
def run(mini_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Run:
    return _run(mini_dir, tmp_path_factory.mktemp("release") / "out")


def test_the_miniature_run_fills_every_row_of_the_table(run: Run, mini_rows):
    assert run.rows == mini_rows
    counted = {}
    for decision in run.decisions:
        counted.setdefault(decision.row, []).append(decision.key.cleaned_title)
    assert {row: len(titles) for row, titles in counted.items()} == {
        TableRow.LOOKUP: 2,
        TableRow.WAVE1: 4,
        TableRow.MODELS_AGREE: 1,
        TableRow.AGENT_SETTLED: 1,
        TableRow.REVIEW_QUEUE: 1,
        TableRow.GENERIC: 2,
        TableRow.LEGACY: 4,
    }
    assert counted[TableRow.REVIEW_QUEUE] == ["special projects"]
    assert counted[TableRow.GENERIC] == ["payroll specialist", "treasury analyst"]
    reasons = {d.key.cleaned_title: d.legacy_reason for d in run.decisions if d.legacy_reason}
    assert reasons == {
        "staff accountant": ExceptionReason.UNDECIDED_UNSCORED,
        "internal auditor": ExceptionReason.UNDECIDED_BARRED,
        "bdr": ExceptionReason.OVER_GENERIC_CAP,
        "customer success manager [NAME]": ExceptionReason.UNDECIDED_UNSCORED,
    }
    # The cap: a tenth of 22 employees is 2, and two employees hold a generic role.
    assert close.generic_cap(22, run.settings) == 2


def test_the_run_writes_the_four_files(run: Run):
    assert sorted(path.name for path in run.settings.out_dir.iterdir()) == [
        "exceptions.csv",
        "load.csv",
        "reverse.csv",
        "review_queue.csv",
    ]
    load, reverse, exceptions = _files(run.settings.out_dir)
    assert (len(load), len(reverse), len(exceptions)) == (18, 18, 5)


def test_the_files_of_the_run_pass_every_check(run: Run):
    _check_the_files(run)


def test_the_worked_example_reaches_the_load_file(run: Run):
    # 1.3: "ACCT MGR II" (grade 8) and "Senior Account Manager" (grade 7) are
    # one role and one level, and so is "Sr. Acct Mgr - NYC".
    load = {line["employee_id"]: line for line in _files(run.settings.out_dir)[0]}
    for employee_id in ("E001", "E004", "E006"):
        line = load[employee_id]
        assert (line["role_id"], line["long_title"], line["short_title"], line["level"]) == (
            "SAL-01",
            "Senior Account Manager",
            "Sr Acct Mgr",
            "Senior",
        )
        assert (line["status"], line["path"], line["simulated"]) == (
            "Approved",
            "approved in Wave 1",
            "1",
        )
        assert line["decided_by"] in people.PERSONA_SMES
    restored = release.apply_reverse(*_files(run.settings.out_dir)[:2])
    assert [restored[employee_id] for employee_id in ("E001", "E004", "E006")] == [
        "ACCT MGR II",
        "Senior Account Manager",
        "Sr. Acct Mgr - NYC",
    ]


def test_each_kind_of_decision_is_stated_in_the_load_file(run: Run):
    load = {line["employee_id"]: line for line in _files(run.settings.out_dir)[0]}
    shown = {
        employee_id: tuple(
            load[employee_id][column]
            for column in ("long_title", "status", "path", "error_rate", "decided_by")
        )
        for employee_id in ("E013", "E016", "E010", "E022", "E019", "E020")
    }
    assert shown == {
        "E013": ("Associate Sales Engineer", "Approved", "look-up hit", "", LOOKUP_DECIDED_BY),
        "E016": ("Senior Financial Analyst", "Confirmed", "models agree", "", ""),
        "E010": (
            "Associate Sales Operations Analyst",
            "Provisional",
            "agent-settled",
            "0.1000",
            "",
        ),
        # Decided by the reviewer; no grade, so the title has no level word.
        "E022": ("Financial Analyst", "Approved", "review queue", "", people.PERSONA_REVIEWER),
        # The generic role of the family, at each employee's own level. E020 is
        # recorded in family "FIN ", which is family FIN (S05).
        "E019": ("Junior Finance", "Provisional", "generic family role", "", ""),
        "E020": ("Principal Finance", "Provisional", "generic family role", "", ""),
    }
    assert (load["E019"]["role_id"], load["E019"]["short_title"]) == ("FIN-GEN", "Jr Finance")


def test_the_exceptions_list_of_the_run(run: Run):
    exceptions = _files(run.settings.out_dir)[2]
    assert [
        (line["employee_id"], line["job_family_code"], line["reason"], line["released"])
        for line in exceptions
    ] == [
        ("E011", "SAL", "undecided title with a person's name or no usable family", "0"),
        # No grade and left on the legacy title: listed once, as not released (Q05).
        ("E012", "SAL", "over the generic cap", "0"),
        ("E018", "FIN", "undecided widely held or flagged title", "0"),
        ("E021", "", "undecided title with a person's name or no usable family", "0"),
        ("E022", "FIN", "no grade", "1"),
    ]
    # The name in a legacy title is in the exceptions list as received: it is
    # the employer's own record going back to the employer, not a model input.
    assert exceptions[0]["legacy_job_title"] == "Cust Success Mgr (Tavi Quillfeather)"


def test_applying_the_reverse_file_restores_every_legacy_title_of_the_run(run: Run, mini_dir: Path):
    # 2.7, as Q05 spells it out: the restored title equals the employees
    # file's job_title for every load row, and both files hold the same ids.
    load, reverse, exceptions = _files(run.settings.out_dir)
    received = {e.employee_id: e.job_title for e in datafiles.read_employees(mini_dir)}
    restored = release.apply_reverse(load, reverse)
    assert list(restored) == [line["employee_id"] for line in load]
    assert sorted(restored) == sorted(line["employee_id"] for line in reverse)
    for employee_id, title in restored.items():
        assert title == received[employee_id]
    # With the employees who never left their legacy title, that is everybody:
    # after the reverse file is applied the whole employer is where it started.
    kept = {
        line["employee_id"]: line["legacy_job_title"]
        for line in exceptions
        if line["released"] == "0"
    }
    assert len(restored) + len(kept) == 22
    assert {**restored, **kept} == received
    # The load file changed something for the reverse file to undo.
    changed = [line for line in load if line["long_title"] != restored[line["employee_id"]]]
    assert len(changed) == 15


def test_the_reverse_file_holds_the_titles_of_the_employees_file_byte_for_byte(
    run: Run, mini_dir: Path
):
    # Read the employees file with the csv module alone, to compare with what
    # the file holds and not with what a reader of this package returns.
    with (mini_dir / "employees.csv").open(encoding="utf-8", newline="") as handle:
        received = {line["employee_id"]: line["job_title"] for line in csv.DictReader(handle)}
    with (run.settings.out_dir / "reverse.csv").open(encoding="utf-8", newline="") as handle:
        reverse = {line["employee_id"]: line["legacy_job_title"] for line in csv.DictReader(handle)}
    assert len(reverse) == 18
    for employee_id, title in reverse.items():
        assert title.encode() == received[employee_id].encode()
    assert reverse["E006"] == "Sr. Acct Mgr - NYC" and reverse["E013"] == "Sales Engineer #20311"


def test_the_review_queue_file_of_the_run(run: Run):
    rows = datafiles.read_table(
        run.settings.out_dir / release.REVIEW_QUEUE_FILE, release.REVIEW_QUEUE_COLUMNS
    )
    keys = [TitleKey(row["job_family_code"], row["cleaned_title"]) for row in rows]
    assert keys == sorted(entry.row.key for entry in run.queue) and len(keys) == 7
    by_title = {row["cleaned_title"]: row for row in rows}
    assert {title: row["decision"] for title, row in by_title.items()} == {
        "staff accountant": "could not decide",
        "internal auditor": "could not decide",
        "payroll specialist": "could not decide",
        "special projects": "FIN-01",
        "treasury analyst": "could not decide",
        "bdr": "could not decide",
        "customer success manager [NAME]": "could not decide",
    }
    assert {title: row["outcome"] for title, row in by_title.items()} == {
        "staff accountant": "left on legacy title",
        "internal auditor": "left on legacy title",
        "payroll specialist": "generic family role",
        "special projects": "review queue",
        "treasury analyst": "generic family role",
        "bdr": "left on legacy title",
        "customer success manager [NAME]": "left on legacy title",
    }
    for row in rows:
        # People were simulated, and the file says so on every decision (Q16, Q27).
        assert (row["decided_by"], row["simulated"]) == (people.PERSONA_REVIEWER, "1")
        assert row["headcount"] == "1"
    assert by_title["staff accountant"]["reasons"] == "no usable family code"
    assert by_title["staff accountant"]["job_family_code"] == ""


def test_two_runs_write_the_same_bytes(run: Run, mini_dir: Path, tmp_path: Path):
    # Q06: no output holds the wall-clock time or depends on anything but the inputs.
    again = _run(mini_dir, tmp_path / "again")
    names = sorted(path.name for path in again.settings.out_dir.iterdir())
    assert len(names) == 4
    for name in names:
        assert (again.settings.out_dir / name).read_bytes() == (
            run.settings.out_dir / name
        ).read_bytes()


def test_the_order_of_the_employees_file_does_not_change_the_files(
    run: Run, mini_dir: Path, tmp_path: Path
):
    data_dir = tmp_path / "data"
    shutil.copytree(mini_dir, data_dir)
    datafiles.write_employees(data_dir, run.inputs.employees[::-1])
    again = _run(data_dir, tmp_path / "out")
    for path in again.settings.out_dir.iterdir():
        assert path.read_bytes() == (run.settings.out_dir / path.name).read_bytes()


@pytest.mark.parametrize("share", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_the_files_pass_every_check_whoever_cannot_decide(
    mini_dir: Path, tmp_path: Path, rate: float, share: float
):
    for seed in range(5):
        out_dir = tmp_path / f"out-{seed}"
        _check_the_files(
            _run(mini_dir, out_dir, seed=seed, undecided_rate=rate, generic_cap_share=share)
        )


def test_simulated_mistakes_still_give_catalog_titles(mini_dir: Path, tmp_path: Path):
    # 2.4: a wrong label is another role of the catalog, so the release still passes.
    run = _run(mini_dir, tmp_path / "out", sme_error_rate=1.0, undecided_rate=0.0)
    _check_the_files(run)
    truth = {
        (row.job_title, row.job_family_code.strip()): row.true_role_id
        for row in people.read_truth(mini_dir)
    }
    by_person = [d for d in run.decisions if d.source is DecisionSource.PERSON]
    assert len(by_person) == 11
    rows = {row.key: row for row in run.rows}
    for decision in by_person:
        true_roles = {
            truth[(raw, decision.key.family_code)] for raw in rows[decision.key].raw_titles
        }
        assert decision.role_id not in true_roles


def test_a_title_with_no_truth_row_is_left_on_its_legacy_title(mini_dir: Path, tmp_path: Path):
    # Q03, through the files: the title waits for the catalog owner.
    data_dir = tmp_path / "data"
    shutil.copytree(mini_dir, data_dir)
    people.write_truth(
        data_dir, [row for row in people.read_truth(data_dir) if row.job_title != "BDR"]
    )
    run = _run(data_dir, tmp_path / "out", undecided_rate=0.0, generic_cap_share=1.0)
    _check_the_files(run)
    exceptions = _files(run.settings.out_dir)[2]
    assert [
        (line["employee_id"], line["reason"], line["released"])
        for line in exceptions
        if line["released"] == "0"
    ] == [("E012", "no suitable role", "0")]
    queue = datafiles.read_table(
        run.settings.out_dir / release.REVIEW_QUEUE_FILE, release.REVIEW_QUEUE_COLUMNS
    )
    assert [
        (row["reasons"], row["decision"], row["outcome"])
        for row in queue
        if row["cleaned_title"] == "bdr"
    ] == [
        (
            "no match; no suitable role, waiting for catalog owner",
            "no suitable role",
            "left on legacy title",
        )
    ]


def test_a_decision_outside_the_catalog_stops_the_release_of_the_run(run: Run, tmp_path: Path):
    # 2.7 on the real rows: one role the catalog does not hold, and no file is written.
    decisions = [
        dataclasses.replace(decision, role_id="FIN-99")
        if decision.key == TitleKey("FIN", "financial analyst")
        else decision
        for decision in run.decisions
    ]
    with pytest.raises(ReleaseError, match="employee E014: role FIN-99 is not in the catalog"):
        release.write_release_files(
            tmp_path / "out", decisions, run.rows, run.inputs.employees, run.inputs.catalog
        )
    assert not (tmp_path / "out").exists()


def test_a_legacy_title_is_never_released_as_it_stands(run: Run, tmp_path: Path):
    # A role id is the only way into the load file. A decision that carries the
    # legacy title where the role id belongs is refused, even where the legacy
    # title is, word for word, a title of the catalog.
    assert catalog.is_catalog_title(run.inputs.catalog, "Treasury Analyst", "Treasury Analyst")
    decisions = [
        dataclasses.replace(
            decision,
            row=TableRow.MODELS_AGREE,
            source=DecisionSource.MACHINE,
            role_id="Treasury Analyst",
            status=Status.CONFIRMED,
        )
        if decision.key == TitleKey("FIN", "treasury analyst")
        else decision
        for decision in run.decisions
    ]
    with pytest.raises(ReleaseError, match="role Treasury Analyst is not in the catalog"):
        release.write_release_files(
            tmp_path / "out", decisions, run.rows, run.inputs.employees, run.inputs.catalog
        )
    assert not (tmp_path / "out").exists()
