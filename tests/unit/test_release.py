"""Release: the load file, the reverse file, the exceptions list, the review queue.

Two tests of section 2.7 belong to this module. The writer refuses a role or a
title that is not in the catalog, so none can reach the load file; and applying
the reverse file to the load file restores every legacy title. Beside them, the
file checks of Q05 and the review-queue file of Q16 and S12.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from titlestd import catalog, close, datafiles, release
from titlestd.contracts import (
    LOOKUP_DECIDED_BY,
    Catalog,
    Decision,
    DecisionSource,
    Employee,
    ExceptionReason,
    HumanAnswer,
    HumanDecision,
    QueueEntry,
    Reason,
    Record,
    StandardTitle,
    Status,
    TableRow,
    TitleRow,
    TitlestdError,
)
from titlestd.release import ReleaseError

SME = "SME 2 (simulated)"
REVIEWER = "Reviewer (simulated)"


def _employee(employee_id: str, title: str, family: str, grade: int | None) -> Employee:
    return Employee(employee_id, title, family, "Some Department", grade, "")


def _row(cleaned: str, *employees: Employee) -> TitleRow:
    """The title row of employees who share a family: what the prepare step would build."""
    family = employees[0].job_family_code.strip()
    return TitleRow(
        record=Record(cleaned, family, "Some Department", ""),
        employee_ids=tuple(sorted(employee.employee_id for employee in employees)),
        raw_titles=tuple(sorted({employee.job_title for employee in employees})),
    )


def _by_person(row: TitleRow, table_row: TableRow, role_id: str, persona: str = SME) -> Decision:
    return Decision(
        row.key, table_row, DecisionSource.PERSON, role_id, Status.APPROVED, persona, True
    )


def _by_machine(
    row: TitleRow, table_row: TableRow, role_id: str, status: Status, rate: float | None = None
) -> Decision:
    return Decision(row.key, table_row, DecisionSource.MACHINE, role_id, status, error_rate=rate)


def _on_legacy(row: TitleRow, reason: ExceptionReason) -> Decision:
    return Decision(row.key, TableRow.LEGACY, DecisionSource.NONE, legacy_reason=reason)


# One small release with every kind of decision. The catalog is the mini one.
EMPLOYEES = (
    _employee("A1", "Sr. Acct Mgr - NYC", "SAL", 7),
    _employee("A2", "Senior Account Manager", "SAL", 8),
    _employee("A3", "Senior Account Manager", "SAL", 2),
    _employee("B1", "BDR", "SAL", None),
    _employee("C1", "Sales Engineer #20311", "SAL", 6),
    _employee("D1", "Fin Analyst", "FIN", 4),
    _employee("D2", "Fin Analyst", "FIN", 9),
    _employee("E1", "Special Projects", "FIN", None),
    _employee("F1", "Staff Accountant", "", 5),
    _employee("G1", "Payroll Spec", "FIN", None),
    _employee("H1", "Treasury Analyst", "FIN ", 10),
)
_BY_ID = {employee.employee_id: employee for employee in EMPLOYEES}
ACCOUNT = _row("senior account manager", _BY_ID["A1"], _BY_ID["A2"], _BY_ID["A3"])
BDR = _row("bdr", _BY_ID["B1"])
ENGINEER = _row("sales engineer", _BY_ID["C1"])
ANALYST = _row("financial analyst", _BY_ID["D1"], _BY_ID["D2"])
PROJECTS = _row("special projects", _BY_ID["E1"])
STAFF = dataclasses.replace(_row("staff accountant", _BY_ID["F1"]), family_usable=False)
PAYROLL = _row("payroll specialist", _BY_ID["G1"])
TREASURY = _row("treasury analyst", _BY_ID["H1"])
ROWS = (ACCOUNT, BDR, ENGINEER, ANALYST, PROJECTS, STAFF, PAYROLL, TREASURY)
DECISIONS = (
    _by_person(ACCOUNT, TableRow.WAVE1, "SAL-01"),
    _by_machine(BDR, TableRow.MODELS_AGREE, "SAL-05", Status.CONFIRMED),
    Decision(
        ENGINEER.key,
        TableRow.LOOKUP,
        DecisionSource.LOOKUP,
        "SAL-04",
        Status.APPROVED,
        LOOKUP_DECIDED_BY,
    ),
    _by_machine(ANALYST, TableRow.AGENT_SETTLED, "FIN-01", Status.PROVISIONAL, 5 / 60),
    Decision(PROJECTS.key, TableRow.GENERIC, DecisionSource.CODE, "FIN-GEN", Status.PROVISIONAL),
    _on_legacy(STAFF, ExceptionReason.UNDECIDED_UNSCORED),
    _on_legacy(PAYROLL, ExceptionReason.OVER_GENERIC_CAP),
    _by_person(TREASURY, TableRow.REVIEW_QUEUE, "FIN-06", REVIEWER),
)

# The load file of that release, in two halves so that the lines stay readable.
LOAD_TITLES = """
A1,SAL-01,Senior Account Manager,Sr Acct Mgr,Senior,0
A2,SAL-01,Senior Account Manager,Sr Acct Mgr,Senior,0
A3,SAL-01,Junior Account Manager,Jr Acct Mgr,Junior,0
B1,SAL-05,Business Development Representative,Biz Dev Rep,,1
C1,SAL-04,Associate Sales Engineer,Assoc Sales Eng,Associate,0
D1,FIN-01,Associate Financial Analyst,Assoc Fin Analyst,Associate,0
D2,FIN-01,Lead Financial Analyst,Lead Fin Analyst,Lead,0
E1,FIN-GEN,Finance,Finance,,1
H1,FIN-06,Principal Treasury Analyst,Prin Treasury Analyst,Principal,0
"""
LOAD_STATUS = """
A1,Approved,approved in Wave 1,,SME 2 (simulated),1
A2,Approved,approved in Wave 1,,SME 2 (simulated),1
A3,Approved,approved in Wave 1,,SME 2 (simulated),1
B1,Confirmed,models agree,,,0
C1,Approved,look-up hit,,Seed look-up (mock data),0
D1,Provisional,agent-settled,0.0833,,0
D2,Provisional,agent-settled,0.0833,,0
E1,Provisional,generic family role,,,0
H1,Approved,review queue,,Reviewer (simulated),1
"""
REVERSE_TEXT = """\
employee_id,legacy_job_title
A1,Sr. Acct Mgr - NYC
A2,Senior Account Manager
A3,Senior Account Manager
B1,BDR
C1,Sales Engineer #20311
D1,Fin Analyst
D2,Fin Analyst
E1,Special Projects
H1,Treasury Analyst
"""
EXCEPTIONS_TEXT = """\
employee_id,legacy_job_title,job_family_code,reason,released
B1,BDR,SAL,no grade,1
E1,Special Projects,FIN,no grade,1
F1,Staff Accountant,,undecided title with a person's name or no usable family,0
G1,Payroll Spec,FIN,over the generic cap,0
"""


def _columns(rows: list[dict[str, str]], *columns: str) -> str:
    """The named columns of the rows, one comma-joined line per row."""
    lines = [",".join(row[column] for column in columns) for row in rows]
    return "\n" + "\n".join(lines) + "\n"


def _read(out: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """The three files as written: load, reverse and exceptions rows."""
    return (
        datafiles.read_table(out / release.LOAD_FILE, release.LOAD_COLUMNS),
        datafiles.read_table(out / release.REVERSE_FILE, release.REVERSE_COLUMNS),
        datafiles.read_table(out / release.EXCEPTIONS_FILE, release.EXCEPTIONS_COLUMNS),
    )


def _release_one(
    tmp_path: Path, mini_catalog: Catalog, decision_for, *employees: Employee
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Release the employees as one title row whose decision decision_for(row) gives."""
    row = _row("some title", *employees)
    release.write_release_files(tmp_path, [decision_for(row)], [row], employees, mini_catalog)
    return _read(tmp_path)


def _confirmed(role_id: str):
    return lambda row: _by_machine(row, TableRow.MODELS_AGREE, role_id, Status.CONFIRMED)


@pytest.fixture
def out(tmp_path: Path, mini_catalog: Catalog) -> Path:
    """The folder the small release was written into."""
    release.write_release_files(tmp_path / "out", DECISIONS, ROWS, EMPLOYEES, mini_catalog)
    return tmp_path / "out"


# --- The files and their columns (Q05, S12) -----------------------------------------


def test_the_file_names_and_columns_are_those_of_the_record():
    assert (release.LOAD_FILE, release.REVERSE_FILE) == ("load.csv", "reverse.csv")
    assert release.EXCEPTIONS_FILE == "exceptions.csv"
    assert release.REVIEW_QUEUE_FILE == "review_queue.csv"
    assert release.LOAD_COLUMNS == (
        "employee_id",
        "role_id",
        "long_title",
        "short_title",
        "level",
        "status",
        "path",
        "error_rate",
        "no_grade",
        "decided_by",
        "simulated",
    )
    assert release.REVERSE_COLUMNS == ("employee_id", "legacy_job_title")
    assert release.EXCEPTIONS_COLUMNS == (
        "employee_id",
        "legacy_job_title",
        "job_family_code",
        "reason",
        "released",
    )
    assert release.REVIEW_QUEUE_COLUMNS == (
        "cleaned_title",
        "job_family_code",
        "headcount",
        "reasons",
        "ml_first_choice",
        "llm_pick",
        "agent_recommendation",
        "answer_source",
        "decision",
        "decided_by",
        "simulated",
        "outcome",
    )


def test_a_small_release_writes_these_three_files(out: Path):
    assert sorted(path.name for path in out.iterdir()) == [
        "exceptions.csv",
        "load.csv",
        "reverse.csv",
    ]
    load = datafiles.read_table(out / "load.csv", release.LOAD_COLUMNS)
    titles = ("employee_id", "role_id", "long_title", "short_title", "level", "no_grade")
    status = ("employee_id", "status", "path", "error_rate", "decided_by", "simulated")
    assert set(titles) | set(status) == set(release.LOAD_COLUMNS)
    assert _columns(load, *titles) == LOAD_TITLES
    assert _columns(load, *status) == LOAD_STATUS
    assert (out / "reverse.csv").read_text(encoding="utf-8") == REVERSE_TEXT
    assert (out / "exceptions.csv").read_text(encoding="utf-8") == EXCEPTIONS_TEXT


def test_the_files_are_utf8_csv_with_a_header_row_and_line_feeds(out: Path):
    # Q05, S01: the format of the data files, so the same reader reads them back.
    for name, columns in (
        (release.LOAD_FILE, release.LOAD_COLUMNS),
        (release.REVERSE_FILE, release.REVERSE_COLUMNS),
        (release.EXCEPTIONS_FILE, release.EXCEPTIONS_COLUMNS),
    ):
        content = (out / name).read_bytes()
        lines = content.decode("utf-8").split("\n")
        assert lines[0] == ",".join(columns)
        assert lines[-1] == "" and b"\r" not in content
        assert len(datafiles.read_table(out / name, columns)) == len(lines) - 2
    first = (out / "load.csv").read_text(encoding="utf-8").splitlines()[1]
    assert first.startswith("A1,SAL-01,Senior Account Manager,Sr Acct Mgr,Senior,Approved,")
    assert first.endswith(",approved in Wave 1,,0,SME 2 (simulated),1")


def test_the_order_of_the_arguments_does_not_change_a_byte(
    out: Path, tmp_path: Path, mini_catalog: Catalog
):
    again = tmp_path / "again"
    release.write_release_files(again, DECISIONS[::-1], ROWS[::-1], EMPLOYEES[::-1], mini_catalog)
    for name in (release.LOAD_FILE, release.REVERSE_FILE, release.EXCEPTIONS_FILE):
        assert (again / name).read_bytes() == (out / name).read_bytes()


@pytest.mark.parametrize("read_once", ["decisions", "rows", "employees"])
def test_an_argument_that_can_be_read_only_once_gives_the_same_files(
    out: Path, tmp_path: Path, mini_catalog: Catalog, read_once: str
):
    # The writer reads its title rows and employees twice. An iterator must not
    # pass the checks and then leave three files with a header and no line.
    arguments = {"decisions": DECISIONS, "rows": ROWS, "employees": EMPLOYEES}
    arguments[read_once] = iter(arguments[read_once])
    again = tmp_path / "again"
    release.write_release_files(again, catalog=mini_catalog, **arguments)
    for name in (release.LOAD_FILE, release.REVERSE_FILE, release.EXCEPTIONS_FILE):
        assert (again / name).read_bytes() == (out / name).read_bytes()


def test_rows_are_sorted_by_employee_id(out: Path):
    for rows in _read(out):
        ids = [row["employee_id"] for row in rows]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_writing_again_replaces_the_files(out: Path, mini_catalog: Catalog):
    # A run starts from fresh outputs (Q06): nothing of the earlier release stays.
    row = _row("bdr", _BY_ID["B1"])
    release.write_release_files(
        out, [_on_legacy(row, ExceptionReason.STOPPED_PATH)], [row], [_BY_ID["B1"]], mini_catalog
    )
    load, reverse, exceptions = _read(out)
    assert load == [] and reverse == []
    assert [line["employee_id"] for line in exceptions] == ["B1"]


def test_no_output_can_hold_the_time_of_day():
    # Q06: two runs give the same bytes. Neither module can read a clock.
    for module in (release, close):
        imported = set()
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").partition(".")[0])
        assert imported.isdisjoint({"time", "datetime"})


# --- The load file (Q05) ------------------------------------------------------------


def test_the_load_file_has_one_row_per_employee_who_receives_a_new_title(out: Path):
    load, _, _ = _read(out)
    legacy = {"F1", "G1"}
    assert [row["employee_id"] for row in load] == sorted(set(_BY_ID) - legacy)


def test_level_and_titles_are_applied_per_employee_from_the_grade(out: Path):
    # 1.4: the decision is the role; one title row, three employees, two levels.
    load = {row["employee_id"]: row for row in _read(out)[0]}
    assert [load[i]["role_id"] for i in ("A1", "A2", "A3")] == ["SAL-01"] * 3
    assert [load[i]["level"] for i in ("A1", "A2", "A3")] == ["Senior", "Senior", "Junior"]
    # 1.3: grades 7 and 8 both read as Senior.
    for employee_id in ("A1", "A2"):
        assert load[employee_id]["long_title"] == "Senior Account Manager"
        assert load[employee_id]["short_title"] == "Sr Acct Mgr"
    assert (load["A3"]["long_title"], load["A3"]["short_title"]) == (
        "Junior Account Manager",
        "Jr Acct Mgr",
    )


def test_status_is_one_of_three_and_path_is_the_table_row(out: Path):
    load = _read(out)[0]
    assert {row["status"] for row in load} == {"Approved", "Confirmed", "Provisional"}
    released_rows = [row.value for row in TableRow if row is not TableRow.LEGACY]
    assert {row["path"] for row in load} == set(released_rows)
    by_key = {row.key: row for row in ROWS}
    for decision in DECISIONS:
        for employee_id in by_key[decision.key].employee_ids:
            lines = [row for row in load if row["employee_id"] == employee_id]
            if decision.row is TableRow.LEGACY:
                assert lines == []
            else:
                assert [(line["status"], line["path"]) for line in lines] == [
                    (decision.status.value, decision.row.value)
                ]


def test_every_approved_row_says_who_decided_it(out: Path):
    # 1.9: Approved means a named person decided it. S11: a look-up hit carries
    # the fixed label instead of a persona.
    load = {row["employee_id"]: row for row in _read(out)[0]}
    for row in load.values():
        assert bool(row["decided_by"]) is (row["status"] == "Approved")
    assert load["A1"]["decided_by"] == SME
    assert load["H1"]["decided_by"] == REVIEWER
    assert load["C1"]["decided_by"] == LOOKUP_DECIDED_BY == "Seed look-up (mock data)"


def test_the_simulated_mark_is_on_every_simulated_persons_decision_and_nothing_else(out: Path):
    # 2.4, Q27. The seed look-up is input data, not a decision simulated in the run.
    load = {row["employee_id"]: row["simulated"] for row in _read(out)[0]}
    assert load == {
        "A1": "1",
        "A2": "1",
        "A3": "1",
        "B1": "0",
        "C1": "0",
        "D1": "0",
        "D2": "0",
        "E1": "0",
        "H1": "1",
    }


@pytest.mark.parametrize(
    ("wrong", "written"), [(5, "0.0833"), (3, "0.0500"), (12, "0.2000"), (13, "0.2167")]
)
def test_the_error_rate_of_a_provisional_path_has_four_decimal_places(
    tmp_path: Path, mini_catalog: Catalog, wrong: int, written: str
):
    def provisional(row: TitleRow) -> Decision:
        return _by_machine(row, TableRow.MODELS_AGREE, "SAL-02", Status.PROVISIONAL, wrong / 60)

    load, _, _ = _release_one(tmp_path, mini_catalog, provisional, _employee("X1", "Rep", "SAL", 5))
    assert [row["error_rate"] for row in load] == [written]


def test_the_error_rate_is_filled_on_the_provisional_rows_of_a_machine_path_and_no_other(
    out: Path,
):
    load = _read(out)[0]
    filled = [row["employee_id"] for row in load if row["error_rate"]]
    assert filled == ["D1", "D2"]
    for row in load:
        on_a_path = row["path"] in ("models agree", "agent-settled")
        assert bool(row["error_rate"]) is (row["status"] == "Provisional" and on_a_path)
    # A generic family role is Provisional too, and has no path to take a rate from.
    generic = next(row for row in load if row["employee_id"] == "E1")
    assert (generic["status"], generic["error_rate"]) == ("Provisional", "")


# --- No grade (2.4, Q05) ------------------------------------------------------------


def test_an_employee_with_no_grade_gets_the_title_with_no_level_word_and_is_listed(out: Path):
    load, _, exceptions = _read(out)
    by_id = {row["employee_id"]: row for row in load}
    assert by_id["B1"]["long_title"] == "Business Development Representative"
    assert by_id["B1"]["short_title"] == "Biz Dev Rep"
    assert (by_id["B1"]["level"], by_id["B1"]["no_grade"]) == ("", "1")
    for row in load:
        assert (row["level"] == "") is (row["no_grade"] == "1")
        assert (_BY_ID[row["employee_id"]].grade is None) is (row["no_grade"] == "1")
    listed = [row for row in exceptions if row["released"] == "1"]
    assert [(row["employee_id"], row["reason"]) for row in listed] == [
        ("B1", "no grade"),
        ("E1", "no grade"),
    ]


def test_a_grade_hrs_table_does_not_list_is_no_grade(tmp_path: Path, mini_catalog: Catalog):
    # 1.3 says "no usable grade"; titlestd.catalog reads an unlisted grade so.
    assert catalog.level_for_grade(mini_catalog, 11) is None
    load, _, exceptions = _release_one(
        tmp_path, mini_catalog, _confirmed("SAL-02"), _employee("X1", "Rep", "SAL", 11)
    )
    assert load[0]["long_title"] == "Sales Representative"
    assert (load[0]["level"], load[0]["no_grade"]) == ("", "1")
    assert [(row["reason"], row["released"]) for row in exceptions] == [("no grade", "1")]


def test_an_employee_with_no_grade_on_a_legacy_title_is_listed_once(out: Path):
    # Q05: the released = 0 row only.
    assert _BY_ID["G1"].grade is None
    exceptions = _read(out)[2]
    assert [
        (row["reason"], row["released"]) for row in exceptions if row["employee_id"] == "G1"
    ] == [("over the generic cap", "0")]


# --- The exceptions list (Q05, S11) -------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        ExceptionReason.STOPPED_PATH,
        ExceptionReason.OVER_GENERIC_CAP,
        ExceptionReason.UNDECIDED_BARRED,
        ExceptionReason.UNDECIDED_UNSCORED,
        ExceptionReason.NO_SUITABLE_ROLE,
    ],
)
def test_an_employee_left_on_a_legacy_title_is_listed_with_the_reason(
    tmp_path: Path, mini_catalog: Catalog, reason: ExceptionReason
):
    employees = (_employee("X1", "Odd Job", "FIN", 3), _employee("X2", "odd job", "FIN", None))
    load, reverse, exceptions = _release_one(
        tmp_path, mini_catalog, lambda row: _on_legacy(row, reason), *employees
    )
    assert load == [] and reverse == []
    assert exceptions == [
        {
            "employee_id": "X1",
            "legacy_job_title": "Odd Job",
            "job_family_code": "FIN",
            "reason": reason.value,
            "released": "0",
        },
        {
            "employee_id": "X2",
            "legacy_job_title": "odd job",
            "job_family_code": "FIN",
            "reason": reason.value,
            "released": "0",
        },
    ]


def test_the_exceptions_list_keeps_the_family_code_as_received(
    tmp_path: Path, mini_catalog: Catalog
):
    # Like the legacy title beside it: a padded code stays padded, a blank one blank.
    padded = _employee("X1", "Treasury Analyst", " FIN ", None)
    _, _, exceptions = _release_one(tmp_path, mini_catalog, _confirmed("FIN-06"), padded)
    assert [row["job_family_code"] for row in exceptions] == [" FIN "]
    assert (tmp_path / "exceptions.csv").read_text(encoding="utf-8").splitlines()[1] == (
        "X1,Treasury Analyst, FIN ,no grade,1"
    )


def test_load_and_the_unreleased_exceptions_cover_every_employee_exactly_once(out: Path):
    load, reverse, exceptions = _read(out)
    loaded = [row["employee_id"] for row in load]
    unreleased = [row["employee_id"] for row in exceptions if row["released"] == "0"]
    assert sorted(loaded + unreleased) == sorted(_BY_ID)
    assert set(loaded).isdisjoint(unreleased)
    assert [row["employee_id"] for row in reverse] == loaded
    # A released = 1 line is about an employee who is in the load file.
    assert {row["employee_id"] for row in exceptions if row["released"] == "1"} <= set(loaded)


# --- No title outside the catalog can reach the load file (2.7) ---------------------


def test_every_title_in_the_load_file_is_a_catalog_title(out: Path, mini_catalog: Catalog):
    load = _read(out)[0]
    assert len(load) == 9
    for row in load:
        assert catalog.get_role(mini_catalog, row["role_id"]) is not None
        assert catalog.is_catalog_title(mini_catalog, row["long_title"], row["short_title"])
    # The legacy title is in the reverse file and never in the load file.
    assert "Sr. Acct Mgr - NYC" not in (out / "load.csv").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "role_id", ["SAL-99", "sal-01", " SAL-01", "Account Manager", "Sr. Acct Mgr - NYC", "GEN"]
)
def test_the_writer_refuses_a_role_that_is_not_in_the_catalog(
    tmp_path: Path, mini_catalog: Catalog, role_id: str
):
    decisions = [
        dataclasses.replace(decision, role_id=role_id) if decision.key == BDR.key else decision
        for decision in DECISIONS
    ]
    with pytest.raises(ReleaseError, match="is not in the catalog; nothing was written"):
        release.write_release_files(tmp_path / "out", decisions, ROWS, EMPLOYEES, mini_catalog)
    assert not (tmp_path / "out").exists()


def test_the_writer_refuses_a_role_of_another_catalog(tmp_path: Path, mini_catalog: Catalog):
    # The catalog is data: a role removed from it can no longer be released.
    smaller = dataclasses.replace(
        mini_catalog, roles=tuple(role for role in mini_catalog.roles if role.role_id != "SAL-05")
    )
    with pytest.raises(ReleaseError, match="employee B1: role SAL-05 is not in the catalog"):
        release.write_release_files(tmp_path / "out", DECISIONS, ROWS, EMPLOYEES, smaller)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "titles",
    [
        # the legacy title in either column
        ("Sr. Acct Mgr - NYC", "Sr Acct Mgr"),
        ("Senior Account Manager", "Sr. Acct Mgr - NYC"),
        # the Long Title of one role beside the Short Title of another
        ("Senior Account Manager", "Sr Sales Rep"),
        # a level word the ladder does not hold, and a title changed by one space
        ("Chief Account Manager", "Chf Acct Mgr"),
        ("Senior Account Manager ", "Sr Acct Mgr"),
        ("", ""),
    ],
)
def test_the_writer_refuses_a_title_the_naming_rule_cannot_produce(
    tmp_path: Path, mini_catalog: Catalog, monkeypatch: pytest.MonkeyPatch, titles
):
    # Whatever hands the writer its titles, the guard reads what is about to be
    # written. Here the naming rule itself is replaced by a faulty one.
    def faulty(_catalog, _role, _grade) -> StandardTitle:
        return StandardTitle("Senior", *titles)

    monkeypatch.setattr(release, "standard_title", faulty)
    with pytest.raises(ReleaseError, match="are not a Long and Short Title of the catalog"):
        release.write_release_files(tmp_path / "out", DECISIONS, ROWS, EMPLOYEES, mini_catalog)
    assert not (tmp_path / "out").exists()


def test_the_guard_is_the_catalogs_own(
    tmp_path: Path, mini_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
):
    # The writer refuses exactly what titlestd.catalog.is_catalog_title rejects.
    asked = []

    def nothing_passes(_catalog, long_title: str, short_title: str) -> bool:
        asked.append((long_title, short_title))
        return False

    monkeypatch.setattr(release, "is_catalog_title", nothing_passes)
    with pytest.raises(ReleaseError, match="employee A1: the titles 'Senior Account Manager'"):
        release.write_release_files(tmp_path / "out", DECISIONS, ROWS, EMPLOYEES, mini_catalog)
    assert asked == [("Senior Account Manager", "Sr Acct Mgr")]
    assert not (tmp_path / "out").exists()


def test_the_guard_is_asked_about_every_row_of_the_load_file(
    tmp_path: Path, mini_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
):
    # A release that passes: the catalog's guard saw the titles of every row
    # that was written, whatever its path, its level or its role.
    asked = set()

    def spy(catalog_: Catalog, long_title: str, short_title: str) -> bool:
        asked.add((long_title, short_title))
        return catalog.is_catalog_title(catalog_, long_title, short_title)

    monkeypatch.setattr(release, "is_catalog_title", spy)
    release.write_release_files(tmp_path, DECISIONS, ROWS, EMPLOYEES, mini_catalog)
    load = _read(tmp_path)[0]
    assert asked == {(row["long_title"], row["short_title"]) for row in load}
    assert len(load) == 9 and len(asked) == 8  # A1 and A2 hold the same two titles


# What the naming rule is asked for each employee of the small release who gets
# a new title: the role of the title row and the employee's own grade.
_ASKED_FOR = {
    employee_id: (decision.role_id, _BY_ID[employee_id].grade)
    for decision in DECISIONS
    if decision.role_id is not None
    for row in ROWS
    if row.key == decision.key
    for employee_id in row.employee_ids
}


@pytest.mark.parametrize(
    ("employee_id", "titles"),
    [
        # the last row of the file
        ("H1", ("Chief Treasury Analyst", "Chf Treasury Analyst")),
        # the Long Title of the row before it, which passed, beside another role's Short Title
        ("A2", ("Senior Account Manager", "Sr Sales Rep")),
        # an employee with no grade, so a row with a blank level: here the legacy title
        ("B1", ("BDR", "BDR")),
        # the row on the generic family role
        ("E1", ("Special Projects", "Special Projects")),
        # a look-up hit, and a row in the middle of the file
        ("C1", ("Sales Engineer #20311", "Assoc Sales Eng")),
    ],
)
def test_one_title_outside_the_catalog_on_any_row_stops_the_release(
    tmp_path: Path,
    mini_catalog: Catalog,
    monkeypatch: pytest.MonkeyPatch,
    employee_id: str,
    titles: tuple[str, str],
):
    # The naming rule is wrong for one employee and right for all the others,
    # so a guard that reads only some of the rows lets the release through.
    assert len(set(_ASKED_FOR.values())) == len(_ASKED_FOR) == 9  # one employee per question

    def wrong_for_one(catalog_: Catalog, role, grade: int | None) -> StandardTitle:
        title = catalog.standard_title(catalog_, role, grade)
        if (role.role_id, grade) != _ASKED_FOR[employee_id]:
            return title
        return dataclasses.replace(title, long_title=titles[0], short_title=titles[1])

    monkeypatch.setattr(release, "standard_title", wrong_for_one)
    with pytest.raises(ReleaseError, match=f"employee {employee_id}: the titles") as refusal:
        release.write_release_files(tmp_path / "out", DECISIONS, ROWS, EMPLOYEES, mini_catalog)
    assert f"{titles[0]!r} and {titles[1]!r} are not a Long and Short Title" in str(refusal.value)
    assert not (tmp_path / "out").exists()


def test_a_refused_release_leaves_the_files_of_an_earlier_one_alone(
    out: Path, mini_catalog: Catalog
):
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    decisions = [dataclasses.replace(DECISIONS[0], role_id="SAL-99"), *DECISIONS[1:]]
    with pytest.raises(ReleaseError):
        release.write_release_files(out, decisions, ROWS, EMPLOYEES, mini_catalog)
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_the_load_file_never_stands_without_its_reverse_file(
    tmp_path: Path, mini_catalog: Catalog, monkeypatch: pytest.MonkeyPatch
):
    # The load file is written last: a write that fails earlier leaves no
    # load file that nothing could reverse.
    write_table = datafiles.write_table

    def failing(path: Path, columns, rows) -> None:
        if path.name == release.REVERSE_FILE:
            raise OSError("no space left on device")
        write_table(path, columns, rows)

    monkeypatch.setattr(datafiles, "write_table", failing)
    with pytest.raises(OSError, match="no space left"):
        release.write_release_files(tmp_path, DECISIONS, ROWS, EMPLOYEES, mini_catalog)
    assert not (tmp_path / release.LOAD_FILE).exists()


def test_a_generic_family_role_is_a_catalog_role_and_is_released(out: Path, mini_catalog: Catalog):
    # 1.8: at the close a leftover takes it, marked provisional. It is in the
    # catalog, so the guard lets it through, named like any other role.
    load = {row["employee_id"]: row for row in _read(out)[0]}
    assert catalog.get_role(mini_catalog, "FIN-GEN").is_generic
    assert (load["E1"]["role_id"], load["E1"]["long_title"], load["E1"]["status"]) == (
        "FIN-GEN",
        "Finance",
        "Provisional",
    )


def test_the_refusal_is_an_error_for_the_person_running_the_program():
    assert issubclass(ReleaseError, TitlestdError)


# --- Decisions, title rows and employees must fit each other ------------------------


def test_a_title_row_without_a_decision_is_refused(tmp_path: Path, mini_catalog: Catalog):
    with pytest.raises(ReleaseError, match="title 'bdr' in family 'SAL' has no decision"):
        release.write_release_files(
            tmp_path,
            [decision for decision in DECISIONS if decision.key != BDR.key],
            ROWS,
            EMPLOYEES,
            mini_catalog,
        )
    assert list(tmp_path.iterdir()) == []


def test_two_decisions_for_one_title_row_are_refused(tmp_path: Path, mini_catalog: Catalog):
    twice = [*DECISIONS, _on_legacy(BDR, ExceptionReason.STOPPED_PATH)]
    with pytest.raises(ReleaseError, match="title 'bdr' in family 'SAL' has two decisions"):
        release.write_release_files(tmp_path, twice, ROWS, EMPLOYEES, mini_catalog)
    assert list(tmp_path.iterdir()) == []


def test_a_decision_for_no_title_row_is_refused(tmp_path: Path, mini_catalog: Catalog):
    rows = [row for row in ROWS if row.key != STAFF.key]
    employees = [employee for employee in EMPLOYEES if employee.employee_id != "F1"]
    with pytest.raises(ReleaseError, match="'staff accountant' in family '' has a decision and"):
        release.write_release_files(tmp_path, DECISIONS, rows, employees, mini_catalog)
    assert list(tmp_path.iterdir()) == []


def test_an_employee_in_no_title_row_is_refused(tmp_path: Path, mini_catalog: Catalog):
    stray = _employee("Z9", "Night Porter", "FIN", 3)
    with pytest.raises(ReleaseError, match=r"do not hold the same employees: Z9$"):
        release.write_release_files(tmp_path, DECISIONS, ROWS, [*EMPLOYEES, stray], mini_catalog)
    assert list(tmp_path.iterdir()) == []


def test_a_title_row_naming_an_unknown_employee_is_refused(tmp_path: Path, mini_catalog: Catalog):
    employees = [employee for employee in EMPLOYEES if employee.employee_id not in ("A2", "D1")]
    with pytest.raises(ReleaseError, match=r"do not hold the same employees: A2, D1$"):
        release.write_release_files(tmp_path, DECISIONS, ROWS, employees, mini_catalog)
    assert list(tmp_path.iterdir()) == []


def test_an_employee_in_two_title_rows_is_refused(tmp_path: Path, mini_catalog: Catalog):
    rows = [dataclasses.replace(BDR, employee_ids=("A1", "B1")), *ROWS[:1], *ROWS[2:]]
    with pytest.raises(ReleaseError, match="employee A1 is in two title rows"):
        release.write_release_files(tmp_path, DECISIONS, rows, EMPLOYEES, mini_catalog)
    assert list(tmp_path.iterdir()) == []


def test_an_employee_listed_twice_is_refused(tmp_path: Path, mini_catalog: Catalog):
    with pytest.raises(ReleaseError, match="the employees file lists an employee twice"):
        release.write_release_files(
            tmp_path, DECISIONS, ROWS, [*EMPLOYEES, _BY_ID["B1"]], mini_catalog
        )
    assert list(tmp_path.iterdir()) == []


_ACCOUNT_APPROVED = _by_person(ACCOUNT, TableRow.WAVE1, "SAL-01")
_PAYROLL_LEGACY = _on_legacy(PAYROLL, ExceptionReason.OVER_GENERIC_CAP)


@pytest.mark.parametrize(
    ("decision", "fault"),
    [
        # released, and not all a released title needs
        (dataclasses.replace(_ACCOUNT_APPROVED, role_id=None), "no role or no status"),
        (dataclasses.replace(_ACCOUNT_APPROVED, role_id=""), "no role or no status"),
        (dataclasses.replace(_ACCOUNT_APPROVED, status=None), "no role or no status"),
        (
            dataclasses.replace(_ACCOUNT_APPROVED, legacy_reason=ExceptionReason.STOPPED_PATH),
            "has a reason to stay on legacy",
        ),
        # Approved, and nobody named (1.9)
        (dataclasses.replace(_ACCOUNT_APPROVED, decided_by=""), "names nobody who decided it"),
        (dataclasses.replace(_ACCOUNT_APPROVED, decided_by="  "), "names nobody who decided it"),
        # an error rate that is not a Provisional path's (Q05)
        (dataclasses.replace(_ACCOUNT_APPROVED, error_rate=0.1), "has an error rate"),
        (
            _by_machine(ACCOUNT, TableRow.MODELS_AGREE, "SAL-01", Status.CONFIRMED, 0.0),
            "has an error rate",
        ),
        (
            Decision(
                ACCOUNT.key,
                TableRow.GENERIC,
                DecisionSource.CODE,
                "SAL-GEN",
                Status.PROVISIONAL,
                error_rate=0.1,
            ),
            "has an error rate",
        ),
        # a Provisional title of a machine path, and no measured error rate (1.9)
        (
            _by_machine(ACCOUNT, TableRow.MODELS_AGREE, "SAL-01", Status.PROVISIONAL),
            "has no error rate",
        ),
        (
            _by_machine(ACCOUNT, TableRow.AGENT_SETTLED, "SAL-01", Status.PROVISIONAL),
            "has no error rate",
        ),
        # left on its legacy title, and not as such a title must be
        (dataclasses.replace(_PAYROLL_LEGACY, role_id="FIN-03"), "has a role or a status"),
        (
            dataclasses.replace(_PAYROLL_LEGACY, status=Status.PROVISIONAL),
            "has a role or a status",
        ),
        (dataclasses.replace(_PAYROLL_LEGACY, legacy_reason=None), "gives no reason"),
        (
            dataclasses.replace(_PAYROLL_LEGACY, legacy_reason=ExceptionReason.NO_GRADE),
            "gives no reason",
        ),
    ],
)
def test_a_decision_the_files_cannot_state_is_refused(
    tmp_path: Path, mini_catalog: Catalog, decision: Decision, fault: str
):
    decisions = [decision if decision.key == other.key else other for other in DECISIONS]
    with pytest.raises(ReleaseError, match=fault) as refusal:
        release.write_release_files(tmp_path, decisions, ROWS, EMPLOYEES, mini_catalog)
    assert f"title {decision.key.cleaned_title!r} in family" in str(refusal.value)
    assert list(tmp_path.iterdir()) == []


# --- The reverse file, and applying it (2.7, Q05) -----------------------------------


def test_the_reverse_file_holds_the_same_employees_as_the_load_file(out: Path):
    load, reverse, _ = _read(out)
    assert [row["employee_id"] for row in reverse] == [row["employee_id"] for row in load]
    assert {row["employee_id"]: row["legacy_job_title"] for row in reverse} == {
        row["employee_id"]: _BY_ID[row["employee_id"]].job_title for row in load
    }


def test_applying_the_reverse_file_restores_every_legacy_title(out: Path):
    load, reverse, _ = _read(out)
    restored = release.apply_reverse(load, reverse)
    assert list(restored) == [row["employee_id"] for row in load]
    assert len(restored) == 9
    for employee_id, title in restored.items():
        assert title == _BY_ID[employee_id].job_title
    # It restores what the load file replaced, and nothing in the load file is legacy.
    assert restored["A1"] == "Sr. Acct Mgr - NYC"
    assert next(row for row in load if row["employee_id"] == "A1")["long_title"] != restored["A1"]


AWKWARD_TITLES = [
    "  Sr. Acct Mgr - NYC  ",
    'Manager, "Special" Projects',
    "Ingénieur Commercial / Ventes",
    "ACCT MGR II\t",
    "Line One\nLine Two",
    "Carriage\r\nReturn",
    "'=Quoted",
    "経理",
]


def test_a_legacy_title_comes_back_byte_for_byte(tmp_path: Path, mini_catalog: Catalog):
    employees = [
        _employee(f"X{number}", title, "SAL", 5) for number, title in enumerate(AWKWARD_TITLES)
    ]
    rows = [_row(f"title {number}", employee) for number, employee in enumerate(employees)]
    decisions = [_confirmed("SAL-02")(row) for row in rows]
    release.write_release_files(tmp_path, decisions, rows, employees, mini_catalog)
    load, reverse, _ = _read(tmp_path)
    restored = release.apply_reverse(load, reverse)
    assert [restored[f"X{number}"] for number in range(len(AWKWARD_TITLES))] == AWKWARD_TITLES
    for employee in employees:
        assert restored[employee.employee_id].encode() == employee.job_title.encode()


def test_apply_reverse_reads_only_the_two_columns_it_needs():
    load = [{"employee_id": "B", "long_title": "x"}, {"employee_id": "A", "long_title": "y"}]
    reverse = [
        {"employee_id": "A", "legacy_job_title": "Old A"},
        {"employee_id": "B", "legacy_job_title": "Old B"},
    ]
    restored = release.apply_reverse(load, reverse)
    assert restored == {"A": "Old A", "B": "Old B"}
    assert list(restored) == ["B", "A"]  # the order of the load rows
    assert release.apply_reverse([], []) == {}


def test_apply_reverse_refuses_files_that_do_not_hold_the_same_employees(out: Path):
    load, reverse, _ = _read(out)
    with pytest.raises(
        ReleaseError, match=r"reverse\.csv has no legacy title for employee A1 of load\.csv"
    ):
        release.apply_reverse(load, reverse[1:])
    with pytest.raises(
        ReleaseError, match=r"reverse\.csv lists employees that load\.csv does not: A1, A2$"
    ):
        release.apply_reverse(load[2:], reverse)
    with pytest.raises(ReleaseError, match="no legacy title for employee A1"):
        release.apply_reverse(load, [])
    with pytest.raises(ReleaseError, match=r"does not: A1, A2, A3, B1, C1 and 4 more$"):
        release.apply_reverse([], reverse)


def test_apply_reverse_refuses_an_employee_listed_twice(out: Path):
    load, reverse, _ = _read(out)
    with pytest.raises(ReleaseError, match=r"load\.csv: employee A1 is listed twice"):
        release.apply_reverse([*load, load[0]], reverse)
    other = {"employee_id": "A1", "legacy_job_title": "Another Title"}
    with pytest.raises(ReleaseError, match=r"reverse\.csv: employee A1 is listed twice"):
        release.apply_reverse(load, [*reverse, other])


# --- The review-queue file (Q16, S12, Q40) ------------------------------------------


def _human(row: TitleRow, answer: HumanAnswer, role_id: str | None = None) -> HumanDecision:
    return HumanDecision(row.key, answer, role_id, REVIEWER, True)


QUEUE = (
    QueueEntry(
        row=TREASURY,
        reasons=(Reason.MODELS_DISAGREE, Reason.AGENT_ESCALATED),
        ml_first_choice="FIN-01",
        llm_pick="FIN-06",
        agent_recommendation=None,
        answer_source="stand-in",
        decision=_human(TREASURY, HumanAnswer.ROLE, "FIN-06"),
        outcome=TableRow.REVIEW_QUEUE,
    ),
    QueueEntry(
        row=PROJECTS,
        reasons=(Reason.NO_MATCH, Reason.AGENT_ESCALATED),
        ml_first_choice="FIN-01",
        llm_pick=None,
        agent_recommendation=None,
        answer_source="recorded 2026-10-04, claude-haiku-4-5",
        decision=_human(PROJECTS, HumanAnswer.UNDECIDED),
        outcome=TableRow.GENERIC,
    ),
    QueueEntry(
        row=STAFF,
        reasons=(Reason.NO_USABLE_FAMILY,),
        ml_first_choice=None,
        llm_pick=None,
        agent_recommendation=None,
        answer_source="",
        decision=_human(STAFF, HumanAnswer.NO_SUITABLE_ROLE),
        outcome=TableRow.LEGACY,
    ),
    QueueEntry(
        row=ANALYST,
        reasons=(Reason.COMPLIANCE_FLAG, Reason.MODELS_DISAGREE),
        ml_first_choice="FIN-04",
        llm_pick="FIN-01",
        agent_recommendation="FIN-01",
        answer_source="live",
        decision=None,
        outcome=TableRow.LEGACY,
    ),
)

# The review-queue file of that queue, in three parts: why each title is there,
# what the models said, and what the person decided.
QUEUE_WHY = """
staff accountant,,1,no usable family code; no suitable role, waiting for catalog owner
financial analyst,FIN,2,compliance-flagged role; models disagree
special projects,FIN,1,no match; agent escalated
treasury analyst,FIN,1,models disagree; agent escalated
"""
QUEUE_MODELS = """
staff accountant,,,,
financial analyst,FIN-04,FIN-01,FIN-01,live
special projects,FIN-01,,,recorded 2026-10-04, claude-haiku-4-5
treasury analyst,FIN-01,FIN-06,,stand-in
"""
QUEUE_PEOPLE = """
staff accountant,no suitable role,Reviewer (simulated),1,left on legacy title
financial analyst,,,0,left on legacy title
special projects,could not decide,Reviewer (simulated),1,generic family role
treasury analyst,FIN-06,Reviewer (simulated),1,review queue
"""


def test_the_review_queue_file_has_one_row_per_title_sorted_by_key(tmp_path: Path):
    release.write_review_queue(tmp_path / "out", QUEUE)
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["review_queue.csv"]
    written = tmp_path / "out" / "review_queue.csv"
    rows = datafiles.read_table(written, release.REVIEW_QUEUE_COLUMNS)
    why = ("cleaned_title", "job_family_code", "headcount", "reasons")
    models = ("ml_first_choice", "llm_pick", "agent_recommendation", "answer_source")
    people = ("decision", "decided_by", "simulated", "outcome")
    assert why + models + people == release.REVIEW_QUEUE_COLUMNS
    assert _columns(rows, *why) == QUEUE_WHY
    assert _columns(rows, "cleaned_title", *models) == QUEUE_MODELS
    assert _columns(rows, "cleaned_title", *people) == QUEUE_PEOPLE
    # The same format as the other files; a label with a comma is quoted.
    text = written.read_text(encoding="utf-8")
    assert text.split("\n")[0] == ",".join(release.REVIEW_QUEUE_COLUMNS)
    assert text.endswith("\n") and "\r" not in text
    assert ',"recorded 2026-10-04, claude-haiku-4-5",could not decide,' in text
    # The order the entries come in does not change a byte.
    again = tmp_path / "again"
    release.write_review_queue(again, QUEUE[::-1])
    assert (again / "review_queue.csv").read_bytes() == written.read_bytes()


def test_the_review_queue_shows_the_persons_decision_marked_as_simulated(tmp_path: Path):
    release.write_review_queue(tmp_path, QUEUE)
    rows = datafiles.read_table(tmp_path / "review_queue.csv", release.REVIEW_QUEUE_COLUMNS)
    by_title = {row["cleaned_title"]: row for row in rows}
    # The person's role id, or one of the two answers that are not a role (S12).
    decisions = [row["decision"] for row in rows]
    assert decisions == ["no suitable role", "", "could not decide", "FIN-06"]
    assert by_title["special projects"]["decision"] == HumanAnswer.UNDECIDED.value
    assert by_title["staff accountant"]["decision"] == HumanAnswer.NO_SUITABLE_ROLE.value
    for row in rows:
        decided = row["cleaned_title"] != "financial analyst"
        assert row["simulated"] == ("1" if decided else "0")
        assert row["decided_by"] == (REVIEWER if decided else "")
    # The answer label of Q40 and the final outcome are shown as given.
    assert by_title["special projects"]["answer_source"] == "recorded 2026-10-04, claude-haiku-4-5"
    assert {row["outcome"] for row in rows} <= {table_row.value for table_row in TableRow}


def test_reasons_keep_the_order_they_were_given_in(tmp_path: Path):
    entry = dataclasses.replace(
        QUEUE[0], reasons=(Reason.QUOTE_NOT_IN_RECORD, Reason.COMPLIANCE_FLAG, Reason.ML_ALONE)
    )
    release.write_review_queue(tmp_path, [entry])
    rows = datafiles.read_table(tmp_path / "review_queue.csv", release.REVIEW_QUEUE_COLUMNS)
    assert rows[0]["reasons"] == "quote not in record; compliance-flagged role; ML alone"


def test_a_title_marked_no_suitable_role_is_shown_waiting_for_the_catalog_owner(tmp_path: Path):
    # Q03: written with that reason, whether or not the entry lists it already.
    marked = _human(STAFF, HumanAnswer.NO_SUITABLE_ROLE)
    waiting = "no suitable role, waiting for catalog owner"
    for given, written in (
        ((), waiting),
        ((Reason.PERSON_NAME,), f"person's name; {waiting}"),
        ((Reason.NO_SUITABLE_ROLE,), waiting),
        ((Reason.NO_SUITABLE_ROLE, Reason.NO_USABLE_FAMILY), f"{waiting}; no usable family code"),
    ):
        entry = dataclasses.replace(QUEUE[2], reasons=given, decision=marked)
        release.write_review_queue(tmp_path, [entry])
        rows = datafiles.read_table(tmp_path / "review_queue.csv", release.REVIEW_QUEUE_COLUMNS)
        assert [(row["reasons"], row["decision"]) for row in rows] == [
            (written, "no suitable role")
        ]
    # Nobody else's title gets the reason: not an undecided one, not a decided one.
    release.write_review_queue(tmp_path, QUEUE[:2])
    rows = datafiles.read_table(tmp_path / "review_queue.csv", release.REVIEW_QUEUE_COLUMNS)
    assert [waiting in row["reasons"] for row in rows] == [False, False]


def test_a_decision_that_is_not_simulated_carries_no_mark(tmp_path: Path):
    real = HumanDecision(TREASURY.key, HumanAnswer.ROLE, "FIN-06", "Catalog owner", False)
    release.write_review_queue(tmp_path, [dataclasses.replace(QUEUE[0], decision=real)])
    rows = datafiles.read_table(tmp_path / "review_queue.csv", release.REVIEW_QUEUE_COLUMNS)
    assert (rows[0]["decided_by"], rows[0]["simulated"]) == ("Catalog owner", "0")


def test_an_empty_queue_writes_the_header_alone(tmp_path: Path):
    release.write_review_queue(tmp_path, [])
    text = (tmp_path / "review_queue.csv").read_text(encoding="utf-8")
    assert text == ",".join(release.REVIEW_QUEUE_COLUMNS) + "\n"


def test_a_title_in_the_queue_twice_is_refused(tmp_path: Path):
    with pytest.raises(ReleaseError, match="'treasury analyst' in family 'FIN' is in the review"):
        release.write_review_queue(tmp_path, [*QUEUE, QUEUE[0]])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("role_id", [None, ""])
def test_a_role_decision_that_names_no_role_is_refused(tmp_path: Path, role_id):
    entry = dataclasses.replace(QUEUE[0], decision=_human(TREASURY, HumanAnswer.ROLE, role_id))
    with pytest.raises(ReleaseError, match="is a role and names none"):
        release.write_review_queue(tmp_path, [entry])
    assert list(tmp_path.iterdir()) == []
