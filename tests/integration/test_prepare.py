"""The prepare step on the mini data set, read through the real file readers.

It must reproduce the hand-written title rows of tests/conftest.py exactly
(MINI_ROWS), and its records must not depend on the order of the employees
file (Q11). The other tests follow the mini files through the step: the seed
look-up (Q09), the padded and the blank family code (S05, Q26), and the names
and the email that must never be shown (Q14).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from titlestd import datafiles, reference, seeding
from titlestd.contracts import EMAIL_MASK, NAME_MASK, Employee, InputData, TitleKey, TitleRow
from titlestd.prepare import prepare, title_key

NAMES = (*reference.GIVEN_NAMES, *reference.SURNAMES)


def family_codes(data: InputData) -> set[str]:
    """What a caller passes: the family codes of the catalog."""
    return {role.family_code for role in data.catalog.roles}


def prepare_inputs(data: InputData) -> tuple[TitleRow, ...]:
    return prepare(data.employees, data.abbreviations, family_codes(data))


@pytest.fixture(scope="module")
def prepared(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    return prepare_inputs(mini_inputs)


@pytest.fixture
def data_dir(tmp_path: Path, mini_dir: Path) -> Path:
    """A private copy of the mini data set that a test may change."""
    target = tmp_path / "data"
    shutil.copytree(mini_dir, target)
    return target


def shown(row: TitleRow) -> str:
    """Everything of a row that a model, a rule or a reviewer may read."""
    record = row.record
    return " ".join(
        [record.cleaned_title, record.family_code, record.department, record.description]
    )


# --- The hand-written rows ----------------------------------------------------------------


def test_it_reproduces_the_hand_written_rows(prepared, mini_rows):
    assert [row.key for row in prepared] == [row.key for row in mini_rows]
    for built, written in zip(prepared, mini_rows, strict=True):
        assert built == written, written.key
    assert prepared == mini_rows


def test_the_rows_cover_the_employees_file(mini_inputs: InputData, prepared):
    listed = [employee_id for row in prepared for employee_id in row.employee_ids]
    assert sorted(listed) == sorted(employee.employee_id for employee in mini_inputs.employees)
    assert len(prepared) == 15 and sum(row.headcount for row in prepared) == 22


# --- The order of the employees file (Q11) --------------------------------------------------


def _reordered(employees: list[Employee], order: str) -> list[Employee]:
    if order == "reversed":
        return employees[::-1]
    if order == "by description":
        return sorted(employees, key=lambda e: (e.job_description, e.employee_id), reverse=True)
    shuffled = list(employees)
    seeding.rng(20260101, "test_prepare", order).shuffle(shuffled)
    return shuffled


@pytest.mark.parametrize(
    "order", ["reversed", "by description", "shuffle 1", "shuffle 2", "shuffle 3", "shuffle 4"]
)
def test_the_rows_do_not_depend_on_the_order_of_the_employees_file(
    order, data_dir: Path, mini_inputs: InputData, prepared
):
    employees = _reordered(list(mini_inputs.employees), order)
    assert [e.employee_id for e in employees] != [e.employee_id for e in mini_inputs.employees]
    datafiles.write_employees(data_dir, employees)

    reread = datafiles.read_inputs(data_dir)
    assert list(reread.employees) == employees  # the file really is in the new order
    rows = prepare_inputs(reread)
    assert rows == prepared
    # The row that file order could change: its two descriptions are held once
    # each, and the one of the lowest employee_id (E007) wins in every order.
    (tied,) = [row for row in rows if row.employee_ids == ("E007", "E008", "E009")]
    assert tied.record.description == "Sells products to new customers by phone and closes deals."


# --- The seed look-up and the family codes (Q09, S05, Q26) ----------------------------------


def test_the_seed_rows_key_onto_prepared_rows(mini_inputs: InputData, prepared):
    by_key = {row.key: row for row in prepared}
    keys = [
        title_key(seed.job_title, seed.job_family_code, mini_inputs.abbreviations)
        for seed in mini_inputs.seed_lookup
    ]
    assert keys == [TitleKey("SAL", "sales engineer"), TitleKey("FIN", "accounts payable clerk")]
    for seed, key in zip(mini_inputs.seed_lookup, keys, strict=True):
        assert seed.job_title in by_key[key].raw_titles


def test_every_employee_is_in_the_row_its_title_key_names(mini_inputs: InputData, prepared):
    by_key = {row.key: row for row in prepared}
    for employee in mini_inputs.employees:
        key = title_key(employee.job_title, employee.job_family_code, mini_inputs.abbreviations)
        assert employee.employee_id in by_key[key].employee_ids
        assert employee.job_title in by_key[key].raw_titles


def test_the_padded_family_code_is_its_family_and_the_blank_one_is_not_usable(
    mini_inputs: InputData, prepared
):
    recorded = {e.employee_id: e.job_family_code for e in mini_inputs.employees}
    assert (recorded["E020"], recorded["E021"]) == ("FIN ", "")
    by_employee = {employee_id: row for row in prepared for employee_id in row.employee_ids}
    assert by_employee["E020"].key == TitleKey("FIN", "treasury analyst")
    assert by_employee["E020"].family_usable
    assert by_employee["E021"].key == TitleKey("", "staff accountant")
    assert [row.key for row in prepared if not row.family_usable] == [by_employee["E021"].key]


def test_the_usable_family_codes_are_the_catalogs(mini_inputs: InputData, prepared):
    assert family_codes(mini_inputs) == {"FIN", "SAL"}
    # A catalog without the Finance family: its rows lose the flag and nothing else.
    without = prepare(mini_inputs.employees, mini_inputs.abbreviations, {"SAL"})
    assert {row.key.family_code for row in without if not row.family_usable} == {"", "FIN"}
    assert without == tuple(
        dataclasses.replace(row, family_usable=row.key.family_code == "SAL") for row in prepared
    )


# --- What is never shown (1.4, Q14) -----------------------------------------------------------


def test_the_file_holds_names_and_an_email_and_no_record_shows_them(
    mini_inputs: InputData, prepared
):
    received = " ".join(
        f"{e.job_title} {e.department} {e.job_description}" for e in mini_inputs.employees
    ).lower()
    held = [name for name in NAMES if name in received]
    assert held == ["maren", "oswin", "tavi", "ashgrove", "quillfeather", "thistlewood"]
    assert f"@{reference.EMAIL_DOMAIN}" in received

    everything = " ".join(shown(row) for row in prepared).lower()
    assert not [name for name in NAMES if name in everything]
    assert "@" not in everything and reference.EMAIL_DOMAIN not in everything
    # One name in a title, one name in a description held by two employees
    # of one row, one email.
    assert everything.count(NAME_MASK.lower()) == 2
    assert everything.count(EMAIL_MASK.lower()) == 1


def test_only_the_title_with_a_name_is_flagged(prepared):
    flagged = [row for row in prepared if row.has_person_name]
    assert [row.raw_titles for row in flagged] == [("Cust Success Mgr (Tavi Quillfeather)",)]
    assert [row.key for row in prepared if NAME_MASK in row.key.cleaned_title] == [
        row.key for row in flagged
    ]


# --- A changed employees file ---------------------------------------------------------------


def test_new_spellings_in_the_file_join_their_rows(
    data_dir: Path, mini_inputs: InputData, prepared
):
    added = (
        Employee("E101", "SR ACCT MGR (REQ-55120) - LDN", " SAL", "Enterprise Sales", None, ""),
        Employee("E102", 'Fin. "Analyst", #4410 / Remote', "FIN", "Group Reporting", 3, ""),
        Employee("E103", "Payroll Spec - Calista Fennimore", "FIN", "Accounting", 2, ""),
        Employee("E104", "Sales Rep", "OPS", "Inside Sales", 4, "Sells by phone.\nCloses deals."),
    )
    datafiles.write_employees(data_dir, (*mini_inputs.employees, *added))
    reread = datafiles.read_inputs(data_dir)
    assert reread.employees[-4:] == added  # commas, quotes and a line break survive the file

    rows = {row.key: row for row in prepare_inputs(reread)}
    before = {row.key: row for row in prepared}

    senior = rows[TitleKey("SAL", "senior account manager")]
    assert senior.employee_ids == ("E004", "E005", "E006", "E101")
    assert senior.raw_titles == (
        "SR ACCT MGR (REQ-55120) - LDN",
        "Senior Account Manager",
        "Sr. Acct Mgr - NYC",
    )
    assert senior.record == before[senior.key].record

    analyst = rows[TitleKey("FIN", "financial analyst")]
    assert analyst.employee_ids == ("E014", "E015", "E102")
    assert analyst.record == before[analyst.key].record  # two of three are in Corporate Finance

    named = rows[TitleKey("FIN", f"payroll specialist {NAME_MASK}")]
    assert named.has_person_name and named.employee_ids == ("E103",)

    unknown = rows[TitleKey("OPS", "sales representative")]
    assert not unknown.family_usable
    assert unknown.record.description == "Sells by phone.\nCloses deals."

    untouched = set(before) - {senior.key, analyst.key}
    assert set(rows) == set(before) | {named.key, unknown.key}
    assert all(rows[key] == before[key] for key in untouched)


# --- With nothing installed (2.5) -------------------------------------------------------------


def test_the_step_runs_on_the_standard_library_alone(bare_python, import_guard, mini_dir: Path):
    script = (
        import_guard
        + """
from pathlib import Path

from titlestd import datafiles
from titlestd.prepare import prepare

data = datafiles.read_inputs(Path(sys.argv[1]))
rows = prepare(data.employees, data.abbreviations, {r.family_code for r in data.catalog.roles})
print(len(rows), sum(row.headcount for row in rows), attempted_imports)
"""
    )
    result = bare_python(script, str(mini_dir))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "15 22 []"
