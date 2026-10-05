"""The prepare step: grouping employees into title rows and their records (Q11, Q14, Q26)."""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from titlestd.contracts import (
    DESCRIPTION_WORDS,
    EMAIL_MASK,
    NAME_MASK,
    Abbreviation,
    Employee,
    Record,
    TitleKey,
    TitleRow,
)
from titlestd.prepare import prepare, title_key

ABBREVIATIONS = (
    Abbreviation("sr", "senior"),
    Abbreviation("acct", "account"),
    Abbreviation("mgr", "manager"),
    Abbreviation("rep", "representative"),
)
FAMILIES = ("FIN", "SAL")


def employee(
    employee_id: str,
    job_title: str = "Sales Rep",
    family: str = "SAL",
    department: str = "Inside Sales",
    description: str = "",
    grade: int | None = 5,
) -> Employee:
    return Employee(
        employee_id=employee_id,
        job_title=job_title,
        job_family_code=family,
        department=department,
        grade=grade,
        job_description=description,
    )


def rows_of(*employees: Employee, families=FAMILIES) -> tuple[TitleRow, ...]:
    return prepare(employees, ABBREVIATIONS, families)


def only_row(*employees: Employee, families=FAMILIES) -> TitleRow:
    (row,) = rows_of(*employees, families=families)
    return row


# --- Grouping (1.4, S05) -------------------------------------------------------------


def test_no_employees_give_no_rows():
    assert prepare((), ABBREVIATIONS, FAMILIES) == ()


def test_one_employee_gives_one_row_with_its_record():
    row = only_row(
        employee("E1", "Sr. Acct Mgr - NYC", "SAL", "Enterprise Sales", "Owns accounts.")
    )
    assert row == TitleRow(
        record=Record(
            cleaned_title="senior account manager",
            family_code="SAL",
            department="Enterprise Sales",
            description="Owns accounts.",
        ),
        employee_ids=("E1",),
        raw_titles=("Sr. Acct Mgr - NYC",),
        has_person_name=False,
        family_usable=True,
    )


def test_spellings_of_one_title_are_one_row():
    row = only_row(
        employee("E3", "senior account manager REQ-10442"),
        employee("E1", "Sr. Acct Mgr - NYC"),
        employee("E4", "Sr. Acct Mgr - NYC"),
        employee("E2", "Senior Account Manager"),
    )
    assert row.key == TitleKey("SAL", "senior account manager")
    assert row.employee_ids == ("E1", "E2", "E3", "E4")
    assert row.headcount == 4
    # The distinct titles as received, sorted.
    assert row.raw_titles == (
        "Senior Account Manager",
        "Sr. Acct Mgr - NYC",
        "senior account manager REQ-10442",
    )


def test_the_same_title_in_two_families_is_two_rows():
    rows = rows_of(employee("E1", "Analyst", "SAL"), employee("E2", "Analyst", "FIN"))
    assert [row.key for row in rows] == [TitleKey("FIN", "analyst"), TitleKey("SAL", "analyst")]
    assert [row.employee_ids for row in rows] == [("E2",), ("E1",)]


def test_rows_are_sorted_by_family_code_then_cleaned_title():
    rows = rows_of(
        employee("E1", "Sales Rep", "SAL"),
        employee("E2", "Treasury Analyst", "FIN"),
        employee("E3", "Account Mgr", "SAL"),
        employee("E4", "Staff Accountant", ""),
        employee("E5", "Clerk", "FIN"),
    )
    keys = [row.key for row in rows]
    assert keys == sorted(keys)
    assert keys == [
        TitleKey("", "staff accountant"),
        TitleKey("FIN", "clerk"),
        TitleKey("FIN", "treasury analyst"),
        TitleKey("SAL", "account manager"),
        TitleKey("SAL", "sales representative"),
    ]


def test_every_employee_is_in_the_row_of_its_title_key():
    employees = (
        employee("E1", "Sr. Acct Mgr - NYC", "SAL"),
        employee("E2", "Senior Account Manager", " SAL "),
        employee("E3", "Assistant to Tavi Quillfeather", "FIN"),
        employee("E4", "Clerk #20311", "  "),
        employee("E5", "Clerk", "XX"),
    )
    rows = {row.key: row for row in rows_of(*employees)}
    for member in employees:
        key = title_key(member.job_title, member.job_family_code, ABBREVIATIONS)
        assert member.employee_id in rows[key].employee_ids
    assert sum(row.headcount for row in rows.values()) == len(employees)


def test_the_grade_plays_no_part():
    # The level is applied per employee from the grade, after the decision (1.4).
    graded = (employee("E1", grade=2), employee("E2", grade=9), employee("E3", "Clerk", grade=4))
    ungraded = tuple(dataclasses.replace(member, grade=None) for member in graded)
    assert rows_of(*graded) == rows_of(*ungraded)


# --- The description of a row (Q11) ------------------------------------------------------


def test_the_description_is_the_most_common_one():
    row = only_row(
        employee("E1", description="Visits prospects."),
        employee("E2", description="Sells by phone."),
        employee("E3", description="Sells by phone."),
    )
    assert row.record.description == "Sells by phone."


def test_a_tie_goes_to_the_description_of_the_lowest_employee_id():
    low = employee("E07", description="Sells by phone.")
    high = employee("E09", description="Visits prospects.")
    # Whichever comes first in the file.
    assert only_row(low, high).record.description == "Sells by phone."
    assert only_row(high, low).record.description == "Sells by phone."


def test_a_tie_is_settled_by_the_lowest_holder_of_each_description():
    row = only_row(
        employee("E4", description="Visits prospects."),
        employee("E3", description="Sells by phone."),
        employee("E2", description="Sells by phone."),
        employee("E1", description="Visits prospects."),
    )
    assert row.record.description == "Visits prospects."


def test_empty_descriptions_are_never_chosen():
    row = only_row(
        employee("E1", description=""),
        employee("E2", description=""),
        employee("E3", description="   "),
        employee("E4", description="Sells by phone."),
    )
    assert row.record.description == "Sells by phone."


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_row_with_no_description_has_an_empty_one(blank):
    # 2.4: no description, use the title alone.
    row = only_row(employee("E1", description=blank), employee("E2", description=""))
    assert row.record.description == ""


def test_the_description_is_masked():
    row = only_row(
        employee("E1", description="Reports to Oswin Thistlewood; write to oswin@example.com.")
    )
    assert row.record.description == f"Reports to {NAME_MASK}; write to {EMAIL_MASK}."


def test_descriptions_are_counted_after_masking():
    row = only_row(
        employee("E1", description="Runs the eastern territory."),
        employee("E2", description="Reports to Oswin Thistlewood."),
        employee("E3", description="Reports to Calista Fennimore."),
    )
    assert row.record.description == f"Reports to {NAME_MASK}."


def test_a_short_description_is_kept_as_recorded():
    text = "Sells  products (new customers), by phone;\tcloses DEALS. "
    assert only_row(employee("E1", description=text)).record.description == text
    exactly = " ".join(f"w{number}" for number in range(DESCRIPTION_WORDS))
    assert only_row(employee("E1", description=exactly)).record.description == exactly


def test_a_long_description_is_cut_to_its_first_200_words():
    words = [f"w{number}" for number in range(1, 251)]
    row = only_row(employee("E1", description=" ".join(words)))
    assert row.record.description == " ".join(words[:200])
    assert len(row.record.description.split()) == DESCRIPTION_WORDS == 200


def test_the_cut_keeps_the_text_between_the_words():
    text = "one,  two\tthree\nfour " + "more " * 300
    row = only_row(employee("E1", description=text))
    assert row.record.description.startswith("one,  two\tthree\nfour more more")
    assert row.record.description.endswith("more more")
    assert len(row.record.description.split()) == DESCRIPTION_WORDS


def test_the_cut_comes_after_masking():
    # Masking turns the two words of the name into one, so the 200 words kept
    # reach one word further than a cut made before masking would.
    words = [f"w{number}" for number in range(1, 251)]
    row = only_row(employee("E1", description="Oswin Thistlewood " + " ".join(words)))
    assert row.record.description == " ".join([NAME_MASK, *words[:199]])


def test_descriptions_are_counted_whole_and_the_chosen_one_is_cut():
    # Cut first, the two long ones would be the same text and tie with the
    # short one, and the tie would go to E1.
    shared = " ".join(f"w{number}" for number in range(1, 201))
    row = only_row(
        employee("E1", description=shared + " first ending"),
        employee("E2", description=shared + " second ending"),
        employee("E3", description="Sells by phone."),
        employee("E4", description="Sells by phone."),
    )
    assert row.record.description == "Sells by phone."
    row = only_row(
        employee("E1", description=shared + " first ending"),
        employee("E2", description=shared + " second ending"),
        employee("E3", description=shared + " second ending"),
    )
    assert row.record.description == shared


# --- The department of a row (Q11) -------------------------------------------------------


def test_the_department_is_the_most_common_one():
    row = only_row(
        employee("E1", department="Field Sales"),
        employee("E2", department="Inside Sales"),
        employee("E3", department="Inside Sales"),
    )
    assert row.record.department == "Inside Sales"


def test_a_department_tie_goes_to_the_lowest_employee_id():
    low = employee("E1", department="Field Sales")
    high = employee("E2", department="Inside Sales")
    assert only_row(low, high).record.department == "Field Sales"
    assert only_row(high, low).record.department == "Field Sales"


def test_an_empty_department_is_never_chosen_and_none_gives_an_empty_one():
    row = only_row(
        employee("E1", department=""),
        employee("E2", department=" "),
        employee("E3", department="Inside Sales"),
    )
    assert row.record.department == "Inside Sales"
    assert only_row(employee("E1", department="")).record.department == ""


def test_the_department_is_masked_and_not_cut():
    long_name = " ".join(f"Unit{number}" for number in range(300))
    row = only_row(employee("E1", department=f"Office of Calista Fennimore {long_name}"))
    assert row.record.department == f"Office of {NAME_MASK} {long_name}"


def test_description_and_department_are_chosen_separately():
    row = only_row(
        employee("E1", department="Field Sales", description="Visits prospects."),
        employee("E2", department="Field Sales", description="Sells by phone."),
        employee("E3", department="Inside Sales", description="Sells by phone."),
    )
    assert (row.record.department, row.record.description) == ("Field Sales", "Sells by phone.")


# --- The order of the employees file (Q11) -------------------------------------------------


def test_the_rows_do_not_depend_on_the_order_of_the_employees():
    # Ties in the description and the department, empty descriptions, two
    # spellings of a title, a padded family code and a name: every order of
    # these six employees gives the same rows.
    employees = (
        employee("E1", "Sales Rep", "SAL", "Field Sales", "Visits prospects."),
        employee("E2", "Sales Rep", "SAL ", "Inside Sales", "Sells by phone."),
        employee("E3", "sales representative REQ-10442", "SAL", "Inside Sales", ""),
        employee("E4", "Sales Rep", "SAL", "Field Sales", ""),
        employee("E5", "Assistant to Tavi Quillfeather", "FIN", "Treasury", "Keeps the diary."),
        employee("E6", "Assistant to Oswin Thistlewood", "FIN", "Accounting", "Books travel."),
    )
    expected = rows_of(*employees)
    assert [row.key for row in expected] == [
        TitleKey("FIN", f"assistant to {NAME_MASK}"),
        TitleKey("SAL", "sales representative"),
    ]
    assert expected[0].record == Record(
        f"assistant to {NAME_MASK}", "FIN", "Treasury", "Keeps the diary."
    )
    assert expected[1].record == Record(
        "sales representative", "SAL", "Field Sales", "Visits prospects."
    )
    for order in itertools.permutations(employees):
        assert rows_of(*order) == expected


# --- A person's name in the title (Q14) ------------------------------------------------------


def test_a_title_with_a_name_is_flagged_and_keyed_on_the_masked_title():
    plain, named = rows_of(
        employee("E1", "Cust Success Mgr (Tavi Quillfeather)", description="Helps customers."),
        employee("E2", "Cust Success Mgr"),
    )
    assert named.key == TitleKey("SAL", f"cust success manager {NAME_MASK}")
    assert named.has_person_name and not named.model_eligible
    assert named.record.description == "Helps customers."
    assert plain.key == TitleKey("SAL", "cust success manager")
    assert not plain.has_person_name and plain.model_eligible


def test_titles_that_differ_only_in_the_name_are_one_flagged_row():
    row = only_row(
        employee("E1", "Assistant to Tavi Quillfeather"),
        employee("E2", "Assistant to Oswin Thistlewood"),
    )
    assert row.has_person_name
    assert row.raw_titles == ("Assistant to Oswin Thistlewood", "Assistant to Tavi Quillfeather")
    assert row.employee_ids == ("E1", "E2")


def test_a_name_in_a_description_does_not_flag_the_row():
    row = only_row(employee("E1", description="Reports to Oswin Thistlewood."))
    assert not row.has_person_name and row.model_eligible


# --- No usable family code (Q26, S05) -------------------------------------------------------


def test_a_family_code_of_the_catalog_is_usable():
    row = only_row(employee("E1", family="SAL"))
    assert row.family_usable and row.model_eligible


@pytest.mark.parametrize("blank", ["", " ", " \t "])
def test_a_blank_family_code_is_not_usable_and_stays_blank(blank):
    row = only_row(employee("E1", "Staff Accountant", blank, "Accounting", "Keeps the ledger."))
    assert row.key == TitleKey("", "staff accountant")
    assert not row.family_usable and not row.model_eligible
    # It is still a row with its record: a person decides it (2.4).
    assert row.record == Record("staff accountant", "", "Accounting", "Keeps the ledger.")


def test_a_blank_family_code_is_not_usable_even_if_the_caller_lists_it():
    row = only_row(employee("E1", family=" "), families=("", " ", "SAL"))
    assert not row.family_usable


@pytest.mark.parametrize("unknown", ["XX", "sal", "SALES", "S AL"])
def test_an_unknown_family_code_is_not_usable_and_is_kept_as_recorded(unknown):
    row = only_row(employee("E1", family=unknown))
    assert row.key == TitleKey(unknown, "sales representative")
    assert not row.family_usable


def test_a_padded_family_code_is_the_same_family():
    row = only_row(employee("E1", family="SAL "), employee("E2", family=" SAL"), employee("E3"))
    assert row.key == TitleKey("SAL", "sales representative")
    assert row.family_usable
    assert row.employee_ids == ("E1", "E2", "E3")


def test_blank_family_codes_of_any_width_are_one_row():
    row = only_row(employee("E1", family=""), employee("E2", family="   "))
    assert row.key.family_code == "" and row.headcount == 2


def test_the_usable_codes_are_the_ones_the_caller_passes():
    employees = (employee("E1", family="SAL"), employee("E2", family="FIN"))
    for families in (["SAL"], {"SAL"}, frozenset({"SAL"}), ("SAL",)):
        fin, sal = rows_of(*employees, families=families)
        assert (fin.family_usable, sal.family_usable) == (False, True)
    assert [row.family_usable for row in rows_of(*employees, families=())] == [False, False]
