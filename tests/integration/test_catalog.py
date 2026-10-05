"""The catalog functions on the mini data set, read by the real readers.

Every employee of the mini data set is given the standard title of its true
role, as a release would give it, and the guard of "no title outside the
catalog can reach the load file" is tried on what the files hold. The catalog
is data: a changed catalog file changes the titles.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from titlestd import catalog, datafiles, people
from titlestd.catalog import CatalogError
from titlestd.contracts import Catalog, CatalogRole, InputData, StandardTitle, TitleRow
from titlestd.datafiles import DataError


@pytest.fixture(scope="module")
def true_roles(mini_dir: Path) -> dict[tuple[str, str], str]:
    """(raw job title, trimmed family code) to the true role id (S05)."""
    return {
        (row.job_title, row.job_family_code.strip()): row.true_role_id
        for row in people.read_truth(mini_dir)
    }


@pytest.fixture(scope="module")
def titles(mini_inputs: InputData, true_roles) -> dict[str, StandardTitle]:
    """employee_id to the standard title of the employee's true role at its grade."""
    given = {}
    for employee in mini_inputs.employees:
        role_id = true_roles[(employee.job_title, employee.job_family_code.strip())]
        role = catalog.get_role(mini_inputs.catalog, role_id)
        given[employee.employee_id] = catalog.standard_title(
            mini_inputs.catalog, role, employee.grade
        )
    return given


# --- Standard titles for the employees of the mini data set ---------------------------


def test_the_worked_example_on_the_employees_file(mini_inputs: InputData, titles):
    # 1.3: "ACCT MGR II" (Sales, grade 8) and "Senior Account Manager" (Sales,
    # grade 7) are one role and one level, so one Long and one Short Title.
    by_id = {employee.employee_id: employee for employee in mini_inputs.employees}
    assert (by_id["E001"].job_title, by_id["E001"].grade) == ("ACCT MGR II", 8)
    assert (by_id["E004"].job_title, by_id["E004"].grade) == ("Senior Account Manager", 7)
    assert (by_id["E006"].job_title, by_id["E006"].grade) == ("Sr. Acct Mgr - NYC", 7)
    expected = StandardTitle("Senior", "Senior Account Manager", "Sr Acct Mgr")
    assert [titles[employee_id] for employee_id in ("E001", "E004", "E006")] == [expected] * 3


@pytest.mark.parametrize(
    ("employee_id", "expected"),
    [
        ("E007", StandardTitle("Junior", "Junior Sales Representative", "Jr Sales Rep")),
        ("E008", StandardTitle("Associate", "Associate Sales Representative", "Assoc Sales Rep")),
        ("E016", StandardTitle("Senior", "Senior Financial Analyst", "Sr Fin Analyst")),
        ("E018", StandardTitle("Lead", "Lead Internal Auditor", "Lead Int Auditor")),
        # E020 is recorded in family "FIN ", which is family FIN (S05).
        ("E020", StandardTitle("Principal", "Principal Treasury Analyst", "Prin Treasury Analyst")),
        # E021 has no family code; its true role is still named by the rule.
        ("E021", StandardTitle("Associate", "Associate Accountant", "Assoc Accountant")),
    ],
)
def test_titles_follow_the_grade_of_each_employee(titles, employee_id, expected):
    assert titles[employee_id] == expected


def test_an_employee_with_no_grade_gets_the_title_with_no_level_word(
    mini_inputs: InputData, titles
):
    no_grade = [e.employee_id for e in mini_inputs.employees if e.grade is None]
    assert no_grade == ["E012", "E022"]
    assert titles["E012"] == StandardTitle(
        None, "Business Development Representative", "Biz Dev Rep"
    )
    assert titles["E022"] == StandardTitle(None, "Financial Analyst", "Fin Analyst")
    with_level = [employee_id for employee_id, title in titles.items() if title.level is not None]
    assert len(with_level) == len(mini_inputs.employees) - 2


def test_each_title_carries_the_level_hrs_table_gives_the_grade(mini_inputs: InputData, titles):
    ladder = {level.level: level for level in mini_inputs.catalog.ladder}
    table = {row.grade: row.level for row in mini_inputs.catalog.grade_levels}
    for employee in mini_inputs.employees:
        title = titles[employee.employee_id]
        assert title.level == table.get(employee.grade)
        if title.level is not None:
            words = ladder[title.level]
            assert title.long_title.startswith(f"{words.long_word} ")
            assert title.short_title.startswith(f"{words.short_word} ")


# --- The guard on what the files hold (2.7) -----------------------------------------------


def test_every_title_given_to_an_employee_passes_the_guard(mini_inputs: InputData, titles):
    assert len(titles) == 22
    for title in titles.values():
        assert catalog.is_catalog_title(mini_inputs.catalog, title.long_title, title.short_title)


def test_a_legacy_title_passes_only_where_the_catalog_holds_the_same_words(
    mini_inputs: InputData, mini_rows: tuple[TitleRow, ...]
):
    mini = mini_inputs.catalog
    raw = sorted({employee.job_title for employee in mini_inputs.employees})
    assert len(raw) == 17
    # One legacy title is, as it stands, the name and the short name of a role.
    treasury = catalog.get_role(mini, "FIN-06")
    assert treasury.role_name == treasury.short_name == "Treasury Analyst"
    assert [title for title in raw if catalog.is_catalog_title(mini, title, title)] == [
        "Treasury Analyst"
    ]
    # A cleaned title is lower-case, and no catalog title is.
    for row in mini_rows:
        cleaned = row.record.cleaned_title
        assert not catalog.is_catalog_title(mini, cleaned, cleaned)


def test_a_legacy_title_beside_a_real_title_is_refused(mini_inputs: InputData, titles):
    # A release that put the legacy title in either column would be stopped,
    # unless the legacy title is that very column's standard title already.
    mini = mini_inputs.catalog
    already_standard = []
    for employee in mini_inputs.employees:
        legacy, title = employee.job_title, titles[employee.employee_id]
        assert legacy != title.short_title
        assert not catalog.is_catalog_title(mini, title.long_title, legacy)
        if legacy == title.long_title:
            already_standard.append(employee.employee_id)
        else:
            assert not catalog.is_catalog_title(mini, legacy, title.short_title)
    # "Senior Account Manager" and "Senior Financial Analyst" are legacy titles
    # that are the Long Title of their employees already.
    assert already_standard == ["E004", "E005", "E016"]


# --- Families, as the other steps use them --------------------------------------------------


def test_usable_families_agree_with_the_hand_written_title_rows(
    mini_inputs: InputData, mini_rows: tuple[TitleRow, ...]
):
    mini = mini_inputs.catalog
    # What the prepare step is handed to tell a usable family code (Q26).
    assert catalog.family_codes(mini) == ("FIN", "SAL")
    for row in mini_rows:
        assert catalog.is_usable_family(mini, row.key.family_code) is row.family_usable
    # The code as recorded gives the same answer as the trimmed code of the key (S05).
    row_of = {employee_id: row for row in mini_rows for employee_id in row.employee_ids}
    for employee in mini_inputs.employees:
        usable = catalog.is_usable_family(mini, employee.job_family_code)
        assert usable is row_of[employee.employee_id].family_usable
    assert [e.employee_id for e in mini_inputs.employees if e.job_family_code == "FIN "] == ["E020"]


def test_decided_roles_are_roles_a_model_may_be_shown(mini_inputs: InputData, true_roles):
    # The seed look-up and the truth name real, non-generic roles, in the
    # recorded family where there is one: the checks of the readers and the
    # catalog functions say the same thing.
    mini = mini_inputs.catalog
    visible = catalog.model_visible_roles(mini)
    decided = [(seed.job_family_code, seed.role_id) for seed in mini_inputs.seed_lookup]
    decided += [(family, role_id) for (_, family), role_id in sorted(true_roles.items())]
    assert len(decided) == 2 + 17
    for family, role_id in decided:
        role = catalog.get_role(mini, role_id)
        assert role in visible
        if catalog.is_usable_family(mini, family):
            assert role in catalog.roles_in_family(mini, family)
    # The one title with no usable family has a true role all the same (Q26).
    assert true_roles[("Staff Accountant", "")] == "FIN-02"
    assert catalog.roles_in_family(mini, "") == ()


def test_the_readers_check_and_the_generic_role_agree(mini_inputs: InputData):
    mini = mini_inputs.catalog
    for family in catalog.family_codes(mini):
        generic = catalog.generic_role(mini, family)
        assert generic.is_generic and generic.family_code == family
        assert generic not in catalog.model_visible_roles(mini)
    # A catalog the readers refuse is one the catalog functions cannot answer for.
    without = dataclasses.replace(
        mini, roles=tuple(role for role in mini.roles if role.role_id != "FIN-GEN")
    )
    with pytest.raises(DataError, match="family FIN must have exactly one generic role"):
        datafiles.validate_inputs(dataclasses.replace(mini_inputs, catalog=without))
    with pytest.raises(CatalogError, match="family 'FIN' has 0 generic roles"):
        catalog.generic_role(without, "FIN")
    assert catalog.generic_role(without, "SAL").role_id == "SAL-GEN"


# --- The catalog is data: a changed file changes the answers ------------------------------


@pytest.fixture
def changed(tmp_path: Path, mini_dir: Path, mini_inputs: InputData) -> Catalog:
    """The mini catalog written with other words and one more family, and read back."""
    data_dir = tmp_path / "data"
    shutil.copytree(mini_dir, data_dir)
    mini = mini_inputs.catalog
    datafiles.write_level_ladder(
        data_dir,
        [
            dataclasses.replace(level, long_word="Staff", short_word="Stf")
            if level.level == "Senior"
            else level
            for level in mini.ladder
        ],
    )
    datafiles.write_grade_levels(
        data_dir, [row for row in mini.grade_levels if row.grade not in (1, 10)]
    )
    datafiles.write_catalog_roles(
        data_dir,
        [
            *mini.roles,
            CatalogRole("OPS-GEN", "OPS", "Operations", "Ops", "Generic role.", False, True),
            CatalogRole("OPS-01", "OPS", "Fleet Planner", "Fleet Plnr", "Plans routes."),
        ],
    )
    return datafiles.read_inputs(data_dir).catalog


def test_the_level_words_are_read_from_the_ladder_file(mini_catalog: Catalog, changed: Catalog):
    role = catalog.get_role(changed, "SAL-01")
    assert catalog.standard_title(changed, role, 8) == StandardTitle(
        "Senior", "Staff Account Manager", "Stf Acct Mgr"
    )
    assert catalog.is_catalog_title(changed, "Staff Account Manager", "Stf Acct Mgr")
    assert not catalog.is_catalog_title(changed, "Senior Account Manager", "Sr Acct Mgr")
    assert catalog.is_catalog_title(mini_catalog, "Senior Account Manager", "Sr Acct Mgr")
    assert not catalog.is_catalog_title(mini_catalog, "Staff Account Manager", "Stf Acct Mgr")


def test_a_grade_removed_from_the_table_is_not_usable(changed: Catalog):
    role = catalog.get_role(changed, "SAL-01")
    for grade in (1, 10):
        assert catalog.level_for_grade(changed, grade) is None
        assert catalog.standard_title(changed, role, grade) == StandardTitle(
            None, "Account Manager", "Acct Mgr"
        )
    assert catalog.level_for_grade(changed, 2) == "Junior"
    assert catalog.level_for_grade(changed, 9) == "Lead"
    # Principal is still on the ladder, so its titles are still catalog titles.
    assert catalog.is_catalog_title(changed, "Principal Account Manager", "Prin Acct Mgr")


def test_a_family_added_to_the_roles_file_is_usable(mini_catalog: Catalog, changed: Catalog):
    assert catalog.family_codes(changed) == ("FIN", "OPS", "SAL")
    assert catalog.is_usable_family(changed, " OPS ")
    assert not catalog.is_usable_family(mini_catalog, "OPS")
    assert [role.role_id for role in catalog.roles_in_family(changed, "OPS")] == ["OPS-01"]
    assert catalog.generic_role(changed, "OPS").role_id == "OPS-GEN"
    visible = [role.role_id for role in catalog.model_visible_roles(changed)]
    assert visible == sorted(visible) and len(visible) == 13
    assert "OPS-01" in visible and "OPS-GEN" not in visible
    assert catalog.is_catalog_title(changed, "Lead Fleet Planner", "Lead Fleet Plnr")
    assert not catalog.is_catalog_title(mini_catalog, "Lead Fleet Planner", "Lead Fleet Plnr")
