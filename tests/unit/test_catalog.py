"""Catalog access, grade to level and the naming rule (design 1.2, 1.3, 2.4).

One test of section 2.7 starts here: model_visible_roles is the only source of
the roles a model may be shown, so it must never hold a generic family role.
"""

from __future__ import annotations

import dataclasses

import pytest

from titlestd import catalog
from titlestd.catalog import CatalogError
from titlestd.contracts import (
    Catalog,
    CatalogRole,
    GradeLevel,
    LadderLevel,
    StandardTitle,
    TitlestdError,
)


def _role(
    role_id: str,
    family: str,
    name: str,
    short: str,
    *,
    generic: bool = False,
    flagged: bool = False,
) -> CatalogRole:
    return CatalogRole(role_id, family, name, short, f"What a {name} does.", flagged, generic)


def _ids(roles) -> list[str]:
    return [role.role_id for role in roles]


# A catalog that gives nothing away by its looks: roles out of role_id order,
# generic roles whose ids and positions are like any other role's, level names
# that are not the level words, a grade table in no order, and a top level that
# no grade leads to.
ODD = Catalog(
    roles=(
        _role("R-30", "OPS", "Dispatcher", "Disp"),
        _role("R-20", "OPS", "Operations", "Ops", generic=True),
        _role("R-10", "OPS", "Planner", "Plnr"),
        _role("R-25", "LAW", "Counsel", "Csl", flagged=True),
        _role("R-05", "LAW", "Legal", "Legal", generic=True, flagged=True),
    ),
    ladder=(
        LadderLevel("L1", "Junior", "Jr"),
        LadderLevel("L2", "Senior", "Sr"),
        LadderLevel("L3", "Principal", "Prin"),
    ),
    grade_levels=(GradeLevel(5, "L1"), GradeLevel(2, "L2"), GradeLevel(9, "L1")),
)


def _with_roles(base: Catalog, *roles: CatalogRole) -> Catalog:
    return dataclasses.replace(base, roles=roles)


# --- Catalog access ---------------------------------------------------------------


def test_family_codes_are_distinct_and_sorted(mini_catalog: Catalog):
    assert mini_catalog.roles[0].family_code == "SAL"  # the file lists SAL first
    assert catalog.family_codes(mini_catalog) == ("FIN", "SAL")
    assert catalog.family_codes(ODD) == ("LAW", "OPS")


@pytest.mark.parametrize("code", ["SAL", "FIN", "FIN ", " SAL", "\tFIN\n"])
def test_a_family_code_is_usable_after_trimming(mini_catalog: Catalog, code: str):
    assert catalog.is_usable_family(mini_catalog, code)


@pytest.mark.parametrize(
    "code", ["", "   ", "XXX", "fin", "Sal", "S AL", "SALFIN", "FIN-GEN", "SAL-01", "Sales"]
)
def test_a_blank_or_unknown_family_code_is_not_usable(mini_catalog: Catalog, code: str):
    assert not catalog.is_usable_family(mini_catalog, code)


def test_a_blank_code_is_never_a_family_whatever_the_catalog_holds():
    # The reader refuses such a catalog; Q26 still says blank is unusable.
    blank = _with_roles(ODD, _role("X-01", "", "Clerk", "Clk"), _role("X-02", " ", "Aide", "Aide"))
    assert not catalog.is_usable_family(blank, "")
    assert not catalog.is_usable_family(blank, " ")
    assert catalog.roles_in_family(blank, "") == ()


def test_get_role_finds_any_role_by_its_id(mini_catalog: Catalog):
    assert catalog.get_role(mini_catalog, "SAL-01").role_name == "Account Manager"
    assert catalog.get_role(mini_catalog, "FIN-GEN").is_generic
    for role in mini_catalog.roles:
        assert catalog.get_role(mini_catalog, role.role_id) is role


@pytest.mark.parametrize("role_id", ["SAL-99", "sal-01", " SAL-01", "SAL", "Account Manager", ""])
def test_get_role_gives_none_for_an_id_the_catalog_does_not_hold(mini_catalog: Catalog, role_id):
    assert catalog.get_role(mini_catalog, role_id) is None


# --- No model is ever shown a generic family role (2.7) --------------------------------


def test_no_generic_family_role_is_model_visible(mini_catalog: Catalog):
    visible = catalog.model_visible_roles(mini_catalog)
    assert not [role.role_id for role in visible if role.is_generic]
    assert {"SAL-GEN", "FIN-GEN"}.isdisjoint(_ids(visible))
    # The catalog does hold them, so a function that returned every role would fail here.
    assert {"SAL-GEN", "FIN-GEN"} <= set(_ids(mini_catalog.roles))


def test_every_other_role_is_model_visible_in_role_id_order(mini_catalog: Catalog):
    assert _ids(catalog.model_visible_roles(mini_catalog)) == [
        *(f"FIN-0{n}" for n in range(1, 7)),
        *(f"SAL-0{n}" for n in range(1, 7)),
    ]
    # A compliance-flagged role is shown to the models; the rules keep a machine
    # from settling it.
    assert catalog.get_role(mini_catalog, "FIN-04").compliance_flag


def test_a_generic_role_is_told_by_its_flag_not_by_its_id_or_its_place():
    assert _ids(catalog.model_visible_roles(ODD)) == ["R-10", "R-25", "R-30"]
    only_generic = _with_roles(ODD, _role("R-20", "OPS", "Operations", "Ops", generic=True))
    assert catalog.model_visible_roles(only_generic) == ()


def test_roles_in_family_are_its_non_generic_roles_in_role_id_order(mini_catalog: Catalog):
    six = range(1, 7)
    assert _ids(catalog.roles_in_family(mini_catalog, "SAL")) == [f"SAL-0{n}" for n in six]
    assert _ids(catalog.roles_in_family(mini_catalog, "FIN ")) == [f"FIN-0{n}" for n in six]
    assert _ids(catalog.roles_in_family(ODD, "OPS")) == ["R-10", "R-30"]
    assert _ids(catalog.roles_in_family(ODD, "LAW")) == ["R-25"]


@pytest.mark.parametrize("code", ["", "  ", "XXX", "sal"])
def test_a_code_that_is_not_a_family_has_no_roles(mini_catalog: Catalog, code: str):
    assert catalog.roles_in_family(mini_catalog, code) == ()


def test_the_families_split_the_model_visible_roles_between_them(mini_catalog: Catalog):
    for subject in (mini_catalog, ODD):
        by_family = [
            role
            for family in catalog.family_codes(subject)
            for role in catalog.roles_in_family(subject, family)
        ]
        assert sorted(_ids(by_family)) == _ids(catalog.model_visible_roles(subject))


# --- The generic family role --------------------------------------------------------


def test_each_family_has_its_one_generic_role(mini_catalog: Catalog):
    assert catalog.generic_role(mini_catalog, "SAL").role_id == "SAL-GEN"
    assert catalog.generic_role(mini_catalog, " FIN ").role_id == "FIN-GEN"
    assert catalog.generic_role(ODD, "OPS").role_id == "R-20"
    assert catalog.generic_role(ODD, "LAW").role_id == "R-05"


def test_generic_and_model_visible_roles_are_the_whole_catalog_and_share_nothing(
    mini_catalog: Catalog,
):
    for subject in (mini_catalog, ODD):
        generic = [catalog.generic_role(subject, code) for code in catalog.family_codes(subject)]
        visible = catalog.model_visible_roles(subject)
        assert sorted(_ids(generic) + _ids(visible)) == sorted(_ids(subject.roles))
        assert set(_ids(generic)).isdisjoint(_ids(visible))


@pytest.mark.parametrize("code", ["", "  ", "XXX", "sal"])
def test_there_is_no_generic_role_without_a_usable_family(mini_catalog: Catalog, code: str):
    # Q26: with no family there is no generic role to take.
    with pytest.raises(CatalogError, match="0 generic roles"):
        catalog.generic_role(mini_catalog, code)


def test_a_family_must_have_exactly_one_generic_role():
    planner = _role("R-10", "OPS", "Planner", "Plnr")
    with pytest.raises(CatalogError, match="family 'OPS' has 0 generic roles"):
        catalog.generic_role(_with_roles(ODD, planner), " OPS")
    two = _with_roles(
        ODD,
        planner,
        _role("R-20", "OPS", "Operations", "Ops", generic=True),
        _role("R-21", "OPS", "Operations Other", "Ops Oth", generic=True),
    )
    with pytest.raises(CatalogError, match="family 'OPS' has 2 generic roles"):
        catalog.generic_role(two, "OPS")


def test_a_catalog_error_is_meant_for_the_person_running_the_program():
    assert issubclass(CatalogError, TitlestdError)


# --- Grade to level (1.2, Q36) ----------------------------------------------------------


@pytest.mark.parametrize(
    ("grade", "level"),
    [
        (1, "Junior"),
        (2, "Junior"),
        (3, "Junior"),
        (4, "Associate"),
        (5, "Associate"),
        (6, "Associate"),
        (7, "Senior"),
        (8, "Senior"),
        (9, "Lead"),
        (10, "Principal"),
    ],
)
def test_the_level_comes_from_the_grade_through_hrs_table(mini_catalog: Catalog, grade, level):
    assert catalog.level_for_grade(mini_catalog, grade) == level


@pytest.mark.parametrize("grade", [None, 0, 11, -1, 70])
def test_a_missing_or_unlisted_grade_has_no_level(mini_catalog: Catalog, grade):
    assert catalog.level_for_grade(mini_catalog, grade) is None


def test_the_level_is_looked_up_never_worked_out():
    # A table in no order, where a higher grade reads as a lower level.
    assert [catalog.level_for_grade(ODD, grade) for grade in (2, 5, 9)] == ["L2", "L1", "L1"]
    assert [catalog.level_for_grade(ODD, grade) for grade in (1, 3, 4, 6, 7, 8)] == [None] * 6


# --- The naming rule (1.3, 2.4) ------------------------------------------------------------


def test_the_worked_example_of_the_design(mini_catalog: Catalog):
    # 1.3 and Q36: "ACCT MGR II" (Sales, grade 8) and "Senior Account Manager"
    # (Sales, grade 7) are both Sales x Account Manager at level Senior.
    role = catalog.get_role(mini_catalog, "SAL-01")
    assert (role.family_code, role.role_name, role.short_name) == (
        "SAL",
        "Account Manager",
        "Acct Mgr",
    )
    expected = StandardTitle("Senior", "Senior Account Manager", "Sr Acct Mgr")
    assert catalog.standard_title(mini_catalog, role, 7) == expected
    assert catalog.standard_title(mini_catalog, role, 8) == expected


@pytest.mark.parametrize(
    ("grade", "level", "long_title", "short_title"),
    [
        (1, "Junior", "Junior Account Manager", "Jr Acct Mgr"),
        (3, "Junior", "Junior Account Manager", "Jr Acct Mgr"),
        (4, "Associate", "Associate Account Manager", "Assoc Acct Mgr"),
        (6, "Associate", "Associate Account Manager", "Assoc Acct Mgr"),
        (9, "Lead", "Lead Account Manager", "Lead Acct Mgr"),
        (10, "Principal", "Principal Account Manager", "Prin Acct Mgr"),
    ],
)
def test_each_level_puts_its_long_and_its_short_word_before_the_names(
    mini_catalog: Catalog, grade, level, long_title, short_title
):
    role = catalog.get_role(mini_catalog, "SAL-01")
    assert catalog.standard_title(mini_catalog, role, grade) == StandardTitle(
        level, long_title, short_title
    )


@pytest.mark.parametrize("grade", [None, 0, 11])
def test_with_no_usable_grade_both_titles_are_the_name_alone(mini_catalog: Catalog, grade):
    # 2.4: "No grade: the title with no level word". A grade the table does not
    # list is no more usable than a missing one.
    role = catalog.get_role(mini_catalog, "SAL-01")
    assert catalog.standard_title(mini_catalog, role, grade) == StandardTitle(
        None, "Account Manager", "Acct Mgr"
    )


def test_a_generic_family_role_is_named_like_any_other_role(mini_catalog: Catalog):
    generic = catalog.generic_role(mini_catalog, "SAL")
    assert catalog.standard_title(mini_catalog, generic, 7) == StandardTitle(
        "Senior", "Senior Sales", "Sr Sales"
    )
    assert catalog.standard_title(mini_catalog, generic, None) == StandardTitle(
        None, "Sales", "Sales"
    )


def test_the_words_are_the_ladders_not_the_name_of_the_level():
    planner = catalog.get_role(ODD, "R-10")
    senior = StandardTitle("L2", "Senior Planner", "Sr Plnr")
    junior = StandardTitle("L1", "Junior Planner", "Jr Plnr")
    assert catalog.standard_title(ODD, planner, 2) == senior
    assert catalog.standard_title(ODD, planner, 9) == junior


def test_the_names_are_used_exactly_as_the_catalog_holds_them():
    role = _role("R-40", "OPS", "Head of R&D, EMEA", "Hd R&D (EMEA)")
    assert catalog.standard_title(ODD, role, 2) == StandardTitle(
        "L2", "Senior Head of R&D, EMEA", "Sr Hd R&D (EMEA)"
    )


def test_a_grade_whose_level_is_not_on_the_ladder_is_an_error():
    # The reader refuses such a catalog; the naming rule does not guess a word.
    broken = dataclasses.replace(ODD, grade_levels=(*ODD.grade_levels, GradeLevel(4, "L9")))
    assert catalog.level_for_grade(broken, 4) == "L9"
    with pytest.raises(CatalogError, match="grade 4 names level L9"):
        catalog.standard_title(broken, catalog.get_role(broken, "R-10"), 4)
