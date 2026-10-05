"""Catalog access (design 1.2; Q26 for a usable family code).

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
