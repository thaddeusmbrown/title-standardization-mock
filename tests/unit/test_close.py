"""The close: leftovers take their family's generic role, under the cap (1.8, Q25).

The section 2.7 test of this module: the generic family role is never assigned
past its cap, or to a Wave 1 or compliance-flagged title. Beside it, the rules
the decisions record adds: a title marked "no suitable role" is never generic
and never counts against the cap (Q03), and a leftover with no usable family
stays on its legacy title (Q26).
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from titlestd import catalog, close, seeding
from titlestd.catalog import CatalogError
from titlestd.contracts import (
    NAME_MASK,
    Catalog,
    Decision,
    DecisionSource,
    ExceptionReason,
    Leftover,
    Record,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings

EMPLOYEES = 100  # with this many employees a cap of n is a share of n hundredths


def _leftover(
    family: str,
    title: str,
    headcount: int = 1,
    *,
    wave1: bool = False,
    flagged: bool = False,
    no_role: bool = False,
    name: bool = False,
    usable: bool = True,
) -> Leftover:
    row = TitleRow(
        record=Record(title, family, "Some Department", ""),
        employee_ids=tuple(f"{family}/{title}/{n}" for n in range(headcount)),
        raw_titles=(title.upper(),),
        has_person_name=name,
        family_usable=usable,
    )
    return Leftover(row, on_wave1_list=wave1, compliance_fired=flagged, no_suitable_role=no_role)


def _close(leftovers, cap: int, mini_catalog: Catalog) -> dict[TitleKey, Decision]:
    """Close the leftovers under a cap of exactly cap employees; decisions by key."""
    settings = Settings(generic_cap_share=cap / EMPLOYEES)
    assert close.generic_cap(EMPLOYEES, settings) == cap
    decisions = close.close_leftovers(leftovers, EMPLOYEES, mini_catalog, settings)
    return {decision.key: decision for decision in decisions}


def _generic(decisions: dict[TitleKey, Decision]) -> list[str]:
    """The cleaned titles that took a generic role, in key order."""
    return [
        key.cleaned_title
        for key, decision in sorted(decisions.items())
        if decision.row is TableRow.GENERIC
    ]


def _legacy(family: str, title: str, reason: ExceptionReason) -> Decision:
    return Decision(
        TitleKey(family, title), TableRow.LEGACY, DecisionSource.NONE, legacy_reason=reason
    )


# --- The cap (Q25) ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("employees", "cap"),
    [(1500, 30), (1549, 30), (1550, 31), (1499, 29), (50, 1), (49, 0), (22, 0), (0, 0)],
)
def test_the_cap_is_two_percent_of_the_employees_rounded_down(employees: int, cap: int):
    assert close.generic_cap(employees, Settings()) == cap


@pytest.mark.parametrize(
    ("share", "employees", "cap"),
    [(0.0, 1500, 0), (1.0, 1500, 1500), (0.5, 7, 3), (0.25, 22, 5), (0.05, 1500, 75)],
)
def test_the_cap_follows_the_setting(share: float, employees: int, cap: int):
    assert close.generic_cap(employees, Settings(generic_cap_share=share)) == cap


@pytest.mark.parametrize(
    ("share", "employees", "cap"), [(0.018, 1500, 27), (0.29, 100, 29), (0.57, 100, 57)]
)
def test_the_cap_is_the_share_as_written_not_its_binary_float(
    share: float, employees: int, cap: int
):
    # Float arithmetic lands just under the whole number and would round a
    # whole employee off the cap.
    assert math.floor(share * employees) == cap - 1
    assert close.generic_cap(employees, Settings(generic_cap_share=share)) == cap


# --- What a decision of the close holds ---------------------------------------------


def test_a_leftover_that_fits_takes_its_familys_generic_role(mini_catalog: Catalog):
    leftovers = [_leftover("FIN", "special projects"), _leftover("SAL", "bdr", 2)]
    decisions = close.close_leftovers(leftovers, 1500, mini_catalog, Settings())
    # Code assigned it: Provisional, nobody named, not simulated, no error rate.
    assert decisions == (
        Decision(
            key=TitleKey("FIN", "special projects"),
            row=TableRow.GENERIC,
            source=DecisionSource.CODE,
            role_id="FIN-GEN",
            status=Status.PROVISIONAL,
            decided_by="",
            simulated=False,
            error_rate=None,
            legacy_reason=None,
        ),
        Decision(
            TitleKey("SAL", "bdr"),
            TableRow.GENERIC,
            DecisionSource.CODE,
            "SAL-GEN",
            Status.PROVISIONAL,
        ),
    )
    for decision in decisions:
        role = catalog.get_role(mini_catalog, decision.role_id)
        assert role.is_generic and role.family_code == decision.key.family_code


def test_a_leftover_that_does_not_fit_stays_on_its_legacy_title(mini_catalog: Catalog):
    decisions = close.close_leftovers(
        [_leftover("FIN", "special projects", 31)], 1500, mini_catalog, Settings()
    )
    assert decisions == (
        Decision(
            key=TitleKey("FIN", "special projects"),
            row=TableRow.LEGACY,
            source=DecisionSource.NONE,
            role_id=None,
            status=None,
            decided_by="",
            simulated=False,
            error_rate=None,
            legacy_reason=ExceptionReason.OVER_GENERIC_CAP,
        ),
    )


def test_there_is_one_decision_per_leftover_sorted_by_key(mini_catalog: Catalog):
    leftovers = [
        _leftover("SAL", "bdr"),
        _leftover("FIN", "treasury analyst", wave1=True),
        _leftover("", "staff accountant", usable=False),
        _leftover("SAL", "account lead", 40),
        _leftover("FIN", "special projects", no_role=True),
    ]
    decisions = close.close_leftovers(leftovers, 1500, mini_catalog, Settings())
    assert [decision.key for decision in decisions] == sorted(
        leftover.row.key for leftover in leftovers
    )
    assert [decision.row for decision in decisions] == [
        TableRow.LEGACY,
        TableRow.LEGACY,
        TableRow.LEGACY,
        TableRow.LEGACY,
        TableRow.GENERIC,
    ]
    # Every decision is one of the two kinds, and only a generic one has a role.
    for decision in decisions:
        generic = decision.row is TableRow.GENERIC
        assert (decision.role_id is not None) is generic
        assert (decision.status is Status.PROVISIONAL) is generic
        assert (decision.legacy_reason is None) is generic
        assert decision.source is (DecisionSource.CODE if generic else DecisionSource.NONE)


def test_no_leftovers_give_no_decisions(mini_catalog: Catalog):
    assert close.close_leftovers([], 1500, mini_catalog, Settings()) == ()


def test_a_title_cannot_be_a_leftover_twice(mini_catalog: Catalog):
    twice = [_leftover("FIN", "special projects"), _leftover("FIN", "special projects", 2)]
    with pytest.raises(ValueError, match="'special projects' in family 'FIN' is a leftover twice"):
        close.close_leftovers(twice, 1500, mini_catalog, Settings())


# --- Never past the cap (2.7, Q25) --------------------------------------------------

SIZES = (("a", 1), ("b", 1), ("c", 2), ("d", 3), ("e", 3), ("f", 5), ("g", 8))


@pytest.mark.parametrize("cap", range(25))
def test_the_generic_role_is_never_assigned_past_its_cap(mini_catalog: Catalog, cap: int):
    leftovers = [_leftover("FIN", title, headcount) for title, headcount in SIZES]
    decisions = _close(leftovers, cap, mini_catalog)
    taken = [headcount for title, headcount in SIZES if title in _generic(decisions)]
    assert sum(taken) <= cap
    # The titles taken are the smallest ones, and the next one would cross the cap.
    assert taken == [headcount for _, headcount in SIZES][: len(taken)]
    if len(taken) < len(SIZES):
        assert sum(taken) + SIZES[len(taken)][1] > cap
    else:
        assert cap >= 23


def test_a_title_that_reaches_the_cap_exactly_fits(mini_catalog: Catalog):
    leftovers = [_leftover("FIN", "a", 2), _leftover("FIN", "b", 3)]
    assert _generic(_close(leftovers, 5, mini_catalog)) == ["a", "b"]
    assert _generic(_close(leftovers, 4, mini_catalog)) == ["a"]


def test_a_title_is_taken_whole_or_not_at_all(mini_catalog: Catalog):
    # Two of the four would fit under the cap; a title is never split.
    leftovers = [_leftover("FIN", "a", 2), _leftover("FIN", "b", 4)]
    decisions = _close(leftovers, 4, mini_catalog)
    assert _generic(decisions) == ["a"]
    assert decisions[TitleKey("FIN", "b")] == _legacy("FIN", "b", ExceptionReason.OVER_GENERIC_CAP)


def test_with_a_cap_of_zero_nobody_takes_the_generic_role(mini_catalog: Catalog):
    leftovers = [_leftover("FIN", "a"), _leftover("SAL", "b")]
    decisions = _close(leftovers, 0, mini_catalog)
    assert [decision.legacy_reason for decision in decisions.values()] == [
        ExceptionReason.OVER_GENERIC_CAP
    ] * 2


def test_the_cap_is_one_for_the_whole_run_not_one_per_family(mini_catalog: Catalog):
    leftovers = [_leftover("FIN", "a", 3), _leftover("SAL", "b", 3)]
    decisions = _close(leftovers, 5, mini_catalog)
    assert _generic(decisions) == ["a"]
    assert decisions[TitleKey("SAL", "b")].legacy_reason is ExceptionReason.OVER_GENERIC_CAP


# --- The fill order (Q25) -----------------------------------------------------------


def test_leftovers_are_taken_in_ascending_headcount(mini_catalog: Catalog):
    leftovers = [_leftover("FIN", "a", 3), _leftover("FIN", "b", 1), _leftover("FIN", "c", 2)]
    assert _generic(_close(leftovers, 3, mini_catalog)) == ["b", "c"]
    assert _generic(_close(leftovers, 2, mini_catalog)) == ["b"]


def test_equal_headcounts_go_by_family_code_then_cleaned_title(mini_catalog: Catalog):
    leftovers = [
        _leftover("SAL", "a"),
        _leftover("SAL", "b"),
        _leftover("FIN", "z"),
        _leftover("FIN", "y"),
    ]
    taken = {cap: _generic(_close(leftovers, cap, mini_catalog)) for cap in range(5)}
    assert taken == {0: [], 1: ["y"], 2: ["y", "z"], 3: ["y", "z", "a"], 4: ["y", "z", "a", "b"]}


def test_the_first_title_that_does_not_fit_ends_the_fill(mini_catalog: Catalog):
    leftovers = [
        _leftover("FIN", title, size) for title, size in zip("abcd", (1, 3, 3, 3), strict=True)
    ]
    decisions = _close(leftovers, 5, mini_catalog)
    assert _generic(decisions) == ["a", "b"]
    over = ExceptionReason.OVER_GENERIC_CAP
    assert [decisions[TitleKey("FIN", title)].legacy_reason for title in "cd"] == [over, over]


def test_the_outcome_does_not_depend_on_the_order_of_the_leftovers(mini_catalog: Catalog):
    leftovers = [
        _leftover("SAL", "a", 2),
        _leftover("FIN", "b", 2),
        _leftover("FIN", "c", 1),
        _leftover("SAL", "d", 1, wave1=True),
        _leftover("FIN", "e", 3),
    ]
    settings = Settings(generic_cap_share=0.04)
    expected = close.close_leftovers(leftovers, EMPLOYEES, mini_catalog, settings)
    assert _generic({decision.key: decision for decision in expected}) == ["b", "c"]
    for shuffled in itertools.permutations(leftovers):
        assert close.close_leftovers(shuffled, EMPLOYEES, mini_catalog, settings) == expected


# --- Never to a Wave 1 or compliance-flagged title (2.7, 1.8) -----------------------


def test_a_wave1_title_never_takes_the_generic_role(mini_catalog: Catalog):
    # It is the smallest leftover and the cap has room for everyone: what
    # keeps it on its legacy title is the Wave 1 list, not the cap.
    leftovers = [_leftover("FIN", "widely held", 1, wave1=True), _leftover("FIN", "other", 2)]
    decisions = _close(leftovers, EMPLOYEES, mini_catalog)
    assert _generic(decisions) == ["other"]
    assert decisions[TitleKey("FIN", "widely held")] == _legacy(
        "FIN", "widely held", ExceptionReason.UNDECIDED_BARRED
    )


def test_a_compliance_flagged_title_never_takes_the_generic_role(mini_catalog: Catalog):
    leftovers = [_leftover("FIN", "internal auditor", 1, flagged=True), _leftover("FIN", "x", 2)]
    decisions = _close(leftovers, EMPLOYEES, mini_catalog)
    assert _generic(decisions) == ["x"]
    assert decisions[TitleKey("FIN", "internal auditor")] == _legacy(
        "FIN", "internal auditor", ExceptionReason.UNDECIDED_BARRED
    )


def test_a_barred_title_leaves_the_cap_to_the_others(mini_catalog: Catalog):
    # If the barred titles counted against the cap, only one of the others would fit.
    leftovers = [
        _leftover("FIN", "a", 1, wave1=True),
        _leftover("FIN", "b", 1, flagged=True),
        _leftover("FIN", "c", 2),
        _leftover("SAL", "d", 2),
    ]
    assert _generic(_close(leftovers, 4, mini_catalog)) == ["c", "d"]


def test_no_mix_of_leftovers_breaks_the_rule(mini_catalog: Catalog):
    # 200 made-up closes: whatever the leftovers and the cap, the employees on
    # a generic role stay at or under the cap, and no barred title is among them.
    # The draws go through titlestd.seeding like every other (S04), on a stream
    # of this test's own that no run uses.
    for case in range(200):
        draw = seeding.rng(case, "test: a mix of leftovers")
        leftovers = [
            _leftover(
                draw.choice(["FIN", "SAL", "", "XXX"]),
                f"title {number}",
                draw.choice([1, 1, 1, 2, 2, 3, 5, 9]),
                wave1=draw.random() < 0.15,
                flagged=draw.random() < 0.15,
                no_role=draw.random() < 0.1,
                name=draw.random() < 0.1,
            )
            for number in range(draw.randint(0, 25))
        ]
        leftovers = [
            dataclasses.replace(
                leftover,
                row=dataclasses.replace(
                    leftover.row, family_usable=leftover.row.key.family_code in ("FIN", "SAL")
                ),
            )
            for leftover in leftovers
        ]
        cap = draw.randint(0, 30)
        decisions = _close(leftovers, cap, mini_catalog)
        assert len(decisions) == len(leftovers)
        on_generic = 0
        for leftover in leftovers:
            decision = decisions[leftover.row.key]
            if decision.row is not TableRow.GENERIC:
                assert decision.row is TableRow.LEGACY and decision.role_id is None
                continue
            on_generic += leftover.row.headcount
            assert not leftover.on_wave1_list and not leftover.compliance_fired
            assert not leftover.no_suitable_role and not leftover.row.has_person_name
            assert decision.role_id == f"{leftover.row.key.family_code}-GEN"
        assert on_generic <= cap
        # Nothing that could still be taken was left behind: an eligible title
        # over the cap holds more employees than the cap has room for.
        for leftover in leftovers:
            if decisions[leftover.row.key].legacy_reason is ExceptionReason.OVER_GENERIC_CAP:
                assert on_generic + leftover.row.headcount > cap


# --- "No suitable role" is never a generic role (Q03) -------------------------------


def test_a_title_marked_no_suitable_role_never_takes_the_generic_role(mini_catalog: Catalog):
    # Otherwise eligible, the smallest leftover, and room for everyone.
    leftovers = [_leftover("FIN", "odd job", 1, no_role=True), _leftover("FIN", "other", 2)]
    decisions = _close(leftovers, EMPLOYEES, mini_catalog)
    assert _generic(decisions) == ["other"]
    assert decisions[TitleKey("FIN", "odd job")] == _legacy(
        "FIN", "odd job", ExceptionReason.NO_SUITABLE_ROLE
    )


def test_a_title_marked_no_suitable_role_never_counts_against_the_cap(mini_catalog: Catalog):
    # With the marked title counted, the cap of 4 would hold only one of the two others.
    marked = _leftover("FIN", "odd job", 1, no_role=True)
    others = [_leftover("FIN", "a", 2), _leftover("SAL", "b", 2)]
    assert _generic(_close([marked, *others], 4, mini_catalog)) == ["a", "b"]
    assert _generic(_close(others, 4, mini_catalog)) == ["a", "b"]


def test_no_suitable_role_is_the_reason_whatever_else_bars_the_title(mini_catalog: Catalog):
    # A Wave 1 title with no truth row is marked by its SME (Q03): it waits for
    # the catalog owner, and the exceptions list says so.
    for extra in (
        {"wave1": True},
        {"flagged": True},
        {"name": True},
        {"usable": False},
        {"wave1": True, "flagged": True, "name": True},
    ):
        leftover = _leftover("FIN", "odd job", no_role=True, **extra)
        decisions = _close([leftover], EMPLOYEES, mini_catalog)
        assert decisions[leftover.row.key] == _legacy(
            "FIN", "odd job", ExceptionReason.NO_SUITABLE_ROLE
        )


# --- No usable family, or no model ever scored it (Q25, Q26) ------------------------


@pytest.mark.parametrize("family", ["", "XXX", "fin"])
def test_a_leftover_with_no_usable_family_stays_on_its_legacy_title(
    mini_catalog: Catalog, family: str
):
    # With no family there is no generic role to take (Q26), however much room
    # the cap has. The close does not ask the catalog for one.
    leftovers = [_leftover(family, "staff accountant", usable=False), _leftover("FIN", "other")]
    decisions = _close(leftovers, EMPLOYEES, mini_catalog)
    assert _generic(decisions) == ["other"]
    assert decisions[TitleKey(family, "staff accountant")] == _legacy(
        family, "staff accountant", ExceptionReason.UNDECIDED_UNSCORED
    )


@pytest.mark.parametrize("family", ["", "XXX"])
def test_a_family_the_catalog_does_not_hold_is_not_usable_whatever_the_row_says(
    mini_catalog: Catalog, family: str
):
    # The row's flag comes from the prepare step. The catalog is asked as well,
    # so a wrong flag cannot send the close after a generic role that is not there.
    leftover = _leftover(family, "staff accountant", usable=True)
    assert leftover.row.model_eligible
    with pytest.raises(CatalogError):
        catalog.generic_role(mini_catalog, family)
    decisions = _close([leftover], EMPLOYEES, mini_catalog)
    assert decisions[leftover.row.key].legacy_reason is ExceptionReason.UNDECIDED_UNSCORED


def test_a_family_the_row_calls_unusable_is_not_given_its_generic_role(mini_catalog: Catalog):
    leftover = _leftover("FIN", "staff accountant", usable=False)
    assert catalog.is_usable_family(mini_catalog, "FIN")
    decisions = _close([leftover], EMPLOYEES, mini_catalog)
    assert decisions[leftover.row.key].legacy_reason is ExceptionReason.UNDECIDED_UNSCORED


def test_a_title_with_a_persons_name_stays_on_its_legacy_title(mini_catalog: Catalog):
    # No model scored it (Q14), so it is not eligible (Q25).
    title = f"customer success manager {NAME_MASK}"
    decisions = _close([_leftover("SAL", title, name=True)], EMPLOYEES, mini_catalog)
    assert decisions[TitleKey("SAL", title)] == _legacy(
        "SAL", title, ExceptionReason.UNDECIDED_UNSCORED
    )


@pytest.mark.parametrize("bar", [{"wave1": True}, {"flagged": True}])
@pytest.mark.parametrize("unscored", [{"name": True}, {"usable": False}])
def test_widely_held_or_flagged_is_the_reason_before_name_or_family(
    mini_catalog: Catalog, bar: dict, unscored: dict
):
    # S11 keeps the other reason for leftovers that are not widely held or flagged.
    leftover = _leftover("FIN", "x", **bar, **unscored)
    decisions = _close([leftover], EMPLOYEES, mini_catalog)
    assert decisions[leftover.row.key].legacy_reason is ExceptionReason.UNDECIDED_BARRED


def test_only_an_eligible_leftover_is_ever_over_the_cap(mini_catalog: Catalog):
    leftovers = [
        _leftover("FIN", "a", wave1=True),
        _leftover("FIN", "b", flagged=True),
        _leftover("FIN", "c", no_role=True),
        _leftover("FIN", "d", name=True),
        _leftover("", "e", usable=False),
        _leftover("FIN", "f"),
    ]
    decisions = _close(leftovers, 0, mini_catalog)
    reasons = [decision.legacy_reason for _, decision in sorted(decisions.items())]
    assert reasons == [
        ExceptionReason.UNDECIDED_UNSCORED,  # "", e
        ExceptionReason.UNDECIDED_BARRED,  # a
        ExceptionReason.UNDECIDED_BARRED,  # b
        ExceptionReason.NO_SUITABLE_ROLE,  # c
        ExceptionReason.UNDECIDED_UNSCORED,  # d
        ExceptionReason.OVER_GENERIC_CAP,  # f
    ]
    # "No grade" is a reason for an employee, never for a title.
    assert ExceptionReason.NO_GRADE not in reasons


# --- The generic role comes from the catalog ----------------------------------------


def test_the_generic_role_is_read_from_the_catalog(mini_catalog: Catalog):
    renamed = dataclasses.replace(
        mini_catalog,
        roles=tuple(
            dataclasses.replace(role, role_id="FIN-ANY") if role.role_id == "FIN-GEN" else role
            for role in mini_catalog.roles
        ),
    )
    decisions = _close([_leftover("FIN", "a"), _leftover("SAL", "b")], EMPLOYEES, renamed)
    assert [decision.role_id for _, decision in sorted(decisions.items())] == [
        "FIN-ANY",
        "SAL-GEN",
    ]


def test_a_family_without_its_generic_role_stops_the_close(mini_catalog: Catalog):
    # The readers refuse such a catalog. The close does not paper over it by
    # leaving the title on its legacy title.
    without = dataclasses.replace(
        mini_catalog, roles=tuple(role for role in mini_catalog.roles if role.role_id != "FIN-GEN")
    )
    with pytest.raises(CatalogError, match="family 'FIN' has 0 generic roles"):
        _close([_leftover("FIN", "a")], EMPLOYEES, without)
    assert _generic(_close([_leftover("SAL", "b")], EMPLOYEES, without)) == ["b"]


def test_at_the_default_share_the_mini_data_set_has_no_room(mini_inputs, mini_rows):
    # 2% of 22 employees is less than one employee.
    settings = Settings()
    assert close.generic_cap(len(mini_inputs.employees), settings) == 0
    decisions = close.close_leftovers(
        [Leftover(row) for row in mini_rows],
        len(mini_inputs.employees),
        mini_inputs.catalog,
        settings,
    )
    assert {decision.row for decision in decisions} == {TableRow.LEGACY}
