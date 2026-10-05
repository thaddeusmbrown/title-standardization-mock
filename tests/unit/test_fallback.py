"""The shared code around choose: the record block and the numbered roles.

docs/design.md 1.6 step 4 and 2.5; docs/build-decisions.md Q18 (the seeded
shuffle), S16 (what a model is shown).

One test of section 2.7 starts here. "No model is ever shown a generic family
role" rests on the two functions that make a numbered list: neither numbers
nor renders one.
"""

from __future__ import annotations

import pytest

from titlestd import fallback, seeding
from titlestd.contracts import (
    EMAIL_MASK,
    NAME_MASK,
    CatalogRole,
    NumberedRole,
    Record,
    TitleKey,
)
from titlestd.settings import Settings

SEED = Settings().seed

RECORD = Record(
    cleaned_title="senior account manager",
    family_code="SAL",
    department="Enterprise Sales",
    description=(
        f"Owns the largest customer accounts and negotiates renewals. Reports to {NAME_MASK}."
    ),
)
BLOCK = fallback.render_record(RECORD)


def role(role_id: str, name: str = "", family: str = "SAL", *, generic: bool = False):
    name = name or f"Role {role_id}"
    return CatalogRole(role_id, family, name, name, f"What a {name} does.", is_generic=generic)


# Ten roles as a shortlist holds them, here in no order at all.
TEN = [role(f"R-{n:02d}") for n in (7, 2, 10, 5, 1, 9, 3, 8, 6, 4)]
GENERIC = role("R-GEN", "Sales", generic=True)
KEY = TitleKey("SAL", "account manager ii")


def ids(numbered) -> list[str]:
    return [shown.role.role_id for shown in numbered]


# --- The record block (Q11, Q15, S16) ---------------------------------------------------------


def test_the_record_block_shows_each_field_on_a_labelled_line_of_its_own():
    assert BLOCK == (
        "Title: senior account manager\n"
        "Family: SAL\n"
        "Department: Enterprise Sales\n"
        "Description: Owns the largest customer accounts and negotiates renewals. "
        "Reports to [NAME]."
    )


def test_every_field_is_shown_verbatim():
    # Nothing is trimmed, re-cased or re-spaced: the stand-ins copy their
    # quotes from the record, and the masks must reach the model as they are.
    odd = Record(
        "customer success manager [NAME]",
        "FIN",
        "R&D / Field  Ops",
        f"Owns  the KEY accounts;\tquestions to {EMAIL_MASK}. ",
    )
    assert fallback.render_record(odd).split("\n") == [
        "Title: customer success manager [NAME]",
        "Family: FIN",
        "Department: R&D / Field  Ops",
        "Description: Owns  the KEY accounts;\tquestions to [EMAIL]. ",
    ]


def test_an_empty_field_leaves_its_label_alone_on_the_line():
    bare = Record("bdr", "SAL", "", "")
    block = fallback.render_record(bare)
    assert block == "Title: bdr\nFamily: SAL\nDepartment:\nDescription:"


def test_the_block_depends_on_the_record_alone():
    again = Record(RECORD.cleaned_title, RECORD.family_code, RECORD.department, RECORD.description)
    assert fallback.render_record(again) == BLOCK
    for changed in (
        Record("account manager", "SAL", RECORD.department, RECORD.description),
        Record(RECORD.cleaned_title, "FIN", RECORD.department, RECORD.description),
        Record(RECORD.cleaned_title, "SAL", "Inside Sales", RECORD.description),
        Record(RECORD.cleaned_title, "SAL", RECORD.department, ""),
    ):
        assert fallback.render_record(changed) != BLOCK


# --- The seeded shuffle and the numbering (Q18) -----------------------------------------------


def test_the_roles_are_numbered_from_one_each_once():
    numbered = fallback.number_roles(TEN, KEY, SEED)
    assert [shown.number for shown in numbered] == list(range(1, 11))
    assert sorted(ids(numbered)) == sorted(each.role_id for each in TEN)
    assert all(shown.role in TEN for shown in numbered)


def test_the_order_is_pinned_on_every_supported_python():
    # tests/unit/test_seeding.py pins this title's shuffle of 1 to 10 as
    # 4, 7, 1, 8, 10, 5, 2, 3, 9, 6: the roles sorted by role_id, so permuted.
    assert ids(fallback.number_roles(TEN, KEY, SEED)) == [
        "R-04",
        "R-07",
        "R-01",
        "R-08",
        "R-10",
        "R-05",
        "R-02",
        "R-03",
        "R-09",
        "R-06",
    ]


def test_the_order_is_the_shuffle_stream_s_permutation_of_the_roles_by_role_id():
    for key in (KEY, TitleKey("FIN", "treasury analyst"), TitleKey("SAL", "")):
        expected = sorted(TEN, key=lambda each: each.role_id)
        seeding.rng(SEED, seeding.STREAM_SHUFFLE, key.family_code, key.cleaned_title).shuffle(
            expected
        )
        assert [shown.role for shown in fallback.number_roles(TEN, key, SEED)] == expected


def test_the_order_does_not_depend_on_the_order_the_roles_arrive_in():
    expected = fallback.number_roles(TEN, KEY, SEED)
    by_id = sorted(TEN, key=lambda each: each.role_id)
    for arrival in (by_id, by_id[::-1], by_id[3:] + by_id[:3], iter(TEN)):
        assert fallback.number_roles(arrival, KEY, SEED) == expected


def test_the_order_depends_on_the_title_the_family_and_the_seed():
    base = ids(fallback.number_roles(TEN, KEY, SEED))
    assert ids(fallback.number_roles(TEN, TitleKey("SAL", "account manager"), SEED)) != base
    assert ids(fallback.number_roles(TEN, TitleKey("FIN", KEY.cleaned_title), SEED)) != base
    assert ids(fallback.number_roles(TEN, KEY, SEED + 1)) != base
    # Really shuffled: over many titles every role comes first, and second, at some point.
    orders = [
        ids(fallback.number_roles(TEN, TitleKey("SAL", f"title {n}"), SEED)) for n in range(200)
    ]
    assert len({tuple(order) for order in orders}) == 200
    for position in (0, 1, 9):
        assert {order[position] for order in orders} == {each.role_id for each in TEN}


def test_a_title_s_order_does_not_depend_on_which_titles_were_numbered_before():
    keys = [TitleKey("SAL", f"title {n}") for n in range(6)]
    forward = {key: fallback.number_roles(TEN, key, SEED) for key in keys}
    backward = {key: fallback.number_roles(TEN, key, SEED) for key in reversed(keys)}
    assert forward == backward


def test_a_shorter_shortlist_gives_a_shorter_list():
    # S02: a catalog with fewer than ten model-visible roles.
    three = fallback.number_roles(TEN[:3], KEY, SEED)
    assert [shown.number for shown in three] == [1, 2, 3]
    assert sorted(ids(three)) == sorted(each.role_id for each in TEN[:3])
    assert fallback.number_roles([], KEY, SEED) == ()


@pytest.mark.parametrize("position", [0, 4, 9])
def test_a_generic_family_role_is_never_numbered(position: int):
    # 2.7: no model is ever shown a generic family role, never in a numbered list.
    roles = [*TEN[:position], GENERIC, *TEN[position:9]]
    with pytest.raises(ValueError, match="R-GEN"):
        fallback.number_roles(roles, KEY, SEED)


def test_a_generic_role_is_told_by_its_flag_not_by_its_id():
    plain_looking = role("R-11", "Sales", generic=True)
    with pytest.raises(ValueError, match="generic family role R-11"):
        fallback.number_roles([*TEN[:9], plain_looking], KEY, SEED)
    # An id that looks generic on a role that is not is numbered like any other.
    looks_generic = role("R-GEN", "Sales Planner")
    assert "R-GEN" in ids(fallback.number_roles([*TEN[:9], looks_generic], KEY, SEED))


# --- The numbered list as a model is shown it (S16) -------------------------------------------

ACCOUNT_MANAGER = CatalogRole(
    "SAL-01",
    "SAL",
    "Account Manager",
    "Acct Mgr",
    "Manages a portfolio of customer accounts and renews their contracts.",
)
AUDITOR = CatalogRole(
    "FIN-04",
    "FIN",
    "Internal Auditor",
    "Int Auditor",
    "Tests financial controls and reports findings to the audit committee.",
    compliance_flag=True,
)


def test_the_list_shows_number_name_family_code_and_definition():
    shown = [NumberedRole(1, AUDITOR), NumberedRole(2, ACCOUNT_MANAGER)]
    assert fallback.render_roles(shown) == (
        "1. Internal Auditor (family FIN): "
        "Tests financial controls and reports findings to the audit committee.\n"
        "2. Account Manager (family SAL): "
        "Manages a portfolio of customer accounts and renews their contracts."
    )


def test_a_role_with_no_number_is_shown_unnumbered():
    # The agent sees a role outside the recorded family without a number (Q10).
    shown = [NumberedRole(1, ACCOUNT_MANAGER), NumberedRole(None, AUDITOR)]
    first, second = fallback.render_roles(shown).split("\n")
    assert first.startswith("1. Account Manager (family SAL): ")
    assert second == (
        "- Internal Auditor (family FIN): "
        "Tests financial controls and reports findings to the audit committee."
    )
    assert not any(character.isdigit() for character in second)


def test_the_list_keeps_the_order_and_the_numbers_it_is_given():
    shown = [NumberedRole(7, TEN[0]), NumberedRole(None, TEN[1]), NumberedRole(2, TEN[2])]
    lines = fallback.render_roles(shown).split("\n")
    assert [line.split(" ")[0] for line in lines] == ["7.", "-", "2."]
    assert [each.role_name in line for each, line in zip(TEN, lines, strict=False)] == [True] * 3
    assert fallback.render_roles([]) == ""


def test_no_role_id_and_no_score_is_shown():
    # S16. Short names and the compliance flag are not for a model either.
    text = fallback.render_roles([NumberedRole(1, AUDITOR), NumberedRole(None, ACCOUNT_MANAGER)])
    for hidden in ("FIN-04", "SAL-01", "Acct Mgr", "Int Auditor", "True", "flag"):
        assert hidden not in text


@pytest.mark.parametrize("number", [3, None])
def test_a_generic_family_role_is_never_rendered(number):
    # The last gate before a prompt: numbered or not, it is refused (2.7).
    shown = [NumberedRole(1, TEN[0]), NumberedRole(number, GENERIC), NumberedRole(2, TEN[1])]
    with pytest.raises(ValueError, match="R-GEN"):
        fallback.render_roles(shown)


def test_what_number_roles_returns_can_be_rendered():
    lines = fallback.render_roles(fallback.number_roles(TEN, KEY, SEED)).split("\n")
    assert [line.partition(". ")[0] for line in lines] == [str(n) for n in range(1, 11)]
