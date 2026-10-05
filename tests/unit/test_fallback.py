"""The shared code around choose: record block, numbered roles, mapping, quote check.

docs/design.md 1.6 step 4, 1.7 and 2.5; docs/build-decisions.md Q15 (the quote
check), Q18 (the seeded shuffle), S16 (what a model is shown).

Two tests of section 2.7 start here. "A quote that is not in the record sends
the title to a person" rests on quote_in_record, and "no model is ever shown a
generic family role" on the two functions that make a numbered list: neither
numbers nor renders one.
"""

from __future__ import annotations

import pytest

from titlestd import fallback, seeding
from titlestd.contracts import (
    EMAIL_MASK,
    NAME_MASK,
    QUOTE_MAX_WORDS,
    QUOTE_MIN_CHARS,
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
    # Nothing is written in place of what is missing, so no word stands there
    # to be quoted. The label left alone on the line is another matter: see
    # test_label_text_alone_passes_the_check_on_every_record.
    for invented in ("none", "empty", "n/a", "not given"):
        assert not fallback.quote_in_record(invented, block)


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


# --- The number-to-role mapping (2.5, Q15) ----------------------------------------------------

# A list as the agent's is: numbers that are not positions, and a role with none.
SHOWN = (
    NumberedRole(2, TEN[0]),
    NumberedRole(None, TEN[1]),
    NumberedRole(1, TEN[2]),
    NumberedRole(5, TEN[3]),
)


def test_a_number_names_the_role_shown_under_it():
    numbered = fallback.number_roles(TEN, KEY, SEED)
    for shown in numbered:
        assert fallback.role_for_number(shown.number, numbered) is shown.role
    assert fallback.role_for_number(2, SHOWN) is TEN[0]
    assert fallback.role_for_number(1, SHOWN) is TEN[2]
    assert fallback.role_for_number(5, SHOWN) is TEN[3]


@pytest.mark.parametrize("number", [0, 11, 12, -1, 100, 10**9])
def test_a_number_out_of_range_names_no_role(number: int):
    # Q15: the rules then fail the catalog check and the title goes to a person.
    assert fallback.role_for_number(number, fallback.number_roles(TEN, KEY, SEED)) is None


@pytest.mark.parametrize("number", [3, 4, 0, 6])
def test_a_number_that_no_shown_role_carries_names_no_role(number: int):
    assert fallback.role_for_number(number, SHOWN) is None


def test_a_role_with_no_number_cannot_be_named():
    # Not by None either: an answer with no number must not become the first
    # role that happens to carry none.
    assert fallback.role_for_number(None, SHOWN) is None
    assert fallback.role_for_number(None, [NumberedRole(None, TEN[0])]) is None
    named = {fallback.role_for_number(number, SHOWN) for number in range(-3, 12)}
    assert TEN[1] not in named


@pytest.mark.parametrize("number", [True, False, 1.0, 2.5, "1", "one", b"1", [1], (1,)])
def test_only_a_whole_number_names_a_role(number):
    # True == 1 and 1.0 == 1 in Python; neither is a number a model was offered.
    assert fallback.role_for_number(number, SHOWN) is None


def test_no_number_names_a_role_of_an_empty_list():
    assert all(fallback.role_for_number(number, ()) is None for number in (None, 0, 1, 10))


# --- The quote check (1.7, Q15) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "senior account manager",  # the whole title
        "account manager",  # a run of its words
        "Owns the largest customer accounts",  # a run of description words
        "negotiates renewals. Reports to [NAME].",  # with its punctuation and the mask
        "[NAME]",
        "Enterprise Sales",  # the department is part of the record
        "SAL",  # exactly three characters
    ],
)
def test_a_quote_copied_from_the_record_passes(quote: str):
    assert fallback.quote_in_record(quote, BLOCK)


@pytest.mark.parametrize(
    "quote",
    [
        "SENIOR ACCOUNT MANAGER",
        "senior Account manager",
        "owns THE largest CUSTOMER accounts",
        "reports to [name]",
    ],
)
def test_case_is_folded_on_both_sides(quote: str):
    assert fallback.quote_in_record(quote, BLOCK)
    assert fallback.quote_in_record(quote.swapcase(), BLOCK.upper())


def test_case_is_folded_not_only_lowered():
    block = fallback.render_record(Record("leiter straße", "SAL", "", ""))
    assert fallback.quote_in_record("LEITER STRASSE", block)


@pytest.mark.parametrize(
    "quote",
    [
        "  senior account manager  ",
        "senior   account\tmanager",
        "senior\naccount\r\nmanager",
        "senior\N{NO-BREAK SPACE}account manager",  # a no-break space
    ],
)
def test_white_space_is_collapsed_in_the_quote(quote: str):
    assert fallback.quote_in_record(quote, BLOCK)


def test_white_space_is_collapsed_in_the_record_too():
    spaced = Record(
        "senior  account manager", "SAL", "", "Owns the\n largest\tcustomer   accounts."
    )
    block = fallback.render_record(spaced)
    assert fallback.quote_in_record("senior account manager", block)
    assert fallback.quote_in_record("the largest customer accounts", block)


@pytest.mark.parametrize(
    "quote",
    [
        "regional sales director",  # words the record does not hold
        "manages a portfolio of accounts",
        "account senior manager",  # its words, in another order
        "senior manager",  # its words, with one left out
        "Owns the largest accounts",
        "Owns the largest customer account",  # one letter short of a word
        "negotiates renewal",
        "negotiates renewals and reports to [NAME]",  # a paraphrase
        "senior account manager.",  # punctuation the record does not have there
        "senior-account-manager",
        "senioraccountmanager",
        "Reports to Oswin Thistlewood",  # the name the mask stands for
    ],
)
def test_a_quote_that_is_not_word_for_word_in_the_record_fails(quote: str):
    # 2.7: this is what sends the title to a person.
    assert not fallback.quote_in_record(quote, BLOCK)


@pytest.mark.parametrize(
    ("quote", "holds"),
    [
        ("count", "account"),  # starts inside a word
        ("account", "accounts"),  # ends inside a word
        ("ccoun", "account"),  # both
        ("nager", "manager"),
        ("senior acc", "senior account"),
        ("ior account manager", "senior account manager"),
        ("sum", "résumé"),  # a letter of another script is a letter
        ("cost", "cost_centre"),  # the underscore is a word character
        ("level", "level2"),  # and so is a digit
        ("2nd", "x2nd"),
    ],
)
def test_a_quote_must_start_and_end_on_a_word_boundary(quote: str, holds: str):
    block = fallback.render_record(Record(f"the {holds} desk", "SAL", "", ""))
    assert quote in block  # it is there, letter for letter
    assert not fallback.quote_in_record(quote, block)
    # The word it was cut from is a quote from the same record.
    assert fallback.quote_in_record(holds, block)


def test_one_place_where_the_quote_stands_as_words_is_enough():
    # "count" is first met inside "account", then as a word of its own.
    block = fallback.render_record(Record("account count clerk", "FIN", "", ""))
    assert fallback.quote_in_record("count", block)
    assert fallback.quote_in_record("count clerk", block)
    assert not fallback.quote_in_record("ccount", block)


@pytest.mark.parametrize(
    "quote",
    [
        "renewals.",  # ends with the record's full stop
        ". Reports to",  # starts with it
        "(EMEA)",
        "end",  # a hyphen ends a word
        "end-to-end",
        "to-end sales",
        "R&D",
        "[EMAIL]",
        "questions to [EMAIL];",
    ],
)
def test_punctuation_that_stands_in_the_record_may_begin_or_end_a_quote(quote: str):
    record = Record(
        "end-to-end sales lead (EMEA)",
        "SAL",
        "R&D",
        f"Negotiates renewals. Reports to {NAME_MASK}; questions to {EMAIL_MASK}; more.",
    )
    assert fallback.quote_in_record(quote, fallback.render_record(record))


@pytest.mark.parametrize("quote", ["", " ", "\n\t ", "a", "ii", "  ii  ", "i\n", "to"])
def test_an_empty_or_too_short_quote_fails(quote: str):
    # "ii" and "to" do stand in this record; they are under three characters.
    block = fallback.render_record(Record("account manager ii", "SAL", "", "Reports to a lead."))
    assert len(" ".join(quote.split())) < QUOTE_MIN_CHARS
    assert not fallback.quote_in_record(quote, block)


@pytest.mark.parametrize("quote", ["123", "2 3", "#20311", "--- 12", "(42)"])
def test_a_quote_without_a_letter_fails(quote: str):
    block = fallback.render_record(Record("analyst 123", "FIN", "2 3", "Ref #20311 --- 12 (42)."))
    assert " ".join(quote.split()) in block  # it is there, word for word
    assert not fallback.quote_in_record(quote, block)


def test_three_characters_with_a_letter_are_enough():
    block = fallback.render_record(Record("bdr", "SAL", "IT", "Grade a 1 role, tier b2."))
    for quote in ("bdr", "a 1", "b2."):
        assert len(quote) == QUOTE_MIN_CHARS
        assert fallback.quote_in_record(quote, block)


def test_a_quote_of_more_than_twenty_five_words_fails():
    words = [f"word{n}" for n in range(40)]
    block = fallback.render_record(Record("analyst", "FIN", "", " ".join(words)))
    at_the_limit = " ".join(words[5 : 5 + QUOTE_MAX_WORDS])
    assert len(at_the_limit.split()) == QUOTE_MAX_WORDS == 25
    assert fallback.quote_in_record(at_the_limit, block)
    assert fallback.quote_in_record("  ".join(words[5 : 5 + QUOTE_MAX_WORDS]) + "\n", block)
    assert not fallback.quote_in_record(" ".join(words[5 : 6 + QUOTE_MAX_WORDS]), block)
    assert not fallback.quote_in_record(" ".join(words), block)


def test_the_quote_is_checked_against_the_block_as_rendered():
    # Q15: the same string serves the prompt and the check, so its labels and
    # line breaks belong to it. A quote may therefore run across two fields.
    assert fallback.quote_in_record("Family: SAL", BLOCK)
    assert fallback.quote_in_record("senior account manager Family: SAL", BLOCK)
    # What the block does not hold fails, however near the record it is.
    assert not fallback.quote_in_record("senior account manager SAL", BLOCK)
    assert not fallback.quote_in_record("job title: senior account manager", BLOCK)


def test_label_text_alone_passes_the_check_on_every_record():
    # Decided in the module docstring: the labels are text of the block that
    # Q15 checks a quote against. A quote of nothing else is evidence of
    # nothing and passes all the same, also where the field under the label
    # is empty.
    bare = fallback.render_record(Record("bdr", "SAL", "", ""))
    for label_text in ("Title", "family", "Department", "Description:"):
        assert fallback.quote_in_record(label_text, BLOCK)
        assert fallback.quote_in_record(label_text, bare)
    # Two labels in a row are text of the block only where the field between them is empty.
    assert fallback.quote_in_record("Department: Description:", bare)
    assert not fallback.quote_in_record("Department: Description:", BLOCK)


def test_the_check_reads_nothing_but_the_quote_and_the_block():
    other = fallback.render_record(Record("sales engineer", "SAL", "Solutions", ""))
    assert fallback.quote_in_record("sales engineer", other)
    assert not fallback.quote_in_record("sales engineer", BLOCK)
    assert not fallback.quote_in_record("senior account manager", other)
    assert not fallback.quote_in_record("senior account manager", "")


def stands_as_words(quote: str, block: str) -> bool:
    """The word-boundary rule spelled out place by place, without the re module."""

    def in_a_word(character: str) -> bool:
        return character.isalnum() or character == "_"

    for start in range(len(block) - len(quote) + 1):
        end = start + len(quote)
        if block[start:end] != quote:
            continue
        cuts_a_word_before = start > 0 and in_a_word(block[start - 1]) and in_a_word(quote[0])
        cuts_a_word_after = end < len(block) and in_a_word(quote[-1]) and in_a_word(block[end])
        if not cuts_a_word_before and not cuts_a_word_after:
            return True
    return False


def test_every_slice_of_the_block_is_judged_by_the_word_boundary_rule():
    # Every piece of the block of 3 to 40 characters, wherever it starts and
    # ends: it passes exactly when it cuts no word in two.
    block = " ".join(BLOCK.casefold().split())
    slices = {
        block[start:end]
        for start in range(len(block))
        for end in range(start + QUOTE_MIN_CHARS, min(len(block), start + 40) + 1)
    }
    slices = {
        piece
        for piece in slices
        if piece == piece.strip() and any(character.isalpha() for character in piece)
    }
    passed = {piece for piece in slices if fallback.quote_in_record(piece, BLOCK)}
    assert passed == {piece for piece in slices if stands_as_words(piece, block)}
    assert len(passed) > 100 and len(slices - passed) > 1000
    assert {"account", "account manager", "sal", "[name]"} <= passed
    assert {"ccount", "account manage", "enior account manager", "sale"} <= slices - passed
