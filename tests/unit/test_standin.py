"""The stand-in LLM: word overlap after abbreviation expansion (2.4, Q15, Q17)."""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

from titlestd import standin
from titlestd.contracts import (
    EMAIL_MASK,
    LABEL_STAND_IN,
    NAME_MASK,
    QUOTE_MAX_WORDS,
    Abbreviation,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    LadderLevel,
    NumberedRole,
    Record,
)
from titlestd.settings import Settings
from titlestd.standin import StandIn, build_llm

LADDER = (
    LadderLevel("Junior", "Junior", "Jr"),
    LadderLevel("Associate", "Associate", "Assoc"),
    LadderLevel("Senior", "Senior", "Sr"),
    LadderLevel("Lead", "Lead", "Lead"),
    LadderLevel("Principal", "Principal", "Prin"),
)
ABBREVIATIONS = (
    Abbreviation("sr", "senior"),
    Abbreviation("acct", "account"),
    Abbreviation("mgr", "manager"),
    Abbreviation("ap", "accounts payable"),
)


def role(role_id: str, name: str, definition: str = "", family: str = "SAL") -> CatalogRole:
    return CatalogRole(role_id, family, name, name, definition)


def numbered(*roles: CatalogRole) -> list[NumberedRole]:
    """The roles numbered from 1 in the order given."""
    return [NumberedRole(number, each) for number, each in enumerate(roles, start=1)]


def record(title: str, description: str = "") -> Record:
    return Record(title, "SAL", "Enterprise Sales", description)


def make(**settings) -> StandIn:
    return StandIn(Settings(**settings), LADDER, ABBREVIATIONS)


def picked(answer: ChooseAnswer, roles: list[NumberedRole]) -> str | None:
    """The role id an answer's number stands for."""
    return next((n.role.role_id for n in roles if n.number == answer.number), None)


ACCOUNT_MANAGER = role(
    "R-01",
    "Account Manager",
    "Manages a portfolio of customer accounts and renews their contracts.",
)
SALES_REP = role(
    "R-02", "Sales Representative", "Sells products to new customers and closes deals."
)
ANALYST = role("R-03", "Financial Analyst", "Builds budgets, forecasts and variance reports.")
THREE = numbered(ACCOUNT_MANAGER, SALES_REP, ANALYST)


# --- Content words ------------------------------------------------------------


def test_content_words_are_lower_case_tokens():
    assert make().content_words("Treasury ANALYST, cash-desk") == {
        "treasury",
        "analyst",
        "cash",
        "desk",
    }


def test_an_abbreviation_is_expanded_when_a_whole_word_equals_it():
    words = make().content_words
    assert words("Acct MGR") == {"account", "manager"}
    assert words("AP Clerk") == {"accounts", "payable", "clerk"}
    assert words("mgrs apply") == {"mgrs", "apply"}  # not a whole word, not expanded


def test_an_expansion_is_not_expanded_again():
    chained = StandIn(Settings(), (), (Abbreviation("x", "y"), Abbreviation("y", "z")))
    assert chained.content_words("x") == {"y"}


def test_stop_words_are_not_content():
    assert make().content_words("Manages a portfolio of the accounts and their contracts") == {
        "manages",
        "portfolio",
        "accounts",
        "contracts",
    }
    assert {"a", "and", "of", "the", "their", "to", "with"} <= standin.STOP_WORDS


def test_it_and_us_stay_content_words():
    # Job titles use both as nouns ("IT Support", "US Payroll").
    assert make().content_words("IT Support for US Payroll") == {"it", "support", "us", "payroll"}


def test_level_words_long_and_short_are_not_content():
    words = make().content_words
    assert words("Senior Account Manager") == {"account", "manager"}
    assert words("Jr Assoc Lead Prin Principal Associate Junior analyst") == {"analyst"}
    assert words("Sr. Acct Mgr") == {"account", "manager"}  # expanded to a level word, then dropped


def test_level_numerals_are_not_content():
    words = make().content_words
    assert words("account manager ii") == {"account", "manager"}
    assert words("Analyst 2 / Analyst IV / analyst v") == {"analyst"}
    assert words("vi 6 analyst") == {"vi", "6", "analyst"}  # five levels: no sixth numeral
    assert {"1", "5", "i", "iii", "v"} <= standin.LEVEL_NUMERALS


def test_level_words_come_from_the_ladder_given():
    other = StandIn(Settings(), (LadderLevel("L1", "Chief", "Ch"),), ())
    assert other.content_words("Chief Senior Accountant Ch") == {"senior", "accountant"}


def test_the_masks_are_not_words():
    words = make().content_words
    assert words(f"Reports to {NAME_MASK}. Questions to {EMAIL_MASK}.") == {"reports", "questions"}
    assert words("name badge and email desk") == {"name", "badge", "email", "desk"}


# --- "insufficient data" (Q17 rule 1) -------------------------------------------


def test_one_title_word_and_no_description_is_insufficient_data():
    answer = make().choose(record("bdr"), THREE)
    assert answer == ChooseAnswer(ChooseKind.INSUFFICIENT_DATA, None, "bdr", LABEL_STAND_IN)


def test_title_words_are_counted_as_content_words_each_once():
    stand_in = make()
    for title in ("senior manager ii", "manager manager", "the manager of", "sr mgr"):
        assert stand_in.choose(record(title), THREE).answer is ChooseKind.INSUFFICIENT_DATA, title


def test_a_blank_description_is_no_description():
    assert make().choose(record("bdr", "  \n "), THREE).answer is ChooseKind.INSUFFICIENT_DATA


def test_two_title_words_are_enough_without_a_description():
    answer = make().choose(record("account manager"), THREE)
    assert (answer.answer, picked(answer, THREE)) == (ChooseKind.ROLE, "R-01")


def test_a_description_is_enough_with_one_title_word():
    answer = make().choose(record("bdr", "Sells products to new customers."), THREE)
    assert (answer.answer, picked(answer, THREE)) == (ChooseKind.ROLE, "R-02")


def test_insufficient_data_is_decided_before_any_role_is_scored():
    # A role named like the title would win on overlap; rule 1 comes first.
    roles = numbered(role("R-09", "BDR", "A bdr finds leads."), *[n.role for n in THREE])
    assert make().choose(record("bdr"), roles).answer is ChooseKind.INSUFFICIENT_DATA


def test_the_word_count_is_a_setting():
    assert (
        make(standin_min_title_words=3).choose(record("account manager"), THREE).answer
        is ChooseKind.INSUFFICIENT_DATA
    )
    # With the rule off the title is scored, and shares nothing with any role.
    assert (
        make(standin_min_title_words=0).choose(record("bdr"), THREE).answer is ChooseKind.NO_MATCH
    )


# --- The score and the ranking (Q17 rule 2) -------------------------------------


def test_a_name_word_in_the_title_counts_twice_more():
    # Account Manager: 2 x {manager} + {manager} = 3.
    # Sales Representative: {sells, products, closes} = 3. A tie on the score,
    # which a name word weighted once would have lost (2 against 3).
    answer = make().choose(record("manager", "Sells products, closes."), THREE)
    assert picked(answer, THREE) == "R-01"


def test_but_only_twice_more():
    # Sales Representative now shares four words: 4 against 3. A name word
    # weighted three times more would have tied and won on the tie-break.
    answer = make().choose(record("manager", "Sells products, closes deals."), THREE)
    assert picked(answer, THREE) == "R-02"


def test_title_words_also_count_against_the_definition():
    answer = make().choose(record("portfolio contracts"), THREE)
    assert (answer.answer, picked(answer, THREE)) == (ChooseKind.ROLE, "R-01")
    assert answer.quote == "portfolio contracts"


def test_description_words_also_count_against_the_role_name():
    answer = make().choose(record("widget wrangler", "The account manager for widgets."), THREE)
    assert (answer.answer, picked(answer, THREE)) == (ChooseKind.ROLE, "R-01")
    assert answer.quote == "account manager"


def test_a_tie_on_the_score_goes_to_more_name_words_in_the_title():
    # Both score 3; the role with the name word has the higher role_id and wins.
    name_word = role("R-08", "Account Manager")
    shared_words = role("R-04", "Clerk", "Sells products, closes.")
    roles = numbered(shared_words, name_word)
    assert picked(make().choose(record("manager", "Sells products, closes."), roles), roles) == (
        "R-08"
    )


def test_a_full_tie_goes_to_the_lowest_role_id_not_to_the_position_in_the_list():
    twins = [role(role_id, "Account Manager") for role_id in ("R-07", "R-04", "R-09")]
    stand_in = make()
    for order in itertools.permutations(twins):
        roles = numbered(*order)
        answer = stand_in.choose(record("account manager"), roles)
        assert picked(answer, roles) == "R-04"
        assert answer.number == [each.role_id for each in order].index("R-04") + 1


def test_the_choice_does_not_depend_on_the_order_of_the_list():
    ten = [
        ACCOUNT_MANAGER,
        SALES_REP,
        ANALYST,
        role("R-04", "Sales Engineer", "Explains the technical side of products during a sale."),
        role("R-05", "Customer Success Manager", "Helps existing customers adopt products."),
        role("R-06", "Accountant", "Keeps the general ledger and prepares monthly accounts."),
        role("R-07", "Payroll Specialist", "Runs payroll and resolves pay queries."),
        role("R-08", "Treasury Analyst", "Manages cash positions and bank accounts."),
        role("R-09", "Internal Auditor", "Tests financial controls and reports findings."),
        role("R-10", "Accounts Payable Clerk", "Processes supplier invoices and payments."),
    ]
    stand_in = make()
    asked = record("customer manager", "Manages customer accounts and their products.")
    orders = [ten, ten[::-1], ten[3:] + ten[:3], ten[1::2] + ten[::2]]
    answers = [(stand_in.choose(asked, numbered(*order)), numbered(*order)) for order in orders]
    assert {picked(answer, roles) for answer, roles in answers} == {"R-05"}
    assert len({answer.number for answer, _ in answers}) > 1  # the number follows the role
    assert {answer.quote for answer, _ in answers} == {"customer manager"}


# --- "no match" (Q17 rule 3) ------------------------------------------------------


def test_one_shared_word_anywhere_is_no_match():
    answer = make().choose(record("widget wrangler", "Sells widgets."), THREE)
    assert answer == ChooseAnswer(ChooseKind.NO_MATCH, None, "widget wrangler", LABEL_STAND_IN)


def test_two_shared_words_are_a_match():
    answer = make().choose(record("widget wrangler", "Sells products."), THREE)
    assert (answer.answer, picked(answer, THREE)) == (ChooseKind.ROLE, "R-02")


def test_the_no_match_line_is_a_setting():
    asked = record("manager", "Sells widgets.")  # Account Manager scores 3
    assert make(standin_no_match_below=3).choose(asked, THREE).answer is ChooseKind.ROLE
    assert make(standin_no_match_below=4).choose(asked, THREE).answer is ChooseKind.NO_MATCH


def test_with_the_line_at_zero_there_is_always_a_role():
    answer = make(standin_no_match_below=0).choose(record("widget wrangler", "Herds."), THREE)
    assert (answer.answer, picked(answer, THREE)) == (ChooseKind.ROLE, "R-01")
    assert answer.quote == "widget wrangler"  # nothing matched: the whole title


def test_no_roles_is_no_match():
    answer = make().choose(record("account manager"), [])
    assert answer == ChooseAnswer(ChooseKind.NO_MATCH, None, "account manager", LABEL_STAND_IN)


def test_a_role_without_a_number_cannot_be_chosen():
    roles = [NumberedRole(None, ACCOUNT_MANAGER), NumberedRole(1, SALES_REP)]
    assert make().choose(record("account manager"), roles).answer is ChooseKind.NO_MATCH
    answer = make().choose(record("account manager", "Sells products."), roles)
    assert (answer.answer, answer.number) == (ChooseKind.ROLE, 1)


# --- The quote (Q15, Q17) ---------------------------------------------------------


def test_the_quote_is_the_longest_run_of_title_words_that_match_the_role():
    assert make().choose(record("senior account manager"), THREE).quote == "account manager"
    assert make().choose(record("account manager ii"), THREE).quote == "account manager"


def test_a_word_that_does_not_match_ends_a_run_and_the_first_longest_run_wins():
    assert make().choose(record("account widget manager"), THREE).quote == "account"
    ops = numbered(role("R-20", "Sales Operations Analyst"))
    answer = make().choose(record("sales and sales operations analyst"), ops)
    assert answer.quote == "sales operations analyst"


def test_without_a_title_word_the_quote_is_a_run_of_description_words_as_written():
    answer = make().choose(
        record("widget wrangler", "Each month: Builds budgets, forecasts and variance reports."),
        THREE,
    )
    assert picked(answer, THREE) == "R-03"
    assert answer.quote == "Builds budgets, forecasts"


def test_a_title_word_is_quoted_before_a_longer_run_of_the_description():
    answer = make().choose(
        record("analyst", "Builds budgets, forecasts and variance reports."), THREE
    )
    assert (picked(answer, THREE), answer.quote) == ("R-03", "analyst")


def test_an_abbreviation_is_quoted_as_written_never_as_expanded():
    asked = record("widget wrangler", "Works as Acct Mgr for key clients.")
    answer = make().choose(asked, THREE)
    assert (picked(answer, THREE), answer.quote) == ("R-01", "Acct Mgr")
    assert "account manager" not in asked.description.lower()


def test_a_mask_matches_nothing_and_ends_a_run():
    badge = numbered(role("R-30", "Name Badge Clerk", "Prints each name badge and email label."))
    masked = record("widget wrangler", f"Reports to {NAME_MASK} about badge orders, {EMAIL_MASK}.")
    assert make().choose(masked, badge).answer is ChooseKind.NO_MATCH  # "badge" alone
    split = record("widget wrangler", f"Tracks budgets {NAME_MASK} forecasts.")
    assert make().choose(split, THREE).quote == "budgets"


def test_a_run_too_short_to_quote_is_passed_over():
    partner = numbered(role("R-40", "HR Partner", "Advises managers on hr matters."))
    answer = make().choose(record("hr", "Advises managers on people matters."), partner)
    assert (answer.answer, answer.quote) == (ChooseKind.ROLE, "Advises managers")


def test_a_run_without_a_letter_is_passed_over():
    review = numbered(role("R-41", "Controller 360", "Runs the 360 review."))
    answer = make().choose(record("360", "Runs the review cycle."), review)
    assert (answer.answer, answer.quote) == (ChooseKind.ROLE, "Runs")


def test_a_quote_is_never_longer_than_the_shared_check_allows():
    answer = make().choose(record("widget wrangler", "budgets forecasts " * 20), THREE)
    assert answer.answer is ChooseKind.ROLE
    assert answer.quote.split() == (["budgets", "forecasts"] * 20)[:QUOTE_MAX_WORDS]


def test_the_quote_of_no_match_and_insufficient_data_is_the_whole_title():
    assert make().choose(record("special projects ii", "Works on assignments."), THREE).quote == (
        "special projects ii"
    )
    assert make().choose(record("senior bdr"), THREE).quote == "senior bdr"


RECORDS = [
    record("senior account manager", "Owns the largest customer accounts. Reports to [NAME]."),
    record("account manager ii", "Manages a portfolio of enterprise accounts."),
    record("sales representative", "Sells products to new customers by phone and closes deals."),
    record("widget wrangler", "Each month: Builds budgets, forecasts and variance reports."),
    record("widget wrangler", "Works as Acct Mgr for key clients."),
    record("special projects", "Works on assignments set by the leadership team."),
    record("bdr"),
    record("financial analyst"),
    record("analyst", "Builds budgets,\n forecasts   and variance reports. Mail [EMAIL]."),
]


@pytest.mark.parametrize("asked", RECORDS, ids=lambda r: f"{r.cleaned_title}/{len(r.description)}")
def test_every_quote_is_copied_character_for_character_from_the_record(asked):
    answer = make().choose(asked, THREE)
    assert answer.quote
    assert answer.quote in asked.cleaned_title or answer.quote in asked.description
    assert answer.quote[0].isalnum() and answer.quote[-1].isalnum()
    assert NAME_MASK not in answer.quote and EMAIL_MASK not in answer.quote
    if answer.answer is not ChooseKind.ROLE:
        assert answer.quote == asked.cleaned_title


# --- The answer ---------------------------------------------------------------


@pytest.mark.parametrize("asked", RECORDS, ids=lambda r: f"{r.cleaned_title}/{len(r.description)}")
def test_every_answer_is_labelled_stand_in_and_well_formed(asked):
    answer = make().choose(asked, THREE)
    assert answer.source == LABEL_STAND_IN == "stand-in"
    assert answer.detail == ""
    assert answer.answer is not ChooseKind.INVALID  # no fault injection
    if answer.answer is ChooseKind.ROLE:
        assert answer.number in {1, 2, 3}
    else:
        assert answer.number is None


def test_the_same_arguments_give_the_same_answer():
    first, second = make(), make()
    for asked in RECORDS:
        answers = {first.choose(asked, THREE), first.choose(asked, THREE)}
        answers.add(second.choose(asked, THREE))
        assert len(answers) == 1


def test_an_answer_does_not_depend_on_what_was_asked_before():
    stand_in = make()
    alone = [make().choose(asked, THREE) for asked in RECORDS]
    forward = [stand_in.choose(asked, THREE) for asked in RECORDS]
    backward = [stand_in.choose(asked, THREE) for asked in reversed(RECORDS)]
    assert forward == alone == backward[::-1]


# --- The provider -------------------------------------------------------------


def test_the_provider_returns_one_stand_in_for_both_calls():
    llm = build_llm(Settings(), LADDER, ABBREVIATIONS)
    assert isinstance(llm.chooser, StandIn) and llm.reviewer is llm.chooser
    assert llm.chooser.choose(record("senior account manager"), THREE) == ChooseAnswer(
        ChooseKind.ROLE, 1, "account manager", LABEL_STAND_IN
    )


def test_finish_does_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm = build_llm(Settings(), LADDER, ABBREVIATIONS)
    assert llm.finish() is None
    assert list(tmp_path.iterdir()) == []


def test_the_provider_reads_the_stand_in_settings():
    strict = build_llm(Settings(standin_no_match_below=99), LADDER, ABBREVIATIONS)
    assert strict.chooser.choose(record("account manager"), THREE).answer is ChooseKind.NO_MATCH


# --- What the module may use (2.4, Q22) ---------------------------------------


def test_it_imports_only_the_standard_library_the_contracts_and_the_settings():
    # Neither the people module (the truth) nor the catalog module (generic
    # roles), and no file or network library either.
    tree = ast.parse(Path(standin.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "re",
        "titlestd.contracts",
        "titlestd.settings",
    }
