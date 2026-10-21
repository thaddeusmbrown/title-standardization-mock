"""The rules of 1.7: where code sends every proposal.

DESIGN.md 1.7; docs/build-decisions.md Q04 (the ML-alone route), Q15 (the
quote is checked on every choose answer; a number that names no shown role
fails the catalog check), S10 (an answer that is not usable), S11 (reasons).

Two tests of section 2.7 are owned here, as rules: a quote that is not in the
record sends the title to a person, and a compliance-flagged role is never
settled by a machine. Every answer in this file is injected: the stand-ins
never fail the quote check or answer out of range by themselves (Q17).
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from titlestd import fallback, rules
from titlestd.contracts import (
    Catalog,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    MatchResult,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
    ScoredRole,
)

# The table of 1.7, top to bottom, then the two reasons that are not in it.
TABLE_ORDER = (
    Reason.ROLE_NOT_IN_CATALOG,
    Reason.QUOTE_NOT_IN_RECORD,
    Reason.COMPLIANCE_FLAG,
    Reason.INSUFFICIENT_DATA,
    Reason.MODELS_DISAGREE,
    Reason.FAMILY_MISMATCH,
    Reason.NO_MATCH,
    Reason.AGENT_ESCALATED,
    Reason.INVALID_ANSWER,
)
ON_FAILURE_THE_AGENT = {Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH, Reason.NO_MATCH}


def _role(role_id: str, family: str, name: str, *, flagged: bool = False, generic: bool = False):
    return CatalogRole(role_id, family, name, name, f"What a {name} does.", flagged, generic)


# A flagged role in each family, so that the compliance check and the family
# check can fail apart, and ids that say nothing about either.
CATALOG = Catalog(
    roles=(
        _role("S1", "SAL", "Account Manager"),
        _role("S2", "SAL", "Sales Representative"),
        _role("S3", "SAL", "Sales Engineer"),
        _role("S4", "SAL", "Deal Desk Officer", flagged=True),
        _role("S9", "SAL", "Sales", generic=True),
        _role("F1", "FIN", "Financial Analyst"),
        _role("F2", "FIN", "Accountant"),
        _role("F3", "FIN", "Internal Auditor", flagged=True),
        _role("F9", "FIN", "Finance", generic=True),
    ),
    ladder=(),
    grade_levels=(),
)
BY_ID = {role.role_id: role for role in CATALOG.roles}
FLAGGED = {"S4", "F3"}
IN_FAMILY = {"SAL": {"S1", "S2", "S3", "S4"}, "FIN": {"F1", "F2", "F3"}}

RECORD = Record(
    "senior account manager",
    "SAL",
    "Enterprise Sales",
    "Owns the largest customer accounts and negotiates renewals.",
)
BLOCK = fallback.render_record(RECORD)
QUOTE = "account manager"  # stands in the record
NOT_A_QUOTE = "regional sales director"  # does not


def shown(*role_ids: str) -> tuple[NumberedRole, ...]:
    """The roles numbered from 1 in the order given."""
    return tuple(
        NumberedRole(number, BY_ID[role_id]) for number, role_id in enumerate(role_ids, start=1)
    )


def ml(*role_ids: str) -> MatchResult:
    """An ML match that ranks these roles, best first."""
    return MatchResult(
        tuple(ScoredRole(role_id, 0.9 - 0.1 * at) for at, role_id in enumerate(role_ids))
    )


# What the LLM is shown: numbers 1 to 7.
LIST = shown("S1", "S2", "S3", "S4", "F1", "F2", "F3")
# The ML model's first choice is S1; both flagged roles are below its top three.
MATCH = ml("S1", "S2", "S3", "S4", "F3")


def picks(number, quote: str = QUOTE) -> ChooseAnswer:
    return ChooseAnswer(ChooseKind.ROLE, number, quote, "test")


def no_match(quote: str = QUOTE) -> ChooseAnswer:
    return ChooseAnswer(ChooseKind.NO_MATCH, None, quote, "test")


def insufficient(quote: str = QUOTE) -> ChooseAnswer:
    return ChooseAnswer(ChooseKind.INSUFFICIENT_DATA, None, quote, "test")


INVALID = ChooseAnswer(ChooseKind.INVALID, None, "", "test", "the model refused")


def llm(answer, roles=LIST, block=BLOCK, match=MATCH, family="SAL", catalog=CATALOG):
    return rules.check_llm(answer, roles, block, match, family, catalog)


def in_table_order(reasons) -> bool:
    places = [TABLE_ORDER.index(reason) for reason in reasons]
    return places == sorted(set(places))


# --- check_llm: the table of 1.7, one failed check at a time ----------------------------------


def test_agreement_with_every_check_passed_is_accepted():
    assert llm(picks(1)) == RuleOutcome(Route.ACCEPT, (), "S1")
    assert llm(picks(6), match=ml("F2", "F1", "S1"), family="FIN") == RuleOutcome(
        Route.ACCEPT, (), "F2"
    )


@pytest.mark.parametrize(
    ("answer", "match", "route", "reason", "role_id"),
    [
        (picks(8), MATCH, Route.PERSON, Reason.ROLE_NOT_IN_CATALOG, None),
        (picks(1, NOT_A_QUOTE), MATCH, Route.PERSON, Reason.QUOTE_NOT_IN_RECORD, "S1"),
        (picks(4), ml("S4", "S1", "S2"), Route.PERSON, Reason.COMPLIANCE_FLAG, "S4"),
        (picks(1), ml("S1", "S2", "S4"), Route.PERSON, Reason.COMPLIANCE_FLAG, "S1"),
        (insufficient(), MATCH, Route.PERSON, Reason.INSUFFICIENT_DATA, None),
        (picks(2), MATCH, Route.AGENT, Reason.MODELS_DISAGREE, "S2"),
        (picks(5), ml("F1", "S1", "S2"), Route.AGENT, Reason.FAMILY_MISMATCH, "F1"),
        (no_match(), MATCH, Route.AGENT, Reason.NO_MATCH, None),
    ],
    ids=[
        "the role exists in the catalog",
        "the quote appears word for word in the record",
        "the proposed role is compliance-flagged",
        "one of the ML top three is compliance-flagged",
        "the LLM answered insufficient data",
        "the LLM and the ML first choice agree",
        "the proposed role is in the recorded family",
        "the LLM answered no match",
    ],
)
def test_each_check_of_the_table_sends_its_failure_where_the_table_says(
    answer, match, route, reason, role_id
):
    assert llm(answer, match=match) == RuleOutcome(route, (reason,), role_id)


def test_the_reasons_read_as_the_review_queue_file_shows_them():
    # S11. The values are the scaffold's; the rules only choose among them.
    assert [reason.value for reason in TABLE_ORDER] == [
        "role not in catalog",
        "quote not in record",
        "compliance-flagged role",
        "insufficient data",
        "models disagree",
        "role outside recorded family",
        "no match",
        "agent escalated",
        "invalid model answer",
    ]


# --- check_llm: the catalog check and the number (Q15) ----------------------------------------


@pytest.mark.parametrize("number", [0, 8, 11, -1, 10**6, None, True, 1.0, "1"])
def test_a_number_that_names_no_shown_role_fails_the_catalog_check(number):
    # The out-of-range number, with an injected answer (Q15, Q17): to a person,
    # and no role comes out of it.
    assert llm(picks(number)) == RuleOutcome(Route.PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)


def test_a_shown_role_that_the_catalog_does_not_hold_fails_the_catalog_check():
    invented = _role("X1", "SAL", "Chief Vibes Officer")
    roles = (*LIST, NumberedRole(8, invented))
    outcome = llm(picks(8), roles=roles, match=ml("X1", "S1", "S2"))
    assert outcome == RuleOutcome(Route.PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)


def test_a_generic_family_role_is_never_a_machine_s_proposal():
    # 1.2: only code assigns it. number_roles refuses to number one; if a list
    # holds one all the same, the rules do not let it through.
    roles = (*LIST, NumberedRole(8, BY_ID["S9"]))
    outcome = llm(picks(8), roles=roles, match=ml("S9", "S1", "S2"))
    assert outcome == RuleOutcome(Route.PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)


def test_only_an_answer_of_the_kind_role_proposes_a_role():
    # A number on any other answer is not read, so it cannot become a role.
    assert llm(ChooseAnswer(ChooseKind.NO_MATCH, 1, QUOTE, "test")) == RuleOutcome(
        Route.AGENT, (Reason.NO_MATCH,), None
    )
    assert llm(ChooseAnswer(ChooseKind.INSUFFICIENT_DATA, 1, QUOTE, "test")) == RuleOutcome(
        Route.PERSON, (Reason.INSUFFICIENT_DATA,), None
    )
    assert llm(ChooseAnswer(ChooseKind.INVALID, 1, QUOTE, "test")) == RuleOutcome(
        Route.PERSON, (Reason.INVALID_ANSWER,), None
    )


def test_the_family_and_the_flag_are_the_catalog_s_not_those_of_the_role_shown():
    # The same id shown with another family and no flag: the catalog decides.
    disguised = dataclasses.replace(BY_ID["F3"], family_code="SAL", compliance_flag=False)
    outcome = llm(picks(1), roles=(NumberedRole(1, disguised),), match=ml("F3", "S1", "S2"))
    assert outcome == RuleOutcome(
        Route.PERSON, (Reason.COMPLIANCE_FLAG, Reason.FAMILY_MISMATCH), "F3"
    )
    # With the flagged role nowhere in the ML top three, the flag is still found.
    assert llm(picks(1), roles=(NumberedRole(1, disguised),)).reasons == (
        Reason.COMPLIANCE_FLAG,
        Reason.MODELS_DISAGREE,
        Reason.FAMILY_MISMATCH,
    )


# --- check_llm: the quote (2.7) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        NOT_A_QUOTE,
        "",
        "   ",
        "senior manager",  # words of the record with one left out
        "ccount manager",  # starts inside a word
        "account manage",  # ends inside one
        "an account manager who owns the largest accounts",  # a paraphrase
        "account manager.",  # punctuation the record does not have there
    ],
)
def test_a_quote_that_is_not_in_the_record_sends_the_title_to_a_person(quote: str):
    # Everything else passes: the two models agree on an unflagged role of the family.
    assert llm(picks(1, quote)) == RuleOutcome(Route.PERSON, (Reason.QUOTE_NOT_IN_RECORD,), "S1")


def test_the_quote_is_checked_on_every_kind_of_answer():
    # Q15. "No match" would otherwise go to the agent: a person beats the agent.
    assert llm(no_match(NOT_A_QUOTE)) == RuleOutcome(
        Route.PERSON, (Reason.QUOTE_NOT_IN_RECORD, Reason.NO_MATCH), None
    )
    assert llm(no_match("")) == RuleOutcome(
        Route.PERSON, (Reason.QUOTE_NOT_IN_RECORD, Reason.NO_MATCH), None
    )
    assert llm(insufficient(NOT_A_QUOTE)) == RuleOutcome(
        Route.PERSON, (Reason.QUOTE_NOT_IN_RECORD, Reason.INSUFFICIENT_DATA), None
    )
    assert llm(picks(8, NOT_A_QUOTE)) == RuleOutcome(
        Route.PERSON, (Reason.ROLE_NOT_IN_CATALOG, Reason.QUOTE_NOT_IN_RECORD), None
    )


def test_the_quote_is_checked_against_the_block_it_is_given():
    other = fallback.render_record(Record("sales engineer", "SAL", "Solutions", "Runs demos."))
    assert llm(picks(1), block=other).reasons == (Reason.QUOTE_NOT_IN_RECORD,)
    assert llm(picks(1, "Runs demos."), block=other).route is Route.ACCEPT
    assert llm(picks(1, "Runs demos.")).reasons == (Reason.QUOTE_NOT_IN_RECORD,)


# --- check_llm: compliance (2.7) --------------------------------------------------------------


@pytest.mark.parametrize("flagged", sorted(FLAGGED))
@pytest.mark.parametrize("place", [0, 1, 2])
def test_a_flagged_role_among_the_ml_top_three_sends_the_title_to_a_person(flagged, place):
    # The LLM picked an unflagged role and, where it can, agrees with the ML model.
    top = ["S1", "S2", "S3"]
    top[place] = flagged
    outcome = llm(picks(1), match=ml(*top))
    assert outcome.route is Route.PERSON
    assert Reason.COMPLIANCE_FLAG in outcome.reasons
    assert outcome.role_id == "S1"


def test_a_flagged_role_below_the_top_three_does_not_fire():
    assert llm(picks(1), match=ml("S1", "S2", "S3", "S4", "F3")).route is Route.ACCEPT
    # Nor does one that was only shown to the LLM and not picked.
    assert {"S4", "F3"} <= {each.role.role_id for each in LIST}
    assert llm(picks(1), match=ml("S1", "S2", "S3")).route is Route.ACCEPT


@pytest.mark.parametrize(("number", "role_id"), [(4, "S4"), (7, "F3")])
@pytest.mark.parametrize("top", [("S1", "S2", "S3"), ("S4", "S1", "S2"), ("F3", "S1", "S2")])
def test_a_flagged_role_the_llm_proposes_is_never_settled_by_a_machine(number, role_id, top):
    outcome = llm(picks(number), match=ml(*top))
    assert outcome.route is Route.PERSON
    assert Reason.COMPLIANCE_FLAG in outcome.reasons
    assert outcome.role_id == role_id


def test_the_ml_top_three_is_read_whatever_the_llm_answered():
    # Q25 asks at the close whether the compliance check ever fired for a title.
    flagged_top = ml("S1", "S4", "S2")
    assert llm(no_match(), match=flagged_top) == RuleOutcome(
        Route.PERSON, (Reason.COMPLIANCE_FLAG, Reason.NO_MATCH), None
    )
    assert llm(insufficient(), match=flagged_top) == RuleOutcome(
        Route.PERSON, (Reason.COMPLIANCE_FLAG, Reason.INSUFFICIENT_DATA), None
    )
    assert llm(INVALID, match=flagged_top) == RuleOutcome(
        Route.PERSON, (Reason.COMPLIANCE_FLAG, Reason.INVALID_ANSWER), None
    )
    assert llm(picks(8), match=flagged_top) == RuleOutcome(
        Route.PERSON, (Reason.ROLE_NOT_IN_CATALOG, Reason.COMPLIANCE_FLAG), None
    )


def test_a_top_three_role_the_catalog_does_not_hold_carries_no_flag():
    assert llm(picks(1), match=ml("S1", "X1", "X2")).route is Route.ACCEPT


# --- check_llm: agreement and the family ------------------------------------------------------


def test_disagreement_keeps_the_llm_s_role_for_the_agent_to_confirm_or_correct():
    for number, role_id in ((2, "S2"), (3, "S3")):
        assert llm(picks(number)) == RuleOutcome(Route.AGENT, (Reason.MODELS_DISAGREE,), role_id)
    # With no ML first choice there is nothing to agree with.
    assert llm(picks(1), match=MatchResult(())).reasons == (Reason.MODELS_DISAGREE,)


@pytest.mark.parametrize("family", ["SAL", " SAL", "SAL ", "\tSAL\n"])
def test_the_recorded_family_is_compared_after_trimming(family: str):
    # S05: the employees file may hold a padded code.
    assert llm(picks(1), family=family).route is Route.ACCEPT


@pytest.mark.parametrize("family", ["FIN", "sal", "Sal", "", "   ", "XXX", "S", "SAL-FIN"])
def test_a_role_outside_the_recorded_family_goes_to_the_agent(family: str):
    assert llm(picks(1), family=family) == RuleOutcome(Route.AGENT, (Reason.FAMILY_MISMATCH,), "S1")


def test_the_checks_about_the_proposed_role_are_not_listed_when_there_is_none():
    # "No match" disagrees with nothing and is outside no family.
    for answer in (no_match(), insufficient(), picks(8), INVALID):
        reasons = llm(answer, family="FIN").reasons
        assert Reason.MODELS_DISAGREE not in reasons
        assert Reason.FAMILY_MISMATCH not in reasons


# --- check_llm: several failed checks ---------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "match", "reasons"),
    [
        (
            picks(2, NOT_A_QUOTE),
            MATCH,
            (Reason.QUOTE_NOT_IN_RECORD, Reason.MODELS_DISAGREE),
        ),
        (
            picks(5, NOT_A_QUOTE),
            ml("F1", "S1", "S2"),
            (Reason.QUOTE_NOT_IN_RECORD, Reason.FAMILY_MISMATCH),
        ),
        (
            picks(7),
            ml("F3", "S1", "S2"),
            (Reason.COMPLIANCE_FLAG, Reason.FAMILY_MISMATCH),
        ),
        (
            picks(2),
            ml("S1", "S4", "S3"),
            (Reason.COMPLIANCE_FLAG, Reason.MODELS_DISAGREE),
        ),
        (
            picks(7, NOT_A_QUOTE),
            MATCH,
            (
                Reason.QUOTE_NOT_IN_RECORD,
                Reason.COMPLIANCE_FLAG,
                Reason.MODELS_DISAGREE,
                Reason.FAMILY_MISMATCH,
            ),
        ),
    ],
)
def test_a_person_beats_the_agent(answer, match, reasons):
    # 1.7: "If several fail, a person beats the agent."
    outcome = llm(answer, match=match)
    assert outcome.route is Route.PERSON
    assert outcome.reasons == reasons


def test_two_failures_that_are_both_the_agent_s_stay_with_the_agent():
    assert llm(picks(5)) == RuleOutcome(
        Route.AGENT, (Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH), "F1"
    )


# --- check_llm: an answer that is not usable (S10) --------------------------------------------


def test_an_invalid_answer_goes_to_a_person():
    assert llm(INVALID) == RuleOutcome(Route.PERSON, (Reason.INVALID_ANSWER,), None)
    # Its quote is not checked: there is no answer to check.
    assert Reason.QUOTE_NOT_IN_RECORD not in llm(INVALID).reasons


def test_an_answer_of_an_unknown_kind_is_invalid():
    for kind in ("maybe", "", None, "ROLE", ReviewKind.ESCALATE):
        answer = ChooseAnswer(kind, 1, QUOTE, "test")
        assert llm(answer) == RuleOutcome(Route.PERSON, (Reason.INVALID_ANSWER,), None)


# --- check_llm: every case at once ------------------------------------------------------------

MODEL_ROLE_IDS = ("S1", "S2", "S3", "S4", "F1", "F2", "F3")


def every_llm_case():
    answers = [picks(number, quote) for number in range(9) for quote in (QUOTE, NOT_A_QUOTE)]
    answers += [make(quote) for make in (no_match, insufficient) for quote in (QUOTE, NOT_A_QUOTE)]
    answers.append(INVALID)
    for top in itertools.permutations(MODEL_ROLE_IDS, 3):
        for answer, family in itertools.product(answers, ("SAL", "FIN")):
            yield answer, ml(*top, "S9"), family


def test_over_every_case_the_route_follows_from_the_failed_checks():
    cases = 0
    for answer, match, family in every_llm_case():
        outcome = llm(answer, match=match, family=family)
        cases += 1
        proposed = None
        if answer.answer is ChooseKind.ROLE and answer.number in range(1, 8):
            proposed = BY_ID[MODEL_ROLE_IDS[answer.number - 1]]
        assert outcome.role_id == (proposed.role_id if proposed else None)
        assert in_table_order(outcome.reasons)
        # 2.7: a compliance-flagged role is never settled by a machine.
        touched = set(match.top_three) | ({proposed.role_id} if proposed else set())
        assert (Reason.COMPLIANCE_FLAG in outcome.reasons) == bool(touched & FLAGGED)
        if touched & FLAGGED:
            assert outcome.route is Route.PERSON
        # 2.7: a quote that is not in the record sends the title to a person.
        if answer.quote == NOT_A_QUOTE:
            assert outcome.route is Route.PERSON
            assert Reason.QUOTE_NOT_IN_RECORD in outcome.reasons
        # The route is the table's: nothing failed, only the agent's checks, or a person's.
        if outcome.route is Route.ACCEPT:
            assert outcome.reasons == ()
            assert answer.answer is ChooseKind.ROLE and answer.quote == QUOTE
            assert proposed.role_id == match.first_choice
            assert proposed.role_id in IN_FAMILY[family] - FLAGGED
        elif outcome.route is Route.AGENT:
            assert outcome.reasons and set(outcome.reasons) <= ON_FAILURE_THE_AGENT
        else:
            assert set(outcome.reasons) - ON_FAILURE_THE_AGENT
    assert cases == 210 * 23 * 2


def test_every_route_and_every_reason_of_the_llm_check_occurs():
    cases = every_llm_case()
    outcomes = [llm(answer, match=match, family=family) for answer, match, family in cases]
    assert {outcome.route for outcome in outcomes} == set(Route)
    reasons = {reason for outcome in outcomes for reason in outcome.reasons}
    assert reasons == set(TABLE_ORDER) - {Reason.AGENT_ESCALATED}


# --- check_agent: the same rules except agreement ---------------------------------------------

# What the agent was shown for a title recorded in SAL (Q10): the roles of the
# family numbered, the others without a number.
AGENT_LIST = (
    NumberedRole(1, BY_ID["S1"]),
    NumberedRole(None, BY_ID["F1"]),
    NumberedRole(2, BY_ID["S2"]),
    NumberedRole(3, BY_ID["S3"]),
    NumberedRole(None, BY_ID["F3"]),
    NumberedRole(4, BY_ID["S4"]),
)
ESCALATES = ReviewAnswer(ReviewKind.ESCALATE, None, "", "test")
BROKE_THE_PROTOCOL = ReviewAnswer(ReviewKind.INVALID, None, "", "test", "a fifth search")


def settles(number, quote: str = QUOTE) -> ReviewAnswer:
    return ReviewAnswer(ReviewKind.ROLE, number, quote, "test")


def agent(answer, roles=AGENT_LIST, block=BLOCK, match=MATCH, family="SAL", catalog=CATALOG):
    return rules.check_agent(answer, roles, block, match, family, catalog)


def test_the_agent_s_role_stands_when_every_check_passes():
    assert agent(settles(1)) == RuleOutcome(Route.ACCEPT, (), "S1")


def test_the_agent_s_answer_is_not_checked_for_agreement():
    # 1.7: "the same rules except agreement". The ML first choice is S1.
    assert agent(settles(2)) == RuleOutcome(Route.ACCEPT, (), "S2")
    assert agent(settles(3), match=MatchResult(())) == RuleOutcome(Route.ACCEPT, (), "S3")


def test_an_escalation_goes_to_a_person_and_needs_no_quote():
    assert agent(ESCALATES) == RuleOutcome(Route.PERSON, (Reason.AGENT_ESCALATED,), None)
    with_a_number = ReviewAnswer(ReviewKind.ESCALATE, 1, NOT_A_QUOTE, "test")
    assert agent(with_a_number) == RuleOutcome(Route.PERSON, (Reason.AGENT_ESCALATED,), None)


def test_an_invalid_agent_answer_goes_to_a_person():
    # Q20: the loop reports a broken protocol this way; Q39: so does a live backend.
    expected = RuleOutcome(Route.PERSON, (Reason.INVALID_ANSWER,), None)
    assert agent(BROKE_THE_PROTOCOL) == expected
    assert agent(ReviewAnswer(ReviewKind.INVALID, 1, QUOTE, "test")) == expected
    assert agent(ReviewAnswer("maybe", 1, QUOTE, "test")) == expected
    assert agent(ReviewAnswer(ChooseKind.NO_MATCH, 1, QUOTE, "test")) == expected


@pytest.mark.parametrize("number", [0, 5, 6, 11, -1, None, True])
def test_the_agent_cannot_settle_on_a_role_it_was_not_shown_under_a_number(number):
    # F1 and F3 were shown, outside the family and so without a number: no
    # answer reaches them.
    assert agent(settles(number)) == RuleOutcome(Route.PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)


def test_a_role_outside_the_family_goes_to_a_person_not_back_to_the_agent():
    # Were a list ever to number such a role, the family check still stops it.
    roles = (*AGENT_LIST, NumberedRole(5, BY_ID["F1"]))
    assert agent(settles(5), roles=roles) == RuleOutcome(
        Route.PERSON, (Reason.FAMILY_MISMATCH,), "F1"
    )
    assert agent(settles(1), family="FIN") == RuleOutcome(
        Route.PERSON, (Reason.FAMILY_MISMATCH,), "S1"
    )


@pytest.mark.parametrize("family", ["SAL", " SAL", "SAL ", "\tSAL\n"])
def test_the_agent_s_role_is_compared_with_the_recorded_family_after_trimming(family: str):
    # S05, as for the LLM's answer: a padded code is still the family.
    assert agent(settles(1), family=family) == RuleOutcome(Route.ACCEPT, (), "S1")


@pytest.mark.parametrize("family", ["FIN", "sal", "Sal", "", "   ", "XXX", "S", "SAL-FIN"])
def test_the_agent_s_role_is_inside_no_family_code_but_its_own(family: str):
    # Case counts, and a blank or unknown code holds no role: whatever the
    # agent picks is outside it, and its failed check goes to a person.
    assert agent(settles(1), family=family) == RuleOutcome(
        Route.PERSON, (Reason.FAMILY_MISMATCH,), "S1"
    )


@pytest.mark.parametrize("quote", [NOT_A_QUOTE, "", "ccount manager", "senior manager"])
def test_an_agent_quote_that_is_not_in_the_record_sends_the_title_to_a_person(quote: str):
    assert agent(settles(1, quote)) == RuleOutcome(
        Route.PERSON, (Reason.QUOTE_NOT_IN_RECORD,), "S1"
    )


def test_a_flagged_role_is_never_settled_by_the_agent():
    assert agent(settles(4)) == RuleOutcome(Route.PERSON, (Reason.COMPLIANCE_FLAG,), "S4")
    for top in (("S4", "S1", "S2"), ("S1", "F3", "S2"), ("S1", "S2", "S4")):
        assert agent(settles(1), match=ml(*top)) == RuleOutcome(
            Route.PERSON, (Reason.COMPLIANCE_FLAG,), "S1"
        )
    assert agent(settles(1), match=ml("S1", "S2", "S3", "S4")).route is Route.ACCEPT


def test_the_agent_s_failed_checks_are_listed_in_table_order():
    roles = (*AGENT_LIST, NumberedRole(5, BY_ID["F3"]))
    assert agent(settles(5, NOT_A_QUOTE), roles=roles).reasons == (
        Reason.QUOTE_NOT_IN_RECORD,
        Reason.COMPLIANCE_FLAG,
        Reason.FAMILY_MISMATCH,
    )
    assert agent(settles(9, NOT_A_QUOTE)).reasons == (
        Reason.ROLE_NOT_IN_CATALOG,
        Reason.QUOTE_NOT_IN_RECORD,
    )
    flagged_top = ml("S1", "S4", "S2")
    assert agent(ESCALATES, match=flagged_top).reasons == (
        Reason.COMPLIANCE_FLAG,
        Reason.AGENT_ESCALATED,
    )
    assert agent(BROKE_THE_PROTOCOL, match=flagged_top).reasons == (
        Reason.COMPLIANCE_FLAG,
        Reason.INVALID_ANSWER,
    )


def test_over_every_case_the_agent_s_answer_is_accepted_or_goes_to_a_person():
    numbered_everything = shown(*MODEL_ROLE_IDS)
    answers = [settles(number, quote) for number in range(9) for quote in (QUOTE, NOT_A_QUOTE)]
    answers += [ESCALATES, BROKE_THE_PROTOCOL]
    routes = set()
    for top in itertools.permutations(MODEL_ROLE_IDS, 3):
        for answer, family in itertools.product(answers, ("SAL", "FIN")):
            outcome = agent(answer, roles=numbered_everything, match=ml(*top), family=family)
            routes.add(outcome.route)
            assert in_table_order(outcome.reasons)
            assert (outcome.route is Route.ACCEPT) == (outcome.reasons == ())
            assert Reason.MODELS_DISAGREE not in outcome.reasons
            touched = set(top) | ({outcome.role_id} if outcome.role_id else set())
            assert (Reason.COMPLIANCE_FLAG in outcome.reasons) == bool(touched & FLAGGED)
            if outcome.route is Route.ACCEPT:
                assert answer.answer is ReviewKind.ROLE and answer.quote == QUOTE
                assert outcome.role_id in IN_FAMILY[family] - FLAGGED
    assert routes == {Route.ACCEPT, Route.PERSON}  # never back to the agent


# --- check_ml_alone (Q04) ---------------------------------------------------------------------


def test_an_ml_alone_title_goes_to_a_person_with_the_ml_role_as_the_proposal():
    assert rules.check_ml_alone(MATCH, "SAL", CATALOG) == RuleOutcome(
        Route.PERSON, (Reason.ML_ALONE,), "S1"
    )
    assert Reason.ML_ALONE.value == "ML alone"


@pytest.mark.parametrize(
    ("top", "role_id"),
    [(("S4", "S1", "S2"), "S4"), (("S1", "S4", "S2"), "S1"), (("S1", "S2", "F3"), "S1")],
)
def test_the_compliance_check_still_runs_on_an_ml_alone_title(top, role_id):
    assert rules.check_ml_alone(ml(*top), "SAL", CATALOG) == RuleOutcome(
        Route.PERSON, (Reason.ML_ALONE, Reason.COMPLIANCE_FLAG), role_id
    )


@pytest.mark.parametrize("match", [ml("X1", "S1", "S2"), ml("S9", "S1", "S2"), MatchResult(())])
def test_the_catalog_check_still_runs_on_an_ml_alone_title(match: MatchResult):
    # A role the catalog does not hold, a generic family role, no role at all.
    assert rules.check_ml_alone(match, "SAL", CATALOG) == RuleOutcome(
        Route.PERSON, (Reason.ML_ALONE, Reason.ROLE_NOT_IN_CATALOG), None
    )


def test_an_ml_alone_role_outside_the_family_it_is_given_says_so():
    assert rules.check_ml_alone(ml("F1", "S1", "S2"), "SAL", CATALOG) == RuleOutcome(
        Route.PERSON, (Reason.ML_ALONE, Reason.FAMILY_MISMATCH), "F1"
    )
    assert rules.check_ml_alone(ml("F3", "S1", "S2"), " SAL ", CATALOG).reasons == (
        Reason.ML_ALONE,
        Reason.COMPLIANCE_FLAG,
        Reason.FAMILY_MISMATCH,
    )
    assert rules.check_ml_alone(ml("F1", "F2", "S1"), "FIN ", CATALOG).reasons == (Reason.ML_ALONE,)


@pytest.mark.parametrize("family", ["SAL", " SAL", "SAL ", "\tSAL\n"])
def test_an_ml_alone_role_is_compared_with_the_family_after_trimming(family: str):
    assert rules.check_ml_alone(MATCH, family, CATALOG) == RuleOutcome(
        Route.PERSON, (Reason.ML_ALONE,), "S1"
    )


@pytest.mark.parametrize("family", ["FIN", "sal", "Sal", "", "   ", "XXX", "S", "SAL-FIN"])
def test_an_ml_alone_role_is_inside_no_family_code_but_its_own(family: str):
    # The one reading of the module docstring: case counts, and a blank or
    # unknown code holds no role.
    assert rules.check_ml_alone(MATCH, family, CATALOG) == RuleOutcome(
        Route.PERSON, (Reason.ML_ALONE, Reason.FAMILY_MISMATCH), "S1"
    )


def test_ml_alone_never_stands_without_a_person():
    # Q04: "stand alone" means without the LLM, not without a person.
    for top in itertools.permutations([*MODEL_ROLE_IDS, "S9", "X1"], 3):
        for family in ("SAL", "FIN", ""):
            outcome = rules.check_ml_alone(ml(*top), family, CATALOG)
            assert outcome.route is Route.PERSON
            assert outcome.reasons[0] is Reason.ML_ALONE
            assert len(set(outcome.reasons)) == len(outcome.reasons)
