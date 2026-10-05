"""The stand-in agent: a fixed script, one turn at a time (Q19, Q20).

The loop, the searches and the numbering are shared code built later
(titlestd.agent). Here the transcripts are written out by hand, as that code
will hand them over: the case, then every search that ran with its hits.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    LABEL_STAND_IN,
    Abbreviation,
    AgentCase,
    AgentTranscript,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    LadderLevel,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
)
from titlestd.settings import Settings
from titlestd.standin import DESCRIPTION_QUERY_WORDS, StandIn

LADDER = (
    LadderLevel("Junior", "Junior", "Jr"),
    LadderLevel("Senior", "Senior", "Sr"),
)
ABBREVIATIONS = (Abbreviation("acct", "account"), Abbreviation("mgr", "manager"))
ESCALATE = ReviewAnswer(ReviewKind.ESCALATE, None, "", LABEL_STAND_IN)


def role(role_id: str, name: str, definition: str = "", family: str = "SAL") -> CatalogRole:
    return CatalogRole(role_id, family, name, name, definition)


ACCOUNT_MANAGER = role(
    "SAL-01",
    "Account Manager",
    "Manages a portfolio of customer accounts and renews their contracts.",
)
SUCCESS_MANAGER = role(
    "SAL-06", "Customer Success Manager", "Helps existing customers adopt products after the sale."
)
SALES_ANALYST = role("SAL-03", "Sales Operations Analyst", "Maintains sales forecasts.")
FIN_ANALYST = role(
    "FIN-01", "Financial Analyst", "Builds budgets, forecasts and variance reports.", "FIN"
)
TREASURY_ANALYST = role(
    "FIN-06", "Treasury Analyst", "Manages cash positions and bank accounts.", "FIN"
)

RECORD = Record(
    "account manager ii",
    "SAL",
    "Enterprise Sales",
    "Manages a portfolio of enterprise accounts and renews their contracts.",
)


def transcript(
    record: Record, roles: Sequence[NumberedRole], *steps: SearchStep
) -> AgentTranscript:
    case = AgentCase(
        record=record,
        roles=tuple(roles),
        llm_answer=ChooseAnswer(ChooseKind.NO_MATCH, None, record.cleaned_title, LABEL_STAND_IN),
        llm_role_id=None,
        ml_first_choice=None,
        reasons=(Reason.NO_MATCH,),
    )
    return AgentTranscript(case, tuple(steps))


def catalog(*hits: NumberedRole, query: str = "q") -> SearchStep:
    request = SearchRequest(SearchTool.CATALOG, query)
    return SearchStep(request, tuple(SearchHit(hit) for hit in hits))


def approved(*hits: NumberedRole, query: str = "q") -> SearchStep:
    request = SearchRequest(SearchTool.APPROVED_TITLES, query)
    return SearchStep(request, tuple(SearchHit(hit, "acct mgr", "SAL") for hit in hits))


def make() -> StandIn:
    return StandIn(Settings(), LADDER, ABBREVIATIONS)


TWO = (NumberedRole(1, ACCOUNT_MANAGER), NumberedRole(2, SUCCESS_MANAGER))


# --- The script (Q19) -----------------------------------------------------------


def test_the_first_turn_searches_the_catalog_with_the_cleaned_title():
    assert make().next_turn(transcript(RECORD, TWO)) == SearchRequest(
        SearchTool.CATALOG, "account manager ii"
    )


def test_the_second_turn_searches_the_approved_titles_with_the_cleaned_title():
    assert make().next_turn(transcript(RECORD, TWO, catalog())) == SearchRequest(
        SearchTool.APPROVED_TITLES, "account manager ii"
    )


def test_with_two_roles_to_pick_from_the_third_turn_is_the_answer():
    turn = make().next_turn(transcript(RECORD, TWO, catalog(), approved()))
    assert turn == ReviewAnswer(ReviewKind.ROLE, 1, "account manager", LABEL_STAND_IN)


def test_with_fewer_than_two_it_searches_the_catalog_with_the_description():
    long = Record("bdr", "SAL", "Inside Sales", " ".join(f"word{n}" for n in range(1, 16)))
    for roles in ((), (NumberedRole(1, ACCOUNT_MANAGER),), (NumberedRole(None, FIN_ANALYST),)):
        turn = make().next_turn(transcript(long, roles, catalog(), approved()))
        assert turn == SearchRequest(
            SearchTool.CATALOG, " ".join(f"word{n}" for n in range(1, DESCRIPTION_QUERY_WORDS + 1))
        )
    assert DESCRIPTION_QUERY_WORDS == 12


MISFILED = Record(
    "financial analyst",
    "SAL",
    "Sales Operations",
    "Builds budgets and forecasts for the sales teams.",
)
OUTSIDE = (NumberedRole(None, FIN_ANALYST), NumberedRole(None, TREASURY_ANALYST))


@pytest.mark.parametrize(
    "inside", [(), (NumberedRole(1, SALES_ANALYST),)], ids=["none of the family", "one"]
)
def test_only_roles_of_the_family_count_toward_the_two(inside):
    # Q19 counts the in-family roles shown, and those are the numbered ones.
    # Two roles of another family were shown as well, without a number, so
    # three roles are on the table and still fewer than two can be picked.
    third = SearchRequest(SearchTool.CATALOG, MISFILED.description)
    stand_in = make()
    in_the_case = transcript(MISFILED, (*OUTSIDE, *inside), catalog(), approved())
    assert stand_in.next_turn(in_the_case) == third
    by_the_searches = transcript(MISFILED, inside, catalog(*OUTSIDE), approved(*OUTSIDE))
    assert stand_in.next_turn(by_the_searches) == third
    by_both = transcript(MISFILED, (*OUTSIDE, *inside), catalog(*OUTSIDE), approved(OUTSIDE[0]))
    assert stand_in.next_turn(by_both) == third


def test_a_role_of_the_family_shown_again_is_still_one_role():
    # The searches often return a role the case already holds: it keeps its
    # number and is one role however often it is shown.
    one = NumberedRole(1, SALES_ANALYST)
    again = transcript(MISFILED, (one,), catalog(one, OUTSIDE[0]), approved(one, one))
    assert make().next_turn(again) == SearchRequest(SearchTool.CATALOG, MISFILED.description)
    # A second role of the family, in the case or found by a search, ends it.
    two = NumberedRole(2, ACCOUNT_MANAGER)
    for enough in (
        transcript(MISFILED, (*OUTSIDE, one, two), catalog(), approved()),
        transcript(MISFILED, (*OUTSIDE, one), catalog(one, two), approved()),
        transcript(MISFILED, (*OUTSIDE, one), catalog(), approved(two)),
    ):
        assert isinstance(make().next_turn(enough), ReviewAnswer)


def test_the_description_query_is_its_first_words_as_written():
    spaced = Record("bdr", "SAL", "Inside Sales", "Finds  and qualifies\nnew leads, by outreach.")
    turn = make().next_turn(transcript(spaced, (), catalog(), approved()))
    assert turn == SearchRequest(SearchTool.CATALOG, "Finds and qualifies new leads, by outreach.")


def test_without_a_description_there_is_no_third_search():
    bare = Record("account manager ii", "SAL", "Enterprise Sales", "")
    one = (NumberedRole(1, ACCOUNT_MANAGER),)
    turn = make().next_turn(transcript(bare, one, catalog(), approved()))
    assert turn == ReviewAnswer(ReviewKind.ROLE, 1, "account manager", LABEL_STAND_IN)
    blank = Record("bdr", "SAL", "Inside Sales", "   ")
    assert make().next_turn(transcript(blank, (), catalog(), approved())) == ESCALATE


def test_roles_numbered_by_a_search_count_as_shown():
    # One role in the case, one more numbered by the first search: two shown.
    one = (NumberedRole(1, SUCCESS_MANAGER),)
    found = catalog(NumberedRole(2, ACCOUNT_MANAGER), NumberedRole(None, FIN_ANALYST))
    turn = make().next_turn(transcript(RECORD, one, found, approved()))
    assert turn == ReviewAnswer(ReviewKind.ROLE, 2, "account manager", LABEL_STAND_IN)


def test_after_the_third_search_it_always_answers():
    stand_in = make()
    for steps in ([catalog()] * 3, [catalog()] * 4, [catalog(), approved(), catalog(), catalog()]):
        assert isinstance(stand_in.next_turn(transcript(RECORD, (), *steps)), ReviewAnswer)


@pytest.mark.parametrize(
    "record",
    [
        RECORD,
        Record("bdr", "SAL", "Inside Sales", ""),
        Record("bdr", "SAL", "Inside Sales", "Finds and qualifies new sales leads by outreach."),
        Record("special projects", "FIN", "Finance", "Works on assignments."),
    ],
    ids=lambda record: f"{record.cleaned_title}/{len(record.description)}",
)
@pytest.mark.parametrize("found", [(), (NumberedRole(1, ACCOUNT_MANAGER),), TWO])
def test_never_a_fourth_search(record, found):
    # A loop as plain as the shared one: run what is asked until an answer comes.
    stand_in = make()
    steps: list[SearchStep] = []
    requests: list[SearchRequest] = []
    while True:
        turn = stand_in.next_turn(transcript(record, (), *steps))
        if isinstance(turn, ReviewAnswer):
            break
        requests.append(turn)
        steps.append(SearchStep(turn, tuple(SearchHit(hit) for hit in found)))
        assert len(steps) < AGENT_MAX_SEARCHES
    assert [request.tool for request in requests[:2]] == [
        SearchTool.CATALOG,
        SearchTool.APPROVED_TITLES,
    ]
    assert len(requests) in (2, 3)
    if len(requests) == 3:
        assert requests[2].tool is SearchTool.CATALOG
        assert record.description and len(found) < 2


# --- The answer (Q19) -----------------------------------------------------------


def answer(record: Record, roles: Sequence[NumberedRole], *steps: SearchStep) -> ReviewAnswer:
    """The final answer once the script's searches have run."""
    ran = steps or (catalog(), approved())
    turn = make().next_turn(transcript(record, roles, *ran, catalog()))
    assert isinstance(turn, ReviewAnswer)
    return turn


def test_it_answers_with_the_top_role_when_that_beats_the_runner_up():
    # Account Manager scores 11, Customer Success Manager 3.
    assert answer(RECORD, TWO) == ReviewAnswer(
        ReviewKind.ROLE, 1, "account manager", LABEL_STAND_IN
    )
    swapped = (NumberedRole(1, SUCCESS_MANAGER), NumberedRole(2, ACCOUNT_MANAGER))
    assert answer(RECORD, swapped).number == 2


def test_a_tie_at_the_top_escalates():
    twins = (NumberedRole(1, role("SAL-07", "Account Manager")), NumberedRole(2, ACCOUNT_MANAGER))
    bare = Record("account manager ii", "SAL", "Enterprise Sales", "")
    assert answer(bare, twins) == ESCALATE


def test_a_similar_approved_title_breaks_the_tie():
    twin = role("SAL-07", "Account Manager")
    twins = (NumberedRole(1, twin), NumberedRole(2, ACCOUNT_MANAGER))
    bare = Record("account manager ii", "SAL", "Enterprise Sales", "")
    for number in (1, 2):
        found = approved(twins[number - 1])
        assert answer(bare, twins, catalog(), found) == ReviewAnswer(
            ReviewKind.ROLE, number, "account manager", LABEL_STAND_IN
        )


def test_the_bonus_is_worth_one_shared_word_and_is_given_once():
    # "portfolio" is the one word more that the leader shares: 7 against 6.
    leader = NumberedRole(1, role("SAL-01", "Account Manager", "Owns a portfolio."))
    chaser = NumberedRole(2, role("SAL-07", "Account Manager"))
    record = Record("account manager", "SAL", "Enterprise Sales", "A portfolio.")
    pair = (leader, chaser)
    assert answer(record, pair).number == 1
    assert answer(record, pair, catalog(), approved(chaser)) == ESCALATE  # 7 against 7
    assert answer(record, pair, catalog(), approved(chaser, chaser), approved(chaser)) == ESCALATE
    assert answer(record, pair, catalog(), approved(leader)).number == 1


def test_a_catalog_hit_earns_no_bonus():
    twin = NumberedRole(1, role("SAL-07", "Account Manager"))
    twins = (twin, NumberedRole(2, ACCOUNT_MANAGER))
    bare = Record("account manager ii", "SAL", "Enterprise Sales", "")
    assert answer(bare, twins, catalog(twin), approved()) == ESCALATE


def test_the_top_role_must_share_a_content_word_with_the_title():
    # The description alone points at a role, which is enough for the stand-in
    # LLM but not for the agent.
    record = Record(
        "widget wrangler", "FIN", "Finance", "Builds budgets, forecasts and variance reports."
    )
    roles = (NumberedRole(1, FIN_ANALYST), NumberedRole(2, role("FIN-02", "Accountant", "", "FIN")))
    assert make().choose(record, roles).answer is ChooseKind.ROLE
    assert answer(record, roles) == ESCALATE
    titled = Record("budgets wrangler", "FIN", "Finance", record.description)
    assert answer(titled, roles) == ReviewAnswer(ReviewKind.ROLE, 1, "budgets", LABEL_STAND_IN)


def test_a_role_shown_without_a_number_is_never_picked():
    # The out-of-family role is the better match; only the numbered one can be picked.
    record = Record("financial analyst", "SAL", "Sales Operations", "Builds budgets and forecasts.")
    roles = (NumberedRole(None, FIN_ANALYST), NumberedRole(1, SALES_ANALYST))
    found = catalog(NumberedRole(None, FIN_ANALYST))
    assert answer(record, roles, found, approved(NumberedRole(None, FIN_ANALYST))) == ReviewAnswer(
        ReviewKind.ROLE, 1, "analyst", LABEL_STAND_IN
    )


def test_with_no_numbered_role_it_escalates():
    assert answer(RECORD, ()) == ESCALATE
    assert answer(RECORD, (NumberedRole(None, FIN_ANALYST),)) == ESCALATE


def test_a_single_candidate_has_no_runner_up_to_beat():
    assert answer(RECORD, (NumberedRole(1, ACCOUNT_MANAGER),)).number == 1
    assert answer(RECORD, (NumberedRole(1, SALES_ANALYST),)) == ESCALATE  # no title word shared


def test_a_role_found_by_a_search_can_be_the_answer():
    one = (NumberedRole(1, SUCCESS_MANAGER),)
    found = catalog(NumberedRole(None, FIN_ANALYST), NumberedRole(2, ACCOUNT_MANAGER))
    assert answer(RECORD, one, found, approved()) == ReviewAnswer(
        ReviewKind.ROLE, 2, "account manager", LABEL_STAND_IN
    )


def test_the_answer_is_one_of_the_numbers_shown_or_an_escalation():
    stand_in = make()
    records = [
        RECORD,
        Record("customer success manager", "SAL", "Customer Success", ""),
        Record("sales analyst", "SAL", "Sales Operations", "Maintains sales forecasts."),
        Record("special projects", "SAL", "Sales", "Works on assignments."),
    ]
    shown = (NumberedRole(None, FIN_ANALYST), NumberedRole(3, SALES_ANALYST), *TWO)
    for record in records:
        script = transcript(record, shown, catalog(), approved(NumberedRole(3, SALES_ANALYST)))
        turn = stand_in.next_turn(script)
        assert isinstance(turn, ReviewAnswer) and turn.source == LABEL_STAND_IN
        assert turn.answer in (ReviewKind.ROLE, ReviewKind.ESCALATE)  # never INVALID
        if turn.answer is ReviewKind.ROLE:
            assert turn.number in {1, 2, 3}
            assert turn.quote and turn.quote in record.cleaned_title + "\n" + record.description
        else:
            assert (turn.number, turn.quote) == (None, "")


def test_the_quote_is_built_as_the_stand_in_llm_builds_it():
    stand_in = make()
    for record in (
        RECORD,
        Record("sr acct mgr", "SAL", "Enterprise Sales", "Looks after accounts."),
        Record("manager of accounts", "SAL", "Enterprise Sales", ""),
    ):
        chosen = stand_in.choose(record, [NumberedRole(1, ACCOUNT_MANAGER)])
        assert chosen.answer is ChooseKind.ROLE
        assert answer(record, (NumberedRole(1, ACCOUNT_MANAGER),)).quote == chosen.quote


# --- One turn at a time (Q20) ---------------------------------------------------


def test_a_turn_depends_only_on_the_transcript():
    first, second = make(), make()
    scripts = [
        transcript(RECORD, TWO),
        transcript(RECORD, TWO, catalog()),
        transcript(RECORD, TWO, catalog(), approved(TWO[1])),
        transcript(Record("bdr", "SAL", "Inside Sales", "Finds new leads."), ()),
        transcript(
            Record("bdr", "SAL", "Inside Sales", "Finds new leads."), (), catalog(), approved()
        ),
    ]
    forward = [first.next_turn(script) for script in scripts]
    backward = [first.next_turn(script) for script in reversed(scripts)]
    assert forward == backward[::-1] == [second.next_turn(script) for script in scripts]
