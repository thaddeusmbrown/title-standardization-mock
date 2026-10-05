"""The agent's case, its loop and what a model is shown of them (Q10, Q20).

Tests of section 2.7 on the shared loop, with a scripted backend: the agent
cannot pick a role it was not shown or one outside the family (T4), and a
compliance-flagged role is never settled by a machine, here with an injected
reviewer that returns one (T3). The cap, not-shown, out-of-family and
zero-search tests of Q20 are here too. The searches and the numbering have
their own file, tests/unit/test_agent_searches.py.

The embedder gives each word a bucket of its own, so a cosine is plain word
overlap and what a search returns can be worked out by hand.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import pytest

from titlestd import agent, fallback, rules
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    AgentCase,
    AgentResult,
    AgentTranscript,
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
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    TitleRow,
    TitlestdError,
    Vector,
)
from titlestd.lookup import Knowledge
from titlestd.settings import Settings

ACCEPT, PERSON = Route.ACCEPT, Route.PERSON
LABEL = "test label"


class WordEmbedder:
    """One bucket per word, L2-normalised."""

    def __init__(self) -> None:
        self._buckets: dict[str, int] = {}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
        norm = math.sqrt(sum(count * count for count in counts.values()))
        return {
            self._buckets.setdefault(word, len(self._buckets)): count / norm
            for word, count in counts.items()
        }


def _role(
    role_id: str, name: str, definition: str, *, flagged: bool = False, generic: bool = False
) -> CatalogRole:
    return CatalogRole(role_id, role_id[:4], name, name[:4], definition, flagged, generic)


# Two families. The bread taster is compliance-flagged.
BREWER = _role("FOOD-01", "Brewer", "brews beer")
BAKER = _role("FOOD-02", "Baker", "bakes bread and rolls")
BUTCHER = _role("FOOD-03", "Butcher", "cuts meat")
TASTER = _role("FOOD-04", "Bread Taster", "tastes bread", flagged=True)
FOOD = _role("FOOD-00", "Food", "Generic role for the Food family.", generic=True)
CASHIER = _role("SHOP-01", "Cashier", "sells bread and beer")
MANAGER = _role("SHOP-02", "Shop Manager", "runs the shop")
SHOP = _role("SHOP-00", "Shop", "Generic role for the Shop family.", generic=True)
CATALOG = Catalog(
    roles=(BREWER, BAKER, BUTCHER, TASTER, FOOD, CASHIER, MANAGER, SHOP), ladder=(), grade_levels=()
)
ROLE_IDS = [role.role_id for role in CATALOG.roles]

# A title recorded in the food family. The LLM picked the cashier, a shop
# role; the ML model's first choice is the baker. Its top three do not hold
# the flagged role.
ROW = TitleRow(
    Record("night baker", "FOOD", "Kitchen", "bakes bread at night"), ("E1",), ("Night Baker",)
)
BLOCK = fallback.render_record(ROW.record)
MATCH = MatchResult(
    (ScoredRole("FOOD-02", 0.9), ScoredRole("SHOP-01", 0.5), ScoredRole("FOOD-01", 0.2))
)
LLM_ROLES = (NumberedRole(1, CASHIER), NumberedRole(2, BAKER), NumberedRole(3, BREWER))
LLM_ANSWER = ChooseAnswer(ChooseKind.ROLE, 1, "night baker", LABEL)
REASONS = (Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH)
CASE = agent.build_case(ROW, LLM_ROLES, LLM_ANSWER, "SHOP-01", MATCH, REASONS)


class Scripted:
    """A backend that plays its turns in order and keeps every transcript it is
    handed. A turn may be a function of the transcript."""

    def __init__(self, *turns: object) -> None:
        self._turns = turns
        self.transcripts: list[AgentTranscript] = []

    def next_turn(self, transcript: AgentTranscript):
        self.transcripts.append(transcript)
        turn = self._turns[len(self.transcripts) - 1]
        return turn(transcript) if callable(turn) else turn


class Counting:
    """The real searches, with a record of every search that was run."""

    def __init__(self, tools: agent.SearchTools) -> None:
        self._tools = tools
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest, transcript: AgentTranscript) -> SearchStep:
        self.requests.append(request)
        return self._tools.search(request, transcript)


def _knowledge() -> Knowledge:
    """One approved title, of a shop role."""
    knowledge = Knowledge()
    seller = TitleRow(Record("bread seller", "SHOP", "Counter", ""), ("E2",), ("Bread Seller",))
    knowledge.learn(seller, "SHOP-01")
    return knowledge


def _run(
    *turns: object, case: AgentCase = CASE, catalog: Catalog = CATALOG
) -> tuple[AgentResult, Scripted, Counting]:
    backend = Scripted(*turns)
    tools = Counting(agent.SearchTools(catalog, _knowledge(), WordEmbedder(), Settings()))
    return agent.review(case, tools, backend), backend, tools


def _checked(result: AgentResult, catalog: Catalog = CATALOG) -> RuleOutcome:
    """The agent's answer through the rules, as the caller sends it (1.7)."""
    numbered = result.transcript.numbered_roles
    return rules.check_agent(result.answer, numbered, BLOCK, MATCH, "FOOD", catalog)


def _pick(number: object, quote: str = "night baker") -> ReviewAnswer:
    return ReviewAnswer(ReviewKind.ROLE, number, quote, LABEL)


def _catalog(query: str) -> SearchRequest:
    return SearchRequest(SearchTool.CATALOG, query)


def _approved(query: str) -> SearchRequest:
    return SearchRequest(SearchTool.APPROVED_TITLES, query)


def _number_of(role_id: str) -> Callable[[AgentTranscript], ReviewAnswer]:
    """A final turn that picks the role by whatever number the transcript gives it."""

    def turn(transcript: AgentTranscript) -> ReviewAnswer:
        (number,) = [
            shown.number for shown in transcript.numbered_roles if shown.role.role_id == role_id
        ]
        return _pick(number)

    return turn


ESCALATE = ReviewAnswer(ReviewKind.ESCALATE, None, "", LABEL)


# --- The case (Q10) -----------------------------------------------------------------


def test_the_case_is_the_record_the_llm_s_list_renumbered_and_the_two_opinions():
    case = agent.build_case(ROW, LLM_ROLES, LLM_ANSWER, "SHOP-01", MATCH, REASONS)
    assert case == AgentCase(
        record=ROW.record,
        roles=(NumberedRole(None, CASHIER), NumberedRole(1, BAKER), NumberedRole(2, BREWER)),
        llm_answer=LLM_ANSWER,
        llm_role_id="SHOP-01",
        ml_first_choice="FOOD-02",
        reasons=REASONS,
    )
    assert [shown.role for shown in case.roles] == [shown.role for shown in LLM_ROLES]


def test_the_case_takes_its_reasons_and_opinions_as_given():
    no_match = ChooseAnswer(ChooseKind.NO_MATCH, None, "night baker", LABEL)
    case = agent.build_case(
        ROW, list(LLM_ROLES), no_match, None, MatchResult(()), [Reason.NO_MATCH]
    )
    assert (case.llm_answer, case.llm_role_id, case.ml_first_choice) == (no_match, None, None)
    assert case.reasons == (Reason.NO_MATCH,)
    assert case.roles == CASE.roles


@pytest.mark.parametrize("flag", ["has_person_name", "family_usable"])
def test_a_row_no_model_may_see_gets_no_case(flag):
    # 1.4, Q14, Q26: to a person and to no model, and the agent is a model.
    row = dataclasses.replace(ROW, **{flag: flag == "has_person_name"})
    assert not row.model_eligible
    with pytest.raises(ValueError, match="to no model"):
        agent.build_case(row, LLM_ROLES, LLM_ANSWER, "SHOP-01", MATCH, REASONS)


# --- The loop: zero searches (Q20) --------------------------------------------------


def test_zero_searches_is_allowed():
    answer = _pick(1)
    result, backend, tools = _run(answer)
    assert result.answer is answer
    assert result.role_id == "FOOD-02"
    assert result.transcript == AgentTranscript(CASE)
    assert backend.transcripts == [AgentTranscript(CASE)]
    assert tools.requests == []
    assert _checked(result) == RuleOutcome(ACCEPT, (), "FOOD-02")


# --- The loop: one turn at a time (Q20) ----------------------------------------------


def test_each_turn_is_handed_the_case_and_every_search_so_far():
    requests = [_catalog("bread"), _approved("bread"), _catalog("beer")]
    result, backend, tools = _run(*requests, _pick(2))
    assert tools.requests == requests
    assert [len(transcript.steps) for transcript in backend.transcripts] == [0, 1, 2, 3]
    assert [transcript.searches_left for transcript in backend.transcripts] == [4, 3, 2, 1]
    for earlier, later in itertools.pairwise(backend.transcripts):
        assert later.case is CASE and later.steps[:-1] == earlier.steps  # oldest first
    assert result.transcript == backend.transcripts[-1]
    assert [step.request for step in result.transcript.steps] == requests
    assert result.role_id == "FOOD-01"


def test_a_role_found_by_a_search_can_be_picked_by_its_number():
    # "cuts meat" finds the butcher, new to the case: number 3.
    result, _, _ = _run(_catalog("cuts meat"), _pick(3))
    first = result.transcript.steps[0].hits[0]
    assert first == SearchHit(NumberedRole(3, BUTCHER))
    assert result.role_id == "FOOD-03"
    assert not result.confirms_llm  # picking another role corrects the LLM
    assert _checked(result) == RuleOutcome(ACCEPT, (), "FOOD-03")


def test_a_number_names_nothing_before_a_search_has_shown_its_role():
    result, _, _ = _run(_pick(3))
    assert result.role_id is None
    assert _checked(result) == RuleOutcome(PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)


def test_picking_the_llm_s_role_confirms_it_and_picking_another_corrects_it():
    # 1.6 step 6. Here the LLM had picked the baker and the ML model the brewer.
    answer = ChooseAnswer(ChooseKind.ROLE, 2, "night baker", LABEL)
    disagree = (Reason.MODELS_DISAGREE,)
    case = agent.build_case(ROW, LLM_ROLES, answer, "FOOD-02", MATCH, disagree)
    confirmed, _, _ = _run(_pick(1), case=case)
    assert (confirmed.role_id, confirmed.confirms_llm) == ("FOOD-02", True)
    corrected, _, _ = _run(_pick(2), case=case)
    assert (corrected.role_id, corrected.confirms_llm) == ("FOOD-01", False)
    escalated, _, _ = _run(ESCALATE, case=case)
    assert (escalated.role_id, escalated.confirms_llm) == (None, False)


# --- The loop: the cap of four searches (2.4, Q20) ----------------------------------


def test_four_searches_and_then_the_answer_stands():
    requests = [_catalog("bread"), _approved("bread"), _catalog("beer"), _approved("beer")]
    result, backend, tools = _run(*requests, _pick(1))
    assert AGENT_MAX_SEARCHES == 4
    assert tools.requests == requests
    assert result.transcript.searches_left == 0
    assert (result.answer, result.role_id) == (_pick(1), "FOOD-02")
    assert len(backend.transcripts) == 5
    assert _checked(result).route is ACCEPT


@pytest.mark.parametrize(
    "kinds",
    ["ccccc", "aaaaa", "cacac", "ccaaa", "aaccc", "caaac"],
)
def test_a_fifth_search_is_not_run_and_the_title_goes_to_a_person(kinds):
    # A search of either kind counts one toward the four.
    requests = [
        _catalog(f"bread {at}") if kind == "c" else _approved(f"bread {at}")
        for at, kind in enumerate(kinds)
    ]
    result, backend, tools = _run(*requests)
    assert tools.requests == requests[:4]
    assert [step.request for step in result.transcript.steps] == requests[:4]
    assert len(backend.transcripts) == 5  # not asked again after the request it may not make
    assert result.answer == ReviewAnswer(
        ReviewKind.INVALID, None, "", LABEL, "a search was requested after the 4 a case may run"
    )
    assert result.role_id is None
    assert _checked(result) == RuleOutcome(PERSON, (Reason.INVALID_ANSWER,), None)


def test_a_backend_that_never_answers_is_stopped_at_the_cap():
    def another(transcript: AgentTranscript) -> SearchRequest:
        return _catalog(f"bread {len(transcript.steps)}")

    result, backend, tools = _run(*[another] * 50)
    assert len(tools.requests) == 4 and len(backend.transcripts) == 5
    assert result.answer.answer is ReviewKind.INVALID


def test_a_search_within_the_cap_is_run_even_when_it_repeats_an_earlier_one():
    # Each executed search counts, a repeated one too.
    same = _catalog("bread")
    result, _, tools = _run(same, same, same, same, _pick(1))
    assert tools.requests == [same] * 4
    assert len({step.hits for step in result.transcript.steps}) == 1  # same hits, same numbers
    assert result.role_id == "FOOD-02"


# --- The agent cannot pick a role it was not shown (2.7) ----------------------------


@pytest.mark.parametrize("number", [0, -1, 3, 4, 7, 10, 99, 10**9])
def test_a_number_that_was_never_shown_names_no_role(number):
    # The case numbers two roles, 1 and 2.
    answer = _pick(number)
    result, _, _ = _run(answer)
    assert result.answer is answer  # not rewritten: the rules fail the catalog check (S11)
    assert result.role_id is None
    assert _checked(result) == RuleOutcome(PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)


@pytest.mark.parametrize("number", [None, True, "1", 1.0, 1.5, [1], "FOOD-02", "Baker"])
def test_only_a_whole_number_that_was_shown_names_a_role(number):
    result, _, _ = _run(_pick(number))
    assert result.role_id is None
    assert _checked(result).route is PERSON


def test_a_role_of_the_family_that_was_never_shown_cannot_be_picked():
    # The butcher and the taster are food roles, and neither is in the case.
    for number in range(-3, 60):
        result, _, _ = _run(_pick(number))
        assert result.role_id in (None, "FOOD-02", "FOOD-01")
        assert (result.role_id is None) == (number not in (1, 2))


def test_no_text_the_backend_writes_in_place_of_a_number_reaches_a_role():
    # 2.5: no model output can become a role by itself. A role id or a role
    # name is not a number, wherever the backend puts it.
    for role_id in ROLE_IDS:
        for answer in (
            _pick(role_id),
            _pick(1, quote=role_id),
            ReviewAnswer(ReviewKind.ROLE, None, "night baker", LABEL, detail=role_id),
        ):
            result, _, _ = _run(answer)
            assert result.role_id in (None, "FOOD-02")
            assert _checked(result).role_id in (None, "FOOD-02")


# --- The agent cannot pick a role outside the family (1.4, 2.7) ----------------------


def test_an_out_of_family_role_in_the_case_has_no_number_to_pick_it_by():
    assert CASE.roles[0] == NumberedRole(None, CASHIER)
    # The LLM's own number for the cashier names the baker in the agent's list.
    result, _, _ = _run(_pick(LLM_ANSWER.number))
    assert result.role_id == "FOOD-02"
    for number in range(-3, 60):
        result, _, _ = _run(_pick(number))
        assert result.role_id != "SHOP-01"


def test_an_out_of_family_role_found_by_a_search_cannot_be_picked_either():
    searches = [_catalog("runs the shop"), _approved("bread seller"), _catalog("bread and beer")]
    shown: set[str] = set()
    for number in range(-3, 60):
        result, _, _ = _run(*searches, _pick(number))
        steps = result.transcript.steps
        shown = {hit.role.role.role_id for step in steps for hit in step.hits}
        assert steps[0].hits[0] == SearchHit(NumberedRole(None, MANAGER))
        assert steps[1].hits == (SearchHit(NumberedRole(None, CASHIER), "bread seller", "SHOP"),)
        numbered = {role.role.role_id: role.number for role in result.transcript.numbered_roles}
        assert set(numbered) == {"FOOD-01", "FOOD-02", "FOOD-03", "FOOD-04"}
        assert result.role_id == {n: role_id for role_id, n in numbered.items()}.get(number)
        outcome = _checked(result)
        assert Reason.FAMILY_MISMATCH not in outcome.reasons
        assert outcome.role_id is None or outcome.role_id.startswith("FOOD-")
    assert {"SHOP-01", "SHOP-02"} <= shown  # both were shown, and neither could be picked


def test_whatever_the_backend_does_the_role_is_one_it_was_shown_in_the_family():
    # Every script of up to two searches over a few queries, and every number.
    queries = ("bread", "runs the shop", "food generic role", "")
    requests = [make(query) for make in (_catalog, _approved) for query in queries]
    scripts = [(), *((first,) for first in requests)]
    scripts += [(first, second) for first in requests[:3] for second in requests[4:7]]
    picked = set()
    for script in scripts:
        for number in range(-1, 9):
            result, _, _ = _run(*script, _pick(number))
            if result.role_id is None:
                continue
            picked.add(result.role_id)
            (role,) = [role for role in CATALOG.roles if role.role_id == result.role_id]
            assert role.family_code == "FOOD" and not role.is_generic
            assert NumberedRole(number, role) in result.transcript.numbered_roles
    assert picked == {"FOOD-01", "FOOD-02", "FOOD-03", "FOOD-04"}


# --- An escalation, and an answer that is not usable (1.7, S10) ---------------------


def test_an_escalation_names_no_role_whatever_number_comes_with_it():
    for answer in (ESCALATE, ReviewAnswer(ReviewKind.ESCALATE, 1, "night baker", LABEL)):
        result, _, _ = _run(_catalog("bread"), answer)
        assert result.answer is answer
        assert result.role_id is None
        assert _checked(result) == RuleOutcome(PERSON, (Reason.AGENT_ESCALATED,), None)


def test_an_invalid_answer_from_the_backend_passes_through_unchanged():
    # How a live backend reports a refusal or a cut-off (S10, Q39).
    refusal = ReviewAnswer(ReviewKind.INVALID, 1, "night baker", "live", "the model refused")
    result, _, _ = _run(refusal)
    assert result.answer is refusal
    assert result.role_id is None
    assert _checked(result) == RuleOutcome(PERSON, (Reason.INVALID_ANSWER,), None)


def test_an_answer_of_an_unknown_kind_names_no_role():
    result, _, _ = _run(ReviewAnswer("maybe", 1, "night baker", LABEL))
    assert result.role_id is None
    assert _checked(result) == RuleOutcome(PERSON, (Reason.INVALID_ANSWER,), None)


@pytest.mark.parametrize(
    "turn",
    [
        None,
        "search_catalog",
        (_catalog("bread"), _approved("bread")),  # more than one tool request in a turn
        [_catalog("bread")],
        LLM_ANSWER,  # the other call's answer
        {"tool": "search_catalog", "query": "bread"},
        1,
    ],
    ids=lambda turn: type(turn).__name__,
)
def test_a_turn_that_is_neither_a_request_nor_an_answer_goes_to_a_person(turn):
    result, backend, tools = _run(_catalog("bread"), turn)
    assert tools.requests == [_catalog("bread")]  # the search before it stays in the transcript
    assert len(result.transcript.steps) == 1 and len(backend.transcripts) == 2
    assert result.answer.answer is ReviewKind.INVALID
    assert (result.answer.number, result.answer.quote, result.role_id) == (None, "", None)
    assert type(turn).__name__ in result.answer.detail
    assert _checked(result) == RuleOutcome(PERSON, (Reason.INVALID_ANSWER,), None)


@pytest.mark.parametrize(
    "request_",
    [
        SearchRequest("search_the_web", "bread"),
        SearchRequest(None, "bread"),
        SearchRequest(SearchTool.CATALOG, None),
        SearchRequest(SearchTool.APPROVED_TITLES, ["bread"]),
    ],
    ids=["unknown tool", "no tool", "no query", "a list for a query"],
)
def test_a_request_that_is_no_search_is_not_run_and_goes_to_a_person(request_):
    result, _, tools = _run(request_)
    assert tools.requests == []
    assert result.answer.answer is ReviewKind.INVALID and result.role_id is None
    assert "one of the two searches" in result.answer.detail
    assert _checked(result).route is PERSON


def test_the_tool_s_name_as_plain_text_is_a_search():
    result, _, tools = _run(SearchRequest("search_approved_titles", "bread"), _pick(1))
    assert len(tools.requests) == 1
    assert result.transcript.steps[0].hits[0].approved_title == "bread seller"


def test_the_answer_the_loop_makes_carries_the_label_of_the_case_s_llm_answer():
    # One llm setting answers both calls (contracts.Llm), and a search request has no label.
    for label in ("stand-in", "live", "recorded 2026-01-05, some-model"):
        case = dataclasses.replace(CASE, llm_answer=dataclasses.replace(LLM_ANSWER, source=label))
        result, _, _ = _run(None, case=case)
        assert (result.answer.answer, result.answer.source) == (ReviewKind.INVALID, label)


def test_what_the_backend_raises_passes_through():
    # A replay miss or a live-mode stop ends the run (2.5, Q39); it is no answer.
    def miss(transcript: AgentTranscript):
        raise TitlestdError("no recorded answer for this turn")

    with pytest.raises(TitlestdError, match="no recorded answer"):
        _run(_catalog("bread"), miss)


# --- A compliance-flagged role is never settled by a machine (2.7) --------------------


def _without_flags() -> Catalog:
    roles = tuple(dataclasses.replace(role, compliance_flag=False) for role in CATALOG.roles)
    return dataclasses.replace(CATALOG, roles=roles)


def test_an_injected_reviewer_that_returns_a_flagged_role_does_not_settle_the_title():
    assert TASTER.compliance_flag and "FOOD-04" not in MATCH.top_three
    # The reviewer finds the flagged role by a search and answers with its number.
    result, _, _ = _run(_catalog("tastes bread"), _number_of("FOOD-04"))
    assert result.transcript.steps[0].hits[0].role.role == TASTER
    assert (result.answer.answer, result.role_id) == (ReviewKind.ROLE, "FOOD-04")
    assert _checked(result) == RuleOutcome(PERSON, (Reason.COMPLIANCE_FLAG,), "FOOD-04")
    # It is the flag that stops it: the same answer on a catalog without one stands.
    unflagged = _without_flags()
    result, _, _ = _run(_catalog("tastes bread"), _number_of("FOOD-04"), catalog=unflagged)
    assert _checked(result, unflagged) == RuleOutcome(ACCEPT, (), "FOOD-04")


def test_a_flagged_role_in_the_case_is_not_settled_without_a_search_either():
    shortlist = (NumberedRole(1, TASTER), NumberedRole(2, BAKER))
    case = agent.build_case(ROW, shortlist, LLM_ANSWER, "FOOD-04", MATCH, (Reason.MODELS_DISAGREE,))
    result, _, _ = _run(_pick(1), case=case)
    assert (result.role_id, result.confirms_llm) == ("FOOD-04", True)
    assert _checked(result) == RuleOutcome(PERSON, (Reason.COMPLIANCE_FLAG,), "FOOD-04")
    # The other role of the same case is the agent's to settle.
    result, _, _ = _run(_pick(2), case=case)
    assert _checked(result) == RuleOutcome(ACCEPT, (), "FOOD-02")


def test_no_number_the_reviewer_can_return_settles_the_flagged_role():
    # All four food roles are shown after this search; only the flagged one is refused.
    settled = set()
    for number in range(-1, 9):
        result, _, _ = _run(_catalog("tastes bread and cuts meat"), _pick(number))
        outcome = _checked(result)
        if outcome.route is ACCEPT:
            settled.add(outcome.role_id)
        else:
            assert outcome.role_id in (None, "FOOD-04")
    assert settled == {"FOOD-01", "FOOD-02", "FOOD-03"}


# --- What a model is shown (Q10, S16) -------------------------------------------------


def test_the_case_as_a_model_is_shown_it():
    assert agent.render_case(CASE) == (
        "Title: night baker\n"
        "Family: FOOD\n"
        "Department: Kitchen\n"
        "Description: bakes bread at night\n"
        "\n"
        "Roles:\n"
        "- Cashier (family SHOP): sells bread and beer [the LLM's pick]\n"
        "1. Baker (family FOOD): bakes bread and rolls [the ML model's first choice]\n"
        "2. Brewer (family FOOD): brews beer\n"
        "\n"
        "Referred because: models disagree; role outside recorded family"
    )


def test_the_case_starts_with_the_record_block_the_quote_check_reads():
    # Q15: one string serves the prompt and the check.
    text = agent.render_case(CASE)
    assert text.startswith(BLOCK + "\n\n")
    assert fallback.quote_in_record("bakes bread at night", text[: len(BLOCK)])
    lines = text.splitlines()
    assert lines[6:9] == [
        line + mark
        for line, mark in zip(
            fallback.render_roles(CASE.roles).splitlines(),
            [f" {agent.LLM_PICK_MARK}", f" {agent.ML_FIRST_CHOICE_MARK}", ""],
            strict=True,
        )
    ]


def test_both_opinions_on_one_role_are_both_marked_and_a_no_match_marks_none():
    agree = dataclasses.replace(CASE, llm_role_id="FOOD-02", reasons=(Reason.FAMILY_MISMATCH,))
    assert (
        "1. Baker (family FOOD): bakes bread and rolls "
        "[the LLM's pick] [the ML model's first choice]\n"
    ) in agent.render_case(agree)
    no_match = dataclasses.replace(
        CASE, llm_role_id=None, ml_first_choice="SHOP-02", reasons=(Reason.NO_MATCH,)
    )
    text = agent.render_case(no_match)
    assert "[" not in text  # the ML first choice is not among the roles shown
    assert text.endswith("\n\nReferred because: no match")


def test_no_rendered_text_shows_a_role_id_or_a_score():
    # S16.
    result, _, _ = _run(_catalog("bread"), _approved("bread"), _catalog("beer"), _pick(1))
    texts = [agent.render_case(CASE), *map(agent.render_step, result.transcript.steps)]
    for text in texts:
        assert not any(role_id in text for role_id in ROLE_IDS)
        assert not re.search(r"\d\.\d", text)  # no similarity score
        assert "Generic role" not in text


def test_a_catalog_search_as_a_model_is_shown_it():
    result, _, _ = _run(_catalog("bread"), _pick(1))
    assert agent.render_step(result.transcript.steps[0]) == (
        "3. Bread Taster (family FOOD): tastes bread\n"
        "1. Baker (family FOOD): bakes bread and rolls\n"
        "- Cashier (family SHOP): sells bread and beer\n"
        "2. Brewer (family FOOD): brews beer\n"
        "4. Butcher (family FOOD): cuts meat"
    )


def test_an_approved_title_search_as_a_model_is_shown_it():
    request = _approved("bread")
    step = SearchStep(
        request,
        (
            SearchHit(NumberedRole(None, CASHIER), "bread seller", "SHOP"),
            SearchHit(NumberedRole(3, BUTCHER), "meat cutter ii", "FOOD"),
            SearchHit(NumberedRole(1, BAKER), "night baker", ""),  # no usable family code (Q26)
        ),
    )
    assert agent.render_step(step) == (
        '- Cashier: approved for the title "bread seller" (family SHOP)\n'
        '3. Butcher: approved for the title "meat cutter ii" (family FOOD)\n'
        '1. Baker: approved for the title "night baker" (family none recorded)'
    )
    assert agent.NO_FAMILY == "none recorded"


def test_a_search_that_found_nothing_says_so():
    for request in (_catalog("bread"), _approved("bread")):
        assert agent.render_step(SearchStep(request, ())) == "No results." == agent.NO_RESULTS


def test_which_lines_a_step_gets_follows_the_search_that_ran():
    # An approved title can be the empty string (a title made only of removable
    # parts), so the hit does not say which search it came from; the request does.
    hit = SearchHit(NumberedRole(1, BAKER), "", "FOOD")
    assert agent.render_step(SearchStep(_approved("x"), (hit,))) == (
        '1. Baker: approved for the title "" (family FOOD)'
    )
    assert agent.render_step(SearchStep(_catalog("x"), (hit,))) == (
        "1. Baker (family FOOD): bakes bread and rolls"
    )


@pytest.mark.parametrize("generic", [FOOD, SHOP])
def test_no_renderer_shows_a_generic_family_role(generic):
    # 1.2, T5: the last thing a role passes before a model reads it.
    for number in (None, 3):
        shown = NumberedRole(number, generic)
        with pytest.raises(ValueError, match="generic family role"):
            agent.render_case(dataclasses.replace(CASE, roles=(*CASE.roles, shown)))
        for request in (_catalog("x"), _approved("x")):
            with pytest.raises(ValueError, match="generic family role"):
                agent.render_step(SearchStep(request, (SearchHit(shown, "food person", "FOOD"),)))
