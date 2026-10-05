"""The agent on the mini data set, with the real modules of stages 1 and 2.

Every title row goes the way a run takes it: the real readers and prepare
step, the seed look-up, the ML match on the offline embedder, the seeded
numbering, the stand-in LLM, the rules of 1.7, and then the agent: the case,
the shared loop, the two searches and the stand-in agent built by the switch,
whose answer goes back through the rules (docs/design.md 1.6 step 6, 1.7;
docs/build-decisions.md Q10, Q19, Q20).

Three tests of section 2.7 are tried on that: the agent cannot pick a role it
was not shown or one outside the family, no model is ever shown a generic
family role in a search result, and a compliance-flagged role is never settled
by a machine, the first and the last with an injected reviewer.

No expectation comes from the truth file: the answers are worked out by hand
from Q17 and Q19, and what a search must return is ranked a second time here,
straight from the embedder.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence

import pytest

from titlestd import agent, catalog, datafiles, fallback, people, prepare, rules, switch
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    LABEL_STAND_IN,
    NAME_MASK,
    AgentCase,
    AgentResult,
    AgentTranscript,
    Catalog,
    ChooseAnswer,
    InputData,
    MatchResult,
    Models,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
    SearchRequest,
    SearchTool,
    TitleKey,
    TitleRow,
    Vector,
)
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher
from titlestd.settings import Settings

SETTINGS = Settings()
FLAGGED = "FIN-04"  # Internal Auditor, the one compliance-flagged role of the mini catalog
GENERIC = {"FIN-GEN", "SAL-GEN"}
ACCEPT, AGENT, PERSON = Route.ACCEPT, Route.AGENT, Route.PERSON
ESCALATE = ReviewAnswer(ReviewKind.ESCALATE, None, "", LABEL_STAND_IN)

# What the stand-in LLM answers for each model-eligible mini row, worked out
# by hand from Q17: the role and the quote. The stand-in agent scores the same
# words (Q19), so it confirms each of them, and escalates the other two rows.
LLM_PICKS = {
    TitleKey("FIN", "accounts payable clerk"): ("FIN-05", "accounts payable clerk"),
    TitleKey("FIN", "financial analyst"): ("FIN-01", "financial analyst"),
    TitleKey("FIN", "internal auditor"): ("FIN-04", "internal auditor"),
    TitleKey("FIN", "payroll specialist"): ("FIN-03", "payroll specialist"),
    TitleKey("FIN", "senior financial analyst"): ("FIN-01", "financial analyst"),
    TitleKey("FIN", "treasury analyst"): ("FIN-06", "treasury analyst"),
    TitleKey("SAL", "account manager ii"): ("SAL-01", "account manager"),
    TitleKey("SAL", "sales engineer"): ("SAL-04", "sales engineer"),
    TitleKey("SAL", "sales operations analyst"): ("SAL-03", "sales operations analyst"),
    TitleKey("SAL", "sales representative"): ("SAL-02", "sales representative"),
    TitleKey("SAL", "senior account manager"): ("SAL-01", "account manager"),
}
NO_ROLE = [TitleKey("FIN", "special projects"), TitleKey("SAL", "bdr")]

# Titles the rules send to the agent, which the mini data set has none of:
# three recorded in the wrong family (1.7), and one on which the two methods
# differ, by its words a customer success manager and by its letters nearest to
# the sales representative.
TREASURY_IN_SALES = TitleRow(
    Record("treasury analyst", "SAL", "Treasury", "Manages cash positions and bank accounts."),
    ("X001",),
    ("Treasury Analyst",),
)
ENGINEER_IN_FINANCE = TitleRow(
    Record(
        "sales engineer",
        "FIN",
        "Solutions",
        "Explains the technical side of products and runs demonstrations during a sale.",
    ),
    ("X002",),
    ("Sales Engineer",),
)
ANALYST_IN_SALES = TitleRow(
    Record(
        "financial analyst",
        "SAL",
        "Sales Operations",
        "Builds budgets and forecasts for the sales teams.",
    ),
    ("X003",),
    ("Financial Analyst",),
)
SUCCESS_REPRESENTATIVE = TitleRow(
    Record("customer success representative", "SAL", "Customer Success", ""),
    ("X004",),
    ("Customer Success Rep",),
)
HAND_MADE = (TREASURY_IN_SALES, ENGINEER_IN_FINANCE, ANALYST_IN_SALES, SUCCESS_REPRESENTATIVE)


# --- A run up to the agent ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scored:
    """One title row with everything the agent and the rules are given about it."""

    row: TitleRow
    match: MatchResult
    shown: tuple[NumberedRole, ...]  # as the LLM saw them
    block: str
    llm_answer: ChooseAnswer
    referral: RuleOutcome

    @property
    def family(self) -> str:
        return self.row.record.family_code

    def case(self) -> AgentCase:
        return agent.build_case(
            self.row,
            self.shown,
            self.llm_answer,
            self.referral.role_id,
            self.match,
            self.referral.reasons,
        )

    def settle(self, result: AgentResult, cat: Catalog) -> RuleOutcome:
        """The agent's answer through the rules (1.7)."""
        numbered = result.transcript.numbered_roles
        return rules.check_agent(result.answer, numbered, self.block, self.match, self.family, cat)


def score_all(
    models: Models, rows: Sequence[TitleRow], knowledge: Knowledge, cat: Catalog
) -> dict[TitleKey, Scored]:
    """Look-up aside, the steps of 1.6 before the agent, for every row a model may see."""
    matches = Matcher(cat, models.embedder).match(rows, knowledge)
    scored = {}
    for row in rows:
        if not row.model_eligible:
            continue
        match = matches[row.key]
        shortlist = [catalog.get_role(cat, each.role_id) for each in match.shortlist]
        shown = fallback.number_roles(shortlist, row.key, SETTINGS.seed)
        block = fallback.render_record(row.record)
        llm_answer = models.llm.chooser.choose(row.record, shown)
        referral = rules.check_llm(llm_answer, shown, block, match, row.record.family_code, cat)
        scored[row.key] = Scored(row, match, shown, block, llm_answer, referral)
    return scored


def without_flags(cat: Catalog) -> Catalog:
    roles = tuple(dataclasses.replace(role, compliance_flag=False) for role in cat.roles)
    return dataclasses.replace(cat, roles=roles)


class SpyingEmbedder:
    """The real embedder, with a record of every text it was given."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.texts += texts
        return self._embedder.embed(texts)


@pytest.fixture(scope="module")
def cat(mini_inputs: InputData) -> Catalog:
    return mini_inputs.catalog


@pytest.fixture(scope="module")
def models(mini_inputs: InputData) -> Models:
    """The default models, from the real switch."""
    return switch.build_models(SETTINGS, mini_inputs.catalog.ladder, mini_inputs.abbreviations)


@pytest.fixture(scope="module")
def rows(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    return prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )


@pytest.fixture(scope="module")
def knowledge(mini_inputs: InputData, rows) -> Knowledge:
    """What people have decided before any wave: the seed look-up."""
    known = Knowledge()
    load_seed(known, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    return known


@pytest.fixture(scope="module")
def scored(models, rows, knowledge, cat) -> dict[TitleKey, Scored]:
    return score_all(models, [*rows, *HAND_MADE], knowledge, cat)


@pytest.fixture(scope="module")
def tools(cat, knowledge, models) -> agent.SearchTools:
    return agent.SearchTools(cat, knowledge, models.embedder, SETTINGS)


@pytest.fixture(scope="module")
def results(scored, tools, models) -> dict[TitleKey, AgentResult]:
    """Every row a model may see through the agent, referred or not."""
    return {
        key: agent.review(each.case(), tools, models.llm.reviewer) for key, each in scored.items()
    }


# --- The stand-in agent through the shared loop (Q19, Q20) --------------------------


def test_the_tables_cover_every_row_a_model_may_see(scored, mini_rows):
    eligible = [row.key for row in mini_rows if row.model_eligible]
    assert sorted([*LLM_PICKS, *NO_ROLE]) == eligible
    assert sorted(scored) == sorted([*eligible, *(row.key for row in HAND_MADE)])


def test_the_agent_runs_its_script_through_the_capped_searches(results):
    for key, result in results.items():
        requests = [step.request for step in result.transcript.steps]
        # Ten shortlisted roles of twelve always show two of the family: no third search.
        assert requests == [
            SearchRequest(SearchTool.CATALOG, key.cleaned_title),
            SearchRequest(SearchTool.APPROVED_TITLES, key.cleaned_title),
        ], key
        assert result.transcript.searches_left == AGENT_MAX_SEARCHES - 2
        assert result.answer.source == LABEL_STAND_IN


@pytest.mark.parametrize("key", sorted(LLM_PICKS), ids=" / ".join)
def test_on_a_plain_row_the_agent_confirms_what_the_llm_picked(scored, results, key):
    role_id, quote = LLM_PICKS[key]
    each, result = scored[key], results[key]
    assert each.referral.role_id == role_id
    assert (result.answer.answer, result.role_id, result.answer.quote) == (
        ReviewKind.ROLE,
        role_id,
        quote,
    )
    assert result.confirms_llm
    # The agent's numbers are its own: the LLM's list numbers all ten roles.
    assert [shown.role for shown in result.transcript.case.roles] == [n.role for n in each.shown]
    assert result.answer.number <= len(result.transcript.numbered_roles) <= 6


def test_the_agent_escalates_what_shares_no_word_with_any_role(results):
    for key in NO_ROLE:
        assert results[key].answer == ESCALATE
        assert results[key].role_id is None and not results[key].confirms_llm


def test_a_title_filed_in_the_wrong_family_is_corrected_to_a_role_of_its_family(
    scored, results, cat
):
    each, result = scored[TREASURY_IN_SALES.key], results[TREASURY_IN_SALES.key]
    # Both models say Treasury Analyst, a finance role; the title is recorded in sales.
    assert each.referral == RuleOutcome(AGENT, (Reason.FAMILY_MISMATCH,), "FIN-06")
    assert each.match.first_choice == "FIN-06"
    case = result.transcript.case
    (marked,) = [shown for shown in case.roles if shown.role.role_id == "FIN-06"]
    assert marked.number is None  # shown, marked as both opinions, and not to be picked
    assert (case.llm_role_id, case.ml_first_choice) == ("FIN-06", "FIN-06")
    # Sales Operations Analyst shares "analyst" with the title and beats the runner-up.
    assert (result.answer.answer, result.role_id, result.answer.quote) == (
        ReviewKind.ROLE,
        "SAL-03",
        "analyst",
    )
    assert not result.confirms_llm  # picking another role corrects the LLM
    assert each.settle(result, cat) == RuleOutcome(ACCEPT, (), "SAL-03")


def test_two_models_that_disagree_are_settled_by_the_agent_confirming_the_llm(scored, results, cat):
    each, result = scored[SUCCESS_REPRESENTATIVE.key], results[SUCCESS_REPRESENTATIVE.key]
    assert each.referral == RuleOutcome(AGENT, (Reason.MODELS_DISAGREE,), "SAL-06")
    assert each.match.first_choice == "SAL-02"
    assert (result.role_id, result.answer.quote) == ("SAL-06", "customer success")
    assert result.confirms_llm
    # The agent's answer is not checked for agreement with the ML model (1.7).
    assert each.settle(result, cat) == RuleOutcome(ACCEPT, (), "SAL-06")


def test_an_escalation_by_the_agent_goes_to_a_person(scored, results, cat):
    each, result = scored[ENGINEER_IN_FINANCE.key], results[ENGINEER_IN_FINANCE.key]
    assert each.referral == RuleOutcome(AGENT, (Reason.FAMILY_MISMATCH,), "SAL-04")
    # No finance role shares a word with "sales engineer".
    assert result.answer == ESCALATE
    assert each.settle(result, cat) == RuleOutcome(PERSON, (Reason.AGENT_ESCALATED,), None)


def test_a_no_match_goes_to_the_agent_which_finds_no_role_either(scored, results, cat):
    # With no flag in the catalog "special projects" is a plain "no match".
    each = scored[TitleKey("FIN", "special projects")]
    unflagged = without_flags(cat)
    referral = rules.check_llm(
        each.llm_answer, each.shown, each.block, each.match, each.family, unflagged
    )
    assert referral == RuleOutcome(AGENT, (Reason.NO_MATCH,), None)
    result = results[each.row.key]
    assert result.transcript.case.llm_role_id is None
    assert each.settle(result, unflagged) == RuleOutcome(PERSON, (Reason.AGENT_ESCALATED,), None)


def test_near_the_flagged_role_the_agent_s_answer_cannot_stand(scored, results, cat):
    # A family mismatch alone would be the agent's to settle. The ML top three
    # hold the internal auditor, so a person decides (1.7: a person beats the agent).
    each, result = scored[ANALYST_IN_SALES.key], results[ANALYST_IN_SALES.key]
    assert FLAGGED in each.match.top_three
    assert each.referral == RuleOutcome(
        PERSON, (Reason.COMPLIANCE_FLAG, Reason.FAMILY_MISMATCH), "FIN-01"
    )
    assert each.settle(result, cat) == RuleOutcome(PERSON, (Reason.COMPLIANCE_FLAG,), "SAL-03")
    assert each.settle(result, without_flags(cat)) == RuleOutcome(ACCEPT, (), "SAL-03")


def test_every_quote_of_the_agent_passes_the_shared_quote_check(scored, results):
    for key, result in results.items():
        if result.answer.answer is ReviewKind.ROLE:
            assert fallback.quote_in_record(result.answer.quote, scored[key].block), key
            # The block is in the case as the agent is shown it, word for word.
            assert agent.render_case(result.transcript.case).startswith(scored[key].block + "\n")


# --- The two searches, ranked a second time (Q10) ------------------------------------


def dot(a: Vector, b: Vector) -> float:
    return math.fsum(weight * b.get(index, 0.0) for index, weight in a.items())


def nearest_roles(models: Models, query: str, cat: Catalog, top: int) -> list[str]:
    """The role ids a catalog search must return: every role but the generic
    ones, nearest first, ties by role_id."""
    visible = [role for role in cat.roles if not role.is_generic]
    texts = [f"{role.role_name}. {role.definition}" for role in visible]
    target, *vectors = models.embedder.embed([query, *texts])
    ranked = sorted(
        zip(visible, vectors, strict=True),
        key=lambda each: (-dot(target, each[1]), each[0].role_id),
    )
    return [role.role_id for role, _ in ranked[:top]]


def nearest_titles(
    models: Models, query: str, known: Knowledge, top: int
) -> list[tuple[str, str, str]]:
    """The approved titles a search must return, each with its family code and role."""
    examples = known.examples()
    texts = [f"{e.record.cleaned_title} {e.record.description}".strip() for e in examples]
    target, *vectors = models.embedder.embed([query, *texts])
    ranked = sorted(
        zip(examples, vectors, strict=True),
        key=lambda each: (
            -dot(target, each[1]),
            each[0].key.cleaned_title,
            each[0].key.family_code,
        ),
    )
    return [(e.key.cleaned_title, e.key.family_code, e.role_id) for e, _ in ranked[:top]]


def queries(rows: Sequence[TitleRow]) -> list[str]:
    asked = [row.record.cleaned_title for row in rows]
    asked += [" ".join(row.record.description.split()[:12]) for row in rows]
    return [*dict.fromkeys(asked), "Sales", "Finance", "Generic role for the Finance family."]


def test_a_catalog_search_returns_the_nearest_roles_of_the_whole_catalog(
    rows, tools, models, cat, scored
):
    transcript = AgentTranscript(scored[TREASURY_IN_SALES.key].case())
    for query in queries(rows):
        step = tools.search(SearchRequest(SearchTool.CATALOG, query), transcript)
        found = [hit.role.role.role_id for hit in step.hits]
        assert found == nearest_roles(models, query, cat, SETTINGS.agent_search_k), query
        assert len(found) == 5
    # The title's own role comes first, whatever family the case is in.
    step = tools.search(SearchRequest(SearchTool.CATALOG, "treasury analyst"), transcript)
    assert step.hits[0].role == NumberedRole(None, catalog.get_role(cat, "FIN-06"))


def test_an_approved_title_search_returns_the_nearest_titles_people_decided(
    rows, tools, models, knowledge, scored
):
    transcript = AgentTranscript(scored[TREASURY_IN_SALES.key].case())
    # Before any wave: the two titles of the mini seed look-up, cleaned.
    assert [tuple(example.key) for example in knowledge.examples()] == [
        ("FIN", "accounts payable clerk"),
        ("SAL", "sales engineer"),
    ]
    for query in queries(rows):
        step = tools.search(SearchRequest(SearchTool.APPROVED_TITLES, query), transcript)
        found = [
            (hit.approved_title, hit.title_family_code, hit.role.role.role_id) for hit in step.hits
        ]
        assert found == nearest_titles(models, query, knowledge, SETTINGS.agent_search_k), query
        assert len(found) == 2
    step = tools.search(SearchRequest(SearchTool.APPROVED_TITLES, "sales engineer"), transcript)
    assert agent.render_step(step).splitlines()[0] == (
        '5. Sales Engineer: approved for the title "sales engineer" (family SAL)'
    )


def test_what_people_approve_in_one_wave_is_found_by_the_searches_of_the_next(
    mini_inputs, mini_settings, rows, models, cat, scored
):
    # Q18: a wave is scored first and its decisions are learned afterwards. The
    # same SearchTools serves both waves.
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0, undecided_rate=0.0)
    simulated = people.load_people(settings, cat)
    golden_key = TitleKey("FIN", "treasury analyst")  # long tail, and not in the seed look-up
    known = Knowledge(golden_keys=[golden_key])
    load_seed(known, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    spy = SpyingEmbedder(models.embedder)
    both_waves = agent.SearchTools(cat, known, spy, SETTINGS)
    transcript = AgentTranscript(scored[SUCCESS_REPRESENTATIVE.key].case())
    request = SearchRequest(SearchTool.APPROVED_TITLES, "senior account manager")

    def found() -> list[tuple[str, str, str]]:
        hits = both_waves.search(request, transcript).hits
        return [(hit.approved_title, hit.title_family_code, hit.role.role.role_id) for hit in hits]

    assert found() == nearest_titles(models, request.query, known, 5)
    assert len(found()) == 2
    # The Wave 1 list of the mini settings, the row with a person's name and a
    # golden-set title are decided by the simulated people.
    wave1 = sorted(rows, key=lambda row: (-row.headcount, row.key))[: settings.wave1_size]
    named = next(row for row in rows if row.has_person_name)
    golden = next(row for row in rows if row.key == golden_key)
    assert golden not in wave1
    for row in (*wave1, named, golden):
        known.learn(row, simulated.wave1_approval(row).role_id)
    assert found() == nearest_titles(models, request.query, known, 5)
    assert found()[0] == ("senior account manager", "SAL", "SAL-01")
    assert [title for title, _, _ in found()][:2] == [
        "senior account manager",
        "account manager ii",
    ]
    # A person's decision on a title with a name joins the look-up and no search (Q14);
    # a golden-set title joins neither (Q07).
    assert known.lookup(named.key) == "SAL-06" and known.lookup(golden.key) is None
    every = agent.SearchTools(cat, known, spy, dataclasses.replace(SETTINGS, agent_search_k=100))
    titles = [hit.approved_title for hit in every.search(request, transcript).hits]
    assert sorted(titles) == sorted(
        [
            "accounts payable clerk",
            "sales engineer",
            "account manager ii",
            "sales representative",
            "senior account manager",
            "financial analyst",
        ]
    )
    # Its text reaches the embedder in no way. (A name masked inside the
    # description of another title is embedded with it, as 1.4 allows.)
    assert NAME_MASK in named.record.cleaned_title
    assert not any(named.record.cleaned_title in text for text in spy.texts)
    assert not any(text.startswith(golden_key.cleaned_title) for text in spy.texts)


# --- No model is ever shown a generic family role (2.7) -----------------------------


def test_no_search_result_numbered_list_or_rendered_text_holds_a_generic_role(
    rows, scored, models, cat, knowledge
):
    assert {role.role_id for role in cat.roles if role.is_generic} == GENERIC
    generic_texts = [
        f"{role.role_name}. {role.definition}" for role in cat.roles if role.is_generic
    ]
    assert generic_texts[0] == "Sales. Generic role for the Sales family."
    for top in (SETTINGS.agent_search_k, 12, 100):
        spy = SpyingEmbedder(models.embedder)
        settings = dataclasses.replace(SETTINGS, agent_search_k=top)
        wide = agent.SearchTools(cat, knowledge, spy, settings)
        for each in scored.values():
            # What the stand-in agent asks for, then a generic role asked for by name.
            transcripts = [agent.review(each.case(), wide, models.llm.reviewer).transcript]
            asked = AgentTranscript(each.case())
            for query in ("Sales", "Finance", *generic_texts):
                step = wide.search(SearchRequest(SearchTool.CATALOG, query), asked)
                asked = AgentTranscript(asked.case, (*asked.steps, step))
            for transcript in (*transcripts, asked):
                hits = [hit for step in transcript.steps for hit in step.hits]
                shown = [*transcript.case.roles, *(hit.role for hit in hits)]
                assert not any(role.role.is_generic for role in shown)
                assert not GENERIC & {role.role.role_id for role in shown}
                assert not any(role.role.is_generic for role in transcript.numbered_roles)
                texts = [
                    agent.render_case(transcript.case),
                    *map(agent.render_step, transcript.steps),
                ]
                assert not any("Generic role" in text for text in texts)
                for step in transcript.steps:
                    if step.request.tool is SearchTool.CATALOG:
                        assert len(step.hits) == min(top, 12)  # every role but the two generic
        # Never embedded for a search either: only as the query that asked for it.
        assert spy.texts.count(generic_texts[0]) == spy.texts.count(generic_texts[1]) == 1
        assert len(set(spy.texts)) == len(spy.texts)


# --- The agent cannot pick a role it was not shown or one outside the family (2.7) ----


class AnsweringWith:
    """An injected reviewer: the stand-in's searches, then a number of the test's
    choosing with a quote that stands in the record."""

    def __init__(self, reviewer, number: int) -> None:
        self._reviewer = reviewer
        self._number = number

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        turn = self._reviewer.next_turn(transcript)
        if isinstance(turn, SearchRequest):
            return turn
        return ReviewAnswer(
            ReviewKind.ROLE, self._number, transcript.case.record.cleaned_title, "test"
        )


def test_whatever_number_the_reviewer_returns_the_role_was_shown_and_is_of_the_family(
    scored, tools, models, cat
):
    picked: set[tuple[TitleKey, str]] = set()
    for key, each in scored.items():
        in_family = {role.role_id for role in catalog.roles_in_family(cat, each.family)}
        for number in range(-1, 15):
            reviewer = AnsweringWith(models.llm.reviewer, number)
            result = agent.review(each.case(), tools, reviewer)
            outcome = each.settle(result, cat)
            assert outcome.role_id == result.role_id
            if result.role_id is None:
                assert outcome.route is PERSON and outcome.reasons[0] is Reason.ROLE_NOT_IN_CATALOG
                continue
            picked.add((key, result.role_id))
            transcript = result.transcript
            shown = [
                *transcript.case.roles,
                *(hit.role for s in transcript.steps for hit in s.hits),
            ]
            assert result.role_id in in_family
            assert NumberedRole(number, catalog.get_role(cat, result.role_id)) in shown
            assert Reason.FAMILY_MISMATCH not in outcome.reasons
    # Every role of the family that was shown could be picked, and nothing else:
    # in the mini catalog that is at most the family's six roles.
    assert all(len({role for k, role in picked if k == key}) in range(2, 7) for key in scored)


def test_the_out_of_family_role_both_models_chose_cannot_be_the_agent_s_answer(
    scored, tools, models, cat
):
    for row, outside in ((TREASURY_IN_SALES, "FIN-06"), (ENGINEER_IN_FINANCE, "SAL-04")):
        each = scored[row.key]
        assert each.referral.role_id == each.match.first_choice == outside
        for number in range(-1, 15):
            result = agent.review(each.case(), tools, AnsweringWith(models.llm.reviewer, number))
            assert result.role_id != outside
            assert each.settle(result, cat).role_id != outside
        # It was shown, more than once, and never with a number.
        result = agent.review(each.case(), tools, models.llm.reviewer)
        transcript = result.transcript
        shown = [*transcript.case.roles, *(hit.role for s in transcript.steps for hit in s.hits)]
        numbers = [role.number for role in shown if role.role.role_id == outside]
        assert len(numbers) >= 2 and set(numbers) == {None}
        # The LLM's own number for it names another role in the agent's list, or none.
        llm_number = each.llm_answer.number
        assert fallback.role_for_number(llm_number, each.shown).role_id == outside
        other = fallback.role_for_number(llm_number, transcript.numbered_roles)
        assert other is None or other.family_code == row.record.family_code


# --- The cap on the real searches (2.4, Q20) ------------------------------------------


class NeverAnswering:
    """An injected reviewer that asks for one search after another.

    Were the loop to go on asking, it gives up at its tenth turn, so a loop
    with no cap fails the test instead of hanging it.
    """

    def __init__(self) -> None:
        self.turns = 0

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        self.turns += 1
        if self.turns == 10:
            return ReviewAnswer(ReviewKind.ESCALATE, None, "", "test")
        tool = SearchTool.CATALOG if self.turns % 2 else SearchTool.APPROVED_TITLES
        return SearchRequest(tool, f"{transcript.case.record.cleaned_title} {self.turns}")


def test_a_reviewer_that_only_searches_is_stopped_after_four_and_goes_to_a_person(
    scored, tools, cat
):
    for each in scored.values():
        reviewer = NeverAnswering()
        result = agent.review(each.case(), tools, reviewer)
        assert reviewer.turns == AGENT_MAX_SEARCHES + 1
        assert len(result.transcript.steps) == AGENT_MAX_SEARCHES
        assert result.answer.answer is ReviewKind.INVALID and result.role_id is None
        assert result.answer.source == LABEL_STAND_IN  # the label of the case's LLM answer
        outcome = each.settle(result, cat)
        assert outcome.route is PERSON and outcome.reasons[-1] is Reason.INVALID_ANSWER


# --- A compliance-flagged role is never settled by a machine (2.7) --------------------


class PickingTheAuditor:
    """An injected reviewer that returns the flagged role: it searches the
    catalog for the internal auditor until that role has a number, then answers
    with it and a quote that stands in the record."""

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        for shown in transcript.numbered_roles:
            if shown.role.role_name == "Internal Auditor":
                title = transcript.case.record.cleaned_title
                return ReviewAnswer(ReviewKind.ROLE, shown.number, title, "test")
        if transcript.searches_left:
            return SearchRequest(SearchTool.CATALOG, "internal auditor")
        return ReviewAnswer(ReviewKind.ESCALATE, None, "", "test")


# The finance rows whose ML top three do not hold the flagged role: nothing but
# the reviewer's own answer brings the flag in.
FAR_FROM_THE_FLAG = [
    TitleKey("FIN", "accounts payable clerk"),
    TitleKey("FIN", "payroll specialist"),
    TitleKey("FIN", "treasury analyst"),
]


def test_an_injected_reviewer_that_returns_the_flagged_role_settles_nothing(scored, tools, cat):
    assert catalog.get_role(cat, FLAGGED).compliance_flag
    for key, each in scored.items():
        result = agent.review(each.case(), tools, PickingTheAuditor())
        outcome = each.settle(result, cat)
        assert outcome.route is PERSON, key
        if each.family == "FIN":
            assert result.role_id == FLAGGED
            assert Reason.COMPLIANCE_FLAG in outcome.reasons
        else:
            # Not of the family: found by every search and never numbered.
            assert result.answer.answer is ReviewKind.ESCALATE
            assert len(result.transcript.steps) == AGENT_MAX_SEARCHES
    for key in FAR_FROM_THE_FLAG:
        each = scored[key]
        assert FLAGGED not in each.match.top_three
        result = agent.review(each.case(), tools, PickingTheAuditor())
        assert each.settle(result, cat) == RuleOutcome(PERSON, (Reason.COMPLIANCE_FLAG,), FLAGGED)


def test_the_flag_in_the_catalog_file_is_what_stops_the_injected_reviewer(
    tmp_path, mini_inputs, models, rows
):
    # The same files with the flag taken off the internal auditor: the same
    # reviewer, loop and searches now settle the title on that role.
    roles = [dataclasses.replace(role, compliance_flag=False) for role in mini_inputs.catalog.roles]
    datafiles.write_inputs(tmp_path, mini_inputs)
    datafiles.write_catalog_roles(tmp_path, roles)
    data = datafiles.read_inputs(tmp_path)
    assert not any(role.compliance_flag for role in data.catalog.roles)
    known = Knowledge()
    load_seed(known, data.seed_lookup, rows, data.abbreviations)
    unflagged = score_all(models, rows, known, data.catalog)
    searches = agent.SearchTools(data.catalog, known, models.embedder, SETTINGS)
    for key in FAR_FROM_THE_FLAG:
        each = unflagged[key]
        result = agent.review(each.case(), searches, PickingTheAuditor())
        assert each.settle(result, data.catalog) == RuleOutcome(ACCEPT, (), FLAGGED)


# --- The same in every process (Q06, Q18) -----------------------------------------------

# Prints, for every mini row a model may see, what the agent was shown and what it answered.
TRANSCRIPTS_SCRIPT = """
import json
from pathlib import Path

from titlestd import agent, catalog, datafiles, fallback, prepare, rules, switch
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher
from titlestd.settings import Settings

settings = Settings()
data = datafiles.read_inputs(Path("tests/fixtures/mini"))
cat = data.catalog
rows = prepare.prepare(data.employees, data.abbreviations, catalog.family_codes(cat))
models = switch.build_models(settings, cat.ladder, data.abbreviations)
knowledge = Knowledge()
load_seed(knowledge, data.seed_lookup, rows, data.abbreviations)
matches = Matcher(cat, models.embedder).match(rows, knowledge)
tools = agent.SearchTools(cat, knowledge, models.embedder, settings)
plain = []
for row in rows:
    if not row.model_eligible:
        continue
    match = matches[row.key]
    shortlist = [catalog.get_role(cat, each.role_id) for each in match.shortlist]
    shown = fallback.number_roles(shortlist, row.key, settings.seed)
    block = fallback.render_record(row.record)
    answer = models.llm.chooser.choose(row.record, shown)
    referral = rules.check_llm(answer, shown, block, match, row.record.family_code, cat)
    case = agent.build_case(row, shown, answer, referral.role_id, match, referral.reasons)
    result = agent.review(case, tools, models.llm.reviewer)
    plain.append(
        [
            agent.render_case(case),
            [
                [step.request.tool, step.request.query, agent.render_step(step)]
                for step in result.transcript.steps
            ],
            [result.answer.answer, result.answer.number, result.answer.quote, result.role_id],
        ]
    )
print(json.dumps(plain))
"""


def test_another_process_shows_the_agent_the_same_and_gets_the_same_answers(
    repo_root, rows, scored, results
):
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", TRANSCRIPTS_SCRIPT],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == outputs[2]
    here = [
        [
            agent.render_case(results[row.key].transcript.case),
            [
                [step.request.tool, step.request.query, agent.render_step(step)]
                for step in results[row.key].transcript.steps
            ],
            [
                results[row.key].answer.answer,
                results[row.key].answer.number,
                results[row.key].answer.quote,
                results[row.key].role_id,
            ],
        ]
        for row in rows
        if row.model_eligible
    ]
    assert outputs[0] == json.loads(json.dumps(here))
    assert len(here) == 13


# --- One turn at a time: a turn can be recorded and replayed (Q20) -----------------------


def shown_so_far(transcript: AgentTranscript) -> str:
    """Everything a backend is shown before a turn: the case, then each search
    with its request and its result text. A recording keys a turn on this (Q33)."""
    parts = [agent.render_case(transcript.case)]
    for step in transcript.steps:
        parts += [f"{step.request.tool.value}: {step.request.query}", agent.render_step(step)]
    return "\n\n".join(parts)


class Recording:
    """A backend in miniature that keeps each turn of another under what it was shown."""

    def __init__(self, reviewer) -> None:
        self._reviewer = reviewer
        self.turns: dict[str, SearchRequest | ReviewAnswer] = {}

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        turn = self._reviewer.next_turn(transcript)
        assert self.turns.setdefault(shown_so_far(transcript), turn) is turn  # one entry per turn
        return turn


class Replaying:
    """A backend in miniature that only looks turns up. A turn it was not
    recorded for is a KeyError: it never falls back."""

    def __init__(self, turns: dict[str, SearchRequest | ReviewAnswer]) -> None:
        self._turns = turns

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        return self._turns[shown_so_far(transcript)]


def test_recorded_turns_replay_to_the_same_result_with_the_searches_run_again(
    scored, results, models, cat, knowledge, mini_inputs, rows
):
    # Q20: one entry per model turn; replay looks each turn up and re-runs the
    # searches for real. The transcript and the two renderers are all it needs.
    recording = Recording(models.llm.reviewer)
    first = agent.SearchTools(cat, knowledge, models.embedder, SETTINGS)
    recorded = {key: agent.review(each.case(), first, recording) for key, each in scored.items()}
    assert recorded == results
    assert len(recording.turns) == sum(len(r.transcript.steps) + 1 for r in recorded.values())

    again = agent.SearchTools(cat, knowledge, models.embedder, SETTINGS)
    replaying = Replaying(recording.turns)
    replayed = {key: agent.review(each.case(), again, replaying) for key, each in scored.items()}
    assert replayed == recorded

    # With one more title approved, a search returns something else: the turn
    # after it was never recorded, and the replay stops instead of answering.
    more = Knowledge()
    load_seed(more, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    more.learn(scored[TREASURY_IN_SALES.key].row, "SAL-03")
    changed = agent.SearchTools(cat, more, models.embedder, SETTINGS)
    with pytest.raises(KeyError):
        agent.review(scored[SUCCESS_REPRESENTATIVE.key].case(), changed, replaying)
