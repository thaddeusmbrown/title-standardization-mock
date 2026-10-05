"""The agent on the generated mock data: what holds for every title the rules refer.

The mini data set has twelve roles in two families, so the ten shortlisted
roles always show most of a family. The generated data has 48 roles in eight
families: a shortlist may hold one role of the recorded family or none, and
the roles the agent may pick are then numbered by its searches.

The shape of the default run follows from the generator's difficulty
constants, which a tuning round may change (Q23), so nothing here counts
titles or names one. Each assertion is a rule of docs/design.md 1.6 step 6 and
1.7 or of docs/build-decisions.md Q10, Q19 and Q20, and holds whatever the
data. Nothing here reads the truth file.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from titlestd import agent, catalog, datafiles, fallback, prepare, rules, switch
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    AgentCase,
    AgentResult,
    AgentTranscript,
    Catalog,
    MatchResult,
    Models,
    NumberedRole,
    Reason,
    ReviewAnswer,
    ReviewKind,
    Route,
    SearchRequest,
    SearchTool,
    TitleRow,
)
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher
from titlestd.settings import Settings

SETTINGS = Settings()


@dataclasses.dataclass(frozen=True)
class Referred:
    """A title the rules sent to the agent, and what the agent made of it."""

    row: TitleRow
    match: MatchResult
    block: str
    case: AgentCase
    result: AgentResult

    @property
    def family(self) -> str:
        return self.row.record.family_code

    @property
    def shown(self) -> list[NumberedRole]:
        """Every role the agent was shown, in the case and by its searches."""
        steps = self.result.transcript.steps
        return [*self.case.roles, *(hit.role for step in steps for hit in step.hits)]


@dataclasses.dataclass(frozen=True)
class World:
    cat: Catalog
    models: Models
    knowledge: Knowledge
    tools: agent.SearchTools
    referred: tuple[Referred, ...]


@pytest.fixture(scope="module")
def world(repo_root: Path) -> World:
    """The steps of 1.6 up to the agent on the committed data, before any wave:
    the seed look-up is all that people have decided."""
    data = datafiles.read_inputs(repo_root / "data")
    cat = data.catalog
    rows = prepare.prepare(data.employees, data.abbreviations, catalog.family_codes(cat))
    models = switch.build_models(SETTINGS, cat.ladder, data.abbreviations)
    knowledge = Knowledge()
    load_seed(knowledge, data.seed_lookup, rows, data.abbreviations)
    scored = [row for row in rows if row.model_eligible and knowledge.lookup(row.key) is None]
    matches = Matcher(cat, models.embedder).match(scored, knowledge)
    tools = agent.SearchTools(cat, knowledge, models.embedder, SETTINGS)
    referred = []
    for row in scored:
        match = matches[row.key]
        shortlist = [catalog.get_role(cat, each.role_id) for each in match.shortlist]
        shown = fallback.number_roles(shortlist, row.key, SETTINGS.seed)
        block = fallback.render_record(row.record)
        answer = models.llm.chooser.choose(row.record, shown)
        referral = rules.check_llm(answer, shown, block, match, row.record.family_code, cat)
        if referral.route is Route.AGENT:
            case = agent.build_case(row, shown, answer, referral.role_id, match, referral.reasons)
            result = agent.review(case, tools, models.llm.reviewer)
            referred.append(Referred(row, match, block, case, result))
    return World(cat, models, knowledge, tools, tuple(referred))


def test_some_titles_are_referred_to_the_agent(world):
    # Q23: the default run has an agent-settled row, so the rules refer titles.
    assert world.referred
    assert all(each.case.reasons for each in world.referred)


def test_every_case_ends_within_the_search_cap(world):
    for each in world.referred:
        requests = [step.request for step in each.result.transcript.steps]
        title = each.row.record.cleaned_title
        # The stand-in's script (Q19): two searches, and a third only when it
        # has fewer than two roles to pick from and a description to ask with.
        assert requests[:2] == [
            SearchRequest(SearchTool.CATALOG, title),
            SearchRequest(SearchTool.APPROVED_TITLES, title),
        ]
        assert len(requests) in (2, 3) and len(requests) <= AGENT_MAX_SEARCHES
        if len(requests) == 3:
            assert requests[2].tool is SearchTool.CATALOG and each.row.record.description.strip()


def test_only_roles_of_the_recorded_family_are_numbered_and_each_keeps_its_number(world):
    for each in world.referred:
        in_family = {role.role_id for role in catalog.roles_in_family(world.cat, each.family)}
        numbers: dict[str, int] = {}
        for shown in each.shown:
            assert not shown.role.is_generic
            assert (shown.number is not None) == (shown.role.role_id in in_family)
            if shown.number is not None:
                assert numbers.setdefault(shown.role.role_id, shown.number) == shown.number
        # One running list: the numbers are 1 to n, each given once.
        assert sorted(numbers.values()) == list(range(1, len(numbers) + 1))
        numbered = each.result.transcript.numbered_roles
        assert {shown.role.role_id: shown.number for shown in numbered} == numbers
        # The roles of the case come first, in the order the LLM saw them.
        in_case = [shown.number for shown in each.case.roles if shown.number is not None]
        assert in_case == list(range(1, len(in_case) + 1))


def test_a_search_returns_the_top_few_and_approved_titles_come_from_the_examples(world):
    approved = {tuple(example.key): example.role_id for example in world.knowledge.examples()}
    for each in world.referred:
        for step in each.result.transcript.steps:
            if step.request.tool is SearchTool.CATALOG:
                assert len(step.hits) == SETTINGS.agent_search_k
                assert len({hit.role.role.role_id for hit in step.hits}) == len(step.hits)
                continue
            assert len(step.hits) == min(SETTINGS.agent_search_k, len(approved))
            for hit in step.hits:
                key = (hit.title_family_code, hit.approved_title)
                assert approved[key] == hit.role.role.role_id


def test_the_agent_s_answer_names_a_role_it_was_shown_in_the_family_or_escalates(world):
    kinds = set()
    for each in world.referred:
        result = each.result
        numbered = result.transcript.numbered_roles
        outcome = rules.check_agent(
            result.answer, numbered, each.block, each.match, each.family, world.cat
        )
        kinds.add(result.answer.answer)
        if result.answer.answer is ReviewKind.ESCALATE:
            assert result.role_id is None
            assert (outcome.route, outcome.reasons[-1]) == (Route.PERSON, Reason.AGENT_ESCALATED)
            continue
        assert result.answer.answer is ReviewKind.ROLE
        role = catalog.get_role(world.cat, result.role_id)
        assert NumberedRole(result.answer.number, role) in numbered
        assert role in catalog.roles_in_family(world.cat, each.family)
        assert fallback.quote_in_record(result.answer.quote, each.block)
        assert result.confirms_llm == (result.role_id == each.case.llm_role_id)
        assert outcome.role_id == result.role_id
        # What stands is never a flagged role or a title near one (1.7).
        flagged = role.compliance_flag or Reason.COMPLIANCE_FLAG in outcome.reasons
        assert outcome.route is (Route.PERSON if flagged else Route.ACCEPT)
    assert kinds <= {ReviewKind.ROLE, ReviewKind.ESCALATE}  # the stand-in is never invalid


class AnsweringWith:
    """An injected reviewer: the stand-in's searches, then a number of the test's choosing."""

    def __init__(self, reviewer, number: int) -> None:
        self._reviewer = reviewer
        self._number = number

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        turn = self._reviewer.next_turn(transcript)
        if isinstance(turn, SearchRequest):
            return turn
        title = transcript.case.record.cleaned_title
        return ReviewAnswer(ReviewKind.ROLE, self._number, title, "test")


def test_no_number_an_injected_reviewer_returns_reaches_a_role_outside_the_family(world):
    # 2.7, on every referred title: the role is one of the family that the
    # agent was shown, or the title goes to a person.
    reached = 0
    for each in world.referred:
        outside = {
            shown.role.role_id for shown in each.shown if shown.role.family_code != each.family
        }
        for number in (0, 1, 2, 3, 6, 11, 40):
            reviewer = AnsweringWith(world.models.llm.reviewer, number)
            result = agent.review(each.case, world.tools, reviewer)
            numbered = result.transcript.numbered_roles
            outcome = rules.check_agent(
                result.answer, numbered, each.block, each.match, each.family, world.cat
            )
            if result.role_id is None:
                assert outcome.route is Route.PERSON
                assert outcome.reasons[0] is Reason.ROLE_NOT_IN_CATALOG
                continue
            reached += 1
            role = catalog.get_role(world.cat, result.role_id)
            assert role.family_code == each.family and not role.is_generic
            assert result.role_id not in outside
            assert NumberedRole(number, role) in numbered
            assert Reason.FAMILY_MISMATCH not in outcome.reasons
    assert reached
