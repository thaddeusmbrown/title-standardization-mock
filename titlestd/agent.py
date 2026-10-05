"""The agent: one plain loop in shared code, two read-only searches.

Owner: feat/agent (stage 3).
Governed by: docs/design.md 1.6 step 6, 1.7 (last paragraph), 2.4 ("The
agent"), 2.5 (the search cap and the number-to-role mapping stay in shared
code); docs/build-decisions.md Q10 (the case, the searches, the numbering),
Q19 (the stand-in's script, in titlestd.standin), Q20 (the loop).
Section 2.7 tests: the agent cannot pick a role it was not shown or one
outside the family; no model is ever shown a generic family role (never in a
search result).

The loop (Q20): call reviewer.next_turn(transcript) until it returns a final
answer. A turn is one SearchRequest or one ReviewAnswer. Each executed search
of either kind counts one toward contracts.AGENT_MAX_SEARCHES; zero searches
is allowed. A search request after the cap, or an answer whose number is not
in the case's map, sends the title to a person. No agent framework.

The searches (Q10): search_catalog(query) and search_approved_titles(query),
one free-text argument, no family filter, ranked by cosine similarity with the
active embedder, ties by role_id or title, top settings.agent_search_k.
Generic family roles are dropped. Numbering is one running list per case:
in-family shortlist roles first, then each new in-family role on first
appearance; a number never changes and is never reused; an out-of-family role
is shown without a number.

The agent's answer goes back through titlestd.rules.check_agent. Picking the
LLM's role confirms it; picking another corrects it. In Wave 1 the agent only
recommends; the pipeline decides what stands.

Decided here, where the record leaves the detail open:

- Who calls the rules. review returns the answer, the role its number named
  and the transcript, and checks nothing else. The caller hands
  AgentResult.answer and AgentResult.transcript.numbered_roles to
  rules.check_agent, with the record block of fallback.render_record.
- The recorded family. A role is of the case's family when its family code
  equals the record's, trimmed, as titlestd.catalog reads a code. A blank
  code is no family, so no role is numbered for it.
- build_case numbers the in-family roles from 1 in the order the LLM saw
  them and takes llm_role_id as given: the role rules.check_llm resolved, or
  None. It raises ValueError for a generic family role in the list, and for
  a row no model may see (a person's name, no usable family code: Q14, Q26),
  because the agent is a model.
- What a search compares. A query is embedded as written. A catalog role is
  embedded as matcher.role_text and an approved title as matcher.embed_text
  of its example's record, so a search meets the same two sets of vectors
  as the ML match (1.6 step 3, Q12). The approved titles are the examples
  of titlestd.lookup: a title with a person's name is none (Q14), so no
  search embeds or returns it.
- What a search returns. Generic family roles are never embedded, and an
  approved title whose role no model may be shown is left out, both before
  ranking, so a search returns the nearest settings.agent_search_k of what
  is left. There is no score threshold: a query that resembles nothing
  still gets that many results, as "top 5" says. Catalog ties go by
  role_id, approved-title ties by cleaned title, then family code. An
  approved-title result is one title, so two titles approved for one role
  are two results that carry the same number.
- When a search reads the approved titles: when it runs. The caller learns
  a wave's decisions only after the wave is scored (Q18), so within a wave
  every search sees the same titles, and one SearchTools may serve the next
  wave too. It keeps the vector of every text it has embedded.
- Numbers. A role new to the case gets one more than the highest number
  in the transcript. SearchTools.search raises ValueError for a search past
  the cap and for a tool that is not one of the two, whoever calls it.
- A number that names no numbered role. The answer stays as the backend
  gave it and AgentResult.role_id is None; rules.check_agent then fails the
  catalog check (S11). Only an answer of the kind "role" has its number
  read.
- A broken protocol (S10). The loop answers INVALID itself, with a detail,
  and runs nothing further when a turn is neither a search request nor a
  final answer, when a request names no search tool or its query is not
  text, and when a search is requested after the cap. Such an answer
  carries the label of the case's LLM answer: the chooser and the reviewer
  of a run come from one llm setting (contracts.Llm), and a search request
  carries no label of its own. Whatever the backend raises passes through.
- render_case is the record block of fallback.render_record, an empty
  line, "Roles:", one line per shown role as fallback.render_roles writes
  it, followed by LLM_PICK_MARK and ML_FIRST_CHOICE_MARK where they apply,
  an empty line and "Referred because: " with the reasons joined by "; ".
  An LLM answer that named no role ("no match") marks no line; the reasons
  say it. The LLM's quote is not shown.
- render_step is the text of one tool result: one line per hit, NO_RESULTS
  when there is none, and no heading, because the request is a turn of its
  own. A catalog hit is the role line of fallback.render_roles. An
  approved-title hit is '3. Role Name: approved for the title "cleaned
  title" (family SAL)', with "-" for the number of an out-of-family role
  and the title's own family code, or NO_FAMILY when that is blank (Q26).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from titlestd.catalog import model_visible_roles
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    AgentCase,
    AgentResult,
    AgentTranscript,
    Catalog,
    CatalogRole,
    ChooseAnswer,
    Embedder,
    MatchResult,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    Reviewer,
    ReviewKind,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    TitleRow,
    Vector,
)
from titlestd.fallback import render_record, render_roles, role_for_number
from titlestd.lookup import Knowledge
from titlestd.matcher import cosine, embed_text, role_text
from titlestd.settings import Settings

# What render_case writes after the line of a role (Q10).
LLM_PICK_MARK = "[the LLM's pick]"
ML_FIRST_CHOICE_MARK = "[the ML model's first choice]"
# What render_step writes for a search that found nothing.
NO_RESULTS = "No results."
# What render_step writes for an approved title with a blank family code (Q26).
NO_FAMILY = "none recorded"


class _Candidate(NamedTuple):
    """Something a search can return: the text it is embedded as, what breaks
    a tie, and the hit it makes."""

    text: str
    tie: tuple[str, ...]
    role: CatalogRole
    approved_title: str = ""
    title_family_code: str = ""


class SearchTools:
    """The two searches for one wave, over the catalog and the approved titles."""

    def __init__(
        self, catalog: Catalog, knowledge: Knowledge, embedder: Embedder, settings: Settings
    ) -> None:
        self._knowledge = knowledge
        self._embedder = embedder
        self._top = settings.agent_search_k
        # By role_id, and never a generic family role (1.2).
        self._roles = {role.role_id: role for role in model_visible_roles(catalog)}
        self._vectors: dict[str, Vector] = {}  # text to vector, kept while these tools live

    def search(self, request: SearchRequest, transcript: AgentTranscript) -> SearchStep:
        """Run one search and number its hits for the transcript's case."""
        if transcript.searches_left == 0:
            raise ValueError(
                f"a case may run {AGENT_MAX_SEARCHES} searches and this one has run them (2.4)"
            )
        if SearchTool(request.tool) is SearchTool.CATALOG:
            candidates = self._catalog_roles()
        else:
            candidates = self._approved_titles()
        record = transcript.case.record
        numbers = {shown.role.role_id: shown.number for shown in transcript.numbered_roles}
        hits = tuple(
            SearchHit(
                _shown(found.role, record, numbers), found.approved_title, found.title_family_code
            )
            for found in self._nearest(request.query, candidates)
        )
        return SearchStep(request, hits)

    def _catalog_roles(self) -> list[_Candidate]:
        return [
            _Candidate(role_text(role), (role_id,), role) for role_id, role in self._roles.items()
        ]

    def _approved_titles(self) -> list[_Candidate]:
        """The examples as they stand now, each with the role a person gave it."""
        found = []
        for example in self._knowledge.examples():
            role = self._roles.get(example.role_id)
            if role is not None:
                family, title = example.key
                text = embed_text(example.record)
                found.append(_Candidate(text, (title, family), role, title, family))
        return found

    def _nearest(self, query: str, candidates: Sequence[_Candidate]) -> list[_Candidate]:
        """The candidates nearest to the query, best first, at most the top few."""
        texts = [query, *(candidate.text for candidate in candidates)]
        new = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if new:
            self._vectors.update(zip(new, self._embedder.embed(new), strict=True))
        target = self._vectors[query]
        ranked = sorted(
            candidates,
            key=lambda candidate: (-cosine(target, self._vectors[candidate.text]), candidate.tie),
        )
        return ranked[: self._top]


def build_case(
    row: TitleRow,
    llm_roles: Sequence[NumberedRole],
    llm_answer: ChooseAnswer,
    llm_role_id: str | None,
    match: MatchResult,
    reasons: Sequence[Reason],
) -> AgentCase:
    """The case for one title: the record as the LLM saw it, the shortlisted
    roles in the LLM's order renumbered for the agent, and the referral reasons."""
    if not row.model_eligible:
        raise ValueError(
            f"the title {row.record.cleaned_title!r} in family {row.record.family_code!r} "
            "goes to a person and to no model (Q14, Q26)"
        )
    numbers: dict[str, int] = {}
    return AgentCase(
        record=row.record,
        roles=tuple(_shown(shown.role, row.record, numbers) for shown in llm_roles),
        llm_answer=llm_answer,
        llm_role_id=llm_role_id,
        ml_first_choice=match.first_choice,
        reasons=tuple(reasons),
    )


def review(case: AgentCase, tools: SearchTools, reviewer: Reviewer) -> AgentResult:
    """Run the loop for one case and map the final number to a role."""
    transcript = AgentTranscript(case)
    while True:
        turn = reviewer.next_turn(transcript)
        if isinstance(turn, ReviewAnswer):
            role = None
            if turn.answer == ReviewKind.ROLE:
                role = role_for_number(turn.number, transcript.numbered_roles)
            return AgentResult(turn, role.role_id if role else None, transcript)
        broken = _broken_protocol(turn, transcript)
        if broken:
            invalid = ReviewAnswer(ReviewKind.INVALID, None, "", case.llm_answer.source, broken)
            return AgentResult(invalid, None, transcript)
        transcript = AgentTranscript(case, (*transcript.steps, tools.search(turn, transcript)))


def render_case(case: AgentCase) -> str:
    """The case as a model is shown it: the record block, the shown roles with
    the LLM's pick and the ML first choice marked, and the referral reasons."""
    lines = [render_record(case.record), "", "Roles:"]
    for shown in case.roles:
        line = [render_roles([shown])]
        if shown.role.role_id == case.llm_role_id:
            line.append(LLM_PICK_MARK)
        if shown.role.role_id == case.ml_first_choice:
            line.append(ML_FIRST_CHOICE_MARK)
        lines.append(" ".join(line))
    lines += ["", "Referred because: " + "; ".join(case.reasons)]
    return "\n".join(lines)


def render_step(step: SearchStep) -> str:
    """One search's results as a model is shown them (Q10): catalog hits show
    role name, family code and definition; approved-title hits show title,
    family code and role name. No similarity score."""
    if step.request.tool == SearchTool.APPROVED_TITLES:
        lines = [_approved_title_line(hit) for hit in step.hits]
    else:
        lines = [render_roles([hit.role]) for hit in step.hits]
    return "\n".join(lines) or NO_RESULTS


def _approved_title_line(hit: SearchHit) -> str:
    role = hit.role.role
    _refuse_generic(role)
    mark = "-" if hit.role.number is None else f"{hit.role.number}."
    title, family = hit.approved_title, hit.title_family_code or NO_FAMILY
    return f'{mark} {role.role_name}: approved for the title "{title}" (family {family})'


def _shown(role: CatalogRole, record: Record, numbers: dict[str, int]) -> NumberedRole:
    """The role as the agent is shown it: with its number in the case's running
    list when it is of the recorded family, with none otherwise (Q10).

    numbers holds the role_id of every role numbered so far and gains the
    role when it is new.
    """
    _refuse_generic(role)
    family = record.family_code.strip()
    if not family or role.family_code != family:
        return NumberedRole(None, role)
    if role.role_id not in numbers:
        numbers[role.role_id] = max(numbers.values(), default=0) + 1
    return NumberedRole(numbers[role.role_id], role)


def _broken_protocol(turn: object, transcript: AgentTranscript) -> str:
    """Why a turn that is no final answer cannot be run as a search; empty when it can (Q20)."""
    if not isinstance(turn, SearchRequest):
        return f"a turn must be one search request or the final answer, not {type(turn).__name__}"
    if turn.tool not in tuple(SearchTool) or not isinstance(turn.query, str):
        return "a search request must name one of the two searches and give its query as text"
    if transcript.searches_left == 0:
        return f"a search was requested after the {AGENT_MAX_SEARCHES} a case may run"
    return ""


def _refuse_generic(role: CatalogRole) -> None:
    if role.is_generic:
        raise ValueError(
            f"generic family role {role.role_id} must never be shown to a model (design 1.2)"
        )
