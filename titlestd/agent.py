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
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import (
    AgentCase,
    AgentResult,
    AgentTranscript,
    Catalog,
    ChooseAnswer,
    Embedder,
    MatchResult,
    NumberedRole,
    Reason,
    Reviewer,
    SearchRequest,
    SearchStep,
    TitleRow,
)
from titlestd.lookup import Knowledge
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.agent is built by feat/agent"


class SearchTools:
    """The two searches for one wave, over the catalog and the approved titles."""

    def __init__(
        self, catalog: Catalog, knowledge: Knowledge, embedder: Embedder, settings: Settings
    ) -> None:
        raise NotImplementedError(_NOT_BUILT)

    def search(self, request: SearchRequest, transcript: AgentTranscript) -> SearchStep:
        """Run one search and number its hits for the transcript's case."""
        raise NotImplementedError(_NOT_BUILT)


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
    raise NotImplementedError(_NOT_BUILT)


def review(case: AgentCase, tools: SearchTools, reviewer: Reviewer) -> AgentResult:
    """Run the loop for one case and map the final number to a role."""
    raise NotImplementedError(_NOT_BUILT)


def render_case(case: AgentCase) -> str:
    """The case as a model is shown it: the record block, the shown roles with
    the LLM's pick and the ML first choice marked, and the referral reasons."""
    raise NotImplementedError(_NOT_BUILT)


def render_step(step: SearchStep) -> str:
    """One search's results as a model is shown them (Q10): catalog hits show
    role name, family code and definition; approved-title hits show title,
    family code and role name. No similarity score."""
    raise NotImplementedError(_NOT_BUILT)
