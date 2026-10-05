"""llm = offline: the deterministic stand-ins for the LLM and the agent.

Owner: feat/model-switch-offline (stage 1).
Governed by: docs/design.md 2.4 ("Stand-in models never read the truth file";
word overlap after abbreviation expansion), 2.5; docs/build-decisions.md Q15
(answer shape and the quote), Q17 (the stand-in LLM), Q19 (the stand-in
agent's script), Q20 (one turn at a time), Q40 (label "stand-in").

Stand-in LLM (Q17): content words are lower-cased tokens after abbreviation
expansion, minus a fixed stop-word list and the level words (ladder long and
short words and level numerals). It answers "insufficient data" when the row
has no description and the title has fewer than settings.standin_min_title_words
content words; scores each listed role as 2 x |T and N| + |(T or D) and (N or F)|;
ranks by score, then |T and N|, then lowest role_id, never by position in the
list; answers "no match" below settings.standin_no_match_below.

Stand-in agent (Q19): search_catalog(cleaned title), then
search_approved_titles(cleaned title), then, only if fewer than two in-family
roles have been shown and the record has a description, search_catalog(first
12 words of the description). Never a fourth search. It answers with the top
numbered role if that shares a content word with the title and strictly beats
the runner-up; otherwise it escalates.

Quotes are copied character for character from record.cleaned_title or from a
run of consecutive words of record.description, which titlestd.fallback
renders verbatim in the record block. There is no fault injection: a failed
quote is tested with an injected answer, not produced here.

Every answer carries contracts.LABEL_STAND_IN. This module must not import
titlestd.people or titlestd.catalog.
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import Abbreviation, LadderLevel, Llm
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.standin is built by feat/model-switch-offline"


def build_llm(
    settings: Settings, ladder: Sequence[LadderLevel], abbreviations: Sequence[Abbreviation]
) -> Llm:
    """The stand-in chooser and reviewer. finish does nothing."""
    raise NotImplementedError(_NOT_BUILT)
