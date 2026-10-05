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

Decided here, where Q17 and Q19 leave the detail open:
- Words. A word is a run of word characters as the re module knows them:
  letters of any script, digits and the underscore. A quote therefore starts
  and ends where a word does, and a piece of a longer word is never taken
  for a word. An abbreviation is expanded when a whole word equals its short
  form, once; a long form of several words gives several content words. The
  masks [NAME] and [EMAIL] are not words.
  STOP_WORDS is a short list of English function words, with the "s" an
  apostrophe leaves behind ("customer's"); it leaves out "it" and "us",
  which job titles use as nouns. The level numerals are 1 to 5 and i to v,
  one for each of the design's five levels.
- "No description" means record.description is empty or blank.
- A role's words are its name's and its definition's content words. A word
  of the record matches a role when it shares a content word with them. The
  agent's "shares a content word with the title" is the same test on the
  title, so the agent's quote comes from the title whenever the matching
  title words are long enough to quote.
- The quote for a role is the longest run of consecutive title words that
  all match it, the first such run on a tie, copied from the first to the
  last of its words with whatever stands between them. With no such run in
  the title it is the longest in the description, and with none there the
  whole title. Any other word ends a run, a stop word included. A run is cut
  to its first contracts.QUOTE_MAX_WORDS words, and a run under
  contracts.QUOTE_MIN_CHARS characters or without a letter is passed over
  for the next longest, so the stand-in does not offer a quote the shared
  check must refuse while another matching run would pass.
- For "no match" and "insufficient data" the quote is the whole title as it
  is. A title of one or two characters, or an empty one, then fails the
  shared quote check and goes to a person: that is the record's doing, not a
  fault injected here.
- Only roles that carry a number are candidates, for the LLM and the agent.
  With none, the LLM answers "no match" and the agent escalates.
- The agent's third search asks with the first 12 words of the description
  as written: the pieces between whitespace, with their punctuation, joined
  by single spaces. That is not the word of the first point: a query is
  text for the search to read, so "end-to-end" is one word of the twelve.
- The agent's bonus is one point, once per role, for a role that any hit of
  an approved-title search maps to. With a single candidate there is no
  runner-up to beat. Its turn depends only on the transcript: how many
  searches have run and what they returned.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from titlestd.contracts import (
    EMAIL_MASK,
    LABEL_STAND_IN,
    NAME_MASK,
    QUOTE_MAX_WORDS,
    QUOTE_MIN_CHARS,
    Abbreviation,
    AgentTranscript,
    ChooseAnswer,
    ChooseKind,
    LadderLevel,
    Llm,
    NumberedRole,
    Record,
    ReviewAnswer,
    ReviewKind,
    SearchRequest,
    SearchTool,
)
from titlestd.settings import Settings

# English function words, never content, and the "s" of "customer's".
# "it" and "us" are left out on purpose.
STOP_WORDS = frozenset(
    {
        "a",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "each",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "its",
        "of",
        "on",
        "or",
        "other",
        "our",
        "s",
        "such",
        "that",
        "the",
        "their",
        "them",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "which",
        "who",
        "will",
        "with",
    }
)
# A level written as a number, as in "ACCT MGR II" or "Analyst 2".
LEVEL_NUMERALS = frozenset({"1", "2", "3", "4", "5", "i", "ii", "iii", "iv", "v"})
# Q19: the third search asks with the first words of the description, split at
# whitespace.
DESCRIPTION_QUERY_WORDS = 12
# Q19: worth one shared word.
APPROVED_TITLE_BONUS = 1

_MASKS = (NAME_MASK, EMAIL_MASK)
_WORD = re.compile("|".join([*map(re.escape, _MASKS), r"\w+"]))


@dataclass(frozen=True, slots=True)
class _Word:
    """One written word of a text: where it stands and its content words."""

    start: int
    end: int
    content: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Scored:
    """A candidate role with its overlap score (Q17)."""

    numbered: NumberedRole
    score: int
    name_hits: int  # |T and N|
    words: frozenset[str]  # N or F


def _tokens(text: str) -> list[str]:
    """The lower-cased words of a reference text: a long form or a level word."""
    return [written.lower() for written in _WORD.findall(text) if written not in _MASKS]


def _content(words: Iterable[_Word]) -> frozenset[str]:
    return frozenset().union(*(word.content for word in words))


def _quotable(span: str) -> bool:
    """True when the shared quote check can accept the span's size (Q15)."""
    return (
        len(span) >= QUOTE_MIN_CHARS
        and any(character.isalpha() for character in span)
        and len(span.split()) <= QUOTE_MAX_WORDS
    )


def _longest_run(text: str, words: Sequence[_Word], role_words: frozenset[str]) -> str:
    """The longest run of consecutive words that all match the role, verbatim.

    Empty when the text has no quotable run.
    """
    best, best_length = "", 0
    run: list[_Word] = []
    for word in [*words, None]:
        if word is not None and word.content & role_words:
            run.append(word)
            continue
        kept = run[:QUOTE_MAX_WORDS]
        if len(kept) > best_length:
            span = text[kept[0].start : kept[-1].end]
            if _quotable(span):
                best, best_length = span, len(kept)
        run = []
    return best


class StandIn:
    """The stand-in chooser and reviewer: word overlap, nothing kept between calls."""

    def __init__(
        self,
        settings: Settings,
        ladder: Sequence[LadderLevel],
        abbreviations: Sequence[Abbreviation],
    ) -> None:
        self._min_title_words = settings.standin_min_title_words
        self._no_match_below = settings.standin_no_match_below
        self._expansions = {
            abbreviation.short_form.lower(): tuple(_tokens(abbreviation.long_form))
            for abbreviation in abbreviations
        }
        level_words = {
            token
            for level in ladder
            for word in (level.long_word, level.short_word)
            for token in _tokens(word)
        }
        self._ignored = STOP_WORDS | LEVEL_NUMERALS | level_words

    # --- words ----------------------------------------------------------------

    def _words(self, text: str) -> list[_Word]:
        words = []
        for match in _WORD.finditer(text):
            written = match.group()
            content: frozenset[str] = frozenset()
            if written not in _MASKS:
                token = written.lower()
                content = frozenset(self._expansions.get(token, (token,))) - self._ignored
            words.append(_Word(match.start(), match.end(), content))
        return words

    def content_words(self, text: str) -> frozenset[str]:
        """The content words of a text (Q17)."""
        return _content(self._words(text))

    # --- the overlap score (Q17) ------------------------------------------------

    def _rank(
        self,
        record: Record,
        roles: Iterable[NumberedRole],
        bonus_role_ids: Collection[str] = (),
    ) -> list[_Scored]:
        """The numbered roles, best first: by score, then |T and N|, then role_id."""
        title = self.content_words(record.cleaned_title)
        record_words = title | self.content_words(record.description)
        scored = []
        for numbered in roles:
            if numbered.number is None:
                continue
            name = self.content_words(numbered.role.role_name)
            role_words = name | self.content_words(numbered.role.definition)
            name_hits = len(title & name)
            score = 2 * name_hits + len(record_words & role_words)
            if numbered.role.role_id in bonus_role_ids:
                score += APPROVED_TITLE_BONUS
            scored.append(_Scored(numbered, score, name_hits, role_words))
        return sorted(scored, key=lambda s: (-s.score, -s.name_hits, s.numbered.role.role_id))

    def _quote(self, record: Record, role_words: frozenset[str]) -> str:
        for text in (record.cleaned_title, record.description):
            run = _longest_run(text, self._words(text), role_words)
            if run:
                return run
        return record.cleaned_title

    # --- choose (Q17) -------------------------------------------------------------

    def choose(self, record: Record, roles: Sequence[NumberedRole]) -> ChooseAnswer:
        title = self.content_words(record.cleaned_title)
        if not record.description.strip() and len(title) < self._min_title_words:
            return ChooseAnswer(
                ChooseKind.INSUFFICIENT_DATA, None, record.cleaned_title, LABEL_STAND_IN
            )
        ranked = self._rank(record, roles)
        if not ranked or ranked[0].score < self._no_match_below:
            return ChooseAnswer(ChooseKind.NO_MATCH, None, record.cleaned_title, LABEL_STAND_IN)
        best = ranked[0]
        return ChooseAnswer(
            ChooseKind.ROLE, best.numbered.number, self._quote(record, best.words), LABEL_STAND_IN
        )

    # --- review, one turn at a time (Q19, Q20) -------------------------------------

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        record = transcript.case.record
        searches = len(transcript.steps)
        if searches == 0:
            return SearchRequest(SearchTool.CATALOG, record.cleaned_title)
        if searches == 1:
            return SearchRequest(SearchTool.APPROVED_TITLES, record.cleaned_title)
        if searches == 2 and len(transcript.numbered_roles) < 2 and record.description.strip():
            first_words = record.description.split()[:DESCRIPTION_QUERY_WORDS]
            return SearchRequest(SearchTool.CATALOG, " ".join(first_words))
        return self._final_answer(transcript)

    def _final_answer(self, transcript: AgentTranscript) -> ReviewAnswer:
        record = transcript.case.record
        approved = {
            hit.role.role.role_id
            for step in transcript.steps
            if step.request.tool == SearchTool.APPROVED_TITLES
            for hit in step.hits
        }
        ranked = self._rank(record, transcript.numbered_roles, approved)
        if ranked:
            best = ranked[0]
            shares_title_word = bool(best.words & self.content_words(record.cleaned_title))
            beats_runner_up = len(ranked) == 1 or best.score > ranked[1].score
            if shares_title_word and beats_runner_up:
                return ReviewAnswer(
                    ReviewKind.ROLE,
                    best.numbered.number,
                    self._quote(record, best.words),
                    LABEL_STAND_IN,
                )
        return ReviewAnswer(ReviewKind.ESCALATE, None, "", LABEL_STAND_IN)


def build_llm(
    settings: Settings, ladder: Sequence[LadderLevel], abbreviations: Sequence[Abbreviation]
) -> Llm:
    """The stand-in chooser and reviewer. finish does nothing."""
    standin = StandIn(settings, ladder, abbreviations)
    return Llm(chooser=standin, reviewer=standin)
