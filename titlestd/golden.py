"""The golden set: its draw and its scoring.

Owner: feat/golden-audit-status (stage 2).
Governed by: DESIGN.md 1.2 (golden set), 1.10 (drawn in set-up, labeled
in Wave 1), 2.4 ("Wrong"), 2.6 (footer); docs/build-decisions.md Q07 (golden
titles are released like any long-tail title; their labels are never learned),
Q08 (size, frame, the three scorings), Q14 and Q26 (a title with a person's
name or no usable family code is out of the frame).
Section 2.7 test: golden-set titles never enter the look-up table or the
examples (with the guard in titlestd.lookup).

The draw (Q08) takes settings.golden_size title rows from the frame: rows that
are not on the Wave 1 list, miss the seed look-up, have a usable family code
and carry no person's name. It depends only on the seed and the title keys:
sort the frame, then sample with seeding.rng(seed, STREAM_GOLDEN).

Scoring (Q08): the denominator is every golden title. "no match" and
"insufficient data" count as wrong and are also counted on their own. Wrong
means the machine's role differs from the simulated SME's label (2.4).

This module reads no file and never the truth (Q22): the caller hands in the
SME's labels. It does not call titlestd.lookup either: the caller passes the
keys of the seed look-up.

Decided on this branch, where the record says nothing:
- The frame is a set of keys. A key that two rows share counts once, and a
  Wave 1 or look-up key that no row has changes nothing.
- Only golden titles are scored: the keys of labels. Any other key in the two
  mappings of machine roles is passed over, so the caller may hand in the
  answers of a whole wave.
- "No answer" counts every golden title the LLM gave no role. That is "no
  match" and "insufficient data" (Q08), and also whatever else left the title
  without a role: an answer that is not usable, a number that maps to no role
  shown (S10, Q15), or no call at all. A role id or None cannot tell them
  apart, and each is wrong in the same way.
- A golden title the SME gave no role, which is "no suitable role" (Q03) and
  happens only when the truth has no row for it, has the label None. It stays
  in the denominator and no model is right about it, not even one that named
  no role either.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from titlestd import seeding
from titlestd.contracts import GoldenScore, Stage, TitleKey, TitleRow
from titlestd.settings import Settings


def draw_golden(
    rows: Sequence[TitleRow],
    wave1_keys: Collection[TitleKey],
    lookup_keys: Collection[TitleKey],
    settings: Settings,
) -> tuple[TitleKey, ...]:
    """The golden set, sorted. Fewer than settings.golden_size when the frame is smaller."""
    excluded = {*wave1_keys, *lookup_keys}
    frame = sorted(
        {
            row.key
            for row in rows
            if row.family_usable and not row.has_person_name and row.key not in excluded
        }
    )
    size = min(settings.golden_size, len(frame))
    return tuple(sorted(seeding.rng(settings.seed, seeding.STREAM_GOLDEN).sample(frame, size)))


def score_golden(
    stage: Stage,
    labels: Mapping[TitleKey, str | None],
    ml_first_choices: Mapping[TitleKey, str | None],
    llm_picks: Mapping[TitleKey, str | None],
) -> GoldenScore:
    """Score the golden set at one stage.

    labels maps each golden title to the SME's role, or to None where the SME
    named none. A missing or None entry in the other two mappings is a title
    with no role from that model.
    """
    return GoldenScore(
        stage=stage,
        size=len(labels),
        ml_right=sum(_right(ml_first_choices.get(key), label) for key, label in labels.items()),
        llm_right=sum(_right(llm_picks.get(key), label) for key, label in labels.items()),
        llm_no_answer=sum(llm_picks.get(key) is None for key in labels),
    )


def _right(role_id: str | None, label: str | None) -> bool:
    """True when the model named a role and it is the SME's label (2.4)."""
    return role_id is not None and role_id == label
