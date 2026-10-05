"""The golden set: its draw and its scoring.

Owner: feat/golden-audit-status (stage 2).
Governed by: docs/design.md 1.2 (golden set), 1.10 (drawn in set-up, labeled
in Wave 1), 2.4 ("Wrong"), 2.6 (footer); docs/build-decisions.md Q07 (golden
titles are released like any long-tail title; their labels are never learned),
Q08 (size, frame, the three scorings).
Section 2.7 test: golden-set titles never enter the look-up table or the
examples (with the guard in titlestd.lookup).

The draw (Q08) takes settings.golden_size title rows from the frame: rows that
are not on the Wave 1 list, miss the seed look-up, have a usable family code
and carry no person's name. It depends only on the seed and the title keys:
sort the frame, then sample with seeding.rng(seed, STREAM_GOLDEN).

Scoring (Q08): the denominator is every golden title. "no match" and
"insufficient data" count as wrong and are also counted on their own. Wrong
means the machine's role differs from the simulated SME's label (2.4).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from titlestd.contracts import GoldenScore, Stage, TitleKey, TitleRow
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.golden is built by feat/golden-audit-status"


def draw_golden(
    rows: Sequence[TitleRow],
    wave1_keys: Collection[TitleKey],
    lookup_keys: Collection[TitleKey],
    settings: Settings,
) -> tuple[TitleKey, ...]:
    """The golden set, sorted. Fewer than settings.golden_size when the frame is smaller."""
    raise NotImplementedError(_NOT_BUILT)


def score_golden(
    stage: Stage,
    labels: Mapping[TitleKey, str],
    ml_first_choices: Mapping[TitleKey, str | None],
    llm_picks: Mapping[TitleKey, str | None],
) -> GoldenScore:
    """Score the golden set at one stage.

    labels maps each golden title to the SME's role. A missing or None entry in
    the other two mappings is a title with no role from that model.
    """
    raise NotImplementedError(_NOT_BUILT)
