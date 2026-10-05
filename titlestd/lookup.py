"""The look-up table and the ML model's examples, with the golden-set guard.

Owner: feat/lookup-ml-match (stage 2).
Governed by: docs/design.md 1.2 (look-up table, golden set), 1.6 steps 2 and
10; docs/build-decisions.md Q07 (the guard), Q09 (the seed file holds raw
titles), Q13 (a hit takes its role with no model; seed entries are examples
from the start), Q14 (a name-bearing title joins the look-up, never the
examples), Q26 (a row with no usable family code).
Section 2.7 test: golden-set titles never enter the look-up table or the
examples.

Knowledge.learn is the single function that adds a look-up entry or an
example. It refuses a golden-set key unconditionally, whatever produced the
human decision. A title the agent settled is not learned until a person
approves it (1.6 step 10); the caller decides when to call learn, and applies
a wave's decisions only after the whole wave is scored (Q18).

Held in memory for the run; titlestd.store writes it to the state file.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from titlestd.contracts import Abbreviation, Example, SeedLookupRow, TitleKey, TitleRow

_NOT_BUILT = "titlestd.lookup is built by feat/lookup-ml-match"


class Knowledge:
    """What people have decided so far: the look-up table and the examples."""

    def __init__(self, golden_keys: Iterable[TitleKey] = ()) -> None:
        raise NotImplementedError(_NOT_BUILT)

    def lookup(self, key: TitleKey) -> str | None:
        """The role a person decided for this exact cleaned title and family, or None."""
        raise NotImplementedError(_NOT_BUILT)

    def learn(self, row: TitleRow, role_id: str) -> bool:
        """Record a person's decision. Returns False, and records nothing, for a
        golden-set title. A title with a person's name joins the look-up table
        only, never the examples (Q14)."""
        raise NotImplementedError(_NOT_BUILT)

    def entries(self) -> tuple[tuple[TitleKey, str], ...]:
        """The look-up table, sorted by key."""
        raise NotImplementedError(_NOT_BUILT)

    def examples(self) -> tuple[Example, ...]:
        """The examples, sorted by key."""
        raise NotImplementedError(_NOT_BUILT)

    def example_count(self, role_id: str) -> int:
        """How many human-approved examples the role has (the three-example rule)."""
        raise NotImplementedError(_NOT_BUILT)


def seed_keys(
    seed_rows: Sequence[SeedLookupRow], abbreviations: Sequence[Abbreviation]
) -> tuple[TitleKey, ...]:
    """The decision keys of the seed look-up rows, cleaned with the shared
    prepare function. The golden-set draw excludes them (Q08)."""
    raise NotImplementedError(_NOT_BUILT)


def load_seed(
    knowledge: Knowledge,
    seed_rows: Sequence[SeedLookupRow],
    rows: Sequence[TitleRow],
    abbreviations: Sequence[Abbreviation],
) -> None:
    """Learn every seed look-up row, as a look-up entry and as an example (Q13)."""
    raise NotImplementedError(_NOT_BUILT)
