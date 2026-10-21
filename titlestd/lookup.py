"""The look-up table and the ML model's examples, with the golden-set guard.

Owner: feat/lookup-ml-match (stage 2).
Governed by: DESIGN.md 1.2 (look-up table, golden set), 1.6 steps 2 and
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

Decided here, where the record is silent:
- The look-up is exact on the key as prepare built it: nothing is trimmed or
  case-folded here, so the family code "FIN " is not the key's "FIN" (S05
  leaves the trimming to whoever builds the key).
- One role per key. A later decision on a title replaces the earlier one, in
  the look-up table and in the examples together, so the two never disagree:
  an example always carries the role the look-up gives for its key. When the
  later row carries a person's name, the earlier example is dropped and none
  takes its place (Q14). Within a run neither happens: a title is decided
  once, and the rows prepare builds for one key carry the same flag, because
  the name mask is part of the key.
- learn has no catalog and does not check that the role exists. It is
  handed a person's decision or a seed row, and both are checked where they
  are made (titlestd.people, titlestd.datafiles). The matcher lists only the
  roles a model may be shown, so an example of any other role reaches no
  shortlist.
- A decision that names no role ("could not decide", "no suitable role") is
  nothing to learn. learn raises ValueError for an empty role_id, golden
  title or not: an entry with no role would read as a miss ever after.
- A decision on a title with a person's name is recorded: learn returns
  True and the title is in the look-up table. It is no example, so it does
  not count toward the three examples of 1.6 step 4.
- A decision on a title with no usable family code joins the look-up table
  under its key, blank or unknown code included, and the examples of the
  chosen role (Q26). From then on its record is embedded as an example.
- seed_keys returns each key once, sorted.
- load_seed stops on a seed file the rest of the data contradicts, with a
  DataError naming the title: a seed row that cleans to no title row of the
  employees file (Q09 has the data hold none, and an example needs the
  row's record), or two seed rows that clean to one key and name different
  roles (the result would otherwise depend on the order of the file).
- load_seed raises ValueError when a seed row's key is in the golden set.
  The guard refuses it like any other decision, and a seed title that
  silently stopped being a look-up hit would be worse than a stop: the
  golden set is drawn from titles that miss the seed look-up (Q08), so
  this is a mistake of the caller, not of the data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from titlestd import prepare
from titlestd.contracts import Abbreviation, Example, SeedLookupRow, TitleKey, TitleRow
from titlestd.datafiles import SEED_LOOKUP_FILE, DataError


class Knowledge:
    """What people have decided so far: the look-up table and the examples."""

    def __init__(self, golden_keys: Iterable[TitleKey] = ()) -> None:
        self._golden = frozenset(golden_keys)
        self._roles: dict[TitleKey, str] = {}
        self._examples: dict[TitleKey, Example] = {}

    def lookup(self, key: TitleKey) -> str | None:
        """The role a person decided for this exact cleaned title and family, or None."""
        return self._roles.get(key)

    def learn(self, row: TitleRow, role_id: str) -> bool:
        """Record a person's decision. Returns False, and records nothing, for a
        golden-set title. A title with a person's name joins the look-up table
        only, never the examples (Q14). A decision with no role is a ValueError."""
        key = row.key
        if not role_id:
            raise ValueError(
                f"no role to learn for {key.cleaned_title!r} in family {key.family_code!r}"
            )
        if key in self._golden:
            return False
        self._roles[key] = role_id
        if row.has_person_name:
            # An example left from an earlier decision would name the old role.
            self._examples.pop(key, None)
        else:
            self._examples[key] = Example(row.record, role_id)
        return True

    def entries(self) -> tuple[tuple[TitleKey, str], ...]:
        """The look-up table, sorted by key."""
        return tuple(sorted(self._roles.items()))

    def examples(self) -> tuple[Example, ...]:
        """The examples, sorted by key."""
        return tuple(self._examples[key] for key in sorted(self._examples))

    def example_count(self, role_id: str) -> int:
        """How many human-approved examples the role has (the three-example rule)."""
        return sum(example.role_id == role_id for example in self._examples.values())


def seed_keys(
    seed_rows: Sequence[SeedLookupRow], abbreviations: Sequence[Abbreviation]
) -> tuple[TitleKey, ...]:
    """The decision keys of the seed look-up rows, cleaned with the shared
    prepare function. The golden-set draw excludes them (Q08)."""
    return tuple(sorted({_seed_key(seed, abbreviations) for seed in seed_rows}))


def load_seed(
    knowledge: Knowledge,
    seed_rows: Sequence[SeedLookupRow],
    rows: Sequence[TitleRow],
    abbreviations: Sequence[Abbreviation],
) -> None:
    """Learn every seed look-up row, as a look-up entry and as an example (Q13)."""
    by_key = {row.key: row for row in rows}
    seeded: dict[TitleKey, str] = {}
    for seed in seed_rows:
        key = _seed_key(seed, abbreviations)
        where = f"{SEED_LOOKUP_FILE}: title {seed.job_title!r} in family {seed.job_family_code!r}"
        if key not in by_key:
            raise DataError(
                f"{where} cleans to {key.cleaned_title!r}, which no employee holds in that family"
            )
        known = seeded.setdefault(key, seed.role_id)
        if known != seed.role_id:
            raise DataError(
                f"{where} names role {seed.role_id}, but another spelling of the same "
                f"title names {known}"
            )
        if not knowledge.learn(by_key[key], seed.role_id):
            raise ValueError(
                f"{where} is in the golden set, which is drawn from titles that miss "
                "the seed look-up (Q08)"
            )


def _seed_key(seed: SeedLookupRow, abbreviations: Sequence[Abbreviation]) -> TitleKey:
    """The seed file holds the raw title and the family code as recorded (Q09)."""
    return prepare.title_key(seed.job_title, seed.job_family_code, abbreviations)
