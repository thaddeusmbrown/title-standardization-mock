"""Simulated people, and the only code that reads the truth file.

Owner: feat/simulated-people (stage 1). The truth-file functions below were
built by the scaffold; the owner built SimulatedPeople and load_people around
them.
Governed by: docs/design.md 1.8, 2.3 (the truth file), 2.4 ("People are
simulated from the truth file, with a small seeded error rate, and every
simulated decision is marked as simulated"); docs/build-decisions.md Q01 (could
not decide), Q03 (no suitable role), Q09 (truth is keyed on the raw title),
Q22 (who may read the truth), Q27 (error rate, personas, one label per title).

Truth isolation is structural (Q22), checked by tests/unit/test_truth_isolation.py:
- the truth file's name and its reader and writer exist only in this module;
- only titlestd.pipeline and the data generator titlestd.mockdata may import
  this module; no model module may.

A label is a pure function of (seed, family code, cleaned title), so the same
title gets the same label on every occasion in a run and whatever the order of
calls (Q27). The truth is looked up by a row's raw titles and its family code,
so this module needs no cleaner. It calls neither titlestd.prepare nor
titlestd.catalog: it reads Catalog.roles directly.

Family codes are compared after removing surrounding whitespace on both sides
(S05): a row's key holds the trimmed code, while the truth file holds the code
as recorded, which may be padded or made of spaces only. The mini data set has
one padded code to show it.

Decided on this branch, where the record says nothing:
- Which SME. A Wave 1 approval or a blind label names one of the three SMEs,
  picked by a keyed draw of the title on the sme_error stream. So a title
  falls to the same SME on every occasion, like its label. A review-queue
  decision is always the reviewer's.
- A row's true role is the one its raw titles name. A raw title with no truth
  row is passed over, so a row is "no suitable role" only when none of its raw
  titles has one (Q03). Two different true roles for one row, which Q09 rules
  out in the data, are a DataError: when the people are built if the truth
  alone shows it (one raw title under family codes that differ only in
  padding), otherwise when the row is first asked about.
- What that asks of a truth changed on purpose, as test (b) of Q22 permutes
  true_role_id: every raw spelling of a title row keeps one role, and every
  role stays in its family. One mapping of role ids per family, applied to
  every truth row, does both; so does giving each title row, with all its
  spellings, another role of its family. Moving the roles from truth row to
  truth row does not: such a truth still loads, since each role is a real
  one of its family, and the run then stops at the first decision on a row
  with two spellings.
- "No suitable role" comes before "could not decide". A review-queue title
  with no truth row is always marked, whatever the undecided rate, so it can
  never become a leftover that takes a generic role (Q01, Q03).
- A wrong label needs another non-generic role in the true role's family.
  Where the family has none, the label is the true role.
- The people check the truth against the catalog when they are built
  (validate_truth), not only in load_people, so no way of building them can
  hand out an unknown or a generic role.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from titlestd import datafiles, seeding
from titlestd.contracts import Catalog, HumanAnswer, HumanDecision, TitleRow
from titlestd.settings import Settings

TRUTH_FILE = "truth.csv"
TRUTH_COLUMNS = ("job_title", "job_family_code", "true_role_id")

# Personas are role labels, not invented human names (Q27).
PERSONA_SMES = ("SME 1 (simulated)", "SME 2 (simulated)", "SME 3 (simulated)")
PERSONA_REVIEWER = "Reviewer (simulated)"
PERSONA_CATALOG_OWNER = "Catalog owner (simulated)"
PERSONA_HR_LEAD = "HR lead (simulated)"
PERSONA_CIO = "CIO (simulated)"


@dataclass(frozen=True, slots=True)
class TruthRow:
    """The one true role of a title within a family, keyed as recorded (Q09)."""

    job_title: str
    job_family_code: str
    true_role_id: str


# --- The truth file (built) ---------------------------------------------------


def read_truth(data_dir: Path) -> tuple[TruthRow, ...]:
    """Read the truth file. One row per distinct raw title and family code."""
    rows, seen = [], set()
    for number, row in enumerate(
        datafiles.read_table(Path(data_dir) / TRUTH_FILE, TRUTH_COLUMNS), start=1
    ):
        truth = TruthRow(row["job_title"], row["job_family_code"], row["true_role_id"])
        if not truth.job_title.strip() or not truth.true_role_id.strip():
            raise datafiles.DataError(
                f"{TRUTH_FILE} row {number}: job_title or true_role_id is empty"
            )
        pair = (truth.job_title, truth.job_family_code)
        if pair in seen:
            raise datafiles.DataError(
                f"{TRUTH_FILE} row {number}: title {truth.job_title!r} in family "
                f"{truth.job_family_code!r} appears more than once"
            )
        seen.add(pair)
        rows.append(truth)
    return tuple(rows)


def write_truth(data_dir: Path, rows: Iterable[TruthRow]) -> None:
    """Write the truth file. For the data generator and for tests."""
    datafiles.write_table(
        Path(data_dir) / TRUTH_FILE,
        TRUTH_COLUMNS,
        (
            {
                "job_title": row.job_title,
                "job_family_code": row.job_family_code,
                "true_role_id": row.true_role_id,
            }
            for row in rows
        ),
    )


def validate_truth(rows: Sequence[TruthRow], catalog: Catalog) -> None:
    """Check every true role exists, is not generic and, where the family code
    is one of the catalog's, is in that family (Q09). Raises DataError."""
    datafiles.check_decided_roles(
        TRUTH_FILE,
        catalog,
        ((row.job_title, row.job_family_code, row.true_role_id) for row in rows),
    )


def truth_fingerprint(data_dir: Path) -> str:
    """Fingerprint of the truth file's bytes, for a recording's manifest.

    Simulated decisions feed the examples, so the truth shapes later prompts
    (Q18). The pipeline combines this with datafiles.input_fingerprint.
    """
    return datafiles.fingerprint_files([Path(data_dir) / TRUTH_FILE])


# --- Simulated people ---------------------------------------------------------


class SimulatedPeople:
    """HR SMEs, reviewers and the catalog owner, simulated from the truth.

    Every decision has simulated = True and names a persona. A wrong decision,
    made at settings.sme_error_rate, is a different non-generic role in the
    true role's family (Q27). For a title with no truth row the persona who
    was asked, an SME or the reviewer, returns HumanAnswer.NO_SUITABLE_ROLE
    (Q03). The catalog owner does not decide it: PERSONA_CATALOG_OWNER is whom
    the title then waits for, which the review-queue file shows through the
    reason Reason.NO_SUITABLE_ROLE. A person never returns a generic family
    role.
    """

    def __init__(self, truth: Iterable[TruthRow], catalog: Catalog, settings: Settings) -> None:
        # Read once and kept: the truth is gone through twice below, and an
        # iterator used up by the check would leave the people with no truth.
        truth = tuple(truth)
        validate_truth(truth, catalog)
        self._seed = settings.seed
        self._error_rate = settings.sme_error_rate
        self._undecided_rate = settings.undecided_rate
        # Every role a person may name, with its family: the generic ones are left out.
        self._family_of = {
            role.role_id: role.family_code for role in catalog.roles if not role.is_generic
        }
        # (raw title, trimmed family code) to the true role (Q09, S05).
        self._truth: dict[tuple[str, str], str] = {}
        for row in truth:
            pair = (row.job_title, row.job_family_code.strip())
            known = self._truth.setdefault(pair, row.true_role_id)
            if known != row.true_role_id:
                raise datafiles.DataError(
                    f"{TRUTH_FILE}: title {pair[0]!r} in family {pair[1]!r} has two true "
                    f"roles, {known} and {row.true_role_id}"
                )

    def wave1_approval(self, row: TitleRow) -> HumanDecision:
        """An HR SME's decision on a Wave 1 title. Never undecided (Q01)."""
        sme, role_id = self._label(row)
        return _decision(row, role_id, sme)

    def blind_label(self, row: TitleRow) -> HumanDecision:
        """An HR SME's label, given without seeing the system's answer: the
        golden set, the random audit and a fully labeled small path. Never
        undecided (Q01)."""
        sme, role_id = self._label(row)
        return _decision(row, role_id, sme)

    def queue_decision(self, row: TitleRow) -> HumanDecision:
        """A reviewer's decision on a review-queue title at the close.

        HumanAnswer.NO_SUITABLE_ROLE for a title with no truth row, whatever
        the rate (Q03). Otherwise HumanAnswer.UNDECIDED when
        seeding.unit(seed, STREAM_UNDECIDED, *row.key) is below
        settings.undecided_rate (Q01), and the title's label when it is not.
        """
        _, role_id = self._label(row)
        draw = seeding.unit(self._seed, seeding.STREAM_UNDECIDED, *row.key)
        if role_id is not None and draw < self._undecided_rate:
            return HumanDecision(row.key, HumanAnswer.UNDECIDED, None, PERSONA_REVIEWER, True)
        return _decision(row, role_id, PERSONA_REVIEWER)

    def _label(self, row: TitleRow) -> tuple[str, str | None]:
        """The SME a title falls to and the title's one label: a role, or None
        when the title has no truth row.

        Both come from the title's own draws on the sme_error stream, so they
        are the same on every occasion and whatever was asked before (Q27).
        """
        generator = seeding.rng(self._seed, seeding.STREAM_SME_ERROR, *row.key)
        sme = generator.choice(PERSONA_SMES)
        role_id = self._true_role(row)
        draw = seeding.unit(self._seed, seeding.STREAM_SME_ERROR, *row.key)
        if role_id is not None and draw < self._error_rate:
            family = self._family_of[role_id]
            others = sorted(
                other
                for other, its_family in self._family_of.items()
                if its_family == family and other != role_id
            )
            if others:
                role_id = generator.choice(others)
        return sme, role_id

    def _true_role(self, row: TitleRow) -> str | None:
        """The one true role the row's raw titles name, or None when none has a truth row."""
        family = row.key.family_code.strip()
        roles = sorted(
            {self._truth[(raw, family)] for raw in row.raw_titles if (raw, family) in self._truth}
        )
        if len(roles) > 1:
            raise datafiles.DataError(
                f"{TRUTH_FILE}: the raw titles of {row.key.cleaned_title!r} in family "
                f"{family!r} have different true roles: {', '.join(roles)}"
            )
        return roles[0] if roles else None


def _decision(row: TitleRow, role_id: str | None, persona: str) -> HumanDecision:
    """The persona's decision on a title whose label is role_id."""
    if role_id is None:
        return HumanDecision(row.key, HumanAnswer.NO_SUITABLE_ROLE, None, persona, True)
    return HumanDecision(row.key, HumanAnswer.ROLE, role_id, persona, True)


def load_people(settings: Settings, catalog: Catalog) -> SimulatedPeople:
    """Read the truth file in settings.data_dir and build the people, who check
    it against the catalog."""
    return SimulatedPeople(read_truth(settings.data_dir), catalog, settings)
