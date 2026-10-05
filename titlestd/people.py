"""Simulated people, and the only code that reads the truth file.

Owner: feat/simulated-people (stage 1). The truth-file functions below are
already built by the scaffold; the owner builds SimulatedPeople and
load_people.
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
so this module needs no cleaner and must not call titlestd.prepare or
titlestd.catalog, which are built in parallel: it reads Catalog.roles directly.

Family codes are compared after removing surrounding whitespace on both sides
(S05): a row's key holds the trimmed code, while the truth file holds the code
as recorded, which may be padded or made of spaces only. The mini data set has
one padded code to show it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from titlestd import datafiles
from titlestd.contracts import Catalog, HumanDecision, TitleRow
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.people is built by feat/simulated-people"

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


# --- Simulated people (to build) ----------------------------------------------


class SimulatedPeople:
    """HR SMEs, reviewers and the catalog owner, simulated from the truth.

    Every decision has simulated = True and names a persona. A wrong decision,
    made at settings.sme_error_rate, is a different non-generic role in the
    true role's family (Q27). A title with no truth row gets
    HumanAnswer.NO_SUITABLE_ROLE from PERSONA_CATALOG_OWNER (Q03). A person
    never returns a generic family role.
    """

    def __init__(self, truth: Sequence[TruthRow], catalog: Catalog, settings: Settings) -> None:
        raise NotImplementedError(_NOT_BUILT)

    def wave1_approval(self, row: TitleRow) -> HumanDecision:
        """An HR SME's decision on a Wave 1 title. Never undecided (Q01)."""
        raise NotImplementedError(_NOT_BUILT)

    def blind_label(self, row: TitleRow) -> HumanDecision:
        """An HR SME's label, given without seeing the system's answer: the
        golden set, the random audit and a fully labeled small path. Never
        undecided (Q01)."""
        raise NotImplementedError(_NOT_BUILT)

    def queue_decision(self, row: TitleRow) -> HumanDecision:
        """A reviewer's decision on a review-queue title at the close.

        HumanAnswer.UNDECIDED when seeding.unit(seed, STREAM_UNDECIDED, *row.key)
        is below settings.undecided_rate (Q01); otherwise the title's label.
        """
        raise NotImplementedError(_NOT_BUILT)


def load_people(settings: Settings, catalog: Catalog) -> SimulatedPeople:
    """Read and check the truth file in settings.data_dir and build the people."""
    raise NotImplementedError(_NOT_BUILT)
