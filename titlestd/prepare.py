"""Prepare: clean titles, mask names, group to one row per title and family.

Owner: feat/prepare (stage 1).
Governed by: docs/design.md 1.4 (nothing about the person; the unit of
decision), 1.6 step 1, 2.4 (missing data); docs/build-decisions.md Q09 (raw
titles in the seed and truth files), Q11 (one record per row), Q14 (cleaning
and masking), Q26 (no usable family code).

Cleaning a title, in this order (Q14): lower-case; remove requisition numbers;
strip punctuation; remove trailing location tokens; expand abbreviations as
whole tokens; collapse spaces; then mask names. Names, locations and
requisition patterns come from titlestd.reference. Descriptions and
departments are not cleaned, only masked: emails to contracts.EMAIL_MASK
first, then names to contracts.NAME_MASK.

The record of a row (Q11) is built here, once, and every later step reads only
that. The result must not depend on the order of the employees file.

This module must not call titlestd.catalog, which is built in parallel: the
caller passes the catalog's family codes.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from titlestd.contracts import Abbreviation, Employee, TitleKey, TitleRow

_NOT_BUILT = "titlestd.prepare is built by feat/prepare"


@dataclass(frozen=True, slots=True)
class CleanedTitle:
    """A title after cleaning and masking, and whether it held a person's name."""

    text: str
    has_person_name: bool


def clean_title(job_title: str, abbreviations: Sequence[Abbreviation]) -> CleanedTitle:
    """Clean and mask one job title.

    "Sr. Acct Mgr - NYC" gives "senior account manager". An adjacent given name
    and surname become one NAME_MASK; a lone listed name token is masked too.
    """
    raise NotImplementedError(_NOT_BUILT)


def mask_text(text: str) -> str:
    """Mask emails, then names, in a description or department. Nothing else changes."""
    raise NotImplementedError(_NOT_BUILT)


def title_key(
    job_title: str, job_family_code: str, abbreviations: Sequence[Abbreviation]
) -> TitleKey:
    """The decision key of a raw title in a family, as prepare() would key it.

    The look-up uses it to clean the seed file on load (Q09). The family code
    is trimmed of surrounding whitespace and otherwise kept as recorded (Q26).
    """
    raise NotImplementedError(_NOT_BUILT)


def prepare(
    employees: Sequence[Employee],
    abbreviations: Sequence[Abbreviation],
    family_codes: Collection[str],
) -> tuple[TitleRow, ...]:
    """Group the employees into title rows, sorted by key.

    For each row (Q11): the description is the most common non-empty masked
    description among its employees, ties going to the one held by the lowest
    employee_id, cut to its first contracts.DESCRIPTION_WORDS words after
    masking; the department is chosen the same way. has_person_name is set when
    the title held a name (Q14). family_usable is False when the family code is
    blank after trimming or not in family_codes (Q26).
    """
    raise NotImplementedError(_NOT_BUILT)
