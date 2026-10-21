"""Prepare: clean titles, mask names, group to one row per title and family.

Owner: feat/prepare (stage 1).
Governed by: DESIGN.md 1.4 (nothing about the person; the unit of
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

This module does not call titlestd.catalog: the caller passes the catalog's
family codes.

Decided here, where the record leaves the detail open:

- Punctuation is every character that is not a letter or a digit, the
  underscore included. It separates tokens, so "acct/mgr" is two tokens. A
  cleaned title is lower-case tokens of letters and digits joined by single
  spaces, plus NAME_MASK where a name stood.
- Location tokens are removed from the end only, as many as stand there
  ("- NYC / Remote"). A location anywhere else in the title stays.
- An abbreviation is matched as one whole token, without regard to case, and
  expanded once: the words it expands to are not expanded again.
- A title made only of removable parts cleans to the empty string. It is
  still a title row, keyed on the empty title.
- A name is a token of the given-name or surname list, in any case, with no
  letter or digit directly beside it. A given name followed by a surname
  across white space is one NAME_MASK. Every other listed token is a
  NAME_MASK of its own, so "surname, given name" gives two.
- An email is a run of letters, digits and "._+-", an "@", and a domain of
  letters, digits, dots and hyphens that ends in a letter or digit, so the
  full stop that ends a sentence stays.
- Titles are masked for names only, as the order above says. An email
  address in a job title is not recognised: the punctuation step breaks it
  into tokens, and those that are listed names are masked.
- A description or department of white space only counts as empty. The
  chosen text is otherwise kept as recorded: a word is a run of characters
  that are not white space, and a longer description ends with its
  DESCRIPTION_WORDS-th word. Texts are counted after masking, so two that
  differ only in a name count as one.
- "The lowest employee_id" is the lowest as text, the order in which a row
  lists its employee_ids.
- A family code is usable when, trimmed, it is exactly one of the codes the
  caller passes; "fin" is not "FIN".
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

from titlestd import reference
from titlestd.contracts import (
    DESCRIPTION_WORDS,
    EMAIL_MASK,
    NAME_MASK,
    Abbreviation,
    Employee,
    Record,
    TitleKey,
    TitleRow,
)

_REQUISITION = re.compile("|".join(reference.REQUISITION_PATTERNS), re.IGNORECASE)
_TOKEN = re.compile(r"[^\W_]+")  # a run of letters and digits
_LOCATIONS = frozenset(reference.LOCATIONS)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]*\w")
_GIVEN = "|".join(reference.GIVEN_NAMES)
_SURNAME = "|".join(reference.SURNAMES)
_NAME = re.compile(
    rf"(?<![^\W_])(?:(?:{_GIVEN})\s+(?:{_SURNAME})|{_GIVEN}|{_SURNAME})(?![^\W_])",
    re.IGNORECASE,
)
_WORD = re.compile(r"\S+")


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
    return _clean(job_title, _expansions(abbreviations))


def mask_text(text: str) -> str:
    """Mask emails, then names, in a description or department. Nothing else changes."""
    return _NAME.sub(NAME_MASK, _EMAIL.sub(EMAIL_MASK, text))


def title_key(
    job_title: str, job_family_code: str, abbreviations: Sequence[Abbreviation]
) -> TitleKey:
    """The decision key of a raw title in a family, as prepare() would key it.

    The look-up uses it to clean the seed file on load (Q09). The family code
    is trimmed of surrounding whitespace and otherwise kept as recorded (Q26).
    """
    return TitleKey(job_family_code.strip(), clean_title(job_title, abbreviations).text)


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
    expansions = _expansions(abbreviations)
    members: dict[TitleKey, list[Employee]] = {}
    named: set[TitleKey] = set()
    for employee in employees:
        title = _clean(employee.job_title, expansions)
        key = TitleKey(employee.job_family_code.strip(), title.text)
        members.setdefault(key, []).append(employee)
        if title.has_person_name:
            named.add(key)
    return tuple(
        _title_row(key, members[key], key in named, family_codes) for key in sorted(members)
    )


def _title_row(
    key: TitleKey, members: Sequence[Employee], has_person_name: bool, family_codes: Collection[str]
) -> TitleRow:
    """One row and its record. Sorting the members first is what makes the
    result independent of the order of the employees file."""
    members = sorted(members, key=lambda employee: employee.employee_id)
    description = _most_common(mask_text(employee.job_description) for employee in members)
    return TitleRow(
        record=Record(
            cleaned_title=key.cleaned_title,
            family_code=key.family_code,
            department=_most_common(mask_text(employee.department) for employee in members),
            description=_first_words(description, DESCRIPTION_WORDS),
        ),
        employee_ids=tuple(employee.employee_id for employee in members),
        raw_titles=tuple(sorted({employee.job_title for employee in members})),
        has_person_name=has_person_name,
        family_usable=bool(key.family_code) and key.family_code in family_codes,
    )


def _most_common(texts: Iterable[str]) -> str:
    """The most common text that is not blank, a tie going to the one met
    first; the empty string when every text is blank."""
    counts = Counter(text for text in texts if text.strip())
    # max() keeps the first of equal counts, and a Counter keeps the order met.
    return max(counts, key=counts.__getitem__, default="")


def _first_words(text: str, limit: int) -> str:
    """The text up to the end of its limit-th word, or all of it."""
    words = list(_WORD.finditer(text))
    return text if len(words) <= limit else text[: words[limit - 1].end()]


def _expansions(abbreviations: Sequence[Abbreviation]) -> dict[str, list[str]]:
    """Each short form, lower-case, with the tokens of its long form."""
    return {
        abbreviation.short_form.strip().lower(): _TOKEN.findall(abbreviation.long_form.lower())
        for abbreviation in abbreviations
    }


def _clean(job_title: str, expansions: Mapping[str, Sequence[str]]) -> CleanedTitle:
    """The steps of S06, in order, then the name mask."""
    text = _REQUISITION.sub(" ", job_title.lower())
    tokens = _TOKEN.findall(text)
    while tokens and tokens[-1] in _LOCATIONS:
        tokens.pop()
    words = [word for token in tokens for word in expansions.get(token, (token,))]
    masked, names = _NAME.subn(NAME_MASK, " ".join(words))
    return CleanedTitle(text=masked, has_person_name=names > 0)
