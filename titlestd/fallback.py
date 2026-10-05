"""Shared code around choose: the record block, the numbered roles, the quote check.

Owner: feat/llm-fallback-rules (stage 2).
Governed by: docs/design.md 1.6 step 4, 1.7 (the quote check), 2.5 ("Masking,
numbering, the quote check, the search cap and the number-to-role mapping stay
in shared code, so no model output can become a role by itself");
docs/build-decisions.md Q11 (the record), Q15 (answer shape, the quote check),
Q18 (the seeded shuffle).
Section 2.7 tests: a quote that is not in the record sends the title to a
person; no model is ever shown a generic family role (never in a numbered
list).

The record block is rendered once per title, and the same string serves every
prompt and the quote check. It shows the record's cleaned title, family code,
department and description, each field verbatim on a labelled line of its own;
the stand-ins rely on that to quote (Q15, Q17). No role_id and no similarity
score appears in a rendered text.

These functions are used by every llm setting, so a prompt is the same string
whoever answers it; a recording's key depends on that (Q18, Q33).

Decided here, where the record leaves the detail open:

- The record block is four lines, "Title: ", "Family: ", "Department: " and
  "Description: ", each followed by the field exactly as the record holds it.
  An empty field leaves its label alone on the line, with nothing written in
  its place, so the block never holds a word that the record does not. The
  labels are part of the block that Q15 checks a quote against, so a quote
  may run across them ("Family: SAL").
- The numbered list is one line per role: "3. Account Manager (family SAL):
  the definition". A role with no number starts with "-" instead.
- A generic family role is refused with ValueError, by number_roles and again
  by render_roles: a list that holds one is a fault in the calling code, and
  render_roles is the last thing a role passes before a model reads it.
- A number names a role only when it is a whole number (an int, and not a
  bool) carried by a role in the list. A role with no number is never the
  answer to anything, None included.
- The quote check measures the quote after case-folding and collapsing white
  space: its characters, and its words as the pieces between spaces. White
  space is whatever str.split() takes for it. A word character is one the re
  module's \\w matches (a letter of any script, a digit or the underscore),
  the same as in titlestd.standin. A quote starts and ends on a word boundary
  when neither end falls between two word characters, so "count" is not a
  quote from "account", while a quote may begin or end with punctuation that
  stands in the record. Any place in the block where the quote stands so is
  enough.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from titlestd import seeding
from titlestd.contracts import (
    QUOTE_MAX_WORDS,
    QUOTE_MIN_CHARS,
    CatalogRole,
    NumberedRole,
    Record,
    TitleKey,
)

# Matches where a quote may start or end: anywhere but between two word characters.
_WORD_BOUNDARY = r"(?:(?<!\w)|(?!\w))"


def render_record(record: Record) -> str:
    """The record block: the masked record exactly as a model is shown it."""
    fields = (
        ("Title", record.cleaned_title),
        ("Family", record.family_code),
        ("Department", record.department),
        ("Description", record.description),
    )
    return "\n".join(f"{label}: {value}" if value else f"{label}:" for label, value in fields)


def number_roles(
    roles: Iterable[CatalogRole], key: TitleKey, seed: int
) -> tuple[NumberedRole, ...]:
    """Shuffle and number the shortlisted roles for one title (Q18).

    Sort by role_id, permute with seeding.rng(seed, STREAM_SHUFFLE, *key), then
    number from 1. The order depends only on the seed and the title. A generic
    family role in roles is an error.
    """
    ordered = sorted(roles, key=lambda role: role.role_id)
    for role in ordered:
        _refuse_generic(role)
    seeding.rng(seed, seeding.STREAM_SHUFFLE, *key).shuffle(ordered)
    return tuple(NumberedRole(number, role) for number, role in enumerate(ordered, start=1))


def render_roles(roles: Sequence[NumberedRole]) -> str:
    """The numbered list as a model is shown it: number, role name, family
    code and definition. A role with no number is shown unnumbered."""
    lines = []
    for shown in roles:
        role = shown.role
        _refuse_generic(role)
        mark = "-" if shown.number is None else f"{shown.number}."
        lines.append(f"{mark} {role.role_name} (family {role.family_code}): {role.definition}")
    return "\n".join(lines)


def role_for_number(number: int | None, roles: Sequence[NumberedRole]) -> CatalogRole | None:
    """The number-to-role mapping. None when the number names no shown role;
    the rules then fail the catalog check (Q15)."""
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return next((shown.role for shown in roles if shown.number == number), None)


def quote_in_record(quote: str, record_block: str) -> bool:
    """The quote check (Q15).

    Case-fold and collapse whitespace on both sides; the quote must then be a
    substring of the record block that starts and ends on a word boundary, has
    at least contracts.QUOTE_MIN_CHARS characters including a letter, and at
    most contracts.QUOTE_MAX_WORDS words. An empty quote fails.
    """
    quote, block = _folded(quote), _folded(record_block)
    if len(quote) < QUOTE_MIN_CHARS or not any(character.isalpha() for character in quote):
        return False
    if len(quote.split(" ")) > QUOTE_MAX_WORDS:
        return False
    return re.search(_WORD_BOUNDARY + re.escape(quote) + _WORD_BOUNDARY, block) is not None


def _folded(text: str) -> str:
    """Case-folded, with every run of white space as one space and none at the ends."""
    return " ".join(text.casefold().split())


def _refuse_generic(role: CatalogRole) -> None:
    if role.is_generic:
        raise ValueError(
            f"generic family role {role.role_id} must never be shown to a model (design 1.2)"
        )
