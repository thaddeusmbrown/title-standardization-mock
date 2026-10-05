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
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from titlestd.contracts import CatalogRole, NumberedRole, Record, TitleKey

_NOT_BUILT = "titlestd.fallback is built by feat/llm-fallback-rules"


def render_record(record: Record) -> str:
    """The record block: the masked record exactly as a model is shown it."""
    raise NotImplementedError(_NOT_BUILT)


def number_roles(
    roles: Iterable[CatalogRole], key: TitleKey, seed: int
) -> tuple[NumberedRole, ...]:
    """Shuffle and number the shortlisted roles for one title (Q18).

    Sort by role_id, permute with seeding.rng(seed, STREAM_SHUFFLE, *key), then
    number from 1. The order depends only on the seed and the title. A generic
    family role in roles is an error.
    """
    raise NotImplementedError(_NOT_BUILT)


def render_roles(roles: Sequence[NumberedRole]) -> str:
    """The numbered list as a model is shown it: number, role name, family
    code and definition. A role with no number is shown unnumbered."""
    raise NotImplementedError(_NOT_BUILT)


def role_for_number(number: int | None, roles: Sequence[NumberedRole]) -> CatalogRole | None:
    """The number-to-role mapping. None when the number names no shown role;
    the rules then fail the catalog check (Q15)."""
    raise NotImplementedError(_NOT_BUILT)


def quote_in_record(quote: str, record_block: str) -> bool:
    """The quote check (Q15).

    Case-fold and collapse whitespace on both sides; the quote must then be a
    substring of the record block that starts and ends on a word boundary, has
    at least contracts.QUOTE_MIN_CHARS characters including a letter, and at
    most contracts.QUOTE_MAX_WORDS words. An empty quote fails.
    """
    raise NotImplementedError(_NOT_BUILT)
