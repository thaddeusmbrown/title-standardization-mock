"""The ML match: nearest neighbours on text embeddings.

Owner: feat/lookup-ml-match (stage 2).
Governed by: docs/design.md 1.4 (what is embedded), 1.5, 1.6 step 3, 2.4
("Role score", "Missing data"); docs/build-decisions.md Q04 (the ML-alone
route), Q12 (example vectors), Q18 (stable ties).
Section 2.7 test: no model is ever shown a generic family role (never embedded,
never on a shortlist).

The query text is the cleaned title plus the record's description (already cut
to 200 words), or the title alone when there is none. A role's score is the
best cosine similarity across its catalog text (name and one-line definition)
and its approved titles. The shortlist holds contracts.SHORTLIST_SIZE roles,
best first, ties by role_id. Only titlestd.catalog.model_visible_roles are
embedded or listed.

A row that is not model_eligible (a person's name, no usable family code) is
never passed to embed (Q14, Q26).
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import (
    Catalog,
    CatalogRole,
    Embedder,
    MatchResult,
    Record,
    TitleKey,
    TitleRow,
    Vector,
)
from titlestd.lookup import Knowledge
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.matcher is built by feat/lookup-ml-match"


def role_text(role: CatalogRole) -> str:
    """The catalog text embedded for a role: its name and one-line definition."""
    raise NotImplementedError(_NOT_BUILT)


def embed_text(record: Record) -> str:
    """The text embedded for a title row, and for the record of an approved
    example, so an example's vector is the one its title was queried with (Q12)."""
    raise NotImplementedError(_NOT_BUILT)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two L2-normalised vectors: a sparse dot product."""
    raise NotImplementedError(_NOT_BUILT)


class Matcher:
    """Matches title rows against the catalog and the approved titles."""

    def __init__(self, catalog: Catalog, embedder: Embedder) -> None:
        raise NotImplementedError(_NOT_BUILT)

    def match(self, rows: Sequence[TitleRow], knowledge: Knowledge) -> dict[TitleKey, MatchResult]:
        """Score every row against the examples as they stand now.

        A whole wave is scored in one call, before that wave's decisions are
        learned (Q18, Q22). Rows that are not model_eligible are left out.
        """
        raise NotImplementedError(_NOT_BUILT)


def ml_alone_eligible(
    match: MatchResult, family_code: str, catalog: Catalog, settings: Settings
) -> bool:
    """True when the ML first choice may skip the LLM (Q04): settings.ml_alone
    is on, the margin is at least settings.ml_margin_cutoff, the first choice
    has at least settings.ml_min_examples approved examples and is in the
    recorded family. The title still goes to a person."""
    raise NotImplementedError(_NOT_BUILT)
