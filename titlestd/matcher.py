"""The ML match: nearest neighbours on text embeddings.

Owner: feat/lookup-ml-match (stage 2).
Governed by: DESIGN.md 1.4 (what is embedded), 1.5, 1.6 step 3, 2.4
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

Decided here, where the record is silent:
- The texts. A role is embedded as "name. definition", or as its name alone
  when it has no definition. A record is embedded as its cleaned title, one
  space and its description; the title alone when there is no description
  (2.4), and the description alone in the odd case of an empty title.
  Nothing else of a record is embedded: not its family code and not its
  department, which 1.4 gives to the family rule and to the LLM, the agent
  and reviewers.
- The match is not filtered by family. A title is compared with every role
  a model may be shown and with every example, whatever family either is in
  (1.6 step 3: "every catalog role", "every human-approved title"). The
  family rule is checked afterwards (1.7), and ml_alone_eligible checks the
  family for the ML-alone route (Q04).
- matched_text, for the audit trail, is the role's catalog text when that
  scored best, otherwise the cleaned title of the approved title that did.
  On equal scores the catalog text is named before an approved title, and
  approved titles in key order. The score does not depend on it.
- An example whose role is not one a model may be shown (a generic family
  role, or a role the catalog does not hold) is neither embedded nor scored.
  No person's decision produces one; this is what keeps a generic role off
  every shortlist whatever the examples hold.
- "Never passed to embed" is about a row as a query. Once a person has
  decided a title with no usable family code it is an example of the chosen
  role (Q26), and its record is embedded as one. A title with a person's
  name is never an example (Q14), so its text reaches embed in no way.
- A row that is itself an example is scored like any other and meets its own
  text. In a run it cannot happen: such a title hits the look-up first.
- cosine sums exactly (math.fsum), so the result does not depend on the
  order of a vector's items and cosine(a, b) equals cosine(b, a) bit for bit.
- A Matcher keeps the vector of every text it has embedded, since the same
  text always gives the same vector (contracts.Embedder). So match makes one
  embed call, for the texts that are new, and none when no row is eligible.
- An embedder that does not return exactly one vector per text, as
  contracts.Embedder promises, is not worked around: match raises ValueError
  and gives no result.
- ml_alone_eligible reads the family code as titlestd.catalog does: trimmed,
  then compared exactly. A blank or unknown code, an empty shortlist and a
  generic or unknown first choice are never eligible. "Clears the cut-off"
  is "at least the cut-off".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from titlestd.catalog import model_visible_roles, roles_in_family
from titlestd.contracts import (
    SHORTLIST_SIZE,
    Catalog,
    CatalogRole,
    Embedder,
    MatchResult,
    Record,
    ScoredRole,
    TitleKey,
    TitleRow,
    Vector,
)
from titlestd.lookup import Knowledge
from titlestd.settings import Settings


def role_text(role: CatalogRole) -> str:
    """The catalog text embedded for a role: its name and one-line definition."""
    if not role.definition.strip():
        return role.role_name
    return f"{role.role_name}. {role.definition}"


def embed_text(record: Record) -> str:
    """The text embedded for a title row, and for the record of an approved
    example, so an example's vector is the one its title was queried with (Q12)."""
    return f"{record.cleaned_title} {record.description}".strip()


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two L2-normalised vectors: a sparse dot product."""
    return math.fsum(a[index] * b[index] for index in a.keys() & b.keys())


class Matcher:
    """Matches title rows against the catalog and the approved titles."""

    def __init__(self, catalog: Catalog, embedder: Embedder) -> None:
        self._embedder = embedder
        # By role_id, and never a generic family role (1.2).
        self._roles = model_visible_roles(catalog)
        self._vectors: dict[str, Vector] = {}  # text to vector, kept for the run

    def match(self, rows: Sequence[TitleRow], knowledge: Knowledge) -> dict[TitleKey, MatchResult]:
        """Score every row against the examples as they stand now.

        A whole wave is scored in one call, before that wave's decisions are
        learned (Q18, Q22). Rows that are not model_eligible are left out.
        """
        eligible = [row for row in rows if row.model_eligible]
        if not eligible:
            return {}
        # What a title is compared with: (role_id, matched text, embedded text).
        # Per role its catalog text first, then its approved titles in key order.
        examples = knowledge.examples()
        compared = []
        for role in self._roles:
            catalog_text = role_text(role)
            compared.append((role.role_id, catalog_text, catalog_text))
            compared += [
                (role.role_id, example.record.cleaned_title, embed_text(example.record))
                for example in examples
                if example.role_id == role.role_id
            ]
        queries = [embed_text(row.record) for row in eligible]
        self._embed_new([*(text for _, _, text in compared), *queries])

        results: dict[TitleKey, MatchResult] = {}
        for row, query in zip(eligible, queries, strict=True):
            vector = self._vectors[query]
            best: dict[str, ScoredRole] = {}
            for role_id, matched_text, text in compared:
                score = cosine(vector, self._vectors[text])
                if role_id not in best or score > best[role_id].score:
                    best[role_id] = ScoredRole(role_id, score, matched_text)
            ranked = sorted(best.values(), key=lambda scored: (-scored.score, scored.role_id))
            shortlist = tuple(ranked[:SHORTLIST_SIZE])
            examples_of_first = knowledge.example_count(shortlist[0].role_id) if shortlist else 0
            results[row.key] = MatchResult(shortlist, examples_of_first)
        return results

    def _embed_new(self, texts: Sequence[str]) -> None:
        """Embed the texts this matcher has not embedded yet, each once."""
        new = [text for text in dict.fromkeys(texts) if text not in self._vectors]
        if new:
            self._vectors.update(zip(new, self._embedder.embed(new), strict=True))


def ml_alone_eligible(
    match: MatchResult, family_code: str, catalog: Catalog, settings: Settings
) -> bool:
    """True when the ML first choice may skip the LLM (Q04): settings.ml_alone
    is on, the margin is at least settings.ml_margin_cutoff, the first choice
    has at least settings.ml_min_examples approved examples and is in the
    recorded family. The title still goes to a person."""
    in_family = {role.role_id for role in roles_in_family(catalog, family_code)}
    return (
        settings.ml_alone
        and match.first_choice in in_family
        and match.margin >= settings.ml_margin_cutoff
        and match.first_choice_examples >= settings.ml_min_examples
    )
