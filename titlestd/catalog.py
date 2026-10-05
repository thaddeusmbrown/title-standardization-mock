"""Catalog access and the naming rule.

Owner: feat/catalog-naming (stage 1).
Governed by: docs/design.md 1.2 (generic family role), 1.3 (naming rule), 2.4
(missing grade); docs/build-decisions.md Q05 (the title with no level word),
Q26 (usable family code), Q36 (the worked example).
Section 2.7 tests: no title outside the catalog can reach the load file
(is_catalog_title); no model is ever shown a generic family role
(model_visible_roles).

Every function takes the Catalog read by titlestd.datafiles and is pure. The
catalog is read-only during a run (Q03). Results are ordered by role_id, never
by set order.

Naming rule (1.3): Long Title is the level's long word plus the role name,
Short Title is the level's short word plus the short role name, joined by one
space. With no usable grade both are the name alone.
"""

from __future__ import annotations

from titlestd.contracts import Catalog, CatalogRole, StandardTitle

_NOT_BUILT = "titlestd.catalog is built by feat/catalog-naming"


def family_codes(catalog: Catalog) -> tuple[str, ...]:
    """The catalog's family codes, sorted."""
    raise NotImplementedError(_NOT_BUILT)


def is_usable_family(catalog: Catalog, family_code: str) -> bool:
    """True when the code, after trimming, is one of the catalog's families (Q26)."""
    raise NotImplementedError(_NOT_BUILT)


def get_role(catalog: Catalog, role_id: str) -> CatalogRole | None:
    """The role with this id, generic or not, or None."""
    raise NotImplementedError(_NOT_BUILT)


def model_visible_roles(catalog: Catalog) -> tuple[CatalogRole, ...]:
    """Every role a model may be shown: all roles except the generic family roles.

    This is the only source of roles for embedding, shortlists, numbered lists
    and searches (1.2).
    """
    raise NotImplementedError(_NOT_BUILT)


def roles_in_family(catalog: Catalog, family_code: str) -> tuple[CatalogRole, ...]:
    """The non-generic roles of one family."""
    raise NotImplementedError(_NOT_BUILT)


def generic_role(catalog: Catalog, family_code: str) -> CatalogRole:
    """The family's one generic role. Only the close assigns it (1.8)."""
    raise NotImplementedError(_NOT_BUILT)


def level_for_grade(catalog: Catalog, grade: int | None) -> str | None:
    """The level HR's table gives the grade; None for a missing or unlisted grade."""
    raise NotImplementedError(_NOT_BUILT)


def standard_title(catalog: Catalog, role: CatalogRole, grade: int | None) -> StandardTitle:
    """The role's Long and Short Title at the grade's level.

    Grades 7 and 8 on Sales x Account Manager both give "Senior Account
    Manager" and "Sr Acct Mgr" (1.3). With no usable grade the level is None
    and both titles carry no level word (2.4).
    """
    raise NotImplementedError(_NOT_BUILT)


def is_catalog_title(catalog: Catalog, long_title: str, short_title: str) -> bool:
    """True when the naming rule can produce this pair from some role and level,
    or from some role with no level word. The release guard uses it."""
    raise NotImplementedError(_NOT_BUILT)
