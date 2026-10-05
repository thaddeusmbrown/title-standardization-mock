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

Decided here, where the record is silent:
- Family codes. A code that is asked about is trimmed of surrounding
  whitespace (S05) and then compared exactly, so "fin" is not "FIN". The
  catalog's own codes are taken as recorded, as the cross-file checks of
  titlestd.datafiles take them. A blank code is never a family (Q26).
  is_usable_family, roles_in_family and generic_role all read a code this way.
- generic_role raises CatalogError unless the family has exactly one generic
  role. That covers every code that is not a usable family: with no family
  there is no generic role to take (Q26), so the caller checks usability first.
- A grade HR's table does not list is no more usable than a missing one: no
  level, and the title with no level word (2.4). A grade whose level is not on
  the ladder is a CatalogError, not a title with a guessed word; the readers
  refuse such a catalog.
- A generic family role is named like any other role: "Senior Sales".
"""

from __future__ import annotations

from collections.abc import Iterable

from titlestd.contracts import Catalog, CatalogRole, LadderLevel, StandardTitle, TitlestdError

_NOT_BUILT = "titlestd.catalog is built by feat/catalog-naming"


class CatalogError(TitlestdError):
    """The catalog cannot answer: a family without exactly one generic role, or
    a grade whose level is not on the ladder."""


def _by_role_id(roles: Iterable[CatalogRole]) -> tuple[CatalogRole, ...]:
    return tuple(sorted(roles, key=lambda role: role.role_id))


def _family_roles(catalog: Catalog, family_code: str) -> list[CatalogRole]:
    """Every role of the family, generic included; none for a blank or unknown code."""
    family = family_code.strip()
    if not family:
        return []
    return [role for role in catalog.roles if role.family_code == family]


def _name(role: CatalogRole, rung: LadderLevel | None) -> StandardTitle:
    """The naming rule (1.3): the role at one level, or with no level word."""
    if rung is None:
        return StandardTitle(None, role.role_name, role.short_name)
    return StandardTitle(
        rung.level, f"{rung.long_word} {role.role_name}", f"{rung.short_word} {role.short_name}"
    )


def family_codes(catalog: Catalog) -> tuple[str, ...]:
    """The catalog's family codes, sorted."""
    return tuple(sorted({role.family_code for role in catalog.roles}))


def is_usable_family(catalog: Catalog, family_code: str) -> bool:
    """True when the code, after trimming, is one of the catalog's families (Q26)."""
    return bool(_family_roles(catalog, family_code))


def get_role(catalog: Catalog, role_id: str) -> CatalogRole | None:
    """The role with this id, generic or not, or None."""
    return next((role for role in catalog.roles if role.role_id == role_id), None)


def model_visible_roles(catalog: Catalog) -> tuple[CatalogRole, ...]:
    """Every role a model may be shown: all roles except the generic family roles.

    This is the only source of roles for embedding, shortlists, numbered lists
    and searches (1.2).
    """
    return _by_role_id(role for role in catalog.roles if not role.is_generic)


def roles_in_family(catalog: Catalog, family_code: str) -> tuple[CatalogRole, ...]:
    """The non-generic roles of one family."""
    return _by_role_id(role for role in _family_roles(catalog, family_code) if not role.is_generic)


def generic_role(catalog: Catalog, family_code: str) -> CatalogRole:
    """The family's one generic role. Only the close assigns it (1.8)."""
    found = [role for role in _family_roles(catalog, family_code) if role.is_generic]
    if len(found) != 1:
        raise CatalogError(
            f"family {family_code.strip()!r} has {len(found)} generic roles in the catalog, "
            "expected exactly one"
        )
    return found[0]


def level_for_grade(catalog: Catalog, grade: int | None) -> str | None:
    """The level HR's table gives the grade; None for a missing or unlisted grade."""
    return next((row.level for row in catalog.grade_levels if row.grade == grade), None)


def standard_title(catalog: Catalog, role: CatalogRole, grade: int | None) -> StandardTitle:
    """The role's Long and Short Title at the grade's level.

    Grades 7 and 8 on Sales x Account Manager both give "Senior Account
    Manager" and "Sr Acct Mgr" (1.3). With no usable grade the level is None
    and both titles carry no level word (2.4).
    """
    level = level_for_grade(catalog, grade)
    if level is None:
        return _name(role, None)
    for rung in catalog.ladder:
        if rung.level == level:
            return _name(role, rung)
    raise CatalogError(f"grade {grade} names level {level}, which is not on the level ladder")


def is_catalog_title(catalog: Catalog, long_title: str, short_title: str) -> bool:
    """True when the naming rule can produce this pair from some role and level,
    or from some role with no level word. The release guard uses it."""
    raise NotImplementedError(_NOT_BUILT)
