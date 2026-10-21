"""The close: leftovers take their family's generic role, under the cap.

Owner: feat/close-release-files (stage 2).
Governed by: DESIGN.md 1.8 (last two points), 1.10 (Wave 3);
docs/build-decisions.md Q01 (where leftovers come from), Q03 (no suitable role
is never a generic role), Q25 (the cap and the fill order), Q26 (no family, no
generic role).
Section 2.7 test: the generic family role is never assigned past its cap, or
to a Wave 1 or compliance-flagged title.

The cap (Q25) is floor(settings.generic_cap_share x employees in the employees
file), one cap for the whole run. Whole titles only. A leftover is eligible
when it has a usable family code, is not on the Wave 1 list, was scored by a
model (no person's name in the title) and the compliance check of 1.7 never
fired on any proposal or on the ML top three. Eligible leftovers are taken in
ascending headcount, ties by family code then cleaned title, while the running
total plus the title's headcount stays at or under the cap; the first that
does not fit ends the fill. Every other leftover stays on its legacy title.

Decided here, where the record is silent:
- The cap is worked out from the share as it is written, not from its binary
  float: 1.8% of 1,500 employees is 27, where float arithmetic gives
  26.999... and so 26. The share is read through its shortest decimal form.
- Which reason a leftover carries when several bar it. In this order: no
  suitable role (Q03: it waits for the catalog owner whatever else is true of
  it); widely held or flagged; a person's name or no usable family, which S11
  keeps for the leftovers that are not widely held or flagged; over the cap,
  which only an eligible leftover can be.
- A usable family is asked for twice: the row's own flag, set by the prepare
  step, and the catalog. A leftover either one refuses stays on its legacy
  title, so the generic role is only ever looked up for a family the catalog
  holds (Q26). A catalog whose family has no generic role, or two, stops the
  close with the catalog's own error instead of leaving the title on legacy.
- A generic decision names nobody and has no simulated mark and no error
  rate: code assigned it, and its Provisional status is not a path's (1.9).
- Two leftovers with one key are a ValueError: a title row has one decision
  (S11), and its headcount counts against the cap once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

from titlestd.catalog import generic_role, is_usable_family
from titlestd.contracts import (
    Catalog,
    Decision,
    DecisionSource,
    ExceptionReason,
    Leftover,
    Status,
    TableRow,
    TitleKey,
)
from titlestd.settings import Settings


def generic_cap(employee_count: int, settings: Settings) -> int:
    """The most employees that may hold a generic family role: 30 of 1,500 at 2%."""
    return math.floor(Fraction(str(settings.generic_cap_share)) * employee_count)


def close_leftovers(
    leftovers: Sequence[Leftover], employee_count: int, catalog: Catalog, settings: Settings
) -> tuple[Decision, ...]:
    """One decision per leftover, sorted by key.

    A title given the generic role: row GENERIC, source CODE, status
    Provisional. Every other: row LEGACY, source NONE, with the
    ExceptionReason that applies (no suitable role, undecided and barred,
    undecided and never scored, or over the cap).
    """
    decisions: dict[TitleKey, Decision] = {}
    eligible = []
    for leftover in leftovers:
        key = leftover.row.key
        if key in decisions:
            raise ValueError(
                f"title {key.cleaned_title!r} in family {key.family_code!r} is a leftover twice"
            )
        reason = _barred(leftover, catalog)
        if reason is None:
            eligible.append(leftover)
            reason = ExceptionReason.OVER_GENERIC_CAP  # unless the fill below takes it
        decisions[key] = _legacy(key, reason)

    cap = generic_cap(employee_count, settings)
    used = 0
    for leftover in sorted(eligible, key=lambda item: (item.row.headcount, item.row.key)):
        row = leftover.row
        if used + row.headcount > cap:
            break  # every later title holds at least as many, so the fill ends here (Q25)
        used += row.headcount
        decisions[row.key] = Decision(
            key=row.key,
            row=TableRow.GENERIC,
            source=DecisionSource.CODE,
            role_id=generic_role(catalog, row.key.family_code).role_id,
            status=Status.PROVISIONAL,
        )
    return tuple(decisions[key] for key in sorted(decisions))


def _barred(leftover: Leftover, catalog: Catalog) -> ExceptionReason | None:
    """Why the leftover may not take a generic role at all, or None when it may."""
    row = leftover.row
    if leftover.no_suitable_role:
        return ExceptionReason.NO_SUITABLE_ROLE
    if leftover.on_wave1_list or leftover.compliance_fired:
        return ExceptionReason.UNDECIDED_BARRED
    if not row.model_eligible or not is_usable_family(catalog, row.key.family_code):
        return ExceptionReason.UNDECIDED_UNSCORED
    return None


def _legacy(key: TitleKey, reason: ExceptionReason) -> Decision:
    return Decision(key=key, row=TableRow.LEGACY, source=DecisionSource.NONE, legacy_reason=reason)
