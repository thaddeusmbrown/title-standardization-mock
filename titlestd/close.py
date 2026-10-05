"""The close: leftovers take their family's generic role, under the cap.

Owner: feat/close-release-files (stage 2).
Governed by: docs/design.md 1.8 (last two points), 1.10 (Wave 3);
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
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import Catalog, Decision, Leftover
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.close is built by feat/close-release-files"


def generic_cap(employee_count: int, settings: Settings) -> int:
    """The most employees that may hold a generic family role: 30 of 1,500 at 2%."""
    raise NotImplementedError(_NOT_BUILT)


def close_leftovers(
    leftovers: Sequence[Leftover], employee_count: int, catalog: Catalog, settings: Settings
) -> tuple[Decision, ...]:
    """One decision per leftover, sorted by key.

    A title given the generic role: row GENERIC, source CODE, status
    Provisional. Every other: row LEGACY, source NONE, with the
    ExceptionReason that applies (no suitable role, undecided and barred,
    undecided and never scored, or over the cap).
    """
    raise NotImplementedError(_NOT_BUILT)
