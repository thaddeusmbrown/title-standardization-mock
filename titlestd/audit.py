"""The random audit of a machine path, its three outcomes and the statuses.

Owner: feat/golden-audit-status (stage 2).
Governed by: docs/design.md 1.9, 2.4 ("Audit at mock scale", "Wrong");
docs/build-decisions.md Q07 (golden titles stay in the sampling frame), Q16
(what the path row shows), Q21 (audited titles take the SME's label), Q23 (the
three outcomes, the path of 60 or fewer and the release setting are tested on
constructed counts).

The numbers (2.4), for a sample of n titles:
- confirmed if wrong is at most the largest count that a path at exactly
  contracts.AUDIT_ACCURACY_BAR right would pass at most
  contracts.AUDIT_PASS_CHANCE of the time, from the binomial: 2 of 60, 14 of 200;
- stopped above contracts.AUDIT_STOP_SHARE wrong: more than 12 of 60, 41 or
  more of 200;
- otherwise provisional, with its measured error rate.
A path of contracts.AUDIT_SAMPLE_SIZE titles or fewer is labeled in full: each
title takes its label with status Approved, and the path gets no status.

Statuses (Q21): every audited title takes the SME's label as its role, with
status Approved, whether or not the label matches the machine's role; wrong is
still counted against the machine's role. The outcome applies only to the
unaudited rest: Confirmed, Provisional with the error rate, or, for a stopped
path, left on legacy titles unless settings.release_stopped_paths releases
them as Provisional. Level is not scored.

The sample draws each title once, with seeding.rng(seed, STREAM_AUDIT,
path.value) over the path's sorted keys.

This module reads no file and never the truth (Q22): the caller hands in the
SME's blind labels. It knows nothing of the golden set, so a golden title is
drawn and decided like any other title of its path (Q07, Q21).

Decided on this branch, where the record says nothing:
- The lines are computed exactly. The three shares are read as the decimals
  the design states (0.90 is nine tenths, not the nearest binary float) and
  the binomial is summed in fractions, so a count never falls on the other
  side of a line through rounding.
- The labels handed in must be the blind labels of the sample: one for every
  title of a path of AUDIT_SAMPLE_SIZE or fewer, AUDIT_SAMPLE_SIZE of them
  otherwise, each for a title of the path. Anything else is a ValueError, as
  is a label "could not decide", which a blind label never is (Q01).
- An audited title the SME marked "no suitable role" (Q03; only a title with
  no truth row) counts as wrong: the machine named a role and the SME says
  none suits.
- A path with no title is "all labeled", with nothing audited.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from fractions import Fraction

from titlestd import seeding
from titlestd.contracts import (
    AUDIT_ACCURACY_BAR,
    AUDIT_PASS_CHANCE,
    AUDIT_SAMPLE_SIZE,
    AUDIT_STOP_SHARE,
    AuditOutcome,
    AuditResult,
    Decision,
    HumanAnswer,
    HumanDecision,
    MachinePath,
    TitleKey,
)
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.audit is built by feat/golden-audit-status"


def _exact(share: float) -> Fraction:
    """A share as the decimal the design states: 0.90 gives 9/10."""
    return Fraction(str(share))


def confirm_limit(sample_size: int) -> int:
    """The most wrong answers a confirmed path may have: 2 for 60, 14 for 200.

    Raises ValueError for a sample so small that no count passes (under 22).
    """
    right = _exact(AUDIT_ACCURACY_BAR)  # the chance that one title is right, at the bar
    pass_chance = _exact(AUDIT_PASS_CHANCE)
    limit = -1
    chance = Fraction(0)  # that a path exactly at the bar has at most this many wrong
    for wrong in range(sample_size + 1):
        chance += (
            math.comb(sample_size, wrong) * (1 - right) ** wrong * right ** (sample_size - wrong)
        )
        if chance > pass_chance:
            break
        limit = wrong
    if limit < 0:
        raise ValueError(
            f"a sample of {sample_size} titles cannot confirm a path: with none wrong, a path "
            f"exactly at the bar would still pass more than {AUDIT_PASS_CHANCE:.0%} of the time"
        )
    return limit


def stop_limit(sample_size: int) -> int:
    """The most wrong answers a path may have without being stopped: 12 for 60,
    40 for 200."""
    return math.floor(_exact(AUDIT_STOP_SHARE) * sample_size)


def draw_sample(path: MachinePath, keys: Collection[TitleKey], seed: int) -> tuple[TitleKey, ...]:
    """The titles to label, sorted: every key when the path holds
    contracts.AUDIT_SAMPLE_SIZE or fewer, otherwise a seeded sample of that size."""
    ordered = sorted(set(keys))
    if len(ordered) <= AUDIT_SAMPLE_SIZE:
        return tuple(ordered)
    generator = seeding.rng(seed, seeding.STREAM_AUDIT, path.value)
    return tuple(sorted(generator.sample(ordered, AUDIT_SAMPLE_SIZE)))


def audit_path(
    path: MachinePath,
    machine_roles: Mapping[TitleKey, str],
    labels: Mapping[TitleKey, HumanDecision],
) -> AuditResult:
    """Compare the labels with the machine's roles and decide the outcome.

    machine_roles holds every title of the path; labels holds the SME's blind
    label for each title drawn by draw_sample. Raises ValueError when the
    labels cannot be those of the path's sample.
    """
    _check_labels(machine_roles, labels)
    audited_keys = tuple(sorted(labels))
    # "No suitable role" has no role, so it differs from whatever the machine chose.
    wrong = sum(machine_roles[key] != labels[key].role_id for key in audited_keys)
    return AuditResult(
        path=path,
        path_size=len(machine_roles),
        audited_keys=audited_keys,
        wrong=wrong,
        outcome=_outcome(len(machine_roles), len(audited_keys), wrong),
    )


def path_decisions(
    result: AuditResult,
    machine_roles: Mapping[TitleKey, str],
    labels: Mapping[TitleKey, HumanDecision],
    settings: Settings,
) -> tuple[Decision, ...]:
    """The final decision for every title of an audited path, sorted by key.

    An audited title is counted in the path's row with source PERSON and status
    Approved. An unaudited title of a stopped path that is not released is
    counted in the legacy row with ExceptionReason.STOPPED_PATH (Q16).
    """
    raise NotImplementedError(_NOT_BUILT)


def _check_labels(
    machine_roles: Mapping[TitleKey, str], labels: Mapping[TitleKey, HumanDecision]
) -> None:
    """Refuse labels that cannot be the blind labels of the path's sample."""
    outside = sorted(set(labels) - set(machine_roles))
    if outside:
        raise ValueError(f"a label for a title that is not in the path: {tuple(outside[0])!r}")
    expected = min(len(machine_roles), AUDIT_SAMPLE_SIZE)
    if len(labels) != expected:
        raise ValueError(
            f"the audit of a path of {len(machine_roles)} titles labels {expected} of them, "
            f"got {len(labels)} labels"
        )
    for key in sorted(labels):
        if labels[key].answer is HumanAnswer.UNDECIDED:
            raise ValueError(
                f"a blind label is never {HumanAnswer.UNDECIDED.value!r}: {tuple(key)!r}"
            )


def _outcome(path_size: int, audited: int, wrong: int) -> AuditOutcome:
    """The outcome of one path's audit (2.4)."""
    if path_size <= AUDIT_SAMPLE_SIZE:
        return AuditOutcome.ALL_LABELED
    if wrong <= confirm_limit(audited):
        return AuditOutcome.CONFIRMED
    if wrong > stop_limit(audited):
        return AuditOutcome.STOPPED
    return AuditOutcome.PROVISIONAL
