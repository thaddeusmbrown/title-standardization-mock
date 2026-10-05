"""The random audit of a machine path, its three outcomes and the statuses.

Owner: feat/golden-audit-status (stage 2).
Governed by: docs/design.md 1.9, 2.4 ("Audit at mock scale", "Wrong");
docs/build-decisions.md Q07 (golden titles stay in the sampling frame), Q16
(what the path row shows), Q21 (audited titles take the SME's label).

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
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from titlestd.contracts import AuditResult, Decision, HumanDecision, MachinePath, TitleKey
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.audit is built by feat/golden-audit-status"


def confirm_limit(sample_size: int) -> int:
    """The most wrong answers a confirmed path may have: 2 for 60, 14 for 200.

    Raises ValueError for a sample so small that no count passes (under 22).
    """
    raise NotImplementedError(_NOT_BUILT)


def stop_limit(sample_size: int) -> int:
    """The most wrong answers a path may have without being stopped: 12 for 60,
    40 for 200."""
    raise NotImplementedError(_NOT_BUILT)


def draw_sample(path: MachinePath, keys: Collection[TitleKey], seed: int) -> tuple[TitleKey, ...]:
    """The titles to label, sorted: every key when the path holds
    contracts.AUDIT_SAMPLE_SIZE or fewer, otherwise a seeded sample of that size."""
    raise NotImplementedError(_NOT_BUILT)


def audit_path(
    path: MachinePath,
    machine_roles: Mapping[TitleKey, str],
    labels: Mapping[TitleKey, HumanDecision],
) -> AuditResult:
    """Compare the labels with the machine's roles and decide the outcome.

    machine_roles holds every title of the path; labels holds the SME's blind
    label for each title drawn by draw_sample.
    """
    raise NotImplementedError(_NOT_BUILT)


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
