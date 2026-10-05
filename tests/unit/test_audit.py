"""The random audit: its two lines, the sample and the three outcomes.

docs/design.md 1.9 and 2.4 ("Audit at mock scale", "Wrong");
docs/build-decisions.md Q23 (the three outcomes and the path of 60 or fewer,
on constructed counts), S02 (the lines for 200 are tested beside those for 60)
and S04 (the audit stream).
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from titlestd import audit, contracts, seeding
from titlestd.contracts import (
    AuditOutcome,
    AuditResult,
    HumanAnswer,
    HumanDecision,
    MachinePath,
    TitleKey,
)

SEED = 20260101
SME = "SME 2 (simulated)"
MACHINE_ROLE = "SAL-01"  # what the machine chose for every title of a made-up path
OTHER_ROLE = "SAL-02"  # what the SME says where the machine is wrong

CONFIRMED_COUNTS = (0, 1, 2)
PROVISIONAL_COUNTS = (3, 4, 7, 11, 12)
STOPPED_COUNTS = (13, 14, 30, 60)


def _keys(count: int, family: str = "SAL") -> list[TitleKey]:
    return [TitleKey(family, f"title {number:03d}") for number in range(count)]


def _label(key: TitleKey, role_id: str, persona: str = SME) -> HumanDecision:
    return HumanDecision(key, HumanAnswer.ROLE, role_id, persona, True)


def _no_suitable_role(key: TitleKey) -> HumanDecision:
    return HumanDecision(key, HumanAnswer.NO_SUITABLE_ROLE, None, SME, True)


def _audited_path(
    size: int, wrong: int, path: MachinePath = MachinePath.MODELS_AGREE
) -> tuple[dict[TitleKey, str], dict[TitleKey, HumanDecision], AuditResult]:
    """A path of size titles whose audit finds the given number wrong: the
    machine's roles, the blind labels of the sample and the audit result."""
    keys = _keys(size)
    machine = dict.fromkeys(keys, MACHINE_ROLE)
    sample = audit.draw_sample(path, keys, SEED)
    labels = {
        key: _label(key, OTHER_ROLE if position < wrong else MACHINE_ROLE)
        for position, key in enumerate(sample)
    }
    return machine, labels, audit.audit_path(path, machine, labels)


def _chance_of_at_most(wrong: int, sample_size: int, right: Fraction) -> Fraction:
    """The chance that a path which is right this often shows at most this many
    wrong in a sample: the binomial, summed exactly."""
    return sum(
        (
            math.comb(sample_size, count) * (1 - right) ** count * right ** (sample_size - count)
            for count in range(wrong + 1)
        ),
        Fraction(0),
    )


AT_THE_BAR = Fraction(9, 10)
ONE_IN_TEN = Fraction(1, 10)


# --- The two lines (1.9, 2.4, S02) ---------------------------------------------------


def test_the_lines_the_design_states():
    assert audit.confirm_limit(60) == 2  # 2.4: "2 of 60"
    assert audit.confirm_limit(200) == 14  # 1.9: "At most 14"
    assert audit.stop_limit(60) == 12  # 2.4: stopped at "more than 12 of 60"
    assert audit.stop_limit(200) == 40  # 1.9: stopped at "41 or more"


def test_the_lines_come_from_the_numbers_the_design_fixes():
    assert contracts.AUDIT_SAMPLE_SIZE == 60
    assert (contracts.AUDIT_ACCURACY_BAR, contracts.AUDIT_PASS_CHANCE) == (0.90, 0.10)
    assert contracts.AUDIT_STOP_SHARE == 0.20


def test_this_files_binomial_gives_the_arithmetic_the_record_checked():
    # "Arithmetic checked" in docs/build-decisions.md: 5.3% and 13.7% at 60,
    # 9.3% and 14.3% at 200. It is the measure the next test holds the code to.
    assert round(float(_chance_of_at_most(2, 60, AT_THE_BAR)), 3) == 0.053
    assert round(float(_chance_of_at_most(3, 60, AT_THE_BAR)), 3) == 0.137
    assert round(float(_chance_of_at_most(14, 200, AT_THE_BAR)), 3) == 0.093
    assert round(float(_chance_of_at_most(15, 200, AT_THE_BAR)), 3) == 0.143


@pytest.mark.parametrize("sample_size", [22, 23, 30, 45, 59, 60, 61, 100, 150, 200, 201, 300])
def test_the_confirm_line_is_the_largest_count_a_path_at_the_bar_passes_one_time_in_ten(
    sample_size,
):
    limit = audit.confirm_limit(sample_size)
    assert _chance_of_at_most(limit, sample_size, AT_THE_BAR) <= ONE_IN_TEN
    assert _chance_of_at_most(limit + 1, sample_size, AT_THE_BAR) > ONE_IN_TEN


def test_the_confirm_line_never_falls_as_the_sample_grows():
    limits = [audit.confirm_limit(size) for size in range(22, 301)]
    assert limits == sorted(limits)
    assert limits[0] == 0 and limits[-1] > 14


def test_the_smallest_sample_that_can_confirm_a_path_has_22_titles():
    # 0.9 to the power 22 is the first under one in ten.
    assert audit.confirm_limit(22) == 0
    assert Fraction(9, 10) ** 22 <= ONE_IN_TEN < Fraction(9, 10) ** 21


@pytest.mark.parametrize("sample_size", [21, 20, 10, 1, 0, -1])
def test_a_sample_too_small_to_confirm_any_path_is_refused(sample_size):
    with pytest.raises(ValueError, match=f"a sample of {sample_size} titles cannot confirm"):
        audit.confirm_limit(sample_size)


def test_the_stop_line_is_a_fifth_of_the_sample_rounded_down():
    for sample_size in range(1, 301):
        limit = audit.stop_limit(sample_size)
        # limit of sample_size is at most 20%; one more is above 20%.
        assert 5 * limit <= sample_size < 5 * (limit + 1), sample_size


def test_exactly_a_fifth_wrong_is_not_stopped():
    # "Stopped above 20% wrong": 12 of 60 and 40 of 200 are exactly 20%.
    for sample_size in (5, 10, 60, 65, 200, 1000, 12345):
        assert audit.stop_limit(sample_size) * 5 == sample_size


def test_the_confirm_line_is_below_the_stop_line():
    for sample_size in range(22, 301):
        assert audit.confirm_limit(sample_size) < audit.stop_limit(sample_size)


def test_a_path_that_is_95_percent_right_is_confirmed_less_than_half_the_time():
    # 2.4, "Say what the small audit can show": about 42% at 60, about 92% at 200.
    at_60 = _chance_of_at_most(audit.confirm_limit(60), 60, Fraction(95, 100))
    at_200 = _chance_of_at_most(audit.confirm_limit(200), 200, Fraction(95, 100))
    assert round(float(at_60), 3) == 0.417
    assert round(float(at_200), 3) == 0.922


# --- The sample (2.4, S04) -------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 2, 59, 60])
def test_a_path_of_60_titles_or_fewer_is_labeled_in_full(size):
    keys = _keys(size)
    for path in MachinePath:
        assert audit.draw_sample(path, keys[::-1], SEED) == tuple(keys)


@pytest.mark.parametrize("size", [61, 62, 120, 400])
def test_a_larger_path_gives_a_sample_of_60_titles_each_drawn_once(size):
    keys = _keys(size)
    sample = audit.draw_sample(MachinePath.MODELS_AGREE, keys, SEED)
    assert len(sample) == len(set(sample)) == 60
    assert set(sample) < set(keys)
    assert list(sample) == sorted(sample)
    assert all(isinstance(key, TitleKey) for key in sample)


def test_the_sample_is_the_audit_stream_of_the_path_over_the_sorted_keys():
    # S04: the stream "audit", keyed by the machine path's name.
    keys = [*_keys(80, "SAL"), *_keys(80, "FIN")]
    for path, name in (
        (MachinePath.MODELS_AGREE, "models agree"),
        (MachinePath.AGENT_SETTLED, "agent-settled"),
    ):
        expected = seeding.rng(SEED, "audit", name).sample(sorted(keys), 60)
        assert audit.draw_sample(path, keys, SEED) == tuple(sorted(expected))


def test_the_two_paths_draw_their_own_samples():
    keys = _keys(200)
    agree = audit.draw_sample(MachinePath.MODELS_AGREE, keys, SEED)
    settled = audit.draw_sample(MachinePath.AGENT_SETTLED, keys, SEED)
    assert agree != settled
    assert 0 < len(set(agree) & set(settled)) < 60


def test_the_sample_is_the_same_every_time_and_changes_with_the_seed():
    keys = _keys(200)
    path = MachinePath.AGENT_SETTLED
    assert audit.draw_sample(path, keys, SEED) == audit.draw_sample(path, keys, SEED)
    assert len({audit.draw_sample(path, keys, seed) for seed in (SEED, 1, 2, 3)}) == 4


def test_the_sample_does_not_depend_on_the_order_or_the_kind_of_collection():
    keys = [*_keys(70, "SAL"), *_keys(70, "FIN")]
    path = MachinePath.MODELS_AGREE
    expected = audit.draw_sample(path, keys, SEED)
    assert audit.draw_sample(path, keys[::-1], SEED) == expected
    assert audit.draw_sample(path, set(keys), SEED) == expected
    assert audit.draw_sample(path, frozenset(keys), SEED) == expected
    assert audit.draw_sample(path, dict.fromkeys(keys, MACHINE_ROLE).keys(), SEED) == expected
    assert audit.draw_sample(path, tuple(keys[35:] + keys[:35]), SEED) == expected


def test_a_title_handed_in_twice_is_one_title_of_the_path():
    # 2.4: "The audit draws each title once".
    keys = _keys(100)
    path = MachinePath.MODELS_AGREE
    assert audit.draw_sample(path, keys + keys, SEED) == audit.draw_sample(path, keys, SEED)
    assert audit.draw_sample(path, _keys(40) + _keys(40), SEED) == tuple(_keys(40))


# --- The three outcomes, on constructed counts (2.4, Q23) ------------------------------


@pytest.mark.parametrize("wrong", CONFIRMED_COUNTS)
def test_at_most_2_wrong_of_60_confirms_the_path(wrong):
    _, _, result = _audited_path(200, wrong)
    assert result.outcome is AuditOutcome.CONFIRMED
    assert (result.audited, result.wrong) == (60, wrong)


@pytest.mark.parametrize("wrong", PROVISIONAL_COUNTS)
def test_3_to_12_wrong_of_60_makes_the_path_provisional(wrong):
    _, _, result = _audited_path(200, wrong)
    assert result.outcome is AuditOutcome.PROVISIONAL
    assert (result.audited, result.wrong) == (60, wrong)
    assert result.error_rate == wrong / 60  # 1.9: "with its measured error rate"


@pytest.mark.parametrize("wrong", STOPPED_COUNTS)
def test_13_or_more_wrong_of_60_stops_the_path(wrong):
    _, _, result = _audited_path(200, wrong)
    assert result.outcome is AuditOutcome.STOPPED
    assert (result.audited, result.wrong) == (60, wrong)


def test_every_count_of_60_has_the_outcome_its_two_lines_give():
    outcomes = [_audited_path(61, wrong)[2].outcome for wrong in range(61)]
    assert outcomes == (
        [AuditOutcome.CONFIRMED] * 3 + [AuditOutcome.PROVISIONAL] * 10 + [AuditOutcome.STOPPED] * 48
    )
    assert audit.confirm_limit(60) == 2 and audit.stop_limit(60) == 12


@pytest.mark.parametrize("path", list(MachinePath))
def test_an_audit_result_says_what_was_audited(path):
    machine, labels, result = _audited_path(150, 5, path)
    assert result.path is path
    assert result.path_size == 150
    assert result.audited_keys == audit.draw_sample(path, machine, SEED)
    assert list(result.audited_keys) == sorted(labels)
    assert result.wrong == 5


def test_an_audit_does_not_depend_on_the_order_the_titles_and_labels_come_in():
    machine, labels, result = _audited_path(150, 5)
    backwards = dict(reversed(list(labels.items())))
    turned = dict(reversed(list(machine.items())))
    assert list(backwards) != sorted(backwards)
    again = audit.audit_path(MachinePath.MODELS_AGREE, turned, backwards)
    assert again == result
    assert list(again.audited_keys) == sorted(labels)


def test_wrong_is_counted_against_the_machines_role_title_by_title():
    # 2.4: wrong means the machine's role differs from the SME's label for that
    # title. Roles here differ from title to title, on both sides.
    keys = _keys(100)
    roles = ("SAL-01", "SAL-02", "SAL-03")
    machine = {key: roles[position % 3] for position, key in enumerate(keys)}
    sample = audit.draw_sample(MachinePath.MODELS_AGREE, keys, SEED)
    mistaken = set(sample[::7])  # 9 of the 60
    labels = {key: _label(key, "SAL-06" if key in mistaken else machine[key]) for key in sample}
    result = audit.audit_path(MachinePath.MODELS_AGREE, machine, labels)
    assert result.wrong == len(mistaken) == 9
    assert result.outcome is AuditOutcome.PROVISIONAL


def test_the_roles_of_unaudited_titles_do_not_change_the_audit():
    machine, labels, result = _audited_path(200, 4)
    changed = {key: (role if key in labels else "FIN-01") for key, role in machine.items()}
    assert audit.audit_path(MachinePath.MODELS_AGREE, changed, labels) == result


def test_a_title_the_sme_marked_no_suitable_role_counts_as_wrong():
    machine, labels, result = _audited_path(200, 2)
    marked = sorted(labels)[-1]  # one the machine was right about until now
    labels[marked] = _no_suitable_role(marked)
    again = audit.audit_path(MachinePath.MODELS_AGREE, machine, labels)
    assert (result.wrong, again.wrong) == (2, 3)
    assert (result.outcome, again.outcome) == (AuditOutcome.CONFIRMED, AuditOutcome.PROVISIONAL)


# --- A path of 60 titles or fewer (2.4, Q16, Q23) ---------------------------------------


@pytest.mark.parametrize(
    ("size", "wrong"), [(60, 0), (60, 2), (60, 13), (60, 60), (59, 30), (1, 1)]
)
def test_a_path_of_60_or_fewer_gets_no_status_however_many_are_wrong(size, wrong):
    machine, labels, result = _audited_path(size, wrong)
    assert result.outcome is AuditOutcome.ALL_LABELED
    assert result.audited == result.path_size == size
    assert result.audited_keys == tuple(sorted(machine)) == tuple(sorted(labels))
    assert result.wrong == wrong  # Q16: audited and wrong are still shown


def test_a_path_of_61_titles_is_the_first_to_be_sampled_and_given_a_status():
    machine, labels, result = _audited_path(61, 0)
    assert (result.path_size, result.audited) == (len(machine), len(labels)) == (61, 60)
    assert result.outcome is AuditOutcome.CONFIRMED


def test_a_path_with_no_title_is_all_labeled_with_nothing_audited():
    result = audit.audit_path(MachinePath.AGENT_SETTLED, {}, {})
    assert result == AuditResult(MachinePath.AGENT_SETTLED, 0, (), 0, AuditOutcome.ALL_LABELED)
    assert result.error_rate is None


# --- Labels that cannot be the blind labels of the sample (Q01) ---------------------------


def test_a_label_for_a_title_outside_the_path_is_refused():
    machine, labels, _ = _audited_path(200, 0)
    stranger = TitleKey("FIN", "not in the path")
    labels.pop(sorted(labels)[0])
    labels[stranger] = _label(stranger, MACHINE_ROLE)
    with pytest.raises(ValueError, match=r"not in the path: \('FIN', 'not in the path'\)"):
        audit.audit_path(MachinePath.MODELS_AGREE, machine, labels)


@pytest.mark.parametrize(
    ("size", "labeled", "message"),
    [
        (200, 59, "a path of 200 titles labels 60 of them, got 59 labels"),
        (200, 61, "a path of 200 titles labels 60 of them, got 61 labels"),
        (200, 200, "a path of 200 titles labels 60 of them, got 200 labels"),
        (200, 0, "a path of 200 titles labels 60 of them, got 0 labels"),
        (40, 39, "a path of 40 titles labels 40 of them, got 39 labels"),
        (60, 0, "a path of 60 titles labels 60 of them, got 0 labels"),
    ],
)
def test_the_labels_must_be_those_of_the_whole_sample(size, labeled, message):
    keys = _keys(size)
    machine = dict.fromkeys(keys, MACHINE_ROLE)
    labels = {key: _label(key, MACHINE_ROLE) for key in keys[:labeled]}
    with pytest.raises(ValueError, match=message):
        audit.audit_path(MachinePath.MODELS_AGREE, machine, labels)


def test_a_blind_label_that_could_not_decide_is_refused():
    # Q01: "could not decide" applies to review-queue decisions only.
    machine, labels, _ = _audited_path(200, 0)
    key = sorted(labels)[5]
    labels[key] = HumanDecision(key, HumanAnswer.UNDECIDED, None, "Reviewer (simulated)", True)
    with pytest.raises(ValueError, match="a blind label is never 'could not decide'"):
        audit.audit_path(MachinePath.MODELS_AGREE, machine, labels)
