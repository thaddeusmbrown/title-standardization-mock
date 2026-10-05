"""The random audit: its two lines and the sample.

docs/design.md 1.9 and 2.4 ("Audit at mock scale"); docs/build-decisions.md
S02 (the lines for 200 are tested beside those for 60) and S04 (the audit
stream).
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from titlestd import audit, contracts, seeding
from titlestd.contracts import MachinePath, TitleKey

SEED = 20260101
MACHINE_ROLE = "SAL-01"  # what the machine chose for every title of a made-up path


def _keys(count: int, family: str = "SAL") -> list[TitleKey]:
    return [TitleKey(family, f"title {number:03d}") for number in range(count)]


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
