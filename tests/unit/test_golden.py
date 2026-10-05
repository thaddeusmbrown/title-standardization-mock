"""The golden set: which titles can be drawn, and the seeded draw.

docs/design.md 1.2; docs/build-decisions.md Q08 (size and frame), Q14 and Q26
(a title with a person's name or no usable family code is out of the frame),
Q22 (the draw depends only on the seed and the title keys).
"""

from __future__ import annotations

import dataclasses

import pytest

from titlestd import golden
from titlestd.contracts import NAME_MASK, Record, TitleKey, TitleRow
from titlestd.settings import Settings

SEED = 20260101

# The mini data set, written out by hand. The Wave 1 list of the mini settings
# is the four most widely held rows (headcounts 3, 3, 3 and 2).
MINI_WAVE1 = (
    TitleKey("FIN", "financial analyst"),
    TitleKey("SAL", "account manager ii"),
    TitleKey("SAL", "sales representative"),
    TitleKey("SAL", "senior account manager"),
)
# The two rows of the mini seed look-up, cleaned.
MINI_SEED = (TitleKey("FIN", "accounts payable clerk"), TitleKey("SAL", "sales engineer"))
# What is left of the 15 rows: not the four and the two above, not the title
# with a person's name, not the row with a blank family code.
MINI_FRAME = (
    TitleKey("FIN", "internal auditor"),
    TitleKey("FIN", "payroll specialist"),
    TitleKey("FIN", "senior financial analyst"),
    TitleKey("FIN", "special projects"),
    TitleKey("FIN", "treasury analyst"),
    TitleKey("SAL", "bdr"),
    TitleKey("SAL", "sales operations analyst"),
)


def _row(
    family: str,
    title: str,
    *,
    headcount: int = 1,
    has_person_name: bool = False,
    family_usable: bool = True,
) -> TitleRow:
    return TitleRow(
        record=Record(cleaned_title=title, family_code=family, department="", description=""),
        employee_ids=tuple(f"E{number:03d}" for number in range(headcount)),
        raw_titles=(title.upper(),),
        has_person_name=has_person_name,
        family_usable=family_usable,
    )


def _rows(count: int, family: str = "SAL") -> list[TitleRow]:
    """count plain long-tail rows whose keys sort in the order they are made."""
    return [_row(family, f"title {number:03d}") for number in range(count)]


def _draw(rows, wave1=(), seed_keys=(), **settings) -> tuple[TitleKey, ...]:
    return golden.draw_golden(rows, wave1, seed_keys, Settings(**settings))


# --- The frame (Q08, Q14, Q26) ---------------------------------------------------


def test_the_frame_of_the_mini_data_set(mini_rows):
    # A golden set as large as the data set takes the whole frame, which shows it.
    drawn = _draw(mini_rows, MINI_WAVE1, MINI_SEED, golden_size=len(mini_rows))
    assert drawn == MINI_FRAME


def test_the_mini_settings_draw_three_titles_of_the_frame(mini_rows, mini_settings):
    drawn = golden.draw_golden(mini_rows, MINI_WAVE1, MINI_SEED, mini_settings)
    assert len(drawn) == mini_settings.golden_size == 3
    assert set(drawn) < set(MINI_FRAME)


def test_with_nothing_excluded_every_row_is_in_the_frame():
    rows = _rows(10)
    assert _draw(rows, golden_size=10) == tuple(row.key for row in rows)


@pytest.mark.parametrize(
    "how",
    ["on the Wave 1 list", "a seed look-up hit", "a person's name", "a blank family code"],
)
def test_each_exclusion_alone_takes_a_title_out_of_the_frame(how):
    rows = _rows(10)
    out = rows[4]
    wave1, seed_keys = [], []
    if how == "on the Wave 1 list":
        wave1 = [out.key]
    elif how == "a seed look-up hit":
        seed_keys = [out.key]
    elif how == "a person's name":
        rows[4] = out = _row("SAL", f"title 004 {NAME_MASK}", has_person_name=True)
    else:
        rows[4] = out = _row("", "title 004", family_usable=False)
    drawn = _draw(rows, wave1, seed_keys, golden_size=10)
    assert out.key not in drawn
    assert drawn == tuple(sorted(row.key for row in rows if row is not out))


def test_an_unknown_family_code_is_as_unusable_as_a_blank_one():
    # Q26: unusable is blank after trimming, or not one of the catalog's codes.
    # The prepare step says which; the draw reads the row's flag.
    rows = [*_rows(5), _row("XXX", "title in no family", family_usable=False)]
    assert _draw(rows, golden_size=10) == tuple(row.key for row in rows[:5])


def test_a_widely_held_title_is_in_the_frame_unless_it_is_on_the_wave1_list():
    # "Long tail" means off the Wave 1 list, whatever the headcount (Q08, Q24).
    rows = [_row("SAL", "big title", headcount=50), _row("SAL", "small title")]
    assert _draw(rows, golden_size=5) == (TitleKey("SAL", "big title"), rows[1].key)
    assert _draw(rows, [rows[0].key], golden_size=5) == (rows[1].key,)


def test_the_same_title_in_another_family_is_another_title():
    rows = [_row("FIN", "analyst"), _row("SAL", "analyst")]
    assert _draw(rows, [TitleKey("FIN", "analyst")], golden_size=5) == (TitleKey("SAL", "analyst"),)
    assert _draw(rows, (), [TitleKey("SAL", "analyst")], golden_size=5) == (
        TitleKey("FIN", "analyst"),
    )


def test_exclusions_shrink_the_frame_and_not_the_set():
    # 50 rows, 10 of them excluded, a golden set of 40: exactly the 40 others.
    # A draw from all 50 followed by a filter would give fewer.
    rows = _rows(50)
    drawn = _draw(
        rows, [row.key for row in rows[:5]], [row.key for row in rows[5:10]], golden_size=40
    )
    assert drawn == tuple(row.key for row in rows[10:])


def test_a_key_that_no_row_has_changes_nothing():
    rows = _rows(30)
    strangers = [TitleKey("SAL", "not in the data"), TitleKey("", "")]
    assert _draw(rows, strangers, strangers, golden_size=7) == _draw(rows, golden_size=7)


def test_a_key_that_two_rows_share_counts_once():
    rows = [*_rows(3), *_rows(3)]
    assert _draw(rows, golden_size=10) == tuple(row.key for row in rows[:3])


# --- The size (Q08) ---------------------------------------------------------------


def test_the_default_golden_set_has_forty_titles():
    drawn = _draw(_rows(300))
    assert Settings().golden_size == 40
    assert len(drawn) == len(set(drawn)) == 40


@pytest.mark.parametrize(("frame", "size", "expected"), [(100, 12, 12), (7, 40, 7), (7, 7, 7)])
def test_the_set_has_golden_size_titles_or_the_whole_frame_when_that_is_smaller(
    frame, size, expected
):
    rows = _rows(frame)
    drawn = _draw(rows, golden_size=size)
    assert len(drawn) == len(set(drawn)) == expected
    assert set(drawn) <= {row.key for row in rows}


def test_a_golden_size_of_zero_or_an_empty_frame_gives_an_empty_set():
    assert _draw(_rows(20), golden_size=0) == ()
    assert _draw([]) == ()
    everything = _rows(5)
    assert _draw(everything, [row.key for row in everything]) == ()


# --- The draw (Q06, Q08, Q22) -------------------------------------------------------


def test_the_set_is_sorted_by_key():
    rows = [*_rows(40, "SAL"), *_rows(40, "FIN"), *_rows(40, "OPS")]
    drawn = _draw(rows, golden_size=30)
    assert list(drawn) == sorted(drawn)
    assert {key.family_code for key in drawn} == {"FIN", "OPS", "SAL"}
    assert all(isinstance(key, TitleKey) for key in drawn)


def test_the_draw_is_the_golden_stream_over_the_sorted_frame():
    # tests/unit/test_seeding.py pins the first sample of five from a hundred on
    # the golden stream as the positions 18, 23, 20, 75 and 63. The golden set
    # of five from a frame of a hundred is therefore those titles (S04).
    rows = _rows(100)
    drawn = _draw(rows, golden_size=5, seed=SEED)
    assert drawn == tuple(rows[position].key for position in (18, 20, 23, 63, 75))


def test_the_draw_does_not_depend_on_the_order_of_the_rows_or_of_the_lists():
    rows = [*_rows(60, "SAL"), *_rows(60, "FIN")]
    wave1 = [row.key for row in rows[:8]]
    seed_keys = [row.key for row in rows[60:66]]
    expected = _draw(rows, wave1, seed_keys)
    assert _draw(rows[::-1], wave1[::-1], seed_keys[::-1]) == expected
    assert _draw(rows[30:] + rows[:30], set(wave1), frozenset(seed_keys)) == expected
    assert _draw(tuple(rows), tuple(wave1), tuple(seed_keys)) == expected


def test_the_draw_depends_only_on_the_title_keys():
    # Not on headcount, raw spellings, department or description, and so on
    # nothing a person or a model later says about a title (Q22).
    rows = _rows(120)
    changed = [
        dataclasses.replace(
            row,
            record=dataclasses.replace(row.record, department="Other", description="Other words."),
            employee_ids=("X1", "X2", "X3"),
            raw_titles=("Another Spelling", "another spelling"),
        )
        for row in rows
    ]
    assert _draw(changed) == _draw(rows)


def test_the_draw_is_the_same_every_time_and_changes_with_the_seed():
    rows = _rows(200)
    assert _draw(rows, seed=SEED) == _draw(rows, seed=SEED) == _draw(rows)
    others = {_draw(rows, seed=seed) for seed in range(1, 6)}
    assert _draw(rows, seed=SEED) not in others
    assert len(others) == 5


def test_a_title_added_to_the_wave1_list_leaves_the_rest_of_the_frame_drawable():
    # The frame is rebuilt from the keys, so the set stays full size.
    rows = _rows(100)
    first = _draw(rows, golden_size=10)
    second = _draw(rows, [first[0]], golden_size=10)
    assert first[0] not in second and len(second) == 10
