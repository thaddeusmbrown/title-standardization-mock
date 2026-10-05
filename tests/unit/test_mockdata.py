"""The mock data generator: what it builds, how it draws, what its knobs change.

docs/design.md 2.1 and 2.3; docs/build-decisions.md Q06, Q09, Q23, Q24, Q35.
These tests build the data in memory and never read data/: the committed files
are tested in tests/integration/test_mockdata.py.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections import Counter
from pathlib import Path

import pytest

from titlestd import catalog as naming
from titlestd import cli, datafiles, mockdata, people, prepare, reference, seeding
from titlestd.contracts import NAME_MASK, InputData, TitleRow
from titlestd.settings import Settings

KNOBS = ("WORDING_WEIGHTS", "DESCRIPTION_WEIGHTS", "FLAGGED_ROLE_IDS")

# Two settings of each knob, far apart, as a tuning round might try them (Q23).
# The tests of the knobs compare these with each other and never with the
# values in force, so they hold whatever stage 4 sets the knobs to.
KNOB_SETTINGS = {
    "WORDING_WEIGHTS": ({"name": 1}, {"name": 1, "near": 2, "far": 7}),
    "DESCRIPTION_WEIGHTS": ({"close": 1}, {"loose": 1, "vague": 3}),
    "FLAGGED_ROLE_IDS": (("FIN-04", "LEG-03", "OPS-06"), ("SAL-03", "TEC-05", "CUS-04")),
}

Built = tuple[InputData, tuple[people.TruthRow, ...]]


def build_with(knob: str, value: object) -> Built:
    """The data as it would be built with one knob set to another value."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mockdata, knob, value)
        return mockdata.build()


def rows_of(data: InputData) -> tuple[TitleRow, ...]:
    """The title rows the real prepare step reads from the data."""
    return prepare.prepare(data.employees, data.abbreviations, naming.family_codes(data.catalog))


@pytest.fixture(scope="module")
def built() -> Built:
    return mockdata.build()


@pytest.fixture(scope="module")
def data(built) -> InputData:
    return built[0]


@pytest.fixture(scope="module")
def rows(data: InputData) -> tuple[TitleRow, ...]:
    return rows_of(data)


# --- The knobs and the fixed sizes (Q23, Q35) ----------------------------------------------------


def test_the_difficulty_knobs_are_the_first_constants_of_the_module():
    # "How the build is cut": stage 4 tunes named constants at the top.
    tree = ast.parse(inspect.getsource(mockdata))
    assigned = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert tuple(assigned[: len(KNOBS)]) == KNOBS
    assert all(name.isupper() and not name.startswith("_") for name in KNOBS)


def test_the_knobs_are_plain_values_a_tuning_round_can_edit():
    assert set(mockdata.WORDING_WEIGHTS) <= {"name", "near", "far"}
    assert set(mockdata.DESCRIPTION_WEIGHTS) <= {"close", "partial", "loose", "vague"}
    assert sum(mockdata.WORDING_WEIGHTS.values()) > 0
    assert sum(mockdata.DESCRIPTION_WEIGHTS.values()) > 0
    assert len(set(mockdata.FLAGGED_ROLE_IDS)) == 3


def test_the_generator_has_its_own_seed_and_no_setting_reaches_it():
    assert Settings().seed != mockdata.GENERATOR_SEED  # Q06: separate from the run seed
    parameters = inspect.signature(mockdata.build).parameters
    assert list(parameters) == []


def test_the_fixed_sizes_are_those_of_the_record():
    assert mockdata.EMPLOYEES == 1500
    assert mockdata.WAVE1_SIZE == Settings().wave1_size == 40
    special = len(mockdata.SPECIAL_TITLES)
    assert 48 * mockdata.ROWS_PER_ROLE + special == 400
    assert [len(titles) for titles in _special_lists()] == [5, 3, 4, 4]
    assert sum(held for _, _, _, held, _ in mockdata.NO_FAMILY_TITLES) == 5  # Q26
    assert mockdata.NO_DESCRIPTION_ROWS == 0.15 * 400
    assert mockdata.NO_GRADE_EMPLOYEES == 20


def _special_lists():
    return (
        mockdata.PERSON_NAME_TITLES,
        mockdata.NO_FAMILY_TITLES,
        mockdata.NO_MATCH_TITLES,
        mockdata.BARE_TITLES,
    )


# --- What is built -------------------------------------------------------------------------------


def test_the_build_is_the_same_on_every_call(built):
    assert mockdata.build() == built


def test_the_build_holds_the_planned_sizes_exactly(data, built, rows):
    assert len(data.employees) == mockdata.EMPLOYEES
    assert len(rows) == 400
    assert sum(row.headcount for row in rows) == mockdata.EMPLOYEES
    assert sum(1 for row in rows if not row.record.description) == mockdata.NO_DESCRIPTION_ROWS
    assert sum(1 for e in data.employees if e.grade is None) == mockdata.NO_GRADE_EMPLOYEES
    assert len(data.seed_lookup) == 8
    assert len(built[1]) == len({(e.job_title, e.job_family_code) for e in data.employees})


def test_the_build_passes_the_checks_of_the_real_readers(data, built):
    datafiles.validate_inputs(data)
    people.validate_truth(built[1], data.catalog)


def test_the_catalog_is_eight_families_of_six_roles_and_a_generic_one(data):
    roles = data.catalog.roles
    assert len(roles) == len({role.role_id for role in roles}) == 56
    per_family = Counter(role.family_code for role in roles)
    assert len(per_family) == 8 and set(per_family.values()) == {7}
    for family in per_family:
        assert naming.generic_role(data.catalog, family).role_id == f"{family}-GEN"
        assert len(naming.roles_in_family(data.catalog, family)) == 6
    flagged = {role.role_id for role in roles if role.compliance_flag}
    assert flagged == set(mockdata.FLAGGED_ROLE_IDS)


def test_a_definition_is_one_sentence_made_of_the_roles_first_two_duties(data):
    role = naming.get_role(data.catalog, "SAL-01")
    assert (role.role_name, role.short_name) == ("Account Manager", "Acct Mgr")  # Q36
    assert role.definition == (
        "Manages a portfolio of customer accounts and renews customer contracts."
    )
    for role in naming.model_visible_roles(data.catalog):
        assert role.definition[0].isupper() and role.definition.endswith(".")
        assert role.definition.count(".") == 1


def test_every_role_has_its_eight_rows_and_the_three_every_role_has(data, built, rows):
    true_role = {(t.job_title, t.job_family_code): t.true_role_id for t in built[1]}
    of_role: dict[str, set[str]] = {}
    for row in rows:
        if row.model_eligible:
            (role_id,) = {true_role[(raw, row.record.family_code)] for raw in row.raw_titles}
            of_role.setdefault(role_id, set()).add(row.key.cleaned_title)
    built_by_hand = Counter(
        role_id for _, _, role_id, _, _ in (*mockdata.NO_MATCH_TITLES, *mockdata.BARE_TITLES)
    )
    for role in naming.model_visible_roles(data.catalog):
        titles = of_role[role.role_id]
        assert len(titles) == mockdata.ROWS_PER_ROLE + built_by_hand[role.role_id], role.role_id
        name = role.role_name.lower()
        assert {name, f"senior {name}", f"{name} ii"} <= titles, role.role_id


def test_the_special_rows_are_recorded_as_written(data):
    held = Counter((e.job_title, e.job_family_code) for e in data.employees)
    described = {(e.job_title, e.job_family_code): e.job_description for e in data.employees}
    for job_title, family_code, _, headcount, description in mockdata.SPECIAL_TITLES:
        assert held[(job_title, family_code)] == headcount, job_title
        assert described[(job_title, family_code)] == description, job_title


def test_employee_ids_run_in_file_order_and_titles_are_mixed(data):
    ids = [employee.employee_id for employee in data.employees]
    assert ids == [f"E{number:04d}" for number in range(1, mockdata.EMPLOYEES + 1)]
    families = [employee.job_family_code for employee in data.employees]
    assert families != sorted(families)  # not grouped: the file is in no telling order


# --- Headcounts (2.1, Q24) -----------------------------------------------------------------------


@pytest.mark.parametrize("skew", [0.5, 0.55, 0.6, 0.65, 0.67, 0.7, 0.75, 0.8])
@pytest.mark.parametrize(("row_count", "employees"), [(384, 1480), (400, 1500)])
def test_headcounts_add_up_fall_with_rank_and_leave_a_gap_at_the_cut(
    monkeypatch, skew, row_count, employees
):
    # Q24: whatever the curve, the 40th row holds more than the 41st. Around
    # the cut the curve is flat, so the gap is made, not met by luck: over
    # this many curves a tie would otherwise show.
    monkeypatch.setattr(mockdata, "HEADCOUNT_SKEW", skew)
    counts = mockdata._headcounts(row_count, employees)
    assert len(counts) == row_count and sum(counts) == employees and min(counts) >= 1
    assert counts == sorted(counts, reverse=True)
    assert counts[mockdata.WAVE1_SIZE - 1] > counts[mockdata.WAVE1_SIZE]


def test_the_top_forty_rows_hold_about_forty_percent_of_the_employees():
    counts = mockdata._headcounts(384, 1480)
    assert 0.38 <= sum(counts[: mockdata.WAVE1_SIZE]) / 1500 <= 0.42  # 2.1: "roughly 40%"
    assert counts[0] > 10 * counts[mockdata.WAVE1_SIZE]  # skewed
    assert counts[-1] == 1  # the long tail ends in titles one employee holds


def test_a_skew_that_empties_a_row_is_refused(monkeypatch):
    monkeypatch.setattr(mockdata, "HEADCOUNT_SKEW", 3.0)
    with pytest.raises(mockdata.MockDataError, match="HEADCOUNT_SKEW"):
        mockdata._headcounts(384, 1480)


def test_the_most_widely_held_rows_are_ordinary_titles(data, rows):
    top = sorted(rows, key=lambda row: (-row.headcount, row.key))[: mockdata.WAVE1_SIZE]
    special = {title for title, *_ in mockdata.SPECIAL_TITLES}
    seeded = {seed.job_title for seed in data.seed_lookup}
    for row in top:
        assert row.model_eligible
        assert not special & set(row.raw_titles) and not seeded & set(row.raw_titles)


# --- Seeded draws (Q06) --------------------------------------------------------------------------


def test_a_draw_depends_only_on_the_generator_seed_and_its_key(monkeypatch):
    first = [mockdata._Draw("spellings", "SAL", "a").unit() for _ in range(2)]
    assert first[0] == first[1]
    assert mockdata._Draw("spellings", "SAL", "b").unit() != first[0]
    assert mockdata._Draw("description", "SAL", "a").unit() != first[0]
    expected = seeding.rng(mockdata.GENERATOR_SEED, "mockdata", "spellings", "SAL", "a").random()
    assert first[0] == expected
    monkeypatch.setattr(mockdata, "GENERATOR_SEED", mockdata.GENERATOR_SEED + 1)
    assert mockdata._Draw("spellings", "SAL", "a").unit() != first[0]


def test_the_generator_stream_is_not_a_stream_of_a_run():
    assert mockdata._STREAM not in seeding.STREAMS


def test_a_weighted_draw_follows_the_weights_and_never_takes_a_zero():
    draw = mockdata._Draw("test", "weighted")
    drawn = Counter(draw.weighted({"often": 8, "never": 0, "seldom": 2}) for _ in range(2000))
    assert drawn["never"] == 0
    assert 0.74 < drawn["often"] / 2000 < 0.86
    assert Counter(draw.weighted({"only": 0, "last": 5}) for _ in range(50)) == {"last": 50}


def test_a_pick_stays_inside_its_options():
    draw = mockdata._Draw("test", "pick")
    assert Counter(draw.pick("abc") for _ in range(300)).keys() == set("abc")


def test_the_lowest_draws_are_an_exact_number_and_do_not_move_when_items_are_added():
    def key(item: str) -> tuple[str, ...]:
        return (item,)

    items = [f"item {number}" for number in range(100)]
    chosen = mockdata._lowest(items, 10, "test", key)
    assert len(chosen) == 10 and set(chosen) <= set(items)
    assert mockdata._lowest(reversed(items), 10, "test", key) == chosen
    more = mockdata._lowest([*items, "another"], 10, "test", key)
    assert set(chosen) - set(more) <= {chosen[-1]} and len(set(more) - set(chosen)) <= 1
    assert mockdata._lowest(items, 10, "another stream", key) != chosen


def test_only_the_two_pinned_draws_of_the_generator_are_used():
    # The scaffold pins random() and shuffle() on every supported Python, and
    # nothing else: choice(), sample() and the rest are not to be relied on.
    source = ast.parse(inspect.getsource(mockdata))
    used = {
        node.attr
        for node in ast.walk(source)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_rng"
    }
    assert used == {"random", "shuffle"}


# --- Titles and their spellings (2.3, Q09) -------------------------------------------------------


def test_short_forms_replace_long_forms_the_longest_first():
    assert mockdata._abbreviate("Senior Accounts Payable Clerk") == "Sr AP Clerk"
    assert mockdata._abbreviate("Human Resources Coordinator") == "HR Coord"
    assert mockdata._abbreviate("Account Manager, Senior") == "Acct Mgr, Sr"
    assert mockdata._abbreviate("Technical Recruiter") == "Technical Recruiter"
    assert mockdata._abbreviate("Marketing Manager") == "Mktg Mgr"
    assert mockdata._abbreviate("Market Analyst") == "Mkt Analyst"


def test_a_short_spelling_cleans_back_to_the_long_one(data):
    for title in ("Senior Accounts Payable Clerk", "Quality Assurance Engineer III"):
        short = mockdata._abbreviate(title)
        assert short != title
        assert prepare.clean_title(short, data.abbreviations) == prepare.clean_title(
            title, data.abbreviations
        )


def test_a_title_is_written_by_the_naming_rule_of_the_catalog(data):
    role = naming.get_role(data.catalog, "SAL-01")
    senior = mockdata._title(
        data.catalog, data.abbreviations, role, role.role_name, mockdata._SENIOR
    )
    named = naming.standard_title(data.catalog, role, 7)
    assert (senior.long, senior.short) == (named.long_title, named.short_title)
    assert (senior.long, senior.short) == ("Senior Account Manager", "Sr Acct Mgr")  # 1.3
    assert (senior.cleaned, senior.level, senior.kind) == (
        "senior account manager",
        "Senior",
        "name",
    )

    two = mockdata._title(data.catalog, data.abbreviations, role, role.role_name, mockdata._TWO)
    assert (two.long, two.short, two.cleaned) == (
        "Account Manager II",
        "Acct Mgr II",
        "account manager ii",
    )
    near = mockdata._title(
        data.catalog, data.abbreviations, role, "Key Account Manager", mockdata._PLAIN
    )
    assert (near.long, near.short, near.kind) == ("Key Account Manager", "Key Acct Mgr", "near")
    far = mockdata._title(data.catalog, data.abbreviations, role, "Client Partner", mockdata._PLAIN)
    assert (far.kind, far.words, far.role) == ("far", 2, role)


def test_a_near_wording_shares_a_word_with_the_roles_name_and_a_far_one_shares_none():
    # What the wording knob promises: a "far" title is one its words cannot place.
    def words(text: str) -> set[str]:
        return set(text.lower().replace("-", " ").split())

    wordings = set()
    for role_id, job in mockdata._JOBS.items():
        name = words(job.name)
        for near in job.near:
            assert words(near) & name and near != job.name, (role_id, near)
        assert not words(job.far) & name, (role_id, job.far)
        wordings |= {(role_id[:3], wording) for wording in (job.name, *job.near, job.far)}
    assert len(wordings) == 48 * 4  # no wording twice in a family


def test_a_wording_that_holds_a_listed_name_is_refused(data):
    role = naming.get_role(data.catalog, "SAL-01")
    name = reference.GIVEN_NAMES[0].capitalize()
    with pytest.raises(mockdata.MockDataError, match="no name"):
        mockdata._title(data.catalog, data.abbreviations, role, f"{name} Manager", mockdata._PLAIN)


def test_every_spelling_the_generator_can_write_cleans_to_its_row(data):
    role = naming.get_role(data.catalog, "CUS-01")
    seen: set[str] = set()
    for form in mockdata._FORMS:
        title = mockdata._title(data.catalog, data.abbreviations, role, role.role_name, form)
        draw = mockdata._Draw("test", "spellings", title.cleaned)
        for _ in range(60):
            spelling = mockdata._spelling(title, draw)
            cleaned = prepare.clean_title(spelling, data.abbreviations)
            assert (cleaned.text, cleaned.has_person_name) == (title.cleaned, False), spelling
            seen.add(spelling)
    # The mess of 2.3 is all there: short forms, dots, cases, the two endings.
    assert any("Cust Svc Rep" in spelling for spelling in seen)
    assert any("Svc." in spelling for spelling in seen)
    assert any(spelling.isupper() for spelling in seen)
    assert any(spelling.islower() for spelling in seen)
    assert any("REQ-" in spelling for spelling in seen) and any(
        "#" in spelling for spelling in seen
    )
    places = {place.upper() for place in reference.LOCATIONS if len(place) == 3} | {"Remote"}
    assert sum(1 for place in places if any(place in spelling for spelling in seen)) >= 8


def test_the_employees_of_a_row_are_spread_over_its_spellings(data, rows):
    held = Counter((e.job_title, e.job_family_code) for e in data.employees)
    several = [row for row in rows if len(row.raw_titles) > 1]
    assert len(several) > 100  # Q09: more raw spellings than title rows
    for row in rows:
        counts = [held[(raw, row.record.family_code)] for raw in row.raw_titles]
        assert sum(counts) == row.headcount and min(counts) >= 1
    assert max(len(row.raw_titles) for row in rows) >= 4  # the most widely held rows


def test_the_first_spelling_of_a_row_holds_the_most_employees():
    assert mockdata._split(1, 1) == [1]
    assert mockdata._split(2, 2) == [1, 1]
    assert mockdata._split(10, 3) == [7, 2, 1]
    assert mockdata._split(77, 4) == [45, 19, 9, 4]
    for headcount in range(1, 120):
        for parts in range(1, min(headcount, 4) + 1):
            split = mockdata._split(headcount, parts)
            assert sum(split) == headcount and min(split) >= 1, (headcount, parts)
            assert split == sorted(split, reverse=True), (headcount, parts)


def test_two_titles_of_one_family_that_clean_alike_are_refused(monkeypatch):
    again = (("Sr. Account Manager", "SAL", "SAL-01", 1, ""),)  # "senior account manager"
    monkeypatch.setattr(mockdata, "SPECIAL_TITLES", (*mockdata.SPECIAL_TITLES, *again))
    with pytest.raises(mockdata.MockDataError, match="clean to the same row"):
        mockdata.build()


def test_a_special_title_must_do_what_its_list_says(monkeypatch):
    # A "person's name" title with no listed name in it is no such title.
    plain = (("Paralegal - Night Shift", "LEG", "LEG-02", 1, ""),)
    monkeypatch.setattr(mockdata, "PERSON_NAME_TITLES", plain)
    monkeypatch.setattr(
        mockdata,
        "SPECIAL_TITLES",
        (*plain, *mockdata.NO_FAMILY_TITLES, *mockdata.NO_MATCH_TITLES, *mockdata.BARE_TITLES),
    )
    with pytest.raises(mockdata.MockDataError, match="person's name"):
        mockdata.build()


def test_a_special_title_with_an_unknown_role_is_refused(monkeypatch):
    unknown = (("Night Porter", "OPS", "OPS-99", 1, ""),)
    monkeypatch.setattr(mockdata, "SPECIAL_TITLES", (*mockdata.SPECIAL_TITLES, *unknown))
    with pytest.raises(mockdata.MockDataError, match="OPS-99"):
        mockdata.build()


# --- Descriptions (Q11, Q35) ---------------------------------------------------------------------


def test_the_four_kinds_of_description_say_as_much_as_their_name_promises():
    job = mockdata._JOBS["FIN-05"]
    defining, other = set(job.duties[:2]), set(job.duties[2:])

    def duties_in(kind: str) -> set[str]:
        text = mockdata._description(job, kind, mockdata._Draw("test", kind)).lower()
        return {duty for duty in job.duties if duty in text}

    assert duties_in("close") == defining
    partial = duties_in("partial")
    assert len(partial & defining) == 1 and len(partial & other) == 1
    assert duties_in("loose") == other
    assert duties_in("vague") == set()
    assert mockdata._description(job, "vague", mockdata._Draw("test")) in mockdata._VAGUE


def test_a_one_word_wording_always_has_a_description(data, rows):
    # Q17: only the bare titles built by hand have one word and no description.
    bare = {title for title, *_ in mockdata.BARE_TITLES}
    level_words = {level.long_word.lower() for level in data.catalog.ladder}
    numerals = {"i", "ii", "iii", "1", "2", "3"}
    for row in rows:
        if row.record.description or not row.model_eligible or bare & set(row.raw_titles):
            continue
        words = set(row.key.cleaned_title.split()) - level_words - numerals
        assert len(words) >= 2, row.key


def test_names_and_emails_in_descriptions_come_from_the_reference_lists(data, rows):
    held = {e.employee_id: e.job_description for e in data.employees}
    names, emails = [], []
    for row in rows:
        texts = {held[employee_id] for employee_id in row.employee_ids}
        names += sorted(text for text in texts if "Reports to " in text)
        emails += sorted(text for text in texts if "@" in text)
    assert len(names) == mockdata.NAMES_IN_DESCRIPTIONS  # one row each
    assert len(emails) == mockdata.EMAILS_IN_DESCRIPTIONS
    for description in names:
        given, surname = description.rstrip(".").split("Reports to ")[1].split()
        assert given.lower() in reference.GIVEN_NAMES and surname.lower() in reference.SURNAMES
        assert NAME_MASK in prepare.mask_text(description)
    for description in emails:
        address = description.rstrip(".").split("Questions to ")[1]
        given, _, rest = address.partition(".")
        surname, _, domain = rest.partition("@")
        assert given in reference.GIVEN_NAMES and surname in reference.SURNAMES
        assert domain == reference.EMAIL_DOMAIN


def test_the_mixed_rows_are_half_differing_and_half_partly_empty(data, rows):
    held = {e.employee_id: e.job_description for e in data.employees}
    differing = partly_empty = 0
    for row in rows:
        texts = Counter(held[employee_id] for employee_id in row.employee_ids)
        if len(texts) > 1:
            assert len(texts) == 2
            partly_empty += "" in texts
            differing += "" not in texts
    assert differing == partly_empty == mockdata.MIXED_DESCRIPTION_ROWS // 2


# --- Grades and departments ----------------------------------------------------------------------


def test_grades_follow_the_level_a_title_names_most_of_the_time(data, rows):
    grade = {e.employee_id: e.grade for e in data.employees}
    seniors = [
        grade[employee_id]
        for row in rows
        if row.key.cleaned_title.startswith("senior ")
        for employee_id in row.employee_ids
        if grade[employee_id] is not None
    ]
    at_level = sum(1 for g in seniors if naming.level_for_grade(data.catalog, g) == "Senior")
    assert len(seniors) > 100 and 0.75 < at_level / len(seniors) < 0.95
    assert {e.grade for e in data.employees} == {None, *range(1, 11)}


def test_departments_are_those_of_the_true_roles_family(data, built):
    family_of = {role.role_id: role.family_code for role in data.catalog.roles}
    true_role = {(t.job_title, t.job_family_code): t.true_role_id for t in built[1]}
    for employee in data.employees:
        family = family_of[true_role[(employee.job_title, employee.job_family_code)]]
        assert employee.department in mockdata._DEPARTMENTS[family], employee.employee_id
    main = Counter(e.department for e in data.employees)
    assert len(main) == sum(len(names) for names in mockdata._DEPARTMENTS.values())


# --- What a knob changes, and what it leaves alone (Q23) -----------------------------------------


@pytest.fixture(scope="module", params=sorted(KNOB_SETTINGS))
def tuned(request) -> tuple[str, Built, Built]:
    """A knob and the data built with each of its two settings."""
    one, other = KNOB_SETTINGS[request.param]
    return request.param, build_with(request.param, one), build_with(request.param, other)


def test_whatever_the_knobs_say_the_data_keeps_its_sizes_and_its_rules(tuned):
    for data, truth in tuned[1:]:
        rows = rows_of(data)
        assert len(data.employees) == mockdata.EMPLOYEES and len(rows) == 400
        assert sum(1 for row in rows if not row.record.description) == mockdata.NO_DESCRIPTION_ROWS
        assert sum(1 for e in data.employees if e.grade is None) == mockdata.NO_GRADE_EMPLOYEES
        assert sum(1 for row in rows if row.has_person_name) == len(mockdata.PERSON_NAME_TITLES)
        assert sum(1 for row in rows if not row.family_usable) == len(mockdata.NO_FAMILY_TITLES)
        held = sorted((row.headcount for row in rows), reverse=True)
        assert held[mockdata.WAVE1_SIZE - 1] > held[mockdata.WAVE1_SIZE]
        assert 0.35 <= sum(held[: mockdata.WAVE1_SIZE]) / mockdata.EMPLOYEES <= 0.45
        assert len(data.seed_lookup) == 8
        assert len([role for role in data.catalog.roles if role.compliance_flag]) == 3
        datafiles.validate_inputs(data)
        people.validate_truth(truth, data.catalog)


def test_a_knob_changes_what_it_governs_and_nothing_else(tuned):
    knob, (one, one_truth), (other, other_truth) = tuned
    assert one != other
    if knob == "FLAGGED_ROLE_IDS":
        # Only the flag column of the catalog.
        assert (one.employees, one.seed_lookup, one_truth) == (
            other.employees,
            other.seed_lookup,
            other_truth,
        )
        assert [dataclasses.replace(role, compliance_flag=False) for role in one.catalog.roles] == [
            dataclasses.replace(role, compliance_flag=False) for role in other.catalog.roles
        ]
    if knob == "DESCRIPTION_WEIGHTS":
        # Only the descriptions: the same employees on the same titles and grades.
        assert (one.catalog, one.seed_lookup, one_truth) == (
            other.catalog,
            other.seed_lookup,
            other_truth,
        )
        assert [dataclasses.replace(e, job_description="") for e in one.employees] == [
            dataclasses.replace(e, job_description="") for e in other.employees
        ]
    if knob == "WORDING_WEIGHTS":
        # The titles, and so the employees and the truth, but never the catalog.
        assert one.catalog == other.catalog
        assert one_truth != other_truth


def test_the_wording_knob_sets_how_many_titles_are_worded_like_no_role():
    def far_rows(data: InputData) -> int:
        far = [
            prepare.clean_title(job.far, data.abbreviations).text for job in mockdata._JOBS.values()
        ]
        return sum(1 for row in rows_of(data) if any(text in row.key.cleaned_title for text in far))

    plain, far_fetched = KNOB_SETTINGS["WORDING_WEIGHTS"]
    assert far_rows(build_with("WORDING_WEIGHTS", plain)[0]) == 0
    assert far_rows(build_with("WORDING_WEIGHTS", far_fetched)[0]) > 2 * 48


def test_the_description_knob_sets_how_much_descriptions_say():
    def vague_share(data: InputData) -> float:
        described = [e for e in data.employees if e.job_description]
        return sum(1 for e in described if e.job_description in mockdata._VAGUE) / len(described)

    close, vague = KNOB_SETTINGS["DESCRIPTION_WEIGHTS"]
    assert vague_share(build_with("DESCRIPTION_WEIGHTS", close)[0]) == 0
    assert vague_share(build_with("DESCRIPTION_WEIGHTS", vague)[0]) > 0.5


@pytest.mark.parametrize(
    ("knob", "value", "message"),
    [
        ("FLAGGED_ROLE_IDS", ("FIN-04", "FIN-01", "OPS-06"), "three different families"),
        ("FLAGGED_ROLE_IDS", ("FIN-04", "LEG-03"), "three different families"),
        ("FLAGGED_ROLE_IDS", ("FIN-04", "LEG-03", "OPS-99"), "three different families"),
        ("FLAGGED_ROLE_IDS", ("FIN-04", "LEG-03", "OPS-GEN"), "three different families"),
        ("WORDING_WEIGHTS", {"name": 1, "exotic": 2}, "WORDING_WEIGHTS"),
        ("WORDING_WEIGHTS", {}, "WORDING_WEIGHTS"),
        ("WORDING_WEIGHTS", {"name": 0, "near": 0}, "WORDING_WEIGHTS"),
        ("DESCRIPTION_WEIGHTS", {"close": -1, "vague": 2}, "DESCRIPTION_WEIGHTS"),
        ("DESCRIPTION_WEIGHTS", {"wordy": 1}, "DESCRIPTION_WEIGHTS"),
    ],
)
def test_a_knob_that_breaks_a_rule_is_refused_with_its_name(monkeypatch, knob, value, message):
    monkeypatch.setattr(mockdata, knob, value)
    with pytest.raises(mockdata.MockDataError, match=message):
        mockdata.build()


# --- Writing the files and the command -----------------------------------------------------------


def test_generate_writes_exactly_the_seven_files_and_they_read_back(tmp_path: Path, built):
    target = tmp_path / "fresh" / "data"
    mockdata.generate(target)
    assert sorted(path.name for path in target.iterdir()) == sorted(
        [*datafiles.INPUT_FILES, people.TRUTH_FILE]
    )
    assert (datafiles.read_inputs(target), people.read_truth(target)) == built


def test_generate_writes_the_same_bytes_every_time(tmp_path: Path):
    mockdata.generate(tmp_path / "one")
    mockdata.generate(tmp_path / "two")
    for path in sorted((tmp_path / "one").iterdir()):
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes(), path.name


def test_the_command_writes_into_the_data_folder_of_the_settings(tmp_path: Path, capsys, built):
    target = tmp_path / "data"
    assert mockdata.generate_command(Settings(data_dir=target)) == 0
    assert str(target) in capsys.readouterr().out
    assert datafiles.read_inputs(target) == built[0]


def test_the_command_line_reaches_the_generator_and_the_run_seed_changes_nothing(
    tmp_path: Path, capsys
):
    assert cli.main(["generate-data", "--data-dir", str(tmp_path / "one")]) == 0
    assert cli.main(["generate-data", "--data-dir", str(tmp_path / "two"), "--seed", "7"]) == 0
    capsys.readouterr()
    for path in sorted((tmp_path / "one").iterdir()):
        assert path.read_bytes() == (tmp_path / "two" / path.name).read_bytes(), path.name


def test_a_broken_knob_stops_the_command_with_status_1_and_writes_nothing(
    tmp_path: Path, monkeypatch, capsys
):
    monkeypatch.setattr(mockdata, "FLAGGED_ROLE_IDS", ("FIN-04", "FIN-01", "OPS-06"))
    assert cli.main(["generate-data", "--data-dir", str(tmp_path / "data")]) == 1
    assert "FLAGGED_ROLE_IDS" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()
