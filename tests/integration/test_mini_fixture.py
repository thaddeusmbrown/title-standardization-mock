"""The mini data set: loads through the readers and is consistent with itself.

It is the test bed for every branch until the generated mock data exists, so
its own promises are tested here: it reproduces the worked example of
docs/design.md 1.3, the hand-written title rows cover the employees file, the
truth and seed files agree with them, and everything in it is invented.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from titlestd import datafiles, people, reference
from titlestd.contracts import EMAIL_MASK, NAME_MASK, InputData, TitleKey, TitleRow


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


@pytest.fixture(scope="module")
def truth(mini_dir: Path) -> dict[tuple[str, str], str]:
    """The mini truth: (raw job title, family code) to true role.

    The family code is trimmed, as in a row's key: codes are compared after
    removing surrounding whitespace on both sides (S05).
    """
    return {
        (row.job_title, row.job_family_code.strip()): row.true_role_id
        for row in people.read_truth(mini_dir)
    }


def test_the_seven_files_and_nothing_else(mini_dir: Path):
    assert sorted(path.name for path in mini_dir.iterdir()) == sorted(
        [*datafiles.INPUT_FILES, people.TRUTH_FILE]
    )


def test_it_loads_and_checks(mini_dir: Path):
    data = datafiles.read_inputs(mini_dir)
    people.validate_truth(people.read_truth(mini_dir), data.catalog)


def test_shape(mini_inputs: InputData, mini_rows):
    roles = mini_inputs.catalog.roles
    families = Counter(role.family_code for role in roles if not role.is_generic)
    assert families == {"SAL": 6, "FIN": 6}  # ten on a shortlist always spans both
    assert sorted(role.role_id for role in roles if role.is_generic) == ["FIN-GEN", "SAL-GEN"]
    assert [role.role_id for role in roles if role.compliance_flag] == ["FIN-04"]
    assert len(mini_inputs.employees) == 22
    assert len(mini_rows) == 15


# --- The worked example of 1.3 ---------------------------------------------------


def test_the_catalog_holds_the_worked_example(mini_inputs: InputData):
    catalog = mini_inputs.catalog
    role = next(r for r in catalog.roles if r.role_name == "Account Manager")
    assert (role.family_code, role.short_name) == ("SAL", "Acct Mgr")
    levels = {grade.grade: grade.level for grade in catalog.grade_levels}
    assert levels[7] == levels[8] == "Senior"
    senior = next(level for level in catalog.ladder if level.level == "Senior")
    assert f"{senior.long_word} {role.role_name}" == "Senior Account Manager"
    assert f"{senior.short_word} {role.short_name}" == "Sr Acct Mgr"


def test_the_grade_table_is_the_decided_one(mini_inputs: InputData):
    levels = [grade.level for grade in mini_inputs.catalog.grade_levels]
    assert levels == (
        ["Junior"] * 3 + ["Associate"] * 3 + ["Senior"] * 2 + ["Lead"] + ["Principal"]
    )
    assert [(lv.long_word, lv.short_word) for lv in mini_inputs.catalog.ladder] == [
        ("Junior", "Jr"),
        ("Associate", "Assoc"),
        ("Senior", "Sr"),
        ("Lead", "Lead"),
        ("Principal", "Prin"),
    ]


def test_the_two_legacy_titles_of_the_example_are_present(mini_inputs: InputData, truth):
    by_title = {employee.job_title: employee for employee in mini_inputs.employees}
    assert (by_title["ACCT MGR II"].job_family_code, by_title["ACCT MGR II"].grade) == ("SAL", 8)
    senior = by_title["Senior Account Manager"]
    assert (senior.job_family_code, senior.grade) == ("SAL", 7)
    assert truth[("ACCT MGR II", "SAL")] == truth[("Senior Account Manager", "SAL")] == "SAL-01"


# --- The hand-written title rows ---------------------------------------------------


def test_rows_are_sorted_and_keys_are_unique(mini_rows: tuple[TitleRow, ...]):
    keys = [row.key for row in mini_rows]
    assert keys == sorted(set(keys))
    assert keys[0] == TitleKey("", "staff accountant")


def test_rows_cover_every_employee_exactly_once(mini_inputs: InputData, mini_rows):
    listed = [employee_id for row in mini_rows for employee_id in row.employee_ids]
    assert sorted(listed) == sorted(e.employee_id for e in mini_inputs.employees)
    assert len(listed) == len(set(listed))
    assert sum(row.headcount for row in mini_rows) == len(mini_inputs.employees)


def test_each_row_lists_its_employees_raw_titles_and_family(mini_inputs: InputData, mini_rows):
    employees = {e.employee_id: e for e in mini_inputs.employees}
    for row in mini_rows:
        members = [employees[employee_id] for employee_id in row.employee_ids]
        assert list(row.employee_ids) == sorted(row.employee_ids)
        assert row.raw_titles == tuple(sorted({member.job_title for member in members}))
        assert {member.job_family_code.strip() for member in members} == {row.key.family_code}


def test_each_record_text_comes_from_one_of_its_employees(mini_inputs: InputData, mini_rows):
    employees = {e.employee_id: e for e in mini_inputs.employees}
    for row in mini_rows:
        members = [employees[employee_id] for employee_id in row.employee_ids]
        assert row.record.department in {member.department for member in members}
        if not row.record.description:
            assert all(not member.job_description for member in members)
        else:
            # Equal up to masking: compare the words outside the masks.
            unmasked = row.record.description.replace(NAME_MASK, " ").replace(EMAIL_MASK, " ")
            assert any(_tokens(unmasked) <= _tokens(m.job_description) for m in members)


def test_flags_on_rows(mini_rows: tuple[TitleRow, ...]):
    named = [row.key.cleaned_title for row in mini_rows if row.has_person_name]
    assert named == [f"customer success manager {NAME_MASK}"]
    assert [row.key for row in mini_rows if not row.family_usable] == [
        TitleKey("", "staff accountant")
    ]
    assert sum(not row.model_eligible for row in mini_rows) == 2


def test_no_record_shows_a_name_or_an_email(mini_rows: tuple[TitleRow, ...]):
    names = set(reference.GIVEN_NAMES) | set(reference.SURNAMES)
    for row in mini_rows:
        shown = " ".join([row.record.cleaned_title, row.record.department, row.record.description])
        assert not names & _tokens(shown)
        assert "@" not in shown


def test_the_wave1_cut_of_the_mini_settings_does_not_rest_on_a_tie(mini_rows, mini_settings):
    headcounts = sorted((row.headcount for row in mini_rows), reverse=True)
    size = mini_settings.wave1_size
    assert headcounts[:size] == [3, 3, 3, 2]
    assert headcounts[size - 1] > headcounts[size]


# --- Truth and seed look-up agree with the rows (Q09) ---------------------------------


def test_truth_has_one_row_per_raw_title_and_family(mini_inputs: InputData, mini_dir: Path):
    # As recorded: the truth file repeats the employees file's pairs byte for byte (Q09).
    raw_pairs = {(e.job_title, e.job_family_code) for e in mini_inputs.employees}
    recorded = [(row.job_title, row.job_family_code) for row in people.read_truth(mini_dir)]
    assert set(recorded) == raw_pairs
    assert len(recorded) == len(raw_pairs)


def test_a_padded_family_code_is_the_same_family(
    mini_inputs: InputData, mini_rows, mini_dir: Path, truth
):
    # One code is padded on purpose. Its row is keyed on the trimmed code, the
    # truth file holds the code as recorded, and the two still meet (S05).
    padded = [e for e in mini_inputs.employees if e.job_family_code != e.job_family_code.strip()]
    assert [(e.employee_id, e.job_family_code) for e in padded] == [("E020", "FIN ")]
    assert people.TruthRow("Treasury Analyst", "FIN ", "FIN-06") in people.read_truth(mini_dir)
    (row,) = [row for row in mini_rows if "E020" in row.employee_ids]
    assert row.key == TitleKey("FIN", "treasury analyst") and row.family_usable
    assert truth[("Treasury Analyst", row.key.family_code)] == "FIN-06"


def test_each_cleaned_row_has_one_true_role(mini_rows: tuple[TitleRow, ...], truth):
    for row in mini_rows:
        roles = {truth[(raw, row.key.family_code)] for raw in row.raw_titles}
        assert len(roles) == 1, row.key


def test_seed_rows_are_long_tail_titles_and_agree_with_the_truth(mini_inputs, mini_rows, truth):
    by_raw = {(raw, row.key.family_code): row for row in mini_rows for raw in row.raw_titles}
    for seed in mini_inputs.seed_lookup:
        pair = (seed.job_title, seed.job_family_code.strip())
        assert truth[pair] == seed.role_id
        assert by_raw[pair].headcount == 1  # never on the Wave 1 list (Q13)


# --- Everything is invented ------------------------------------------------------------


def test_emails_use_the_reserved_domain(mini_dir: Path):
    found = []
    for path in sorted(mini_dir.iterdir()):
        found += re.findall(r"[\w.+-]+@([\w.-]+\w)", path.read_text(encoding="utf-8"))
    assert found == [reference.EMAIL_DOMAIN]


def test_reference_words_do_not_collide_with_the_data(mini_inputs: InputData):
    reserved = set(reference.GIVEN_NAMES) | set(reference.SURNAMES) | set(reference.LOCATIONS)
    catalog = mini_inputs.catalog
    words: set[str] = set()
    for role in catalog.roles:
        words |= _tokens(role.role_name) | _tokens(role.short_name) | _tokens(role.definition)
    for level in catalog.ladder:
        words |= _tokens(level.level) | _tokens(level.long_word) | _tokens(level.short_word)
    for abbreviation in mini_inputs.abbreviations:
        words |= _tokens(abbreviation.short_form) | _tokens(abbreviation.long_form)
    for employee in mini_inputs.employees:
        words |= _tokens(employee.department)
    assert not reserved & words


def test_the_names_in_the_data_come_from_the_invented_lists(mini_inputs: InputData):
    names = set(reference.GIVEN_NAMES) | set(reference.SURNAMES)
    titles = " ".join(e.job_title for e in mini_inputs.employees)
    descriptions = " ".join(e.job_description for e in mini_inputs.employees)
    assert {"tavi", "quillfeather"} <= names & _tokens(titles)
    assert {"oswin", "thistlewood"} <= names & _tokens(descriptions)


def test_the_location_and_requisition_suffixes_are_recognisable(mini_inputs: InputData):
    titles = [e.job_title for e in mini_inputs.employees]
    assert any(_tokens(title) & set(reference.LOCATIONS) for title in titles)
    with_requisition = [
        title
        for title in titles
        if any(re.search(p, title, re.IGNORECASE) for p in reference.REQUISITION_PATTERNS)
    ]
    assert with_requisition == ["sales representative REQ-10442", "Sales Engineer #20311"]


def test_abbreviations_are_lower_case_tokens(mini_inputs: InputData):
    for abbreviation in mini_inputs.abbreviations:
        assert re.fullmatch(r"[a-z]+", abbreviation.short_form)
        assert abbreviation.long_form == abbreviation.long_form.lower()
