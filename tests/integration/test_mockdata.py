"""The committed mock data in data/, put through the real modules of stage 1.

The files are read by the real readers (titlestd.datafiles, titlestd.people),
grouped by the real prepare step, named by the catalog's real naming rule,
labeled by the real simulated people and shown to the real stand-in LLM. The
last tests tie the data to the mini data set, which the earlier stages were
built against.

docs/design.md 2.1 and 2.3; docs/build-decisions.md Q03 (every title has a
truth row), Q06 (byte-for-byte regeneration), Q09 (raw titles, one true role
per row, the seed look-up), Q11 (descriptions), Q13 (seed titles in the long
tail), Q14 (the reference lists share no word with the data), Q17 (titles
built for "no match" and "insufficient data"), Q23 (three flagged roles in
three families), Q24 (the Wave 1 list), Q26 (no usable family code), Q35 (the
counts), Q36 (the grade table and the worked example).

Not here: the run shape of Q23, which feat/pipeline-report-cli measures in
stage 4. That branch may change the difficulty knobs of titlestd.mockdata and
regenerate data/; every test in this file must still pass then.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from titlestd import catalog as naming
from titlestd import datafiles, mockdata, people, prepare, reference, switch
from titlestd.contracts import (
    EMAIL_MASK,
    NAME_MASK,
    ChooseKind,
    HumanAnswer,
    InputData,
    Models,
    NumberedRole,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings

SEVEN_FILES = sorted([*datafiles.INPUT_FILES, people.TRUTH_FILE])
NAMES = frozenset(reference.GIVEN_NAMES) | frozenset(reference.SURNAMES)
LOCATIONS = frozenset(reference.LOCATIONS)
REQUISITION = re.compile("|".join(reference.REQUISITION_PATTERNS), re.IGNORECASE)


def words(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.lower()))


@pytest.fixture(scope="module")
def data_dir(repo_root: Path) -> Path:
    """The committed data folder: the default of the data_dir setting."""
    return repo_root / Settings().data_dir


@pytest.fixture(scope="module")
def inputs(data_dir: Path) -> InputData:
    return datafiles.read_inputs(data_dir)


@pytest.fixture(scope="module")
def truth(data_dir: Path) -> tuple[people.TruthRow, ...]:
    return people.read_truth(data_dir)


@pytest.fixture(scope="module")
def rows(inputs: InputData) -> tuple[TitleRow, ...]:
    """The title rows, as the real prepare step groups the employees file."""
    return prepare.prepare(
        inputs.employees, inputs.abbreviations, naming.family_codes(inputs.catalog)
    )


@pytest.fixture(scope="module")
def true_roles(rows, truth) -> dict[TitleKey, set[str]]:
    """The true roles the raw spellings of each row name, joined here from the
    truth file: raw title and family code as recorded (Q09)."""
    recorded = {(t.job_title, t.job_family_code.strip()): t.true_role_id for t in truth}
    return {
        row.key: {
            recorded.get((raw, row.key.family_code), "no truth row") for raw in row.raw_titles
        }
        for row in rows
    }


@pytest.fixture(scope="module")
def wave1(rows) -> list[TitleRow]:
    """The Wave 1 list (Q24): the 40 rows with the highest headcount."""
    ordered = sorted(rows, key=lambda row: (-row.headcount, row.key))
    return ordered[: Settings().wave1_size]


@pytest.fixture(scope="module")
def models(inputs: InputData) -> Models:
    return switch.build_models(Settings(), inputs.catalog.ladder, inputs.abbreviations)


# --- The committed files (Q06, S01) --------------------------------------------------------------


def test_the_data_folder_holds_exactly_the_seven_files_of_the_design(data_dir: Path):
    assert sorted(path.name for path in data_dir.iterdir()) == SEVEN_FILES


def test_regenerating_gives_the_committed_files_byte_for_byte(data_dir: Path, tmp_path: Path):
    mockdata.generate(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == SEVEN_FILES
    for name in SEVEN_FILES:
        assert (tmp_path / name).read_bytes() == (data_dir / name).read_bytes(), name


@pytest.mark.parametrize("hash_seed", ["0", "12345"])
def test_another_process_with_another_hash_seed_writes_the_same_bytes(
    hash_seed, data_dir: Path, tmp_path: Path, repo_root: Path
):
    script = "import sys; from titlestd import mockdata; mockdata.generate(sys.argv[1])"
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
        check=True,
        timeout=120,
    )
    for name in SEVEN_FILES:
        assert (tmp_path / name).read_bytes() == (data_dir / name).read_bytes(), name


def test_the_command_writes_the_same_bytes_with_nothing_installed(
    bare_cli, data_dir: Path, tmp_path: Path
):
    # "python3 -m titlestd generate-data" on a bare interpreter, with another
    # run seed: the data does not depend on it (Q06).
    result = bare_cli("generate-data", "--data-dir", str(tmp_path), "--seed", "7")
    assert result.returncode == 0, result.stderr
    assert result.stderr == "" and str(tmp_path) in result.stdout
    for name in SEVEN_FILES:
        assert (tmp_path / name).read_bytes() == (data_dir / name).read_bytes(), name


def test_the_files_are_in_the_format_of_the_shared_writers(
    data_dir: Path, tmp_path: Path, inputs: InputData, truth
):
    # S01: UTF-8 CSV, a header row, quoted only where needed, lines ending in a line feed.
    datafiles.write_inputs(tmp_path, inputs)
    people.write_truth(tmp_path, truth)
    for name in SEVEN_FILES:
        content = (data_dir / name).read_bytes()
        assert (tmp_path / name).read_bytes() == content, name
        assert content.endswith(b"\n") and b"\r" not in content, name
        content.decode("ascii")  # the mock data needs no character beyond ASCII


def test_the_cross_file_checks_of_the_readers_pass(inputs: InputData, truth):
    datafiles.validate_inputs(inputs)
    people.validate_truth(truth, inputs.catalog)


# --- Sizes (2.1, Q09, Q35) -----------------------------------------------------------------------


def test_about_1500_employees_on_about_400_title_rows(inputs: InputData, rows):
    assert 1350 <= len(inputs.employees) <= 1650  # Q35: plus or minus 10%
    assert 360 <= len(rows) <= 440  # Q09: "about 400" counts cleaned decision rows
    raw_titles = {(e.job_title, e.job_family_code) for e in inputs.employees}
    assert len(raw_titles) > len(rows) * 1.2  # with more raw spellings than that
    assert sum(row.headcount for row in rows) == len(inputs.employees)
    ids = [employee.employee_id for employee in inputs.employees]
    assert len(set(ids)) == len(ids)


def test_eight_families_of_six_roles_and_one_generic_role_each(inputs: InputData):
    catalog = inputs.catalog
    families = naming.family_codes(catalog)
    assert len(families) == 8
    assert len(naming.model_visible_roles(catalog)) == 48
    assert len([role for role in catalog.roles if role.is_generic]) == 8
    for family in families:
        assert len(naming.roles_in_family(catalog, family)) == 6
        assert naming.generic_role(catalog, family).family_code == family
    assert len({role.role_name for role in catalog.roles}) == len(catalog.roles)
    assert all(role.definition.strip() for role in catalog.roles)


def test_the_three_flagged_roles_sit_in_three_different_families(inputs: InputData):
    flagged = [role for role in inputs.catalog.roles if role.compliance_flag]
    assert len(flagged) == 3  # 2.1
    assert len({role.family_code for role in flagged}) == 3  # Q23
    assert not any(role.is_generic for role in flagged)


def test_the_grade_table_and_the_ladder_are_those_of_the_record(inputs: InputData):
    # Q36: grades 1-3 Junior, 4-6 Associate, 7-8 Senior, 9 Lead, 10 Principal.
    levels = {row.grade: row.level for row in inputs.catalog.grade_levels}
    assert levels == {
        1: "Junior",
        2: "Junior",
        3: "Junior",
        4: "Associate",
        5: "Associate",
        6: "Associate",
        7: "Senior",
        8: "Senior",
        9: "Lead",
        10: "Principal",
    }
    assert [(rung.level, rung.long_word, rung.short_word) for rung in inputs.catalog.ladder] == [
        ("Junior", "Junior", "Jr"),
        ("Associate", "Associate", "Assoc"),
        ("Senior", "Senior", "Sr"),
        ("Lead", "Lead", "Lead"),
        ("Principal", "Principal", "Prin"),
    ]


def test_the_worked_example_of_the_design_is_in_the_data(inputs: InputData, rows, true_roles):
    # 1.3: "ACCT MGR II" and "Senior Account Manager" are both Sales x Account
    # Manager, and grades 7 and 8 both give Senior Account Manager / Sr Acct Mgr.
    by_key = {row.key: row for row in rows}
    (role,) = [
        role
        for role in naming.roles_in_family(inputs.catalog, "SAL")
        if (role.role_name, role.short_name) == ("Account Manager", "Acct Mgr")
    ]
    for cleaned in ("account manager ii", "senior account manager"):
        key = TitleKey("SAL", cleaned)
        assert key in by_key and true_roles[key] == {role.role_id}
    for grade in (7, 8):
        named = naming.standard_title(inputs.catalog, role, grade)
        assert (named.long_title, named.short_title) == ("Senior Account Manager", "Sr Acct Mgr")


def test_about_thirty_abbreviations_each_a_single_lower_case_token(inputs: InputData):
    short_forms = [abbreviation.short_form for abbreviation in inputs.abbreviations]
    assert 27 <= len(short_forms) <= 33  # Q35: about 30
    assert len(set(short_forms)) == len(short_forms)
    assert all(re.fullmatch(r"[a-z]+", short) for short in short_forms)
    assert all(re.fullmatch(r"[a-z]+( [a-z]+)*", a.long_form) for a in inputs.abbreviations)


def test_a_short_title_cleans_to_the_same_row_as_its_long_title(inputs: InputData):
    # The catalog's short words are all in the abbreviations file, so a legacy
    # title written like a catalog Short Title is the row of the Long Title.
    catalog = inputs.catalog
    for role in catalog.roles:
        for grade in (None, 1, 4, 7, 9, 10):
            named = naming.standard_title(catalog, role, grade)
            long = prepare.clean_title(named.long_title, inputs.abbreviations)
            short = prepare.clean_title(named.short_title, inputs.abbreviations)
            assert long == short, (named.long_title, named.short_title)
            assert long.text == named.long_title.lower() and not long.has_person_name


# --- The truth and the seed look-up (Q03, Q09, Q13) ----------------------------------------------


def test_every_raw_title_has_exactly_one_truth_row(inputs: InputData, truth):
    # Q09: one row per distinct raw pair, exactly as in the employees file.
    # Q03: every title has a truth row, so nobody marks "no suitable role".
    recorded = Counter((t.job_title, t.job_family_code) for t in truth)
    assert set(recorded) == {(e.job_title, e.job_family_code) for e in inputs.employees}
    assert set(recorded.values()) == {1}


def test_every_title_row_has_one_true_role_across_its_spellings(inputs, rows, true_roles):
    by_id = {role.role_id: role for role in inputs.catalog.roles}
    assert sum(1 for row in rows if len(row.raw_titles) > 1) > 100
    for row in rows:
        assert len(true_roles[row.key]) == 1, row.key
        (role_id,) = true_roles[row.key]
        role = by_id[role_id]  # "no truth row" would fail here
        assert not role.is_generic, row.key
        if row.family_usable:
            assert role.family_code == row.key.family_code, row.key


def test_every_role_of_the_catalog_is_the_true_role_of_some_title(inputs, true_roles):
    held = {role_id for roles in true_roles.values() for role_id in roles}
    assert held == {role.role_id for role in naming.model_visible_roles(inputs.catalog)}


def test_the_seed_look_up_is_eight_long_tail_rows_correct_against_the_truth(
    inputs: InputData, rows, true_roles, wave1
):
    by_key = {row.key: row for row in rows}
    recorded = {(e.job_title, e.job_family_code) for e in inputs.employees}
    listed = {row.key for row in wave1}
    assert len(inputs.seed_lookup) == 8  # Q35
    keys = []
    for seed in inputs.seed_lookup:
        assert (seed.job_title, seed.job_family_code) in recorded  # the raw title, as recorded
        key = prepare.title_key(seed.job_title, seed.job_family_code, inputs.abbreviations)
        row = by_key[key]  # it cleans to a row of the employees file
        assert true_roles[key] == {seed.role_id}
        assert key not in listed  # Q13: in the long tail
        assert row.model_eligible  # no person's name, a usable family
        keys.append(key)
    assert len(set(keys)) == 8
    assert {key.family_code for key in keys} == set(naming.family_codes(inputs.catalog))


def test_the_simulated_people_give_every_row_its_true_role(inputs, rows, truth, true_roles):
    exact = people.SimulatedPeople(truth, inputs.catalog, Settings(sme_error_rate=0.0))
    for row in rows:
        label = exact.blind_label(row)
        assert (label.answer, {label.role_id}) == (HumanAnswer.ROLE, true_roles[row.key]), row.key


def test_nobody_marks_a_title_no_suitable_role_on_the_default_data(data_dir, inputs, rows):
    # Q03, through the loader the pipeline will use and the default settings.
    simulated = people.load_people(Settings(data_dir=data_dir), inputs.catalog)
    answers = Counter()
    for row in rows:
        answers[simulated.wave1_approval(row).answer] += 1
        answers[simulated.blind_label(row).answer] += 1
        answers[simulated.queue_decision(row).answer] += 1
    assert answers[HumanAnswer.NO_SUITABLE_ROLE] == 0
    assert answers[HumanAnswer.ROLE] > 2.5 * len(rows)
    assert 0 < answers[HumanAnswer.UNDECIDED] < 0.2 * len(rows)  # Q01: about one in ten


# --- The Wave 1 list (2.1, Q24) ------------------------------------------------------------------


def test_the_fortieth_and_forty_first_rows_differ_in_headcount(rows, wave1):
    ordered = sorted(rows, key=lambda row: (-row.headcount, row.key))
    assert len(wave1) == 40
    assert ordered[39].headcount > ordered[40].headcount  # membership never rests on a tie


def test_the_forty_most_widely_held_titles_cover_about_forty_percent(inputs, wave1):
    share = sum(row.headcount for row in wave1) / len(inputs.employees)
    assert 0.35 <= share <= 0.45  # Q24


def test_the_wave_1_list_is_made_of_titles_a_model_may_see(wave1):
    assert all(row.model_eligible for row in wave1)


# --- The deliberate mess (2.3, Q14, Q26, Q35) ----------------------------------------------------


def test_five_title_rows_hold_a_persons_name(rows):
    named = [row for row in rows if row.has_person_name]
    assert len(named) == 5  # Q35
    for row in named:
        assert NAME_MASK in row.key.cleaned_title
        assert any(words(raw) & NAMES for raw in row.raw_titles)
        assert not row.model_eligible


def test_a_listed_name_stands_only_where_the_data_means_one(inputs: InputData, rows):
    # Q14: outside the five titles, no title holds a word of the name lists, so
    # no other row is keyed on a mask by accident.
    for row in rows:
        if not row.has_person_name:
            assert NAME_MASK not in row.key.cleaned_title
            assert not any(words(raw) & NAMES for raw in row.raw_titles), row.key
    assert not any(words(e.department) & NAMES for e in inputs.employees)


def test_no_record_shows_a_name_or_an_email_address(rows):
    with_name = with_email = 0
    for row in rows:
        record = row.record
        shown = " ".join([record.cleaned_title, record.department, record.description])
        assert not words(shown) & NAMES, row.key
        assert "@" not in shown, row.key
        with_name += NAME_MASK in record.description
        with_email += EMAIL_MASK in record.description
    assert with_name >= 3 and with_email >= 3  # the masks have something to do


def test_every_email_address_is_on_the_reserved_domain(inputs: InputData):
    addresses = [
        address
        for employee in inputs.employees
        for field in (employee.job_title, employee.department, employee.job_description)
        for address in re.findall(r"\S+@\S+", field)
    ]
    assert len(addresses) >= 3
    for address in addresses:
        assert address.rstrip(".").endswith("@" + reference.EMAIL_DOMAIN), address


def test_five_employees_on_three_rows_have_no_usable_family_code(inputs: InputData, rows):
    unfiled = [row for row in rows if not row.family_usable]
    assert len(unfiled) == 3 and sum(row.headcount for row in unfiled) == 5  # Q26, Q35
    codes = {row.key.family_code for row in unfiled}
    assert "" in codes and len(codes - {""}) >= 1  # blank, and unknown
    for code in codes:
        assert not naming.is_usable_family(inputs.catalog, code)
    for row in unfiled:
        assert not row.has_person_name and not row.model_eligible
    recorded = {employee.job_family_code for employee in inputs.employees}
    assert recorded - codes == set(naming.family_codes(inputs.catalog))


def test_about_fifteen_percent_of_the_rows_have_no_description(rows):
    without = sum(1 for row in rows if not row.record.description)
    assert 0.135 <= without / len(rows) <= 0.165  # Q35: about 15%, plus or minus 10%


def test_a_few_rows_have_differing_or_partly_empty_descriptions(inputs: InputData, rows):
    # Q11: a handful of rows exercise the rule for the record of a row.
    description = {e.employee_id: e.job_description for e in inputs.employees}
    partly_empty = differing = 0
    for row in rows:
        held = Counter(description[employee_id] for employee_id in row.employee_ids)
        texts = [text for text in held if text]
        if len(held) == 1:
            continue
        partly_empty += len(texts) == 1
        differing += len(texts) > 1
        if len(texts) > 1 and len(set(held.values())) == 1:
            # A tie: the description held by the lowest employee_id is the record's.
            assert row.record.description == prepare.mask_text(description[row.employee_ids[0]])
        else:
            assert row.record.description == prepare.mask_text(max(texts, key=held.__getitem__))
    assert 3 <= partly_empty <= 12 and 3 <= differing <= 12


def test_the_records_do_not_depend_on_the_order_of_the_employees_file(inputs: InputData, rows):
    again = prepare.prepare(
        inputs.employees[::-1], inputs.abbreviations, naming.family_codes(inputs.catalog)
    )
    assert again == rows


def test_about_twenty_employees_have_no_grade(inputs: InputData):
    without = [employee for employee in inputs.employees if employee.grade is None]
    assert 18 <= len(without) <= 22  # Q35
    graded = {employee.grade for employee in inputs.employees} - {None}
    assert graded == {row.grade for row in inputs.catalog.grade_levels} == set(range(1, 11))
    assert all(naming.level_for_grade(inputs.catalog, grade) for grade in graded)


def test_the_titles_are_messy_in_every_way_the_design_lists(inputs: InputData):
    titles = sorted({employee.job_title for employee in inputs.employees})
    short_forms = {abbreviation.short_form for abbreviation in inputs.abbreviations}

    def share(test) -> float:
        return sum(1 for title in titles if test(title)) / len(titles)

    assert share(lambda title: bool(words(title) & short_forms)) > 0.2  # abbreviations
    assert share(str.isupper) > 0.05 and share(str.islower) > 0.03  # mixed case
    assert share(lambda title: not title.isupper() and not title.islower()) > 0.5
    assert share(lambda title: bool(words(title) & LOCATIONS)) > 0.1  # location suffixes
    assert share(lambda title: bool(REQUISITION.search(title))) > 0.05  # requisition suffixes
    assert any("REQ-" in title for title in titles) and any("#" in title for title in titles)
    used = {place for title in titles for place in words(title) & LOCATIONS}
    assert used == LOCATIONS
    # Level words in several forms: long, short, dotted, roman, digit, behind the title.
    for form in (r"^Senior ", r"^Sr ", r"^Sr\. ", r" II\b", r" III\b", r" 2\b", r", Senior\b"):
        assert any(re.search(form, title) for title in titles), form
    for word in ("junior", "associate", "senior", "lead", "principal"):
        assert any(title.lower().startswith(word + " ") for title in titles), word


def test_every_location_and_requisition_number_is_cleaned_away(rows):
    for row in rows:
        cleaned = row.key.cleaned_title
        assert not words(cleaned) & LOCATIONS, row.key
        assert not REQUISITION.search(cleaned) and not re.search(r"\d{4}", cleaned), row.key
        assert cleaned.replace(NAME_MASK, "") == cleaned.replace(NAME_MASK, "").lower()


def test_the_reference_lists_share_no_word_with_the_data(inputs: InputData):
    # Q14: names and locations share no word with role names, abbreviations,
    # level words or departments, so cleaning and masking never eat a real word.
    catalog = inputs.catalog
    texts = {
        "role names": [role.role_name for role in catalog.roles],
        "short names": [role.short_name for role in catalog.roles],
        "definitions": [role.definition for role in catalog.roles],
        "family codes": [role.family_code for role in catalog.roles],
        "abbreviations": [
            form for a in inputs.abbreviations for form in (a.short_form, a.long_form)
        ],
        "level words": [
            word
            for rung in catalog.ladder
            for word in (rung.level, rung.long_word, rung.short_word)
        ],
        "departments": [employee.department for employee in inputs.employees],
    }
    for name, strings in texts.items():
        used = set().union(*(words(text) for text in strings))
        assert used, name
        assert not used & NAMES, (name, sorted(used & NAMES))
        assert not used & LOCATIONS, (name, sorted(used & LOCATIONS))
    assert all(employee.department.strip() for employee in inputs.employees)


# --- Titles built for the stand-in LLM (Q17, Q35) ------------------------------------------------


def numbered(inputs: InputData) -> list[NumberedRole]:
    """Every role a model may be shown, numbered: more than any shortlist of ten."""
    visible = naming.model_visible_roles(inputs.catalog)
    return [NumberedRole(number, role) for number, role in enumerate(visible, start=1)]


def rows_titled(rows, titles) -> list[TitleRow]:
    found = [row for row in rows if {title for title, *_ in titles} & set(row.raw_titles)]
    assert len(found) == len(titles)
    return found


def test_about_four_titles_are_built_to_draw_no_match(inputs: InputData, rows, models):
    # Worded like no role: shown the whole catalog the stand-in finds no match,
    # so it finds none in whichever ten roles the ML model shortlists.
    built = rows_titled(rows, mockdata.NO_MATCH_TITLES)
    assert 3 <= len(built) <= 5  # Q35: about 4
    for row in built:
        assert row.model_eligible and row.record.description
        answer = models.llm.chooser.choose(row.record, numbered(inputs))
        assert answer.answer is ChooseKind.NO_MATCH, row.key
        assert answer.quote == row.record.cleaned_title


def test_about_four_bare_titles_draw_insufficient_data_and_no_other_title_does(
    inputs: InputData, rows, models
):
    built = rows_titled(rows, mockdata.BARE_TITLES)
    assert 3 <= len(built) <= 5  # Q35: about 4
    drawn = [
        row.key
        for row in rows
        if row.model_eligible
        and models.llm.chooser.choose(row.record, numbered(inputs)).answer
        is ChooseKind.INSUFFICIENT_DATA
    ]
    assert drawn == [row.key for row in built]
    for row in built:
        assert len(row.key.cleaned_title.split()) == 1 and not row.record.description


def test_a_vague_description_gives_the_stand_in_no_word_of_any_role(inputs: InputData, models):
    # What the description knob promises: "vague" says nothing about the job,
    # so such a row is placed by its title or not at all.
    content_words = models.llm.chooser.content_words
    role_words = set().union(
        *(
            content_words(f"{role.role_name} {role.definition}")
            for role in naming.model_visible_roles(inputs.catalog)
        )
    )
    for description in mockdata._VAGUE:
        assert content_words(description) and not content_words(description) & role_words


def test_the_special_titles_sit_in_the_long_tail(rows, wave1):
    special = {title for title, *_ in mockdata.SPECIAL_TITLES}
    listed = {raw for row in wave1 for raw in row.raw_titles}
    assert not special & listed
    assert len(rows_titled(rows, mockdata.SPECIAL_TITLES)) == 16


# --- The two models can read the data ------------------------------------------------------------


def test_the_stand_in_llm_knows_a_title_that_is_a_roles_name(inputs, rows, true_roles, models):
    # The plainest rows, a role's own name: the stand-in picks that role from the
    # whole catalog, whatever the description says.
    by_key = {row.key: row for row in rows}
    shown = numbered(inputs)
    for role in naming.model_visible_roles(inputs.catalog):
        row = by_key[TitleKey(role.family_code, role.role_name.lower())]
        assert true_roles[row.key] == {role.role_id}
        answer = models.llm.chooser.choose(row.record, shown)
        assert answer.answer is ChooseKind.ROLE, row.key
        assert shown[answer.number - 1].role == role, row.key


def test_the_offline_embedder_puts_a_roles_own_name_nearest_that_role(inputs, models):
    # The other opinion, by another calculation (Q12): cosine of hashed
    # character n-grams between a title and each role's name and definition.
    # Not every time: "data analyst" lies nearer the privacy role, whose
    # definition says "data" twice. Such titles are what the agent is for.
    roles = naming.model_visible_roles(inputs.catalog)
    role_vectors = models.embedder.embed([f"{r.role_name}. {r.definition}" for r in roles])
    titles = [prepare.clean_title(role.role_name, inputs.abbreviations).text for role in roles]
    right = 0
    for role, query in zip(roles, models.embedder.embed(titles), strict=True):
        scores = [
            math.fsum(weight * vector.get(index, 0.0) for index, weight in query.items())
            for vector in role_vectors
        ]
        right += roles[max(range(len(roles)), key=scores.__getitem__)] == role
    assert right >= 0.9 * len(roles)


# --- The mini data set ---------------------------------------------------------------------------


def test_the_ladder_and_the_grade_table_are_those_of_the_mini_data_set(inputs, mini_inputs):
    assert inputs.catalog.ladder == mini_inputs.catalog.ladder
    assert inputs.catalog.grade_levels == mini_inputs.catalog.grade_levels


def test_the_abbreviations_clean_the_mini_data_set_to_its_hand_written_rows(
    inputs: InputData, mini_inputs: InputData, mini_rows
):
    assert set(mini_inputs.abbreviations) <= set(inputs.abbreviations)
    families = naming.family_codes(mini_inputs.catalog)
    assert prepare.prepare(mini_inputs.employees, inputs.abbreviations, families) == mini_rows


def test_the_mini_catalog_is_a_part_of_the_catalog(inputs: InputData, mini_inputs: InputData):
    by_id = {role.role_id: role for role in inputs.catalog.roles}
    for mini_role in mini_inputs.catalog.roles:
        role = by_id[mini_role.role_id]
        assert (role.family_code, role.role_name, role.short_name, role.is_generic) == (
            mini_role.family_code,
            mini_role.role_name,
            mini_role.short_name,
            mini_role.is_generic,
        )


def test_the_mini_truth_and_seed_look_up_hold_against_the_catalog(
    inputs: InputData, mini_inputs: InputData, mini_dir: Path, mini_rows
):
    # Whoever swaps the mini catalog for this one, as a later test might, keeps
    # the people and the look-up of the mini data set.
    mini_truth = people.read_truth(mini_dir)
    people.validate_truth(mini_truth, inputs.catalog)
    datafiles.check_decided_roles(
        datafiles.SEED_LOOKUP_FILE,
        inputs.catalog,
        ((s.job_title, s.job_family_code, s.role_id) for s in mini_inputs.seed_lookup),
    )
    on_mini = people.SimulatedPeople(mini_truth, mini_inputs.catalog, Settings(sme_error_rate=0.0))
    on_default = people.SimulatedPeople(mini_truth, inputs.catalog, Settings(sme_error_rate=0.0))
    for row in mini_rows:
        assert on_default.blind_label(row) == on_mini.blind_label(row), row.key
