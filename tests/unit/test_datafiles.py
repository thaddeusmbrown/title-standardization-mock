"""Readers, writers and checks for the six input files (design 2.3)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from titlestd import datafiles
from titlestd.contracts import CatalogRole, Employee, InputData, SeedLookupRow
from titlestd.datafiles import DataError

# The column names of DESIGN.md 2.3, copied from the design.
DESIGN_COLUMNS = {
    datafiles.ROLES_FILE: (
        "role_id, family_code, role_name, short_name, definition, compliance_flag, is_generic"
    ),
    datafiles.LADDER_FILE: "level, long_word, short_word",
    datafiles.GRADES_FILE: "grade, level",
    datafiles.ABBREVIATIONS_FILE: "short_form, long_form",
    datafiles.EMPLOYEES_FILE: (
        "employee_id, job_title, job_family_code, department, grade, job_description"
    ),
    datafiles.SEED_LOOKUP_FILE: "job_title, job_family_code, role_id",
}
COLUMNS = {
    datafiles.ROLES_FILE: datafiles.ROLES_COLUMNS,
    datafiles.LADDER_FILE: datafiles.LADDER_COLUMNS,
    datafiles.GRADES_FILE: datafiles.GRADES_COLUMNS,
    datafiles.ABBREVIATIONS_FILE: datafiles.ABBREVIATIONS_COLUMNS,
    datafiles.EMPLOYEES_FILE: datafiles.EMPLOYEES_COLUMNS,
    datafiles.SEED_LOOKUP_FILE: datafiles.SEED_LOOKUP_COLUMNS,
}


@pytest.fixture
def data_dir(tmp_path: Path, mini_dir: Path) -> Path:
    """A private copy of the mini data set that a test may damage."""
    target = tmp_path / "data"
    shutil.copytree(mini_dir, target)
    return target


def _rewrite(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_columns_are_exactly_the_ones_the_design_lists():
    assert set(COLUMNS) == set(datafiles.INPUT_FILES) and len(datafiles.INPUT_FILES) == 6
    for name, columns in COLUMNS.items():
        assert ", ".join(columns) == DESIGN_COLUMNS[name]


def test_reading_the_mini_data_set(mini_inputs: InputData):
    assert len(mini_inputs.catalog.roles) == 14
    assert [level.level for level in mini_inputs.catalog.ladder] == [
        "Junior",
        "Associate",
        "Senior",
        "Lead",
        "Principal",
    ]
    assert [grade.grade for grade in mini_inputs.catalog.grade_levels] == list(range(1, 11))
    assert len(mini_inputs.abbreviations) == 12
    assert len(mini_inputs.employees) == 22
    assert len(mini_inputs.seed_lookup) == 2


def test_fields_are_typed(mini_inputs: InputData):
    by_id = {role.role_id: role for role in mini_inputs.catalog.roles}
    assert by_id["FIN-04"].compliance_flag is True and by_id["FIN-04"].is_generic is False
    assert by_id["SAL-GEN"].is_generic is True and by_id["SAL-GEN"].compliance_flag is False
    employees = {employee.employee_id: employee for employee in mini_inputs.employees}
    assert employees["E001"].grade == 8
    assert employees["E012"].grade is None  # a missing grade
    assert employees["E012"].job_description == ""  # an empty description
    assert employees["E021"].job_family_code == ""  # a blank family code


def test_writing_what_was_read_gives_the_same_bytes(
    tmp_path: Path, mini_dir: Path, mini_inputs: InputData
):
    datafiles.write_inputs(tmp_path, mini_inputs)
    for name in datafiles.INPUT_FILES:
        assert (tmp_path / name).read_bytes() == (mini_dir / name).read_bytes(), name
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(datafiles.INPUT_FILES)


def test_files_end_lines_with_a_line_feed_only(tmp_path: Path, mini_inputs: InputData):
    datafiles.write_inputs(tmp_path, mini_inputs)
    for name in datafiles.INPUT_FILES:
        content = (tmp_path / name).read_bytes()
        assert b"\r" not in content and content.endswith(b"\n")


@pytest.mark.parametrize(
    "job_title",
    [
        "  Sr. Acct Mgr - NYC  ",
        'Analyst, "Special" Projects',
        "Ingénieur Commercial\tII",
        "Title with\nline break",
        "#20311",
    ],
)
def test_a_job_title_survives_byte_for_byte(tmp_path: Path, job_title: str):
    employee = Employee("E1", job_title, " SAL ", "Dept", None, "")
    datafiles.write_employees(tmp_path, [employee])
    assert datafiles.read_employees(tmp_path) == (employee,)


def test_blank_lines_are_skipped(data_dir: Path):
    path = data_dir / datafiles.GRADES_FILE
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(datafiles.read_grade_levels(data_dir)) == 10


def test_read_table_and_write_table_round_trip(tmp_path: Path):
    rows = [{"a": "1", "b": "x,y"}, {"a": "", "b": 'say "hi"'}]
    datafiles.write_table(tmp_path / "deep" / "t.csv", ("a", "b"), rows)
    assert datafiles.read_table(tmp_path / "deep" / "t.csv", ("a", "b")) == rows


def test_write_table_refuses_a_row_with_other_columns(tmp_path: Path):
    with pytest.raises(DataError, match=r"t\.csv"):
        datafiles.write_table(tmp_path / "t.csv", ("a", "b"), [{"a": "1"}])


# --- Malformed files -------------------------------------------------------------


def test_a_missing_file_is_named(tmp_path: Path):
    with pytest.raises(DataError, match=datafiles.ROLES_FILE):
        datafiles.read_catalog_roles(tmp_path)


def test_a_folder_in_place_of_a_file_is_reported(tmp_path: Path):
    (tmp_path / datafiles.ROLES_FILE).mkdir()
    with pytest.raises(DataError, match="cannot be read"):
        datafiles.read_catalog_roles(tmp_path)
    with pytest.raises(DataError, match="cannot be read"):
        datafiles.fingerprint_files([tmp_path / datafiles.ROLES_FILE])


def test_an_empty_file_is_refused(data_dir: Path):
    (data_dir / datafiles.LADDER_FILE).write_text("", encoding="utf-8")
    with pytest.raises(DataError, match="header"):
        datafiles.read_level_ladder(data_dir)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("employee_id,job_title,", "employee_id,title,"),  # a renamed column
        (",grade,job_description", ",job_description"),  # a missing column
        (",grade,job_description", ",grade,job_description,salary"),  # an extra column
        ("employee_id,job_title,", "job_title,employee_id,"),  # another order
    ],
)
def test_the_header_must_match_exactly(data_dir: Path, old: str, new: str):
    _rewrite(data_dir / datafiles.EMPLOYEES_FILE, old, new)
    with pytest.raises(DataError) as error:
        datafiles.read_employees(data_dir)
    assert datafiles.EMPLOYEES_FILE in str(error.value)
    assert "employee_id,job_title,job_family_code" in str(error.value)


def test_a_short_row_is_reported_with_its_line(data_dir: Path):
    _rewrite(data_dir / datafiles.GRADES_FILE, "3,Junior", "3")
    with pytest.raises(DataError, match=r"grade_levels\.csv line 4: 1 fields, expected 2"):
        datafiles.read_grade_levels(data_dir)


def test_a_long_row_is_refused(data_dir: Path):
    _rewrite(data_dir / datafiles.GRADES_FILE, "3,Junior", "3,Junior,extra")
    with pytest.raises(DataError, match="3 fields, expected 2"):
        datafiles.read_grade_levels(data_dir)


@pytest.mark.parametrize("flag", ["yes", "", "true", "2"])
def test_a_flag_must_be_0_or_1(data_dir: Path, flag: str):
    _rewrite(data_dir / datafiles.ROLES_FILE, "audit committee.,1,0", f"audit committee.,{flag},0")
    with pytest.raises(DataError, match="compliance_flag must be 0 or 1"):
        datafiles.read_catalog_roles(data_dir)


@pytest.mark.parametrize("grade", ["eight", "-1", "7.5", " 8"])
def test_a_grade_must_be_a_whole_number_or_empty(data_dir: Path, grade: str):
    _rewrite(
        data_dir / datafiles.EMPLOYEES_FILE, "Enterprise Sales,8,", f"Enterprise Sales,{grade},"
    )
    with pytest.raises(DataError, match=r"employees\.csv line 2: grade"):
        datafiles.read_employees(data_dir)


def test_a_grade_table_entry_must_be_a_whole_number(data_dir: Path):
    _rewrite(data_dir / datafiles.GRADES_FILE, "10,Principal", ",Principal")
    with pytest.raises(DataError, match="grade must be a whole number"):
        datafiles.read_grade_levels(data_dir)


@pytest.mark.parametrize(
    ("name", "old", "new", "message"),
    [
        (datafiles.ROLES_FILE, "SAL-02,SAL,", "SAL-01,SAL,", "role_id SAL-01"),
        (datafiles.LADDER_FILE, "Lead,Lead,Lead", "Senior,Lead,Lead", "level Senior"),
        (datafiles.GRADES_FILE, "8,Senior", "7,Senior", "grade 7"),
        (datafiles.ABBREVIATIONS_FILE, "jr,junior", "sr,junior", "short_form sr"),
        (datafiles.EMPLOYEES_FILE, "E002,", "E001,", "employee_id E001"),
        (
            datafiles.SEED_LOOKUP_FILE,
            "AP Clerk,FIN,FIN-05",
            "Sales Engineer #20311,SAL,SAL-01",
            "Sales Engineer #20311",
        ),
    ],
)
def test_a_repeated_key_is_refused(data_dir: Path, name: str, old: str, new: str, message: str):
    _rewrite(data_dir / name, old, new)
    with pytest.raises(DataError) as error:
        datafiles.read_inputs(data_dir)
    assert name in str(error.value)
    assert message in str(error.value)
    assert "more than once" in str(error.value)


@pytest.mark.parametrize(
    ("name", "old", "new", "column"),
    [
        (
            datafiles.ROLES_FILE,
            "SAL-02,SAL,Sales Representative",
            ",SAL,Sales Representative",
            "role_id",
        ),
        (datafiles.ROLES_FILE, "SAL-02,SAL,", "SAL-02,,", "family_code"),
        (datafiles.ROLES_FILE, "Sales Representative,Sales Rep", ",Sales Rep", "role_name"),
        (
            datafiles.ROLES_FILE,
            "Sales Representative,Sales Rep",
            "Sales Representative, ",
            "short_name",
        ),
        (datafiles.LADDER_FILE, "Senior,Senior,Sr", "Senior,,Sr", "long_word"),
        (datafiles.GRADES_FILE, "1,Junior", "1,", "level"),
        (datafiles.ABBREVIATIONS_FILE, "jr,junior", "jr,", "long_form"),
        (datafiles.EMPLOYEES_FILE, "E012,BDR,", "E012,,", "job_title"),
        (datafiles.EMPLOYEES_FILE, "E012,BDR,", ",BDR,", "employee_id"),
        (datafiles.SEED_LOOKUP_FILE, "AP Clerk,FIN,FIN-05", "AP Clerk,FIN,", "role_id"),
    ],
)
def test_a_required_field_may_not_be_empty(data_dir: Path, name: str, old: str, new: str, column):
    _rewrite(data_dir / name, old, new)
    with pytest.raises(DataError, match=f"{column} is empty") as error:
        datafiles.read_inputs(data_dir)
    assert name in str(error.value)


def test_bytes_that_are_not_utf8_are_refused(data_dir: Path):
    path = data_dir / datafiles.ABBREVIATIONS_FILE
    path.write_bytes(path.read_bytes() + b"caf\xe9,coffee\n")
    with pytest.raises(DataError, match="UTF-8"):
        datafiles.read_abbreviations(data_dir)


# --- Checks across files -----------------------------------------------------------


def _with_roles(data: InputData, roles: tuple[CatalogRole, ...]) -> InputData:
    return dataclasses.replace(data, catalog=dataclasses.replace(data.catalog, roles=roles))


def test_the_mini_data_set_is_consistent(mini_inputs: InputData):
    datafiles.validate_inputs(mini_inputs)


def test_every_family_needs_exactly_one_generic_role(mini_inputs: InputData):
    roles = mini_inputs.catalog.roles
    without = tuple(role for role in roles if role.role_id != "FIN-GEN")
    with pytest.raises(DataError, match="family FIN must have exactly one generic role, found 0"):
        datafiles.validate_inputs(_with_roles(mini_inputs, without))
    extra = (*roles, CatalogRole("SAL-GEN2", "SAL", "Sales Other", "Sales Oth", "", False, True))
    with pytest.raises(DataError, match="family SAL must have exactly one generic role, found 2"):
        datafiles.validate_inputs(_with_roles(mini_inputs, extra))


def test_an_empty_catalog_is_refused(mini_inputs: InputData):
    with pytest.raises(DataError, match="no roles"):
        datafiles.validate_inputs(_with_roles(mini_inputs, ()))


def test_a_grade_must_name_a_level_of_the_ladder(data_dir: Path):
    _rewrite(data_dir / datafiles.GRADES_FILE, "9,Lead", "9,Director")
    with pytest.raises(DataError, match="grade 9 names level Director"):
        datafiles.read_inputs(data_dir)


@pytest.mark.parametrize(
    ("seed", "message"),
    [
        (SeedLookupRow("AP Clerk", "FIN", "FIN-99"), "not in the catalog"),
        (SeedLookupRow("AP Clerk", "FIN", "FIN-GEN"), "generic role"),
        (SeedLookupRow("AP Clerk", "FIN", "SAL-01"), "of family SAL"),
        (SeedLookupRow("AP Clerk", " FIN ", "SAL-01"), "of family SAL"),
    ],
)
def test_a_seed_role_must_be_a_real_role_of_the_family(mini_inputs, seed, message):
    data = dataclasses.replace(mini_inputs, seed_lookup=(seed,))
    with pytest.raises(DataError, match=message) as error:
        datafiles.validate_inputs(data)
    assert datafiles.SEED_LOOKUP_FILE in str(error.value)


@pytest.mark.parametrize("family", ["", "   ", "XXX"])
def test_a_seed_title_with_no_usable_family_may_name_any_role(mini_inputs, family):
    data = dataclasses.replace(
        mini_inputs, seed_lookup=(SeedLookupRow("Staff Accountant", family, "FIN-02"),)
    )
    datafiles.validate_inputs(data)


# --- Fingerprints --------------------------------------------------------------------


def test_the_fingerprint_is_stable_and_sees_every_byte(data_dir: Path, mini_dir: Path):
    original = datafiles.input_fingerprint(mini_dir)
    assert len(original) == 64
    assert datafiles.input_fingerprint(data_dir) == original  # a copy elsewhere: same value
    _rewrite(data_dir / datafiles.EMPLOYEES_FILE, "E022,Special Projects", "E022,Special projects")
    assert datafiles.input_fingerprint(data_dir) != original


def test_the_fingerprint_depends_on_file_names_and_order(tmp_path: Path):
    (tmp_path / "a.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("x\n", encoding="utf-8")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    assert datafiles.fingerprint_files([a]) != datafiles.fingerprint_files([b])
    assert datafiles.fingerprint_files([a, b]) != datafiles.fingerprint_files([b, a])


def test_the_fingerprint_of_a_missing_file_is_a_data_error(tmp_path: Path):
    with pytest.raises(DataError, match="not found"):
        datafiles.input_fingerprint(tmp_path)
