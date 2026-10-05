"""The truth file's reader, writer and checks, which live in titlestd.people."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from titlestd import datafiles, people
from titlestd.datafiles import DataError
from titlestd.people import TruthRow


@pytest.fixture
def data_dir(tmp_path: Path, mini_dir: Path) -> Path:
    target = tmp_path / "data"
    shutil.copytree(mini_dir, target)
    return target


def test_columns_are_the_ones_the_design_lists():
    assert ", ".join(people.TRUTH_COLUMNS) == "job_title, job_family_code, true_role_id"


def test_reading_the_mini_truth(mini_dir: Path):
    truth = people.read_truth(mini_dir)
    assert len(truth) == 17
    assert truth[0] == TruthRow("ACCT MGR II", "SAL", "SAL-01")
    assert TruthRow("Staff Accountant", "", "FIN-02") in truth  # keyed on the code as recorded


def test_writing_what_was_read_gives_the_same_bytes(tmp_path: Path, mini_dir: Path):
    people.write_truth(tmp_path, people.read_truth(mini_dir))
    assert (tmp_path / people.TRUTH_FILE).read_bytes() == (
        mini_dir / people.TRUTH_FILE
    ).read_bytes()


def test_one_row_per_title_within_a_family(data_dir: Path):
    path = data_dir / people.TRUTH_FILE
    path.write_text(path.read_text(encoding="utf-8") + "BDR,SAL,SAL-02\n", encoding="utf-8")
    with pytest.raises(DataError, match="'BDR' in family 'SAL' appears more than once"):
        people.read_truth(data_dir)


def test_the_same_title_in_another_family_is_another_row(data_dir: Path):
    path = data_dir / people.TRUTH_FILE
    path.write_text(path.read_text(encoding="utf-8") + "BDR,FIN,FIN-01\n", encoding="utf-8")
    assert len(people.read_truth(data_dir)) == 18


@pytest.mark.parametrize("line", ["BDR2,SAL,\n", ",SAL,SAL-01\n"])
def test_title_and_role_are_required(data_dir: Path, line: str):
    path = data_dir / people.TRUTH_FILE
    path.write_text(path.read_text(encoding="utf-8") + line, encoding="utf-8")
    with pytest.raises(DataError, match="row 18"):
        people.read_truth(data_dir)


def test_a_missing_truth_file_is_named(tmp_path: Path):
    with pytest.raises(DataError, match="not found"):
        people.read_truth(tmp_path)


def test_the_mini_truth_fits_the_catalog(mini_dir: Path, mini_catalog):
    people.validate_truth(people.read_truth(mini_dir), mini_catalog)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (TruthRow("BDR", "SAL", "SAL-99"), "not in the catalog"),
        (TruthRow("BDR", "SAL", "SAL-GEN"), "generic role"),
        (TruthRow("BDR", "SAL", "FIN-01"), "of family FIN"),
    ],
)
def test_a_true_role_is_a_real_role_of_the_family(mini_catalog, row, message):
    with pytest.raises(DataError, match=message):
        people.validate_truth([row], mini_catalog)


def test_a_title_with_no_usable_family_may_have_any_true_role(mini_catalog):
    people.validate_truth(
        [TruthRow("Staff Accountant", "", "FIN-02"), TruthRow("Odd Job", "ZZZ", "SAL-03")],
        mini_catalog,
    )


def test_the_truth_fingerprint_follows_the_file(data_dir: Path, mini_dir: Path):
    original = people.truth_fingerprint(mini_dir)
    assert people.truth_fingerprint(data_dir) == original
    assert original != datafiles.input_fingerprint(mini_dir)
    path = data_dir / people.TRUTH_FILE
    changed = path.read_text(encoding="utf-8").replace("BDR,SAL,SAL-05", "BDR,SAL,SAL-02")
    path.write_text(changed, encoding="utf-8")
    assert people.truth_fingerprint(data_dir) != original


def test_personas_are_role_labels_marked_as_simulated():
    personas = [
        *people.PERSONA_SMES,
        people.PERSONA_REVIEWER,
        people.PERSONA_CATALOG_OWNER,
        people.PERSONA_HR_LEAD,
        people.PERSONA_CIO,
    ]
    assert personas == [
        "SME 1 (simulated)",
        "SME 2 (simulated)",
        "SME 3 (simulated)",
        "Reviewer (simulated)",
        "Catalog owner (simulated)",
        "HR lead (simulated)",
        "CIO (simulated)",
    ]
