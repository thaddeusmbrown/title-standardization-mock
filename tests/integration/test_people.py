"""Simulated people on the mini data set, with the real readers and settings.

The people are loaded from the truth file on disk (titlestd.people.load_people)
against the catalog read by titlestd.datafiles, with settings built by
titlestd.settings, and asked about the hand-written title rows of the mini
data set (docs/build-decisions.md Q01, Q03, Q09, Q27, S05). Two tests show
which permuted truth the people accept, for test (b) of Q22 in stage 4.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from titlestd import people
from titlestd.contracts import HumanAnswer, Record, TitleKey, TitleRow
from titlestd.datafiles import DataError
from titlestd.settings import Settings, load_settings

OCCASIONS = ("wave1_approval", "blind_label", "queue_decision")


@pytest.fixture
def data_dir(tmp_path: Path, mini_dir: Path) -> Path:
    """A private copy of the mini data set that a test may change."""
    target = tmp_path / "data"
    shutil.copytree(mini_dir, target)
    return target


def _rewrite_truth(data_dir: Path, old: str, new: str) -> None:
    path = data_dir / people.TRUTH_FILE
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _true_roles(mini_dir: Path, rows) -> dict[TitleKey, str]:
    """Each row's true role, joined here from the truth file: raw title and
    trimmed family code (Q09, S05)."""
    recorded = {
        (truth.job_title, truth.job_family_code.strip()): truth.true_role_id
        for truth in people.read_truth(mini_dir)
    }
    roles = {}
    for row in rows:
        (roles[row.key],) = {recorded[(raw, row.key.family_code)] for raw in row.raw_titles}
    return roles


def test_without_errors_the_people_give_every_mini_row_its_true_role(
    mini_dir, mini_inputs, mini_rows, mini_settings
):
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0, undecided_rate=0.0)
    simulated = people.load_people(settings, mini_inputs.catalog)
    expected = _true_roles(mini_dir, mini_rows)
    assert len(expected) == 15
    for occasion in OCCASIONS:
        decisions = [getattr(simulated, occasion)(row) for row in mini_rows]
        assert all(d.answer is HumanAnswer.ROLE and d.simulated for d in decisions)
        assert {d.key: d.role_id for d in decisions} == expected


def test_the_padded_and_the_blank_family_code_meet_their_truth(
    mini_inputs, mini_rows, mini_settings
):
    # "FIN " on one employee and its truth row; a blank code on another (S05, Q26).
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0, undecided_rate=0.0)
    simulated = people.load_people(settings, mini_inputs.catalog)
    labels = {row.key: simulated.blind_label(row).role_id for row in mini_rows}
    assert labels[TitleKey("FIN", "treasury analyst")] == "FIN-06"
    assert labels[TitleKey("", "staff accountant")] == "FIN-02"


def test_at_the_default_rates_people_only_name_real_roles_of_the_catalog(
    mini_inputs, mini_rows, mini_settings
):
    simulated = people.load_people(mini_settings, mini_inputs.catalog)
    nameable = {role.role_id for role in mini_inputs.catalog.roles if not role.is_generic}
    for row in mini_rows:
        for occasion in OCCASIONS:
            decision = getattr(simulated, occasion)(row)
            assert decision.answer is not HumanAnswer.NO_SUITABLE_ROLE  # every row has a truth
            if occasion != "queue_decision":
                assert decision.answer is HumanAnswer.ROLE
            if decision.answer is HumanAnswer.ROLE:
                assert decision.role_id in nameable


def test_the_rates_come_from_the_settings_in_force(tmp_path, mini_dir, mini_inputs, mini_rows):
    config = tmp_path / "settings.toml"
    config.write_text("sme_error_rate = 1.0\nundecided_rate = 0.0\n", encoding="utf-8")
    all_wrong = people.load_people(
        load_settings(config, {"data_dir": mini_dir}), mini_inputs.catalog
    )
    all_undecided = people.load_people(
        load_settings(config, {"data_dir": mini_dir, "undecided_rate": 1.0}), mini_inputs.catalog
    )
    true_roles = _true_roles(mini_dir, mini_rows)
    for row in mini_rows:
        assert all_wrong.queue_decision(row).role_id not in (None, true_roles[row.key])
        assert all_undecided.queue_decision(row).answer is HumanAnswer.UNDECIDED
        assert all_undecided.blind_label(row) == all_wrong.blind_label(row)


def test_removing_a_truth_row_marks_that_title_and_no_other(
    data_dir, mini_inputs, mini_rows, mini_settings
):
    # Q03: showing "no suitable role" is a data change, the removal of a truth row.
    _rewrite_truth(data_dir, "BDR,SAL,SAL-05\n", "")
    whole = people.load_people(mini_settings, mini_inputs.catalog)
    cut = people.load_people(
        dataclasses.replace(mini_settings, data_dir=data_dir), mini_inputs.catalog
    )
    marked = []
    for row in mini_rows:
        for occasion in OCCASIONS:
            decision = getattr(cut, occasion)(row)
            if decision.answer is HumanAnswer.NO_SUITABLE_ROLE:
                marked.append((occasion, row.key))
                assert decision.role_id is None
            else:
                assert decision == getattr(whole, occasion)(row)
    assert marked == [(occasion, TitleKey("SAL", "bdr")) for occasion in OCCASIONS]


def test_a_row_keeps_its_role_while_one_of_its_raw_titles_has_a_truth_row(
    data_dir, mini_inputs, mini_rows, mini_settings
):
    _rewrite_truth(data_dir, "Sr. Acct Mgr - NYC,SAL,SAL-01\n", "")
    settings = dataclasses.replace(mini_settings, data_dir=data_dir, sme_error_rate=0.0)
    simulated = people.load_people(settings, mini_inputs.catalog)
    (row,) = [row for row in mini_rows if "Sr. Acct Mgr - NYC" in row.raw_titles]
    assert simulated.wave1_approval(row).role_id == "SAL-01"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("BDR,SAL,SAL-05", "BDR,SAL,SAL-GEN", "generic role SAL-GEN"),
        ("BDR,SAL,SAL-05", "BDR,SAL,SAL-77", "SAL-77, which is not in the catalog"),
        ("BDR,SAL,SAL-05", "BDR,SAL,FIN-01", "FIN-01 of family FIN"),
        ("BDR,SAL,SAL-05", "BDR,SAL,SAL-05\nBDR,SAL ,SAL-02", "'BDR' in family 'SAL' has two"),
        ("BDR,SAL,SAL-05", "BDR,SAL,SAL-05\nBDR,SAL,SAL-05", "appears more than once"),
    ],
)
def test_a_truth_file_that_is_wrong_stops_the_load(
    data_dir, mini_inputs, mini_settings, old, new, message
):
    _rewrite_truth(data_dir, old, new)
    settings = dataclasses.replace(mini_settings, data_dir=data_dir)
    with pytest.raises(DataError, match=message):
        people.load_people(settings, mini_inputs.catalog)


def test_a_missing_truth_file_stops_the_load(tmp_path, mini_inputs):
    with pytest.raises(DataError, match="not found"):
        people.load_people(Settings(data_dir=tmp_path), mini_inputs.catalog)


def test_two_spellings_of_one_row_with_different_true_roles_stop_the_run(
    data_dir, mini_inputs, mini_rows, mini_settings
):
    # Q09 promises one true role per cleaned row; the people do not pick one.
    _rewrite_truth(data_dir, "Sr. Acct Mgr - NYC,SAL,SAL-01", "Sr. Acct Mgr - NYC,SAL,SAL-06")
    settings = dataclasses.replace(mini_settings, data_dir=data_dir)
    simulated = people.load_people(settings, mini_inputs.catalog)
    (row,) = [row for row in mini_rows if "Sr. Acct Mgr - NYC" in row.raw_titles]
    with pytest.raises(DataError, match="'senior account manager' in family 'SAL'"):
        simulated.wave1_approval(row)


# --- A truth permuted on purpose, as test (b) of Q22 needs one in stage 4 -------


def test_one_mapping_of_role_ids_per_family_gives_a_truth_the_people_accept(
    data_dir, mini_dir, mini_inputs, mini_rows, mini_settings
):
    # Every truth row that named role A now names role B of the same family. So
    # each title row keeps one true role across its spellings (Q09), each role
    # stays in its family, and nothing stops.
    by_family: dict[str, list[str]] = {}
    for role in mini_inputs.catalog.roles:
        if not role.is_generic:
            by_family.setdefault(role.family_code, []).append(role.role_id)
    mapping: dict[str, str] = {}
    for role_ids in by_family.values():
        mapping.update(zip(role_ids, role_ids[1:] + role_ids[:1], strict=True))
    people.write_truth(
        data_dir,
        [
            dataclasses.replace(truth, true_role_id=mapping[truth.true_role_id])
            for truth in people.read_truth(mini_dir)
        ],
    )
    settings = dataclasses.replace(
        mini_settings, data_dir=data_dir, sme_error_rate=0.0, undecided_rate=0.0
    )
    simulated = people.load_people(settings, mini_inputs.catalog)
    before = _true_roles(mini_dir, mini_rows)
    expected = {key: mapping[role_id] for key, role_id in before.items()}
    assert all(expected[key] != before[key] for key in before)  # every true role changed
    for occasion in OCCASIONS:
        decisions = [getattr(simulated, occasion)(row) for row in mini_rows]
        assert {d.key: d.role_id for d in decisions} == expected


def test_true_roles_moved_between_truth_rows_stop_the_run_at_a_row_with_two_spellings(
    data_dir, mini_dir, mini_inputs, mini_rows, mini_settings
):
    # The same roles, each moved one truth row on within its family. Every role
    # is still a real one of its family, so the load passes. The run then stops
    # at the first decision on a row whose spellings no longer agree.
    truth = people.read_truth(mini_dir)
    family_of = {role.role_id: role.family_code for role in mini_inputs.catalog.roles}
    positions: dict[str, list[int]] = {}
    for position, row in enumerate(truth):
        positions.setdefault(family_of[row.true_role_id], []).append(position)
    moved = list(truth)
    for in_family in positions.values():
        for position, source in zip(in_family, in_family[1:] + in_family[:1], strict=True):
            moved[position] = dataclasses.replace(
                truth[position], true_role_id=truth[source].true_role_id
            )
    assert sorted(row.true_role_id for row in moved) == sorted(row.true_role_id for row in truth)
    people.write_truth(data_dir, moved)

    settings = dataclasses.replace(mini_settings, data_dir=data_dir)
    simulated = people.load_people(settings, mini_inputs.catalog)
    stopped = []
    for row in mini_rows:
        if len(row.raw_titles) > 1:
            with pytest.raises(DataError, match="have different true roles"):
                simulated.wave1_approval(row)
            stopped.append(row.key.cleaned_title)
        else:
            assert simulated.wave1_approval(row).answer is HumanAnswer.ROLE
    assert stopped == ["sales representative", "senior account manager"]


# --- The same decisions in every process, on the standard library alone (Q06) ---

# Run after the import guard of tests/conftest.py. It cannot import MINI_ROWS,
# which lives beside pytest, so it makes one row per truth row: the same keys
# and raw titles in every process, which is all this comparison needs.
DECIDE_EVERYTHING = """
import json
from pathlib import Path

from titlestd import datafiles, people
from titlestd.contracts import Record, TitleRow
from titlestd.settings import Settings

data_dir = Path(sys.argv[1])
settings = Settings(data_dir=data_dir, sme_error_rate=0.5, undecided_rate=0.5)
simulated = people.load_people(settings, datafiles.read_inputs(data_dir).catalog)
decisions = []
for truth in people.read_truth(data_dir):
    record = Record(truth.job_title.lower(), truth.job_family_code.strip(), "", "")
    row = TitleRow(record, ("E900",), (truth.job_title,))
    for decide in (simulated.wave1_approval, simulated.blind_label, simulated.queue_decision):
        decision = decide(row)
        decisions.append(
            [*decision.key, decision.answer.value, decision.role_id, decision.decided_by]
        )
print(json.dumps({"decisions": decisions, "attempted_imports": attempted_imports}))
"""


def _decide_in_this_process(mini_dir: Path, catalog) -> list[list[str | None]]:
    settings = Settings(data_dir=mini_dir, sme_error_rate=0.5, undecided_rate=0.5)
    simulated = people.load_people(settings, catalog)
    decisions = []
    for truth in people.read_truth(mini_dir):
        record = Record(truth.job_title.lower(), truth.job_family_code.strip(), "", "")
        row = TitleRow(record, ("E900",), (truth.job_title,))
        for occasion in OCCASIONS:
            decision = getattr(simulated, occasion)(row)
            decisions.append(
                [*decision.key, decision.answer.value, decision.role_id, decision.decided_by]
            )
    return decisions


def test_decisions_are_the_same_in_every_process(mini_dir, mini_inputs, repo_root, import_guard):
    expected = _decide_in_this_process(mini_dir, mini_inputs.catalog)
    assert len(expected) == 17 * len(OCCASIONS)
    assert {decision[2] for decision in expected} == {"role", "could not decide"}
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", import_guard + DECIDE_EVERYTHING, str(mini_dir)],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert printed["decisions"] == expected, hash_seed
        assert printed["attempted_imports"] == []


def test_the_people_decide_on_a_bare_interpreter(mini_dir, mini_inputs, bare_python, import_guard):
    # Nothing installed and nothing outside the standard library imported (2.5).
    result = bare_python(import_guard + DECIDE_EVERYTHING, str(mini_dir))
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    assert printed["decisions"] == _decide_in_this_process(mini_dir, mini_inputs.catalog)
    assert printed["attempted_imports"] == []
