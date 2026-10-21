"""The golden set on the mini data set, with the real modules around it.

The title rows come from the real prepare step over the real readers, the
labels from the real simulated people, and the machine's answers from the real
offline embedder and stand-in LLM (DESIGN.md 1.2, 1.10, 2.4;
docs/build-decisions.md Q08, Q14, Q22, Q26).

The modules that sit between these in a run are left out of these tests: the
Wave 1 list and the order of a run (titlestd.pipeline), the keys of the seed
look-up (titlestd.lookup), the nearest-neighbour match (titlestd.matcher), the
shuffle and the numbering (titlestd.fallback). This file does their part in
miniature, by the rules the decisions record gives them (Q13, Q18, Q24). No
expected answer here is read from the truth file: each is written out by hand.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from titlestd import catalog, golden, people, prepare, seeding, switch
from titlestd.contracts import (
    SHORTLIST_SIZE,
    CatalogRole,
    ChooseKind,
    GoldenScore,
    InputData,
    NumberedRole,
    Stage,
    TitleKey,
    TitleRow,
    Vector,
)
from titlestd.settings import Settings

MINI_WAVE1 = (
    TitleKey("FIN", "financial analyst"),
    TitleKey("SAL", "account manager ii"),
    TitleKey("SAL", "sales representative"),
    TitleKey("SAL", "senior account manager"),
)
MINI_SEED = (TitleKey("FIN", "accounts payable clerk"), TitleKey("SAL", "sales engineer"))
# The frame: the long-tail rows that miss the seed look-up, have a usable family
# code and carry no person's name. Each with its true role, written out by hand
# from tests/fixtures/mini.
MINI_FRAME = {
    TitleKey("FIN", "internal auditor"): "FIN-04",
    TitleKey("FIN", "payroll specialist"): "FIN-03",
    TitleKey("FIN", "senior financial analyst"): "FIN-01",
    TitleKey("FIN", "special projects"): "FIN-01",
    TitleKey("FIN", "treasury analyst"): "FIN-06",
    TitleKey("SAL", "bdr"): "SAL-05",
    TitleKey("SAL", "sales operations analyst"): "SAL-03",
}
# What the stand-in LLM answers for the frame, worked out by hand from Q17: it
# names five roles and none for two. "special projects" shares no word with a
# role ("no match"); "bdr" is one word with no description ("insufficient
# data").
LLM_NAMES_NO_ROLE = {
    TitleKey("FIN", "special projects"): ChooseKind.NO_MATCH,
    TitleKey("SAL", "bdr"): ChooseKind.INSUFFICIENT_DATA,
}


# --- The steps around the golden set, in miniature -----------------------------------


def wave1_list(rows: tuple[TitleRow, ...], size: int) -> list[TitleKey]:
    """Q24: the top rows by headcount, then family code, then cleaned title."""
    return [row.key for row in sorted(rows, key=lambda row: (-row.headcount, row.key))[:size]]


def seed_keys(data: InputData) -> list[TitleKey]:
    """Q09, Q13: the seed look-up, cleaned on load with the shared prepare function."""
    return [
        prepare.title_key(seed.job_title, seed.job_family_code, data.abbreviations)
        for seed in data.seed_lookup
    ]


@dataclasses.dataclass(frozen=True)
class MachineAnswers:
    """What the two models said about each title, as the scoring takes it."""

    ml_first_choices: dict[TitleKey, str | None]
    llm_picks: dict[TitleKey, str | None]
    llm_kinds: dict[TitleKey, ChooseKind]


def nearest_roles(
    vector: Vector, role_vectors: list[Vector], roles: tuple[CatalogRole, ...]
) -> list[CatalogRole]:
    """The roles, nearest to the vector first, ties by role_id (Q18)."""

    def score(at: int) -> float:
        return math.fsum(
            weight * role_vectors[at].get(index, 0.0) for index, weight in vector.items()
        )

    order = sorted(range(len(roles)), key=lambda at: (-score(at), roles[at].role_id))
    return [roles[at] for at in order]


def ask_the_models(
    data: InputData, rows: tuple[TitleRow, ...], settings: Settings
) -> MachineAnswers:
    """Embed and choose for every model-eligible row: the ML first choice is the
    nearest model-visible role, the LLM sees the ten nearest, sorted by role_id
    and shuffled per title (Q18), and its number is mapped back to a role."""
    models = switch.build_models(settings, data.catalog.ladder, data.abbreviations)
    roles = catalog.model_visible_roles(data.catalog)
    role_vectors = models.embedder.embed([f"{r.role_name}. {r.definition}" for r in roles])
    answers = MachineAnswers({}, {}, {})
    for row in rows:
        if not row.model_eligible:
            continue
        record = row.record
        (vector,) = models.embedder.embed([f"{record.cleaned_title} {record.description}".strip()])
        nearest = nearest_roles(vector, role_vectors, roles)
        shortlist = sorted(nearest[:SHORTLIST_SIZE], key=lambda role: role.role_id)
        seeding.rng(settings.seed, seeding.STREAM_SHUFFLE, *row.key).shuffle(shortlist)
        shown = [NumberedRole(number, role) for number, role in enumerate(shortlist, start=1)]
        answer = models.llm.chooser.choose(record, shown)
        answers.ml_first_choices[row.key] = nearest[0].role_id
        answers.llm_picks[row.key] = next(
            (numbered.role.role_id for numbered in shown if numbered.number == answer.number), None
        )
        answers.llm_kinds[row.key] = answer.answer
    return answers


@pytest.fixture(scope="module")
def prepared(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    """The title rows of the mini data set, from the real prepare step."""
    return prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )


def _whole_frame(settings: Settings) -> Settings:
    """The same settings with a golden set as large as any frame of the mini data."""
    return dataclasses.replace(settings, golden_size=40)


# --- The draw at set-up (1.10, Q08) ---------------------------------------------------


def test_the_miniature_steps_give_the_lists_written_out_by_hand(mini_inputs, prepared):
    assert sorted(wave1_list(prepared, 4)) == list(MINI_WAVE1)
    assert sorted(seed_keys(mini_inputs)) == list(MINI_SEED)


def test_the_frame_of_the_mini_data_set_from_the_real_prepare_step(
    mini_inputs, prepared, mini_settings
):
    drawn = golden.draw_golden(
        prepared, wave1_list(prepared, 4), seed_keys(mini_inputs), _whole_frame(mini_settings)
    )
    assert drawn == tuple(MINI_FRAME)
    # The 15 rows less the four of Wave 1, the two seed hits, the title with a
    # person's name (Q14) and the row with a blank family code (Q26).
    out = {row.key for row in prepared} - set(drawn) - set(MINI_WAVE1) - set(MINI_SEED)
    assert sorted(out) == [
        TitleKey("", "staff accountant"),
        TitleKey("SAL", "customer success manager [NAME]"),
    ]


def test_the_golden_set_of_the_mini_settings_is_pinned(mini_inputs, prepared, mini_settings):
    # Three titles of the frame on the default seed. A change here changes
    # which titles are never learned, and with that every later prompt (Q18).
    drawn = golden.draw_golden(
        prepared, wave1_list(prepared, 4), seed_keys(mini_inputs), mini_settings
    )
    assert drawn == (
        TitleKey("FIN", "payroll specialist"),
        TitleKey("SAL", "bdr"),
        TitleKey("SAL", "sales operations analyst"),
    )


def test_the_hand_written_rows_and_another_order_of_the_employees_give_the_same_set(
    mini_inputs, mini_rows, prepared, mini_settings
):
    expected = golden.draw_golden(prepared, MINI_WAVE1, MINI_SEED, mini_settings)
    assert golden.draw_golden(mini_rows, MINI_WAVE1, MINI_SEED, mini_settings) == expected
    backwards = prepare.prepare(
        mini_inputs.employees[::-1],
        mini_inputs.abbreviations,
        catalog.family_codes(mini_inputs.catalog),
    )
    assert golden.draw_golden(backwards[::-1], MINI_WAVE1, MINI_SEED, mini_settings) == expected


def test_the_draw_reads_no_file_so_it_cannot_depend_on_the_truth(prepared, mini_settings, tmp_path):
    # Q22: the draw depends only on the seed and the title keys. With a data
    # folder that does not exist it is the same set.
    nowhere = dataclasses.replace(mini_settings, data_dir=tmp_path / "no such folder")
    assert golden.draw_golden(prepared, MINI_WAVE1, MINI_SEED, nowhere) == golden.draw_golden(
        prepared, MINI_WAVE1, MINI_SEED, mini_settings
    )


def test_a_larger_wave1_list_takes_its_titles_out_of_the_frame(
    mini_inputs, prepared, mini_settings
):
    # The four of Wave 1 are the only rows held by more than one employee, so a
    # list of five takes the first single-held row by key, which is the row
    # with a blank family code and was out already. A list of seven also takes
    # the seed hit "accounts payable clerk" and "internal auditor" of the frame.
    seven = wave1_list(prepared, 7)
    assert seven[4:] == [
        TitleKey("", "staff accountant"),
        TitleKey("FIN", "accounts payable clerk"),
        TitleKey("FIN", "internal auditor"),
    ]
    drawn = golden.draw_golden(prepared, seven, seed_keys(mini_inputs), _whole_frame(mini_settings))
    assert drawn == tuple(key for key in MINI_FRAME if key != TitleKey("FIN", "internal auditor"))


# --- Labeled blind in Wave 1, then scored (1.10, 2.4, Q08) ------------------------------


def _labels(simulated: people.SimulatedPeople, rows, keys) -> dict[TitleKey, str | None]:
    by_key = {row.key: row for row in rows}
    return {key: simulated.blind_label(by_key[key]).role_id for key in keys}


def test_the_people_label_the_frame_with_its_true_roles_when_they_make_no_error(
    mini_inputs, prepared, mini_settings
):
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    simulated = people.load_people(settings, mini_inputs.catalog)
    assert _labels(simulated, prepared, MINI_FRAME) == MINI_FRAME


def test_the_stand_ins_scored_on_the_whole_frame(mini_inputs, prepared, mini_settings):
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    drawn = golden.draw_golden(
        prepared, wave1_list(prepared, 4), seed_keys(mini_inputs), _whole_frame(settings)
    )
    labels = _labels(people.load_people(settings, mini_inputs.catalog), prepared, drawn)
    answers = ask_the_models(mini_inputs, prepared, settings)
    assert {key: answers.llm_kinds[key] for key in LLM_NAMES_NO_ROLE} == LLM_NAMES_NO_ROLE

    score = golden.score_golden(Stage.WAVE1, labels, answers.ml_first_choices, answers.llm_picks)
    # The answers hold all 13 model-eligible rows; only the seven golden ones count.
    assert len(answers.llm_picks) == 13
    assert (score.stage, score.size) == (Stage.WAVE1, 7)
    assert score.llm_right == 5  # the five titles that carry their role's name
    assert score.llm_no_answer == 2  # counted as wrong, and on their own (Q08)
    # The embedder's first choice, held against the hand-written true roles.
    assert score.ml_right == sum(
        answers.ml_first_choices[key] == role_id for key, role_id in MINI_FRAME.items()
    )
    assert 5 <= score.ml_right <= 7


def test_the_golden_set_of_the_mini_settings_scored(mini_inputs, prepared, mini_settings):
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    drawn = golden.draw_golden(prepared, wave1_list(prepared, 4), seed_keys(mini_inputs), settings)
    labels = _labels(people.load_people(settings, mini_inputs.catalog), prepared, drawn)
    answers = ask_the_models(mini_inputs, prepared, settings)
    # "payroll specialist" and "sales operations analyst" are role names, which
    # both models find; "bdr" gets "insufficient data" from the LLM, and the ML
    # model's nearest role to those three letters is not the true one.
    assert answers.ml_first_choices[TitleKey("SAL", "bdr")] != "SAL-05"
    for stage in (Stage.WAVE1, Stage.WAVE2, Stage.CLOSE):
        score = golden.score_golden(stage, labels, answers.ml_first_choices, answers.llm_picks)
        assert score == GoldenScore(stage, size=3, ml_right=2, llm_right=2, llm_no_answer=1)


def test_an_smes_mistake_counts_against_the_machine(mini_inputs, prepared, mini_settings):
    # 2.4, Q27: wrong means the machine's role differs from the SME's label.
    # With every label mistaken, the LLM is right about no title: where it
    # names a role, it names the true one.
    settings = dataclasses.replace(mini_settings, sme_error_rate=1.0)
    labels = _labels(people.load_people(settings, mini_inputs.catalog), prepared, MINI_FRAME)
    assert all(labels[key] not in (None, role_id) for key, role_id in MINI_FRAME.items())
    answers = ask_the_models(mini_inputs, prepared, settings)
    score = golden.score_golden(Stage.WAVE2, labels, answers.ml_first_choices, answers.llm_picks)
    assert (score.size, score.llm_right, score.llm_no_answer) == (7, 0, 2)
    assert score.ml_right <= 2  # at most the two titles the ML model had wrong


def test_a_golden_title_without_a_truth_row_stays_in_the_denominator(
    mini_inputs, prepared, mini_settings, mini_dir
):
    # Q03: the SME marks it "no suitable role" and names no role. Q08: the
    # denominator is every golden title.
    truth = [row for row in people.read_truth(mini_dir) if row.job_title != "Payroll Spec"]
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    simulated = people.SimulatedPeople(truth, mini_inputs.catalog, settings)
    labels = _labels(simulated, prepared, MINI_FRAME)
    assert labels[TitleKey("FIN", "payroll specialist")] is None
    answers = ask_the_models(mini_inputs, prepared, settings)
    score = golden.score_golden(Stage.WAVE1, labels, answers.ml_first_choices, answers.llm_picks)
    assert (score.size, score.llm_right, score.llm_no_answer) == (7, 4, 2)


# --- The same set in every process, on the standard library alone (Q06) ----------------

# Run after the import guard of tests/conftest.py. The lists are handed in as
# sets, whose order changes with the hash seed of the process.
DRAW_THE_GOLDEN_SET = """
import json
from pathlib import Path

from titlestd import catalog, datafiles, golden, prepare
from titlestd.settings import Settings

data = datafiles.read_inputs(Path(sys.argv[1]))
rows = prepare.prepare(data.employees, data.abbreviations, catalog.family_codes(data.catalog))
wave1 = {row.key for row in sorted(rows, key=lambda row: (-row.headcount, row.key))[:4]}
seed_keys = {
    prepare.title_key(seed.job_title, seed.job_family_code, data.abbreviations)
    for seed in data.seed_lookup
}
drawn = {
    size: golden.draw_golden(rows, wave1, seed_keys, Settings(golden_size=size))
    for size in (1, 2, 3, 5, 40)
}
print(json.dumps({"drawn": drawn, "attempted_imports": attempted_imports}))
"""


def _drawn_in_this_process(prepared) -> dict[str, list[list[str]]]:
    return {
        str(size): [
            list(key)
            for key in golden.draw_golden(
                prepared, MINI_WAVE1, MINI_SEED, Settings(golden_size=size)
            )
        ]
        for size in (1, 2, 3, 5, 40)
    }


def test_the_golden_set_is_the_same_in_every_process(mini_dir, prepared, repo_root, import_guard):
    expected = _drawn_in_this_process(prepared)
    assert [len(keys) for keys in expected.values()] == [1, 2, 3, 5, 7]
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", import_guard + DRAW_THE_GOLDEN_SET, str(mini_dir)],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert printed["drawn"] == expected, hash_seed
        assert printed["attempted_imports"] == []


def test_the_golden_set_is_drawn_on_a_bare_interpreter(
    mini_dir: Path, prepared, bare_python, import_guard
):
    # Nothing installed and nothing outside the standard library imported (2.5).
    result = bare_python(import_guard + DRAW_THE_GOLDEN_SET, str(mini_dir))
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    assert printed["drawn"] == _drawn_in_this_process(prepared)
    assert printed["attempted_imports"] == []
