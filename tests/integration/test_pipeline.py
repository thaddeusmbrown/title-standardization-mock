"""The pipeline on the mini data set, with the real modules merged before it.

Every test runs the whole of titlestd.pipeline.run_pipeline: set-up, Wave 1,
Wave 2 and the close on tests/fixtures/mini (15 title rows, a Wave 1 list of
four, a golden set of three). Where a rule needs a model that misbehaves, or a
look at what a model was handed, the models go in through the seam of S09
(run_pipeline(settings, models=...)); everything else is the real code.

The mini data set is too small for a sampled audit, a generic role under the
cap or a referral to the agent by the stand-ins. Those are tested on the mock
data in tests/integration/test_pipeline_default_run.py.
"""

from __future__ import annotations

import dataclasses
import shutil
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from titlestd import cli, datafiles, lookup, people, pipeline, release, store, switch
from titlestd.catalog import family_codes, roles_in_family
from titlestd.contracts import (
    LOOKUP_DECIDED_BY,
    AgentTranscript,
    ChooseAnswer,
    ChooseKind,
    DecisionSource,
    Example,
    Llm,
    Models,
    NumberedRole,
    Record,
    ReviewAnswer,
    ReviewKind,
    RunResult,
    SearchRequest,
    Status,
    TableRow,
    TitleKey,
    TitlestdError,
)
from titlestd.matcher import embed_text, role_text
from titlestd.settings import Settings

NAMED = TitleKey("SAL", "customer success manager [NAME]")  # a person's name (Q14)
NO_FAMILY = TitleKey("", "staff accountant")  # no usable family code (Q26)
WAVE1 = (
    TitleKey("FIN", "financial analyst"),
    TitleKey("SAL", "account manager ii"),
    TitleKey("SAL", "sales representative"),
    TitleKey("SAL", "senior account manager"),
)
GOLDEN = (
    TitleKey("FIN", "payroll specialist"),
    TitleKey("SAL", "bdr"),
    TitleKey("SAL", "sales operations analyst"),
)
TREASURY = TitleKey("FIN", "treasury analyst")  # the stand-ins agree on it in Wave 2


# --- helpers ----------------------------------------------------------------------


def real_models(settings: Settings) -> Models:
    data = datafiles.read_inputs(settings.data_dir)
    return switch.build_models(settings, data.catalog.ladder, data.abbreviations)


def rows_of(settings: Settings, query: str, *arguments: object) -> list[tuple]:
    """Read the state file a run left in settings.out_dir."""
    connection = sqlite3.connect(settings.out_dir / store.STATE_FILE)
    try:
        return list(connection.execute(query, arguments))
    finally:
        connection.close()


def output(settings: Settings, name: str, columns: Sequence[str]) -> list[dict[str, str]]:
    return datafiles.read_table(settings.out_dir / name, columns)


def queue_file(settings: Settings) -> dict[TitleKey, dict[str, str]]:
    lines = output(settings, release.REVIEW_QUEUE_FILE, release.REVIEW_QUEUE_COLUMNS)
    return {TitleKey(line["job_family_code"], line["cleaned_title"]): line for line in lines}


def decision_of(result: RunResult, key: TitleKey):
    (decision,) = [each for each in result.decisions if each.key == key]
    return decision


def copy_of_mini(mini_dir: Path, tmp_path: Path) -> Path:
    """A copy of the mini data set that a test may change."""
    return Path(shutil.copytree(mini_dir, tmp_path / "data"))


class Spy:
    """The real models, with a record of everything each call was handed."""

    def __init__(self, models: Models, chooser=None, reviewer=None) -> None:
        self._embedder = models.embedder
        self._chooser = chooser or models.llm.chooser
        self._reviewer = reviewer or models.llm.reviewer
        self.embedded: list[str] = []
        self.chosen: list[tuple[Record, tuple[NumberedRole, ...]]] = []
        self.turns: list[AgentTranscript] = []
        self.finished: list[bool] = []
        self.models = Models(self, Llm(self, self, self.finish))
        self.out_dir: Path | None = None

    def embed(self, texts):
        self.embedded += list(texts)
        return self._embedder.embed(texts)

    def choose(self, record, roles):
        self.chosen.append((record, tuple(roles)))
        return self._chooser.choose(record, roles)

    def next_turn(self, transcript):
        self.turns.append(transcript)
        return self._reviewer.next_turn(transcript)

    def finish(self) -> None:
        # Whether the load file stood when the run called finish.
        self.finished.append(bool(self.out_dir and (self.out_dir / release.LOAD_FILE).is_file()))

    @property
    def chosen_keys(self) -> list[TitleKey]:
        return [TitleKey(record.family_code, record.cleaned_title) for record, _ in self.chosen]

    @property
    def shown_roles(self) -> list[NumberedRole]:
        """Every role the LLM or the agent was shown."""
        shown = [role for _, roles in self.chosen for role in roles]
        for transcript in self.turns:
            shown += transcript.case.roles
            shown += [hit.role for step in transcript.steps for hit in step.hits]
        return shown


class NoMatch:
    """A chooser that finds no role, with a quote the record holds: the rules
    then refer the title to the agent (1.7)."""

    def choose(self, record, roles):
        return ChooseAnswer(ChooseKind.NO_MATCH, None, record.cleaned_title, "test")


class Changing:
    """The stand-in's answer, changed for the titles a test names."""

    def __init__(self, chooser, titles: set[str], **changes: object) -> None:
        self._chooser, self._titles, self._changes = chooser, titles, changes

    def choose(self, record, roles):
        answer = self._chooser.choose(record, roles)
        if record.cleaned_title in self._titles:
            return dataclasses.replace(answer, **self._changes)
        return answer


class AnsweringWith:
    """A reviewer that runs the stand-in's searches and then answers with a
    number of the test's choosing, quoting the title."""

    def __init__(self, reviewer, number: object) -> None:
        self._reviewer, self._number = reviewer, number

    def next_turn(self, transcript):
        turn = self._reviewer.next_turn(transcript)
        if isinstance(turn, SearchRequest):
            return turn
        title = transcript.case.record.cleaned_title
        return ReviewAnswer(ReviewKind.ROLE, self._number, title, "test")


# --- the run as a whole -------------------------------------------------------------


def test_a_run_decides_every_title_once_and_writes_the_files_and_the_state(mini_settings):
    result = pipeline.run_pipeline(mini_settings)
    assert [decision.key for decision in result.decisions] == [row.key for row in result.rows]
    assert len(result.rows) == 15 and result.employee_count == 22
    assert result.wave1_keys == WAVE1 and result.golden_keys == GOLDEN
    by_row = dict.fromkeys(TableRow, 0)
    for decision in result.decisions:
        by_row[decision.row] += 1
    assert by_row == {
        TableRow.LOOKUP: 2,
        TableRow.WAVE1: 4,
        TableRow.MODELS_AGREE: 3,
        TableRow.AGENT_SETTLED: 0,
        TableRow.REVIEW_QUEUE: 6,
        TableRow.GENERIC: 0,
        TableRow.LEGACY: 0,
    }
    for name in (*pipeline.OUTPUT_FILES, store.STATE_FILE):
        assert (mini_settings.out_dir / name).is_file(), name
    assert result.answer_labels == ("stand-in",)
    assert [score.stage.value for score in result.golden_scores] == ["Wave 1", "Wave 2", "close"]
    assert all(score.size == 3 for score in result.golden_scores)
    # E012 and E022 have no grade and received a title with no level word.
    assert result.no_grade_employees == 2


def test_the_rows_of_the_prepare_step_are_the_rows_of_the_run(mini_settings, mini_rows):
    assert pipeline.run_pipeline(mini_settings).rows == mini_rows


def test_two_runs_give_the_same_result_and_the_same_bytes(mini_settings, tmp_path):
    again = dataclasses.replace(mini_settings, out_dir=tmp_path / "again")
    assert pipeline.run_pipeline(mini_settings) == pipeline.run_pipeline(again)
    for name in pipeline.OUTPUT_FILES:
        first = (mini_settings.out_dir / name).read_bytes()
        assert first == (again.out_dir / name).read_bytes(), name


def test_a_run_starts_from_fresh_state_whatever_an_earlier_run_left(mini_settings):
    # Q06: the state file is deleted and recreated at the start of every run.
    first = pipeline.run_pipeline(mini_settings)
    counts = rows_of(mini_settings, "SELECT count(*) FROM scorings")
    (mini_settings.out_dir / release.LOAD_FILE).write_text("stale", encoding="utf-8")
    assert pipeline.run_pipeline(mini_settings) == first
    assert rows_of(mini_settings, "SELECT count(*) FROM scorings") == counts
    assert (mini_settings.out_dir / release.LOAD_FILE).read_text(encoding="utf-8") != "stale"


class FirstLook:
    """The real embedder, which notes what the state file holds the first time
    it is asked: the ML match of the first wave is the first model call of a run."""

    TABLES = ("settings", "thresholds", "title_rows", "scorings", "human_decisions")

    def __init__(self, embedder, settings: Settings) -> None:
        self._embedder, self._settings = embedder, settings
        self.seen: dict[str, list[tuple]] = {}

    def embed(self, texts):
        if not self.seen:
            for table in self.TABLES:
                self.seen[table] = rows_of(self._settings, f"SELECT * FROM {table}")
        return self._embedder.embed(texts)


def test_set_up_records_the_settings_and_thresholds_in_force(mini_settings):
    # Q37: fixed and recorded before any scoring. The state is read when the
    # first model is asked, and again when the run is over.
    models = real_models(mini_settings)
    first = FirstLook(models.embedder, mini_settings)
    pipeline.run_pipeline(mini_settings, models=Models(first, models.llm))
    assert first.seen["scorings"] == first.seen["human_decisions"] == []
    assert dict(first.seen["settings"]) == {
        name: "" if value is None else str(int(value) if isinstance(value, bool) else value)
        for name, value in mini_settings.as_dict().items()
    }
    assert dict(first.seen["thresholds"]) == {
        "audit_sample_size": "60",
        "audit_accuracy_bar": "0.9",
        "audit_confirm_limit": "2",
        "audit_stop_limit": "12",
        "generic_cap_employees": "0",  # 2% of 22 employees, rounded down
        "release_stopped_paths": "0",
    }
    assert len(first.seen["title_rows"]) == 15
    # Nothing a run does later changes what set-up recorded.
    for table in ("settings", "thresholds", "title_rows"):
        assert rows_of(mini_settings, f"SELECT * FROM {table}") == first.seen[table], table
    assert rows_of(mini_settings, "SELECT count(*) FROM scorings") != [(0,)]
    flagged = rows_of(
        mini_settings,
        "SELECT family_code, cleaned_title FROM title_rows WHERE on_wave1_list ORDER BY 1, 2",
    )
    assert [TitleKey(*key) for key in flagged] == list(WAVE1)
    golden = rows_of(
        mini_settings,
        "SELECT family_code, cleaned_title FROM title_rows WHERE in_golden_set ORDER BY 1, 2",
    )
    assert [TitleKey(*key) for key in golden] == list(GOLDEN)


def test_the_data_fingerprint_covers_the_six_input_files_and_the_truth(mini_dir, tmp_path):
    # S09: simulated decisions feed the examples, so the truth shapes later prompts.
    original = pipeline.data_fingerprint(mini_dir)
    assert len(original) == 64 and original == pipeline.data_fingerprint(mini_dir)
    assert original not in (
        datafiles.input_fingerprint(mini_dir),
        people.truth_fingerprint(mini_dir),
    )
    changed_truth = copy_of_mini(mini_dir, tmp_path / "a")
    truth = people.read_truth(changed_truth)
    people.write_truth(changed_truth, truth[:-1])
    changed_input = copy_of_mini(mini_dir, tmp_path / "b")
    with (changed_input / datafiles.ABBREVIATIONS_FILE).open("a", encoding="utf-8") as handle:
        handle.write("vp,vice president\n")
    fingerprints = {original, *map(pipeline.data_fingerprint, (changed_truth, changed_input))}
    assert len(fingerprints) == 3


# --- the models and the seam (S09) ----------------------------------------------------


def test_a_real_run_builds_its_models_with_the_switch(mini_settings, monkeypatch):
    seen = []
    build = switch.build_models

    def recording(settings, ladder, abbreviations, data_fingerprint=""):
        seen.append((settings, data_fingerprint))
        return build(settings, ladder, abbreviations, data_fingerprint)

    monkeypatch.setattr(switch, "build_models", recording)
    pipeline.run_pipeline(mini_settings)
    assert seen == [(mini_settings, pipeline.data_fingerprint(mini_settings.data_dir))]


def test_models_handed_in_are_used_and_the_switch_is_not_called(mini_settings, monkeypatch):
    models = real_models(mini_settings)
    spy = Spy(models)

    def refuse(*arguments, **named):
        raise AssertionError("the switch was called although the models were handed in")

    monkeypatch.setattr(switch, "build_models", refuse)
    result = pipeline.run_pipeline(mini_settings, models=spy.models)
    assert spy.embedded and spy.chosen
    assert result == pipeline.run_pipeline(mini_settings, models=models)


def test_finish_is_called_once_after_the_files_are_written(mini_settings):
    # Q29: live mode completes its recording there, after a run that completed.
    spy = Spy(real_models(mini_settings))
    spy.out_dir = mini_settings.out_dir
    pipeline.run_pipeline(mini_settings, models=spy.models)
    assert spy.finished == [True]
    assert rows_of(mini_settings, "SELECT count(*) FROM decisions") == [(15,)]


def test_titles_are_processed_in_key_order_one_stage_after_another(mini_settings):
    # Q18: sorted by family code, then cleaned title, so prompts repeat.
    spy = Spy(real_models(mini_settings))
    pipeline.run_pipeline(mini_settings, models=spy.models)
    long_tail = rows_of(
        mini_settings,
        "SELECT family_code, cleaned_title FROM scorings WHERE stage = 'Wave 2' ORDER BY 1, 2",
    )
    assert spy.chosen_keys == [*WAVE1, *GOLDEN, *(TitleKey(*key) for key in long_tail), *GOLDEN]


# --- a run that stops ---------------------------------------------------------------


class StopsAt:
    """A chooser that answers like the stand-in until one title, then stops the run."""

    def __init__(self, chooser, title: str) -> None:
        self._chooser, self._title = chooser, title

    def choose(self, record, roles):
        if record.cleaned_title == self._title:
            raise TitlestdError(f"no recorded answer for the title {self._title!r}")
        return self._chooser.choose(record, roles)


def test_a_run_that_stops_leaves_no_output_file_and_does_not_finish(mini_settings):
    # Q29: a miss stops the run at once, with no output files.
    pipeline.run_pipeline(mini_settings)  # an earlier run that completed
    models = real_models(mini_settings)
    spy = Spy(models, chooser=StopsAt(models.llm.chooser, TREASURY.cleaned_title))
    with pytest.raises(TitlestdError, match="no recorded answer"):
        pipeline.run_pipeline(mini_settings, models=spy.models)
    for name in pipeline.OUTPUT_FILES:
        assert not (mini_settings.out_dir / name).exists(), name
    assert spy.finished == []
    # The state holds what was recorded up to the stop: Wave 1, and no decision.
    stages = rows_of(mini_settings, "SELECT DISTINCT stage FROM scorings WHERE route != ''")
    assert ("Wave 1",) in stages
    assert rows_of(mini_settings, "SELECT count(*) FROM decisions") == [(0,)]
    assert rows_of(mini_settings, "SELECT count(*) FROM lookup WHERE joined_at = 'Wave 1'") == [
        (4,)
    ]


def test_a_provider_that_refuses_stops_the_run_with_nothing_in_the_folder(
    mini_settings, monkeypatch
):
    # Replay refuses a missing recording before any work is done (Q29).
    pipeline.run_pipeline(mini_settings)

    def refuse(*arguments, **named):
        raise TitlestdError("the recording is missing")

    monkeypatch.setattr(switch, "build_models", refuse)
    with pytest.raises(TitlestdError, match="recording is missing"):
        pipeline.run_pipeline(mini_settings)
    assert list(mini_settings.out_dir.iterdir()) == []


def test_a_failure_while_finishing_removes_the_output_files(mini_settings):
    models = real_models(mini_settings)

    def cannot_finish() -> None:
        raise TitlestdError("the recording could not be completed")

    failing = Models(models.embedder, dataclasses.replace(models.llm, finish=cannot_finish))
    with pytest.raises(TitlestdError, match="could not be completed"):
        pipeline.run_pipeline(mini_settings, models=failing)
    for name in pipeline.OUTPUT_FILES:
        assert not (mini_settings.out_dir / name).exists(), name


# --- the command line ---------------------------------------------------------------

MINI_FLAGS = ("--data-dir", "tests/fixtures/mini", "--wave1-size", "4", "--golden-size", "3")


def test_the_run_command_prints_the_table_and_says_where_the_files_are(
    mini_settings, tmp_path, capsys, monkeypatch, repo_root
):
    monkeypatch.chdir(repo_root)
    out_dir = tmp_path / "from the command line"
    assert cli.main(["run", *MINI_FLAGS, "--out-dir", str(out_dir)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert lines[0] == "Job-title standardization mock: set-up, Wave 1, Wave 2 and the close"
    assert lines[-1] == (
        f"Files written to the folder {out_dir}: load.csv, reverse.csv, exceptions.csv, "
        "review_queue.csv, state.sqlite."
    )
    for name in ("load.csv", "reverse.csv", "exceptions.csv", "review_queue.csv", "state.sqlite"):
        assert (out_dir / name).is_file(), name
    # The same table as the pipeline called directly, whatever the folder.
    pipeline.run_command(mini_settings)
    assert capsys.readouterr().out.splitlines()[:-1] == lines[:-1]


def test_the_run_command_works_on_a_bare_interpreter(bare_cli, tmp_path):
    # 2.5: the default run installs nothing. The whole mock data is run this
    # way, under the import guard, in test_pipeline_default_run.py.
    result = bare_cli("run", *MINI_FLAGS, "--out-dir", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "approved in Wave 1         4         11" in result.stdout
    assert (tmp_path / "out" / "load.csv").is_file()


def test_a_data_error_stops_the_run_with_status_1(tmp_path, capsys):
    status = cli.main(["run", "--data-dir", str(tmp_path / "absent"), "--out-dir", str(tmp_path)])
    assert status == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("titlestd: data file not found")


# --- Wave 1 (1.10, Q13, Q22, Q24) -----------------------------------------------------


def test_every_wave_1_title_is_approved_by_an_sme_whatever_the_machine_proposed(mini_settings):
    # 1.10: the machine proposes, an HR SME approves every one.
    result = pipeline.run_pipeline(mini_settings)
    proposals = {
        TitleKey(family, title): route
        for family, title, route in rows_of(
            mini_settings,
            "SELECT family_code, cleaned_title, route FROM scorings WHERE stage = 'Wave 1'",
        )
    }
    for key in WAVE1:
        decision = decision_of(result, key)
        assert key in proposals  # the machine did propose
        assert (decision.row, decision.source) == (TableRow.WAVE1, DecisionSource.PERSON)
        assert decision.status is Status.APPROVED and decision.simulated
        assert decision.decided_by.startswith("SME ")
    # A proposal the two models agreed on is still not a machine decision.
    assert proposals[WAVE1[1]] == "models agree"
    audited = {key for audit in result.audits for key in audit.audited_keys}
    assert audited.isdisjoint(WAVE1)


def test_a_wave_1_title_is_released_with_the_sme_s_role_not_the_machine_s(mini_settings):
    # 1.8, 1.10: the machine proposes and the SME's approval is what is released.
    # On the default error rate the two name the same role for every Wave 1
    # title, so the role alone cannot show whose answer the run took. An SME
    # who is wrong about every title names another role than the machine did.
    settings = dataclasses.replace(mini_settings, sme_error_rate=1.0)
    result = pipeline.run_pipeline(settings)
    approved = {
        TitleKey(family, title): role_id
        for family, title, role_id in rows_of(
            settings,
            "SELECT family_code, cleaned_title, role_id FROM human_decisions "
            "WHERE occasion = 'Wave 1 approval'",
        )
    }
    assert set(approved) == set(WAVE1)
    accepted = {
        TitleKey(family, title): role_id
        for family, title, role_id in rows_of(
            settings,
            "SELECT family_code, cleaned_title, role_id FROM model_answers "
            "WHERE stage = 'Wave 1' AND route = 'accept'",
        )
        if TitleKey(family, title) in approved
    }
    assert len(accepted) >= 3, "the test needs Wave 1 titles whose proposal the rules accepted"
    for key, machine_role in accepted.items():
        assert machine_role and machine_role != approved[key], key
    for key in WAVE1:
        assert decision_of(result, key).role_id == approved[key], key


def test_in_wave_1_the_agent_only_recommends(mini_settings):
    # With a chooser that finds no role, the rules refer the titles to the
    # agent, and the stand-in agent settles some of them.
    models = real_models(mini_settings)
    spy = Spy(models, chooser=NoMatch())
    result = pipeline.run_pipeline(mini_settings, models=spy.models)
    settled = rows_of(
        mini_settings,
        "SELECT family_code, cleaned_title FROM scorings "
        "WHERE stage = 'Wave 1' AND route = 'agent-settled'",
    )
    assert settled, "the test needs a Wave 1 title that the agent settled"
    for key in settled:
        decision = decision_of(result, TitleKey(*key))
        assert (decision.row, decision.source) == (TableRow.WAVE1, DecisionSource.PERSON)
    # From Wave 2 on its answer stands: the same answer there is a machine path.
    wave2 = rows_of(
        mini_settings,
        "SELECT family_code, cleaned_title FROM scorings "
        "WHERE stage = 'Wave 2' AND route = 'agent-settled'",
    )
    assert wave2
    assert {decision_of(result, TitleKey(*key)).row for key in wave2} == {TableRow.AGENT_SETTLED}


def test_wave_1_is_scored_before_any_approval_is_learned(mini_dir, mini_settings, tmp_path):
    # Q22 (b): with the true roles changed and every other file as it was, the
    # machine's Wave 1 proposals and its golden-set answers are the same.
    changed = copy_of_mini(mini_dir, tmp_path)
    truth = people.read_truth(changed)

    def another_role(role_id: str) -> str:
        """The next role of the same family: SAL-01 gives SAL-02, SAL-06 gives SAL-01."""
        family, number = role_id.split("-")
        return f"{family}-{int(number) % 6 + 1:02d}"

    people.write_truth(
        changed,
        [dataclasses.replace(row, true_role_id=another_role(row.true_role_id)) for row in truth],
    )
    other = dataclasses.replace(mini_settings, data_dir=changed, out_dir=tmp_path / "other out")
    first, second = pipeline.run_pipeline(mini_settings), pipeline.run_pipeline(other)
    assert first.golden_keys == second.golden_keys and first.wave1_keys == second.wave1_keys

    def machine_answers(settings: Settings, stage: str) -> list[list[tuple]]:
        return [
            rows_of(settings, f"SELECT * FROM {table} WHERE stage = ? ORDER BY rowid", stage)
            for table in ("scorings", "ml_shortlist", "model_answers", "shown_roles")
        ]

    assert machine_answers(mini_settings, "Wave 1") == machine_answers(other, "Wave 1")
    assert len(machine_answers(mini_settings, "Wave 1")[0]) == len(WAVE1) + len(GOLDEN)
    # The change did reach the people, and through what they approved the models:
    # nothing is claimed after the first human decision.
    approvals = "SELECT role_id FROM human_decisions WHERE occasion = 'Wave 1 approval'"
    assert rows_of(mini_settings, approvals) != rows_of(other, approvals)
    assert machine_answers(mini_settings, "Wave 2") != machine_answers(other, "Wave 2")


def test_a_seed_look_up_hit_on_the_wave_1_list_stays_a_look_up_hit(
    mini_dir, mini_settings, tmp_path
):
    # Q13: the look-up runs first; the title is not approved again, and the
    # Wave 1 row shows fewer than wave1_size titles.
    data_dir = copy_of_mini(mini_dir, tmp_path)
    with (data_dir / datafiles.SEED_LOOKUP_FILE).open("a", encoding="utf-8") as handle:
        handle.write("ACCT MGR II,SAL,SAL-01\n")
    settings = dataclasses.replace(mini_settings, data_dir=data_dir)
    spy = Spy(real_models(settings))
    result = pipeline.run_pipeline(settings, models=spy.models)

    hit = TitleKey("SAL", "account manager ii")
    assert result.wave1_keys == WAVE1 and hit in WAVE1  # the list is drawn over all rows
    decision = decision_of(result, hit)
    assert (decision.row, decision.source) == (TableRow.LOOKUP, DecisionSource.LOOKUP)
    assert (decision.role_id, decision.status) == ("SAL-01", Status.APPROVED)
    assert (decision.decided_by, decision.simulated) == (LOOKUP_DECIDED_BY, False)
    in_wave1_row = [each.key for each in result.decisions if each.row is TableRow.WAVE1]
    assert in_wave1_row == [key for key in WAVE1 if key != hit]
    # A hit takes its role with no model and no person.
    assert hit not in spy.chosen_keys
    asked = rows_of(settings, "SELECT family_code, cleaned_title FROM human_decisions")
    assert tuple(hit) not in asked
    assert rows_of(settings, "SELECT count(*) FROM scorings WHERE cleaned_title = ?", hit[1]) == [
        (0,)
    ]


# --- what people decide (1.8, Q14, Q26) -------------------------------------------------


@pytest.mark.parametrize("undecided_rate", [0.1, 1.0])
def test_no_model_is_called_for_a_title_with_a_name_or_no_usable_family(
    mini_settings, mini_rows, undecided_rate
):
    settings = dataclasses.replace(mini_settings, undecided_rate=undecided_rate)
    spy = Spy(real_models(settings))
    result = pipeline.run_pipeline(settings, models=spy.models)
    records = {row.key: row.record for row in mini_rows}
    for key in (NAMED, NO_FAMILY):
        assert key not in spy.chosen_keys
        assert all(transcript.case.record != records[key] for transcript in spy.turns)
    # A title with a person's name reaches embed in no way, decided or not (Q14).
    assert embed_text(records[NAMED]) not in spy.embedded
    assert not any("[NAME]" in text and "customer success" in text for text in spy.embedded)
    if undecided_rate == 1.0:
        # Nobody decided anything in the queue, so nothing of the title with
        # no usable family was learned: it was never embedded either (Q26).
        assert embed_text(records[NO_FAMILY]) not in spy.embedded
        assert decision_of(result, NO_FAMILY).row is TableRow.LEGACY
    queue = queue_file(settings)
    assert queue[NAMED]["reasons"] == "person's name"
    assert queue[NO_FAMILY]["reasons"] == "no usable family code"
    for key in (NAMED, NO_FAMILY):
        line = queue[key]
        assert (line["ml_first_choice"], line["llm_pick"], line["agent_recommendation"]) == (
            "",
            "",
            "",
        )
        assert line["answer_source"] == "" and line["decided_by"] == "Reviewer (simulated)"
    assert spy.chosen and spy.embedded  # the other titles did go to the models


def test_a_decided_title_with_a_name_joins_the_look_up_and_never_the_examples(mini_settings):
    # Q14, Q26: the one with no usable family joins both, under its key.
    result = pipeline.run_pipeline(mini_settings)
    assert decision_of(result, NAMED).row is TableRow.REVIEW_QUEUE
    assert decision_of(result, NO_FAMILY).role_id == "FIN-02"
    looked_up = rows_of(mini_settings, "SELECT family_code, cleaned_title, role_id FROM lookup")
    examples = rows_of(mini_settings, "SELECT family_code, cleaned_title, role_id FROM examples")
    assert (*NAMED, "SAL-06") in looked_up and (*NO_FAMILY, "FIN-02") in looked_up
    assert (*NO_FAMILY, "FIN-02") in examples
    assert NAMED not in [TitleKey(*row[:2]) for row in examples]


def test_what_people_decide_is_learned_after_the_wave_it_was_decided_in(mini_settings):
    # 1.6 step 10, Q18: score a whole wave, then learn its decisions.
    result = pipeline.run_pipeline(mini_settings)
    joined = {}
    for family, title, stage in rows_of(
        mini_settings, "SELECT family_code, cleaned_title, joined_at FROM lookup"
    ):
        joined.setdefault(stage, set()).add(TitleKey(family, title))
    queue_decided = {
        each.key
        for each in result.decisions
        if each.row is TableRow.REVIEW_QUEUE and each.key not in GOLDEN
    }
    assert joined == {
        "set-up": {TitleKey("FIN", "accounts payable clerk"), TitleKey("SAL", "sales engineer")},
        "Wave 1": set(WAVE1),
        "Wave 2": {TREASURY},  # audited; the other two audited titles are golden
        "close": queue_decided,
    }
    # The seed entries are examples from the start (Q13); a title with a
    # person's name is the only look-up entry that is never one (Q14).
    examples = rows_of(mini_settings, "SELECT family_code, cleaned_title, joined_at FROM examples")
    assert {TitleKey(*row[:2]) for row in examples if row[2] == "set-up"} == joined["set-up"]
    assert {TitleKey(*row[:2]) for row in examples} == set().union(*joined.values()) - {NAMED}


def test_each_scoring_of_the_golden_set_meets_the_examples_that_joined_before_it(mini_settings):
    # Q08: scored in Wave 1 before the Wave 1 approvals join, in Wave 2 on the
    # answers of Wave 2, and at the close after the queue decisions join.
    pipeline.run_pipeline(mini_settings)
    joined = dict(rows_of(mini_settings, "SELECT cleaned_title, joined_at FROM examples"))
    approved, audited, queued = "senior account manager", "treasury analyst", "special projects"
    assert [joined[title] for title in (approved, audited, queued)] == ["Wave 1", "Wave 2", "close"]
    # The approved titles that scored best for a role of the shortlist of one golden title.
    met = {
        stage: {
            text
            for (text,) in rows_of(
                mini_settings,
                "SELECT matched_text FROM ml_shortlist "
                "WHERE stage = ? AND family_code = ? AND cleaned_title = ?",
                stage,
                *GOLDEN[0],
            )
        }.intersection(joined)
        for stage in ("Wave 1", "Wave 2", "close")
    }
    assert met["Wave 1"] <= {title for title, stage in joined.items() if stage == "set-up"}
    assert approved in met["Wave 2"] and not {audited, queued} & met["Wave 2"]
    assert {approved, audited, queued} <= met["close"]


# --- the eight rules of 2.7 that need a model handed in ------------------------------------


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"quote": "manages the bank accounts of the group"}, "quote not in record"),
        ({"quote": ""}, "quote not in record"),
        ({"number": 11}, "role not in catalog"),  # Q15: out of range
        ({"number": 0}, "role not in catalog"),
        # What a hosted model could answer and no file or database takes as it is.
        ({"number": 10**30}, "role not in catalog"),
        ({"number": [3]}, "role not in catalog"),
        ({"quote": "treasury \ud800 analyst"}, "quote not in record"),
    ],
)
def test_t2_a_quote_that_is_not_in_the_record_sends_the_title_to_a_person(
    mini_settings, changes, reason
):
    # 2.7, with an injected answer (Q15, Q17): the stand-in never misquotes.
    honest = pipeline.run_pipeline(mini_settings)
    assert decision_of(honest, TREASURY).row is TableRow.MODELS_AGREE

    models = real_models(mini_settings)
    chooser = Changing(models.llm.chooser, {TREASURY.cleaned_title}, **changes)
    spy = Spy(models, chooser=chooser)
    result = pipeline.run_pipeline(mini_settings, models=spy.models)

    decision = decision_of(result, TREASURY)
    assert (decision.row, decision.source) == (TableRow.REVIEW_QUEUE, DecisionSource.PERSON)
    assert decision.decided_by == "Reviewer (simulated)"
    assert queue_file(mini_settings)[TREASURY]["reasons"] == reason
    in_a_path = {key for audit in result.audits for key in audit.audited_keys}
    assert TREASURY not in in_a_path
    # A person beats the agent: the title was not referred to it.
    assert all(turn.case.record.cleaned_title != TREASURY.cleaned_title for turn in spy.turns)
    # Every other title is decided as before.
    assert [each for each in result.decisions if each.key != TREASURY] == [
        each for each in honest.decisions if each.key != TREASURY
    ]
    # The state keeps the answer that was refused, and only the rules read it.
    ((answer, role_id, route),) = rows_of(
        mini_settings,
        "SELECT answer, role_id, route FROM model_answers "
        "WHERE stage = 'Wave 2' AND step = 'choose' AND cleaned_title = ?",
        TREASURY.cleaned_title,
    )
    assert (answer, route) == ("role", "person")
    assert (role_id is None) == (reason == "role not in catalog")


def test_t2_a_chooser_that_never_quotes_the_record_settles_nothing(mini_settings):
    models = real_models(mini_settings)
    misquoting = Changing(
        models.llm.chooser,
        {row.record.cleaned_title for row in pipeline.run_pipeline(mini_settings).rows},
        quote="words the record does not hold",
    )
    result = pipeline.run_pipeline(mini_settings, models=Spy(models, chooser=misquoting).models)
    assert [audit.path_size for audit in result.audits] == [0, 0]
    assert not [each for each in result.decisions if each.source is DecisionSource.MACHINE]
    reasons = [line["reasons"] for line in queue_file(mini_settings).values()]
    assert sum("quote not in record" in each for each in reasons) == 7  # every scored title


class Picking:
    """A reviewer that runs the stand-in's searches and then picks one role by
    its id, with the number the case gave it; it escalates when it has none."""

    def __init__(self, reviewer, role_id: str) -> None:
        self._reviewer, self._role_id = reviewer, role_id

    def next_turn(self, transcript):
        turn = self._reviewer.next_turn(transcript)
        if isinstance(turn, SearchRequest):
            return turn
        for shown in transcript.numbered_roles:
            if shown.role.role_id == self._role_id:
                title = transcript.case.record.cleaned_title
                return ReviewAnswer(ReviewKind.ROLE, shown.number, title, "test")
        return ReviewAnswer(ReviewKind.ESCALATE, None, "", "test")


def test_t3_a_flagged_role_the_agent_picks_is_never_settled_by_a_machine(
    mini_settings, mini_catalog
):
    # 2.7, with an injected reviewer that returns a compliance-flagged role.
    (flagged,) = [role.role_id for role in mini_catalog.roles if role.compliance_flag]
    settings = dataclasses.replace(mini_settings, undecided_rate=0.0)
    models = real_models(settings)
    spy = Spy(models, chooser=NoMatch(), reviewer=Picking(models.llm.reviewer, flagged))
    result = pipeline.run_pipeline(settings, models=spy.models)
    picked = rows_of(
        settings,
        "SELECT family_code, cleaned_title, route, reasons FROM model_answers "
        "WHERE stage = 'Wave 2' AND step = 'review' AND role_id = ?",
        flagged,
    )
    assert picked, "the test needs a title for which the agent picked the flagged role"
    queue = queue_file(settings)
    for family, title, route, reasons in picked:
        key = TitleKey(family, title)
        assert route == "person" and reasons == "compliance-flagged role"
        assert queue[key]["reasons"] == "no match; compliance-flagged role"
        assert queue[key]["agent_recommendation"] == flagged
        decision = decision_of(result, key)
        assert (decision.row, decision.source) == (TableRow.REVIEW_QUEUE, DecisionSource.PERSON)
    assert [audit.path_size for audit in result.audits] == [0, 0]


@pytest.mark.parametrize(
    "number", [0, 1, 2, 3, 4, 5, 6, 7, 11, 40, None, True, "1", 1.0, 10**30, -(10**30), [1]]
)
def test_t4_the_agent_cannot_pick_a_role_it_was_not_shown_or_one_outside_the_family(
    mini_settings, mini_catalog, number
):
    # 2.7, with an injected reviewer: whatever number it returns, the role is
    # one of the recorded family that it was shown, or the title goes to a person.
    settings = dataclasses.replace(mini_settings, undecided_rate=0.0)
    models = real_models(settings)
    reviewer = AnsweringWith(models.llm.reviewer, number)
    spy = Spy(models, chooser=NoMatch(), reviewer=reviewer)
    result = pipeline.run_pipeline(settings, models=spy.models)

    answers = rows_of(
        settings,
        "SELECT family_code, cleaned_title, role_id, route, reasons FROM model_answers "
        "WHERE stage = 'Wave 2' AND step = 'review'",
    )
    assert len(answers) == 4, "the test needs titles that the rules referred to the agent"
    queue = queue_file(settings)
    picked = 0
    for family, title, role_id, route, reasons in answers:
        key = TitleKey(family, title)
        shown = rows_of(
            settings,
            "SELECT number, role_id FROM shown_roles "
            "WHERE stage = 'Wave 2' AND step = 'review' AND cleaned_title = ? AND family_code = ?",
            title,
            family,
        )
        in_family = {role.role_id for role in roles_in_family(mini_catalog, family)}
        # Only roles of the recorded family carry a number at all (Q10).
        assert {each for shown_number, each in shown if shown_number is not None} <= in_family
        assert {each for shown_number, each in shown if shown_number is None}.isdisjoint(in_family)
        if role_id is None:
            assert route == "person" and reasons == "role not in catalog"
            assert queue[key]["reasons"] == "no match; role not in catalog"
            assert decision_of(result, key).source is DecisionSource.PERSON
            assert decision_of(result, key).row is TableRow.REVIEW_QUEUE
            continue
        picked += 1
        assert role_id in in_family and (number, role_id) in shown
        if route == "accept":
            assert decision_of(result, key).row is TableRow.AGENT_SETTLED
        else:  # the one other way out: the role it picked is compliance-flagged (1.7)
            assert reasons == "compliance-flagged role"
            assert decision_of(result, key).row is TableRow.REVIEW_QUEUE
    # In these cases 1 to 6 number the roles of the family, and nothing else does.
    names_a_role = type(number) is int and 1 <= number <= 6
    assert bool(picked) == names_a_role
    if not names_a_role:
        assert result.audits[1].path_size == 0


class AskingOddly:
    """A reviewer whose first search asks with text that is not valid Unicode."""

    def __init__(self, reviewer) -> None:
        self._reviewer = reviewer

    def next_turn(self, transcript):
        turn = self._reviewer.next_turn(transcript)
        if isinstance(turn, SearchRequest) and not transcript.steps:
            return dataclasses.replace(turn, query=turn.query + " \udfff")
        return turn


def test_a_search_is_run_and_recorded_whatever_the_agent_asks_with(mini_settings):
    # A model's text reaches the embedder and the state as it is; neither stops the run.
    models = real_models(mini_settings)
    spy = Spy(models, chooser=NoMatch(), reviewer=AskingOddly(models.llm.reviewer))
    result = pipeline.run_pipeline(mini_settings, models=spy.models)
    asked = rows_of(
        mini_settings,
        "SELECT DISTINCT query FROM shown_roles WHERE shown_in = 'search_catalog'",
    )
    assert asked and all(query.endswith(" \\udfff") for (query,) in asked)
    assert result.audits[1].path_size > 0  # the agent went on to settle titles


def test_t5_no_model_is_ever_shown_a_generic_family_role(mini_settings, mini_catalog):
    # 2.7: never embedded, never on a shortlist, in a numbered list or in a
    # search result. Once with the stand-ins, once with a chooser that sends
    # every title it can to the agent, so that both searches run.
    generic = [role for role in mini_catalog.roles if role.is_generic]
    assert len(generic) == len(family_codes(mini_catalog)) == 2
    models = real_models(mini_settings)
    for spy in (Spy(models), Spy(models, chooser=NoMatch())):
        pipeline.run_pipeline(mini_settings, models=spy.models)
        assert spy.chosen and spy.embedded
        assert not [shown for shown in spy.shown_roles if shown.role.is_generic]
        for role in generic:
            assert role_text(role) not in spy.embedded
            assert not [text for text in spy.embedded if role.definition in text]
        listed = rows_of(mini_settings, "SELECT DISTINCT role_id FROM ml_shortlist")
        shown = rows_of(mini_settings, "SELECT DISTINCT role_id FROM shown_roles")
        assert listed and shown
        assert {role.role_id for role in generic}.isdisjoint(
            role_id for (role_id,) in (*listed, *shown)
        )
    searches = {step.request.tool for turn in spy.turns for step in turn.steps}
    assert len(searches) == 2  # the last run searched the catalog and the approved titles


def test_t6_golden_titles_never_enter_the_look_up_table_or_the_examples(mini_settings):
    # 2.7, after a full run in which people did decide golden titles: two were
    # labeled in an audit and one was decided in the review queue (Q07, Q21).
    result = pipeline.run_pipeline(mini_settings)
    decided_by_people = {
        TitleKey(family, title)
        for family, title in rows_of(
            mini_settings,
            "SELECT family_code, cleaned_title FROM human_decisions "
            "WHERE occasion IN ('audit label', 'review-queue decision') AND answer = 'role'",
        )
    }
    assert set(GOLDEN) <= decided_by_people
    for table in ("lookup", "examples"):
        held = rows_of(mini_settings, f"SELECT family_code, cleaned_title FROM {table}")
        assert held and not set(GOLDEN) & {TitleKey(*key) for key in held}, table
    # The person's decision is still used for the title's release (Q07).
    assert {decision_of(result, key).status for key in GOLDEN} == {Status.APPROVED}


def test_t6_the_run_stops_if_a_golden_title_is_learned(mini_settings, monkeypatch):
    # Q08: asserted after every stage. Here the guard of titlestd.lookup is taken away.
    def learn_anything(self, row, role_id):
        self._roles[row.key] = role_id
        self._examples[row.key] = Example(row.record, role_id)
        return True

    monkeypatch.setattr(lookup.Knowledge, "learn", learn_anything)
    with pytest.raises(TitlestdError, match=r"golden-set title .* entered the look-up table"):
        pipeline.run_pipeline(mini_settings)
    assert not (mini_settings.out_dir / release.LOAD_FILE).exists()


class FlaggedOnce:
    """The stand-in's answers for one golden-set title, which the LLM is asked
    about three times (Q08): in the golden-set pass of Wave 1, in Wave 2 and in
    the golden-set pass of the close. In Wave 2 the quote is not in the
    record, so the title goes to the review queue for a reason that has
    nothing to do with compliance. In the stage the test names, the pick is
    the compliance-flagged role."""

    def __init__(self, chooser, title: str, flagged_role: str, stage: str | None) -> None:
        self._chooser, self._title, self._role, self._stage = chooser, title, flagged_role, stage
        self._stages = iter(("Wave 1", "Wave 2", "close"))

    def choose(self, record, roles):
        answer = self._chooser.choose(record, roles)
        if record.cleaned_title != self._title:
            return answer
        stage = next(self._stages)
        if stage == self._stage:
            (number,) = [shown.number for shown in roles if shown.role.role_id == self._role]
            return dataclasses.replace(answer, answer=ChooseKind.ROLE, number=number)
        if stage == "Wave 2":
            return dataclasses.replace(answer, quote="words the record does not hold")
        return answer


@pytest.mark.parametrize("flagged_in", [None, "Wave 1", "close"])
def test_t7_a_leftover_the_compliance_check_ever_fired_on_takes_no_generic_role(
    mini_settings, mini_catalog, flagged_in
):
    # Q25: eligible only if the check "never fired on any proposal or on the ML
    # top three". A golden-set title is scored three times, against other
    # examples each time, and the check counts wherever it fired.
    (flagged,) = [role.role_id for role in mini_catalog.roles if role.compliance_flag]
    title = GOLDEN[0]
    settings = dataclasses.replace(mini_settings, undecided_rate=1.0, generic_cap_share=1.0)
    models = real_models(settings)
    chooser = FlaggedOnce(models.llm.chooser, title.cleaned_title, flagged, flagged_in)
    llm = dataclasses.replace(models.llm, chooser=chooser)
    result = pipeline.run_pipeline(settings, models=Models(models.embedder, llm))

    scored = dict(
        rows_of(
            settings,
            "SELECT stage, reasons FROM scorings WHERE family_code = ? AND cleaned_title = ?",
            *title,
        )
    )
    fired_in = {stage for stage, reasons in scored.items() if "compliance-flagged role" in reasons}
    assert set(scored) == {"Wave 1", "Wave 2", "close"}
    assert fired_in == ({flagged_in} if flagged_in else set())
    # In Wave 2 nothing about the title touched a flagged role, and nobody decided it.
    line = queue_file(settings)[title]
    assert (line["reasons"], line["decision"]) == ("quote not in record", "could not decide")
    decision = decision_of(result, title)
    if flagged_in is None:
        assert (decision.row, decision.role_id) == (TableRow.GENERIC, "FIN-GEN")
    else:
        assert (decision.row, decision.role_id) == (TableRow.LEGACY, None)
        assert decision.legacy_reason.value == "undecided widely held or flagged title"


# --- the ML-alone route (Q04) ---------------------------------------------------------


def test_the_ml_alone_route_skips_the_llm_and_still_goes_to_a_person(mini_settings):
    settings = dataclasses.replace(
        mini_settings, ml_alone=True, ml_margin_cutoff=0.0, ml_min_examples=1
    )
    spy = Spy(real_models(settings))
    result = pipeline.run_pipeline(settings, models=spy.models)
    scored = rows_of(
        settings,
        "SELECT family_code, cleaned_title, ml_examples, route, reasons FROM scorings "
        "WHERE stage = 'Wave 2'",
    )
    alone = {TitleKey(family, title) for family, title, examples, _, _ in scored if examples >= 1}
    with_llm = {TitleKey(family, title) for family, title, examples, _, _ in scored if not examples}
    assert alone and with_llm, "the test needs titles on both routes"
    # Margin and example count are stored for every title either way.
    assert len(scored) == len(alone) + len(with_llm) == 7

    wave2_chosen = set(spy.chosen_keys) - set(WAVE1) - set(GOLDEN)
    assert wave2_chosen == with_llm - set(GOLDEN)
    queue = queue_file(settings)
    for key in alone:
        assert key not in spy.chosen_keys or key in GOLDEN
        line = queue[key]
        assert line["reasons"].split("; ")[0] == "ML alone"
        assert line["llm_pick"] == "" and line["answer_source"] == ""
        assert line["ml_first_choice"]
        decision = decision_of(result, key)
        assert (decision.row, decision.source) == (TableRow.REVIEW_QUEUE, DecisionSource.PERSON)
        assert decision.status is Status.APPROVED
    # No third path: the two audits are those of the two machine paths, and
    # no title of the route is in either.
    assert [audit.path.value for audit in result.audits] == ["models agree", "agent-settled"]
    in_a_path = {
        each.key
        for each in result.decisions
        if each.row in (TableRow.MODELS_AGREE, TableRow.AGENT_SETTLED)
    }
    assert in_a_path and in_a_path.isdisjoint(alone)
    assert rows_of(settings, "SELECT count(*) FROM model_answers WHERE stage = 'Wave 2'") == [
        (len(with_llm),)
    ]


def test_with_the_ml_alone_route_off_every_scored_title_goes_to_the_llm(mini_settings):
    # Off by default (2.4), whatever the margin and the examples are.
    settings = dataclasses.replace(mini_settings, ml_margin_cutoff=0.0, ml_min_examples=0)
    spy = Spy(real_models(settings))
    pipeline.run_pipeline(settings, models=spy.models)
    scored = rows_of(settings, "SELECT count(*) FROM scorings")
    assert scored == [(len(spy.chosen),)]
    assert "ML alone" not in "".join(line["reasons"] for line in queue_file(settings).values())


# --- "no suitable role" (Q03) and the reasons of the queue ----------------------------------


def test_a_title_marked_no_suitable_role_waits_for_the_catalog_owner(
    mini_dir, mini_settings, tmp_path
):
    # The mark is made only where the truth has no row for the title: here a
    # Wave 1 title, one that an audit labels and one of the review queue.
    data_dir = copy_of_mini(mini_dir, tmp_path)
    marked = {
        TitleKey("FIN", "financial analyst"): "Fin Analyst",  # on the Wave 1 list
        TREASURY: "Treasury Analyst",  # models agree, and the path is labeled in full
        TitleKey("FIN", "internal auditor"): "Internal Auditor",  # in the review queue
    }
    truth = people.read_truth(data_dir)
    people.write_truth(data_dir, [row for row in truth if row.job_title not in marked.values()])
    # Every other queue title is left undecided and may take a generic role.
    settings = dataclasses.replace(
        mini_settings, data_dir=data_dir, undecided_rate=1.0, generic_cap_share=1.0
    )
    result = pipeline.run_pipeline(settings)

    queue = queue_file(settings)
    exceptions = output(settings, release.EXCEPTIONS_FILE, release.EXCEPTIONS_COLUMNS)
    for key in marked:
        decision = decision_of(result, key)
        assert (decision.row, decision.source) == (TableRow.LEGACY, DecisionSource.NONE)
        assert decision.legacy_reason.value == "no suitable role"
        line = queue[key]
        assert line["reasons"].split("; ")[-1] == "no suitable role, waiting for catalog owner"
        assert line["decision"] == "no suitable role" and line["simulated"] == "1"
        assert line["outcome"] == "left on legacy title"
    assert queue[WAVE1[0]]["decided_by"].startswith("SME ")
    assert queue[TREASURY]["decided_by"].startswith("SME ")
    assert queue[TitleKey("FIN", "internal auditor")]["decided_by"] == "Reviewer (simulated)"
    assert (
        sorted(line["reason"] for line in exceptions if line["released"] == "0").count(
            "no suitable role"
        )
        == 4
    )  # two employees hold the Wave 1 title
    # Never a generic role, while an ordinary leftover of the same run takes one.
    generic = [each.key for each in result.decisions if each.row is TableRow.GENERIC]
    assert generic and not set(generic) & set(marked)
    # Nothing was learned for a title with no role.
    looked_up = rows_of(settings, "SELECT family_code, cleaned_title FROM lookup")
    assert not {TitleKey(*key) for key in looked_up} & set(marked)
    # The audit counts the mark as wrong, and the title is not released on the machine's role.
    agree = result.audits[0]
    assert TREASURY in agree.audited_keys and agree.wrong == 1


def test_the_reasons_of_a_queue_title_are_the_llm_s_and_then_the_agent_s(mini_settings):
    models = real_models(mini_settings)
    spy = Spy(models, chooser=NoMatch(), reviewer=AnsweringWith(models.llm.reviewer, 40))
    pipeline.run_pipeline(mini_settings, models=spy.models)
    queue = queue_file(mini_settings)
    referred = [line for line in queue.values() if "role not in catalog" in line["reasons"]]
    assert referred
    for line in referred:
        assert line["reasons"] == "no match; role not in catalog"
        assert line["llm_pick"] == "" and line["agent_recommendation"] == ""
        assert line["answer_source"] == "test"
    # A flagged role among the ML top three sends the title to a person at once.
    flagged = queue[TitleKey("FIN", "internal auditor")]
    assert flagged["reasons"] == "compliance-flagged role; no match"
