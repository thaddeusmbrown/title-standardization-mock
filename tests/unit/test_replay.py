"""llm = replay: recorded answers, and nothing else (2.5, Q29, Q31, Q40).

Every recording here is written by the program's own recorder into a
temporary folder, under the model name "test-double" (Q32): none is written
by hand and none is committed.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from titlestd import recording, replay
from titlestd.contracts import (
    AgentCase,
    AgentTranscript,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    TitlestdError,
    recorded_label,
)
from titlestd.recording import CHOOSE_FILE, MANIFEST_FILE, STEP_CHOOSE, STEP_REVIEW, RecordingError
from titlestd.replay import ReplayMiss
from titlestd.settings import Settings

MODEL = "test-double"
FINGERPRINT = "0f" * 32
RECORDED_ON = datetime(2026, 1, 5, 23, 30, tzinfo=UTC)

RECORD = Record(
    "senior account manager",
    "SAL",
    "Enterprise Sales",
    "Owns the largest customer accounts and negotiates renewals.",
)
OTHER = Record("treasury analyst", "FIN", "Treasury", "Manages cash positions.")
ACCOUNT_MANAGER = CatalogRole(
    "SAL-01", "SAL", "Account Manager", "Acct Mgr", "Manages a portfolio of customer accounts."
)
TREASURY_ANALYST = CatalogRole(
    "FIN-06", "FIN", "Treasury Analyst", "Treasury Analyst", "Manages cash positions."
)
ROLES = (NumberedRole(1, TREASURY_ANALYST), NumberedRole(2, ACCOUNT_MANAGER))
CASE = AgentCase(
    record=RECORD,
    roles=(NumberedRole(None, TREASURY_ANALYST), NumberedRole(1, ACCOUNT_MANAGER)),
    llm_answer=ChooseAnswer(ChooseKind.ROLE, 1, "account manager", "a label"),
    llm_role_id="FIN-06",
    ml_first_choice="SAL-01",
    reasons=(Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH),
)
STEP = SearchStep(
    SearchRequest(SearchTool.CATALOG, "account manager"),
    (SearchHit(NumberedRole(1, ACCOUNT_MANAGER)), SearchHit(NumberedRole(None, TREASURY_ANALYST))),
)
FEWER_HITS = SearchStep(STEP.request, STEP.hits[:1])


def transcript(*steps: SearchStep) -> AgentTranscript:
    return AgentTranscript(CASE, steps)


def make_recording(settings: Settings, model: str = MODEL, fingerprint: str = FINGERPRINT) -> Path:
    """A complete recording, written by the recorder: two choose answers, one of
    them invalid, and the two turns of one agent case."""
    recorder = recording.Recorder(settings, model, fingerprint)
    turns = [
        (STEP_CHOOSE, recording.choose_request(RECORD, ROLES), ("role", 2, "account manager", "")),
        (STEP_CHOOSE, recording.choose_request(OTHER, ROLES), ("invalid", None, "", "refused")),
        (STEP_REVIEW, recording.review_request(transcript()), None),
        (STEP_REVIEW, recording.review_request(transcript(STEP)), ("role", 1, "renewals", "")),
    ]
    for step, request, answer in turns:
        turn = recording.search_turn("search_catalog", "account manager")
        if answer is not None:
            turn = recording.answer_turn(*answer)
        recorder.record(
            recording.request_key(step, model, request),
            step,
            RECORD,
            turn,
            reported_model=f"{model}-0001",
            stop_reason="end_turn",
            input_tokens=1,
            output_tokens=1,
        )
    recorder.finish()
    return recorder.folder


@pytest.fixture(autouse=True)
def recorded_on(monkeypatch) -> None:
    monkeypatch.setattr(recording, "utc_now", lambda: RECORDED_ON)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(llm="replay", recording_dir=tmp_path / "recordings")


LABEL = "recorded 2026-01-05, test-double"


# --- Answers come from the recording (2.5, Q20, Q40) ----------------------------------


def test_a_recorded_answer_is_given_again_under_the_recorded_label(settings: Settings):
    make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    assert llm.chooser.choose(RECORD, ROLES) == ChooseAnswer(
        ChooseKind.ROLE, 2, "account manager", LABEL
    )
    assert recorded_label("2026-01-05", MODEL) == LABEL


def test_an_invalid_answer_is_repeated_with_its_reason(settings: Settings):
    # Q39: recorded so replay repeats it.
    make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    assert llm.chooser.choose(OTHER, ROLES) == ChooseAnswer(
        ChooseKind.INVALID, None, "", LABEL, "refused"
    )


def test_each_agent_turn_is_looked_up_and_a_search_is_handed_back_to_be_run(settings: Settings):
    # Q20: one entry per model turn; the shared loop runs the search again.
    make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    assert llm.reviewer is llm.chooser
    assert llm.reviewer.next_turn(transcript()) == SearchRequest(
        SearchTool.CATALOG, "account manager"
    )
    assert llm.reviewer.next_turn(transcript(STEP)) == ReviewAnswer(
        ReviewKind.ROLE, 1, "renewals", LABEL
    )


def test_the_label_holds_the_day_stored_with_each_answer(settings: Settings, monkeypatch):
    # Q40: the UTC day of the call, read back; a recording may span two days.
    days = [RECORDED_ON, RECORDED_ON, datetime(2026, 1, 6, 0, 10, tzinfo=UTC)]
    monkeypatch.setattr(recording, "utc_now", lambda: days.pop(0) if len(days) > 1 else days[0])
    make_recording(settings)
    monkeypatch.setattr(recording, "utc_now", lambda: datetime(2030, 7, 1, tzinfo=UTC))
    llm = replay.build_llm(settings, FINGERPRINT)
    assert llm.chooser.choose(RECORD, ROLES).source == "recorded 2026-01-05, test-double"
    assert llm.chooser.choose(OTHER, ROLES).source == "recorded 2026-01-06, test-double"


def test_replay_writes_nothing(settings: Settings):
    folder = make_recording(settings)
    before = {path: path.read_bytes() for path in settings.recording_dir.rglob("*.json*")}
    llm = replay.build_llm(settings, FINGERPRINT)
    llm.chooser.choose(RECORD, ROLES)
    llm.reviewer.next_turn(transcript())
    assert llm.finish() is None
    after = {
        path: path.read_bytes() for path in settings.recording_dir.rglob("*") if path.is_file()
    }
    assert after == before and len(before) == 3
    assert sorted(path.name for path in folder.parent.iterdir()) == ["offline"]


# --- Which recording is opened (Q31) -----------------------------------------------------


def test_with_no_model_set_the_one_recording_of_the_embedder_is_used(settings: Settings):
    make_recording(settings)
    make_recording(
        Settings(embedder="local", recording_dir=settings.recording_dir), "other-model"
    )  # another embedder's: not looked at
    assert settings.llm_model is None
    llm = replay.build_llm(settings, FINGERPRINT)
    assert llm.chooser.choose(RECORD, ROLES).source == LABEL


def test_the_model_is_read_from_the_manifest_not_from_the_folder_name(settings: Settings):
    folder = make_recording(settings)
    folder.parent.rename(folder.parent.with_name("renamed-by-someone"))
    llm = replay.build_llm(settings, FINGERPRINT)
    # The key holds the model as configured when recording (Q33), so the answers are found.
    assert llm.chooser.choose(RECORD, ROLES).source == LABEL


def test_with_the_model_set_its_folder_is_opened(settings: Settings):
    make_recording(settings)
    make_recording(settings, "claude-haiku-4-5")
    chosen = Settings(
        llm="replay", llm_model="claude-haiku-4-5", recording_dir=settings.recording_dir
    )
    llm = replay.build_llm(chosen, FINGERPRINT)
    assert llm.chooser.choose(RECORD, ROLES).source == "recorded 2026-01-05, claude-haiku-4-5"


def test_no_recording_is_an_error_at_start_up_that_names_the_record_command(settings: Settings):
    with pytest.raises(RecordingError) as raised:
        replay.build_llm(settings, FINGERPRINT)
    assert str(raised.value) == (
        f"llm = replay found no recording for embedder = offline under {settings.recording_dir}, "
        "nor one for another embedder. Recordings are made with the record command."
    )
    assert isinstance(raised.value, TitlestdError)


def test_a_recording_of_another_embedder_is_no_recording_of_this_one(settings: Settings):
    make_recording(settings)
    local = Settings(llm="replay", embedder="local", recording_dir=settings.recording_dir)
    with pytest.raises(RecordingError) as raised:
        replay.build_llm(local, FINGERPRINT)
    message = str(raised.value)
    assert message.startswith(
        f"llm = replay found no recording for embedder = local under {settings.recording_dir}. "
    )
    # Naming a model cannot help here, so the message does not advise it: it
    # says which embedder the recordings that are there were made with.
    assert "The recordings there were made with embedder = offline:" in message
    assert "--llm-model" not in message and "models found" not in message
    assert "embedder = local with the record command" in message


def test_more_than_one_recording_is_an_error_that_lists_the_models(settings: Settings):
    make_recording(settings, "model-b")
    make_recording(settings, "model-a")
    with pytest.raises(RecordingError) as raised:
        replay.build_llm(settings, FINGERPRINT)
    message = str(raised.value)
    assert "found more than one recording for embedder = offline" in message
    assert "(models found: model-a, model-b)" in message
    assert "--llm-model" in message


def test_a_model_that_differs_from_the_manifest_s_is_an_error_at_start_up(settings: Settings):
    folder = make_recording(settings)
    folder.parent.rename(folder.parent.with_name("claude-haiku-4-5"))
    chosen = Settings(
        llm="replay", llm_model="claude-haiku-4-5", recording_dir=settings.recording_dir
    )
    with pytest.raises(RecordingError) as raised:
        replay.build_llm(chosen, FINGERPRINT)
    message = str(raised.value)
    assert "was made with the model 'test-double'" in message
    assert "llm_model is 'claude-haiku-4-5'" in message


def test_a_model_with_no_recording_is_refused(settings: Settings):
    make_recording(settings)
    chosen = Settings(llm="replay", llm_model="other-model", recording_dir=settings.recording_dir)
    with pytest.raises(RecordingError, match=r"there is no recording in .*other-model"):
        replay.build_llm(chosen, FINGERPRINT)


# --- Refused before any work (Q29) -----------------------------------------------------------


def test_a_run_that_did_not_complete_left_no_recording_to_replay(settings: Settings):
    recorder = recording.Recorder(settings, MODEL, FINGERPRINT)
    key = recording.request_key(STEP_CHOOSE, MODEL, recording.choose_request(RECORD, ROLES))
    recorder.record(
        key,
        STEP_CHOOSE,
        RECORD,
        recording.answer_turn("role", 2, "account manager"),
        reported_model=MODEL,
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
    )
    assert recorder.partial_folder.is_dir()  # the answers are kept, but that is no recording
    with pytest.raises(RecordingError, match="found no recording"):
        replay.build_llm(settings, FINGERPRINT)
    chosen = Settings(llm="replay", llm_model=MODEL, recording_dir=settings.recording_dir)
    with pytest.raises(RecordingError, match="there is no recording in"):
        replay.build_llm(chosen, FINGERPRINT)


def test_a_recording_marked_incomplete_is_refused(settings: Settings):
    folder = make_recording(settings)
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    recording.write_manifest(folder, {**manifest, "complete": False})
    with pytest.raises(RecordingError, match="is not complete"):
        replay.build_llm(settings, FINGERPRINT)


def test_a_damaged_recording_is_refused_at_start_up_not_at_the_first_miss(settings: Settings):
    folder = make_recording(settings)
    text = (folder / CHOOSE_FILE).read_text(encoding="utf-8")
    (folder / CHOOSE_FILE).write_text(text.splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="holds 3 entries and its manifest counts 4"):
        replay.build_llm(settings, FINGERPRINT)


# --- A miss (2.5, Q18, Q29) -------------------------------------------------------------------


def test_a_miss_is_an_error_that_names_the_step_and_the_title(settings: Settings):
    folder = make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    unknown = Record("payroll specialist", "FIN", "Accounting", "")
    with pytest.raises(ReplayMiss) as raised:
        llm.chooser.choose(unknown, ROLES)
    message = str(raised.value)
    key = recording.request_key(STEP_CHOOSE, MODEL, recording.choose_request(unknown, ROLES))
    assert message.startswith(
        "replay miss at step choose for the title 'payroll specialist' in family 'FIN': "
        f"the recording in {folder} holds no answer for this request (key {key})."
    )
    assert "Its manifest matches the settings in force, so the request itself differs" in message
    assert "Replay never falls back to another model" in message
    assert isinstance(raised.value, TitlestdError)  # the command line exits with 1


@pytest.mark.parametrize(
    "roles",
    [ROLES[::-1], ROLES[:1], (NumberedRole(2, TREASURY_ANALYST), NumberedRole(1, ACCOUNT_MANAGER))],
    ids=["another order", "a shorter list", "other numbers"],
)
def test_the_same_record_with_other_roles_shown_is_a_miss(settings: Settings, roles):
    make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    with pytest.raises(ReplayMiss, match="at step choose for the title 'senior account manager'"):
        llm.chooser.choose(RECORD, roles)


def test_an_agent_turn_after_another_search_result_is_a_miss(settings: Settings):
    # Q20: the searches run for real; what they return is part of the next key.
    make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    with pytest.raises(ReplayMiss) as raised:
        llm.reviewer.next_turn(transcript(FEWER_HITS))
    assert "replay miss at step review for the title 'senior account manager' in family 'SAL'" in (
        str(raised.value)
    )
    with pytest.raises(ReplayMiss, match="at step review"):
        llm.reviewer.next_turn(transcript(STEP, STEP))  # a turn that was never recorded


def test_a_miss_prints_how_the_manifest_differs_from_the_settings_in_force(settings: Settings):
    make_recording(settings)
    other = Settings(
        llm="replay", recording_dir=settings.recording_dir, seed=7, agent_search_k=3, wave1_size=4
    )
    # The differences stop nothing by themselves: what was recorded is still found.
    llm = replay.build_llm(other, "ab" * 32)
    assert llm.chooser.choose(RECORD, ROLES).source == LABEL
    with pytest.raises(ReplayMiss) as raised:
        llm.chooser.choose(RECORD, ROLES[::-1])
    message = str(raised.value)
    assert (
        "Its manifest differs from the settings in force: seed: recorded 20260101, now 7; "
        "wave1_size: recorded 40, now 4; agent_search_k: recorded 5, now 3; "
        f"data_fingerprint: recorded '{FINGERPRINT}', now '{'ab' * 32}'."
    ) in message
    assert "matches the settings in force" not in message


def test_a_recording_of_older_prompts_says_so_on_a_miss(settings: Settings, monkeypatch):
    make_recording(settings)
    monkeypatch.setattr(recording, "PROMPT_VERSION", recording.PROMPT_VERSION + 1)
    monkeypatch.setattr(recording, "CHOOSE_SYSTEM", recording.CHOOSE_SYSTEM + "\nBe brief.")
    llm = replay.build_llm(settings, FINGERPRINT)
    with pytest.raises(ReplayMiss, match="prompt_version: recorded 1, now 2"):
        llm.chooser.choose(RECORD, ROLES)


def test_a_miss_is_never_answered_by_anything_else(settings: Settings):
    make_recording(settings)
    llm = replay.build_llm(settings, FINGERPRINT)
    for _ in range(2):  # and asking again does not help
        with pytest.raises(ReplayMiss):
            llm.chooser.choose(RECORD, ROLES[:1])
    assert llm.chooser.choose(RECORD, ROLES).number == 2  # the recorded ones still answer


# --- What replay never touches (2.5) --------------------------------------------------------


def test_replay_names_no_sdk_no_key_and_no_other_provider():
    # It builds its requests with titlestd.recording and imports neither the
    # live provider nor the stand-in (S09: a provider is imported only when
    # its setting is chosen).
    for module in (replay, recording):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {node.module, *(f"{node.module}.{a.name}" for a in node.names)}
        assert not {"os", "anthropic", "httpx2", "socket", "urllib"} & imported, module
        assert not {"titlestd.live", "titlestd.standin", "titlestd.switch"} & imported, module
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "environ" not in names and "getenv" not in names


def test_replay_does_not_read_the_key_even_when_one_is_set(settings: Settings, monkeypatch):
    class Unreadable(dict):
        def __getitem__(self, name):
            raise AssertionError(f"the environment was read for {name}")

        get = __getitem__

    make_recording(settings)
    with monkeypatch.context() as patched:
        patched.setattr("os.environ", Unreadable())
        llm = replay.build_llm(settings, FINGERPRINT)
        answer = llm.chooser.choose(RECORD, ROLES)
    assert answer.source == LABEL
