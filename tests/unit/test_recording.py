"""Recordings on disk: the folder, the manifest, the recorder, reading back (Q18, Q29).

Every recording here is written by the program's own recorder into a
temporary folder. The requests and their key have their own file,
tests/unit/test_recording_requests.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from titlestd import recording
from titlestd.contracts import Record, SearchRequest, SearchTool, TitlestdError
from titlestd.recording import (
    CHOOSE_FILE,
    MANIFEST_FILE,
    REVIEW_FILE,
    STEP_CHOOSE,
    STEP_REVIEW,
    Recorder,
    RecordingError,
)
from titlestd.settings import Settings

MODEL = "test-double"
FINGERPRINT = "0f" * 32

RECORD = Record(
    "senior account manager",
    "SAL",
    "Enterprise Sales",
    "Owns the largest customer accounts and negotiates renewals.",
)


def settings(tmp_path: Path, **changes) -> Settings:
    return Settings(recording_dir=tmp_path / "recordings", **changes)


# --- Where a recording is (Q29, Q31) ---------------------------------------------------


def test_the_folder_is_named_after_the_model_and_the_embedder(tmp_path: Path):
    chosen = settings(tmp_path, embedder="local")
    assert recording.recording_folder(chosen, "claude-haiku-4-5") == (
        tmp_path / "recordings" / "claude-haiku-4-5" / "local"
    )
    assert recording.recording_folder(Settings(), MODEL) == Path("recordings/test-double/offline")


@pytest.mark.parametrize("model", ["", ".", "..", "a/b", "../elsewhere", "/abs", "a\\b"])
def test_a_model_that_cannot_name_a_folder_is_refused(tmp_path: Path, model: str):
    with pytest.raises(RecordingError, match="one plain folder name"):
        recording.recording_folder(settings(tmp_path), model)
    with pytest.raises(RecordingError):
        Recorder(settings(tmp_path), model, FINGERPRINT)
    assert not (tmp_path / "recordings").exists()


def test_the_models_found_are_those_with_a_folder_for_the_embedder_in_force(tmp_path: Path):
    chosen = settings(tmp_path)
    assert recording.recorded_models(chosen) == []  # no recordings folder at all
    root = tmp_path / "recordings"
    (root / "model-b" / "offline").mkdir(parents=True)
    (root / "model-a" / "offline").mkdir(parents=True)
    (root / "model-c" / "local").mkdir(parents=True)  # another embedder
    (root / "model-d" / "offline.partial").mkdir(parents=True)  # a run that did not end
    (root / "model-e").mkdir()
    (root / "notes.txt").write_text("not a model", encoding="utf-8")
    assert recording.recorded_models(chosen) == ["model-a", "model-b"]
    assert recording.recorded_models(settings(tmp_path, embedder="local")) == ["model-c"]


def test_the_embedders_found_are_those_of_the_folders_that_hold_a_manifest(tmp_path: Path):
    chosen = settings(tmp_path)
    assert recording.recorded_embedders(chosen) == []  # no recordings folder at all
    root = tmp_path / "recordings"
    for made in ("model-b/offline", "model-a/offline", "model-c/local", "model-d/local.partial"):
        (root / made).mkdir(parents=True)
        (root / made / MANIFEST_FILE).write_text("{}", encoding="utf-8")
    (root / "model-e" / "another").mkdir(parents=True)  # no manifest: no recording
    (root / "model-f" / "offline" / MANIFEST_FILE).mkdir(parents=True)  # a folder of that name
    (root / "notes.txt").write_text("not a model", encoding="utf-8")
    # Whatever embedder is in force: each once, sorted, a stopped run left out.
    assert recording.recorded_embedders(chosen) == ["local", "offline"]
    assert recording.recorded_embedders(settings(tmp_path, embedder="local")) == [
        "local",
        "offline",
    ]


# --- The manifest (Q18, Q29) -------------------------------------------------------------


def test_a_run_is_described_by_what_can_shape_its_prompts(tmp_path: Path):
    chosen = settings(tmp_path, llm="anthropic", llm_model=MODEL, seed=7, data_dir=tmp_path)
    told = recording.describe_run(chosen, MODEL, FINGERPRINT)
    assert told["model"] == MODEL
    assert (told["embedder"], told["seed"]) == ("offline", 7)
    assert told["data_fingerprint"] == FINGERPRINT
    assert told["prompt_version"] == recording.PROMPT_VERSION
    assert told["key_version"] == recording.KEY_VERSION
    assert (told["agent_search_k"], told["wave1_size"], told["undecided_rate"]) == (5, 40, 0.1)
    # Not the folders, so no manifest holds a path of the machine it was made on.
    assert not {"data_dir", "out_dir", "recording_dir", "llm", "llm_model", "request_cap"} & set(
        told
    )
    assert str(tmp_path) not in json.dumps(told)
    json.dumps(told, allow_nan=False)  # plain JSON


def test_a_manifest_is_written_and_read_back(tmp_path: Path):
    manifest = {"model": MODEL, "data_fingerprint": "é", "complete": True, "requests": 0}
    manifest |= dict.fromkeys(
        ("embedder", "seed", "prompt_version", "key_version", "started", "ended"), 1
    )
    recording.write_manifest(tmp_path, manifest)
    text = (tmp_path / MANIFEST_FILE).read_bytes().decode("utf-8")
    assert text.endswith("}\n") and "\r" not in text and '"é"' in text
    assert recording.read_manifest(tmp_path) == manifest


def _manifest(tmp_path: Path, **changes) -> dict:
    told = recording.describe_run(settings(tmp_path), MODEL, FINGERPRINT)
    told |= {"started": "2026-01-05T10:00:00Z", "ended": "2026-01-05T10:00:09Z"}
    told |= {"requests": 0, "complete": True}
    return told | changes


def test_a_missing_recording_is_refused(tmp_path: Path):
    for folder in (tmp_path, tmp_path / "no" / "such" / "folder"):
        with pytest.raises(RecordingError, match="there is no recording in") as raised:
            recording.read_manifest(folder)
        assert "record command" in str(raised.value)
        assert isinstance(raised.value, TitlestdError)


@pytest.mark.parametrize("complete", [False, None, "true", 1, "absent"])
def test_a_recording_that_is_not_complete_is_refused(tmp_path: Path, complete):
    manifest = _manifest(tmp_path, complete=complete)
    if complete == "absent":
        del manifest["complete"]
    recording.write_manifest(tmp_path, manifest)
    with pytest.raises(RecordingError, match="is not complete"):
        recording.read_manifest(tmp_path)


@pytest.mark.parametrize("text", ["", "{", "[]", '"complete"', "\xff\xfe"])
def test_a_manifest_that_is_no_manifest_is_refused(tmp_path: Path, text: str):
    (tmp_path / MANIFEST_FILE).write_bytes(text.encode("latin-1"))
    with pytest.raises(RecordingError):
        recording.read_manifest(tmp_path)


@pytest.mark.parametrize("name", ["model", "embedder", "seed", "data_fingerprint", "requests"])
def test_a_manifest_that_leaves_something_out_is_refused(tmp_path: Path, name: str):
    manifest = _manifest(tmp_path)
    del manifest[name]
    recording.write_manifest(tmp_path, manifest)
    with pytest.raises(RecordingError, match=f"does not say: {name}"):
        recording.read_manifest(tmp_path)


def test_a_manifest_like_the_settings_in_force_shows_no_difference(tmp_path: Path):
    current = recording.describe_run(settings(tmp_path), MODEL, FINGERPRINT)
    # When and how much a run recorded is not what its prompts were made from.
    assert recording.manifest_differences(_manifest(tmp_path, requests=41), current) == []


def test_each_difference_is_named_with_both_values(tmp_path: Path):
    manifest = _manifest(tmp_path)
    current = recording.describe_run(
        settings(tmp_path, seed=7, agent_search_k=3, embedder="local"), "other-model", "ab" * 32
    )
    assert recording.manifest_differences(manifest, current) == [
        "model: recorded 'test-double', now 'other-model'",
        "embedder: recorded 'offline', now 'local'",
        "seed: recorded 20260101, now 7",
        "agent_search_k: recorded 5, now 3",
        f"data_fingerprint: recorded '{FINGERPRINT}', now '{'ab' * 32}'",
    ]


def test_a_difference_in_prompt_or_key_version_is_named(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(recording, "PROMPT_VERSION", 2)
    monkeypatch.setattr(recording, "KEY_VERSION", 3)
    current = recording.describe_run(settings(tmp_path), MODEL, FINGERPRINT)
    assert recording.manifest_differences(manifest, current) == [
        "prompt_version: recorded 1, now 2",
        "key_version: recorded 1, now 3",
    ]


def test_what_only_one_side_tells_is_a_difference(tmp_path: Path):
    manifest = _manifest(tmp_path, retired_setting=4)
    del manifest["golden_size"]
    current = recording.describe_run(settings(tmp_path), MODEL, FINGERPRINT)
    assert recording.manifest_differences(manifest, current) == [
        "retired_setting: recorded 4, now nothing",
        "golden_size: recorded nothing, now 40",
    ]


# --- The recorder (Q29) ----------------------------------------------------------------------

FIXED_DAY = datetime(2026, 1, 5, 23, 59, 58, tzinfo=UTC)


@pytest.fixture
def clock(monkeypatch) -> list[datetime]:
    """Replaces the clock: the times it will give, oldest first; the last one stays."""
    times = [FIXED_DAY]
    monkeypatch.setattr(recording, "utc_now", lambda: times.pop(0) if len(times) > 1 else times[0])
    return times


def keep(recorder: Recorder, key: str, step: str = STEP_CHOOSE, turn: dict | None = None) -> None:
    if turn is None:
        turn = recording.answer_turn("role", 2, "account manager")
    recorder.record(
        key,
        step,
        RECORD,
        turn,
        reported_model="test-double-0001",
        stop_reason="end_turn",
        input_tokens=321,
        output_tokens=12,
    )


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_nothing_is_written_before_the_first_answer(tmp_path: Path, clock):
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    assert recorder.folder == tmp_path / "recordings" / MODEL / "offline"
    assert recorder.partial_folder == tmp_path / "recordings" / MODEL / "offline.partial"
    assert recorder.requests == 0 and recorder.turn("k1") is None
    assert not (tmp_path / "recordings").exists()


def test_each_answer_is_on_disk_as_soon_as_it_is_recorded(tmp_path: Path, clock):
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(recorder, "k2")
    assert [entry["key"] for entry in lines(recorder.partial_folder / CHOOSE_FILE)] == ["k2"]
    keep(recorder, "k1", STEP_REVIEW, recording.search_turn("search_catalog", "analyst"))
    keep(recorder, "k0")
    # In the order made, each in the file of its step; not yet a recording.
    assert [entry["key"] for entry in lines(recorder.partial_folder / CHOOSE_FILE)] == ["k2", "k0"]
    assert [entry["key"] for entry in lines(recorder.partial_folder / REVIEW_FILE)] == ["k1"]
    assert not (recorder.partial_folder / MANIFEST_FILE).exists()
    assert not recorder.folder.exists()
    assert recorder.requests == 3
    assert recorder.turn("k1") == {"tool": "search_catalog", "query": "analyst"}
    with pytest.raises(RecordingError, match="there is no recording in"):
        recording.read_manifest(recorder.folder)


def test_an_entry_holds_these_fields_and_no_other(tmp_path: Path, clock):
    recorder = Recorder(settings(tmp_path), "claude-haiku-4-5", FINGERPRINT)
    keep(recorder, "k1")
    assert lines(recorder.partial_folder / CHOOSE_FILE) == [
        {
            "key": "k1",
            "key_version": 1,
            "step": "choose",
            "family_code": "SAL",
            "cleaned_title": "senior account manager",
            "model": "claude-haiku-4-5",  # as configured (Q33)
            "reported_model": "test-double-0001",  # as the API said, for information
            "date": "2026-01-05",  # the UTC day of the request (Q40)
            "turn": {"answer": "role", "number": 2, "quote": "account manager", "detail": ""},
            "stop_reason": "end_turn",
            "input_tokens": 321,
            "output_tokens": 12,
        }
    ]


def test_the_date_of_an_entry_is_the_utc_day_of_its_request(tmp_path: Path, clock):
    clock[:] = [FIXED_DAY, FIXED_DAY, datetime(2026, 1, 6, 0, 0, 1, tzinfo=UTC)]
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)  # takes the first time
    keep(recorder, "k1")
    keep(recorder, "k2")
    dates = [entry["date"] for entry in lines(recorder.partial_folder / CHOOSE_FILE)]
    assert dates == ["2026-01-05", "2026-01-06"]


def test_the_clock_is_utc():
    assert recording.utc_now().utcoffset().total_seconds() == 0


def test_finish_sorts_the_files_writes_the_manifest_and_moves_the_folder(tmp_path: Path, clock):
    clock[:] = [FIXED_DAY, datetime(2026, 1, 6, 0, 0, 7, tzinfo=UTC)]
    chosen = settings(tmp_path, seed=7)
    recorder = Recorder(chosen, MODEL, FINGERPRINT)
    for key in ("k3", "k1", "k2"):
        keep(recorder, key)
    keep(recorder, "k0", STEP_REVIEW, recording.answer_turn("escalate", None, ""))
    recorder.finish()

    assert not recorder.partial_folder.exists()
    assert sorted(path.name for path in recorder.folder.iterdir()) == [
        CHOOSE_FILE,
        MANIFEST_FILE,
        REVIEW_FILE,
    ]
    assert [entry["key"] for entry in lines(recorder.folder / CHOOSE_FILE)] == ["k1", "k2", "k3"]
    assert [entry["key"] for entry in lines(recorder.folder / REVIEW_FILE)] == ["k0"]
    manifest = recording.read_manifest(recorder.folder)
    assert manifest == {
        **recording.describe_run(chosen, MODEL, FINGERPRINT),
        "started": "2026-01-05T23:59:58Z",
        "ended": "2026-01-06T00:00:07Z",
        "requests": 4,
        "complete": True,
    }
    assert list(manifest)[-4:] == ["started", "ended", "requests", "complete"]
    assert len(recording.read_entries(recorder.folder, manifest)) == 4


def test_the_manifest_is_written_last(tmp_path: Path, clock, monkeypatch):
    seen = {}

    def spy(folder: Path, manifest) -> None:
        seen["files"] = {path.name: path.read_text(encoding="utf-8") for path in folder.iterdir()}
        seen["folder"] = folder

    monkeypatch.setattr(recording, "write_manifest", spy)
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(recorder, "k2")
    keep(recorder, "k1")
    recorder.finish()
    # Both files were already there, sorted, in the folder the run wrote into.
    assert seen["folder"] == tmp_path / "recordings" / MODEL / "offline.partial"
    assert set(seen["files"]) == {CHOOSE_FILE, REVIEW_FILE}
    assert seen["files"][CHOOSE_FILE] == (recorder.folder / CHOOSE_FILE).read_text(encoding="utf-8")
    assert seen["files"][REVIEW_FILE] == ""


def test_the_previous_recording_stays_until_the_run_completes(tmp_path: Path, clock):
    first = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(first, "old-1")
    keep(first, "old-2", STEP_REVIEW, recording.search_turn("search_catalog", "analyst"))
    first.finish()
    (first.folder / "left-behind.txt").write_text("of the previous recording", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in first.folder.iterdir()}

    second = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    assert second.turn("old-1") is None  # live mode never reads a recording
    keep(second, "new-1")
    assert {path.name: path.read_bytes() for path in first.folder.iterdir()} == before
    second.finish()
    # Replaced as a whole: nothing of the previous one is left.
    assert sorted(path.name for path in second.folder.iterdir()) == [
        CHOOSE_FILE,
        MANIFEST_FILE,
        REVIEW_FILE,
    ]
    assert [entry["key"] for entry in lines(second.folder / CHOOSE_FILE)] == ["new-1"]
    assert (second.folder / REVIEW_FILE).read_text(encoding="utf-8") == ""


def test_a_run_that_stops_keeps_its_answers_and_the_next_starts_fresh(tmp_path: Path, clock):
    stopped = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(stopped, "k1")
    keep(stopped, "k2")
    # No finish: the run stopped. Its answers are there, and there is no recording.
    assert len(lines(stopped.partial_folder / CHOOSE_FILE)) == 2
    assert not stopped.folder.exists()

    after = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    assert len(lines(after.partial_folder / CHOOSE_FILE)) == 2  # untouched until it records
    keep(after, "k9")
    assert [entry["key"] for entry in lines(after.partial_folder / CHOOSE_FILE)] == ["k9"]


def test_a_run_with_no_request_still_completes_its_recording(tmp_path: Path, clock):
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    recorder.finish()
    manifest = recording.read_manifest(recorder.folder)
    assert manifest["requests"] == 0 and manifest["complete"] is True
    assert recording.read_entries(recorder.folder, manifest) == {}


def test_a_key_has_one_answer(tmp_path: Path, clock):
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(recorder, "k1")
    with pytest.raises(ValueError, match="already recorded"):
        keep(recorder, "k1", turn=recording.answer_turn("no_match", None, "bdr"))
    assert recorder.requests == 1
    assert len(lines(recorder.partial_folder / CHOOSE_FILE)) == 1


@pytest.mark.parametrize(
    ("step", "turn"),
    [
        (STEP_CHOOSE, {"answer": "escalate", "number": None, "quote": "", "detail": ""}),
        (STEP_CHOOSE, recording.search_turn("search_catalog", "analyst")),
        (STEP_REVIEW, {"answer": "no_match", "number": None, "quote": "", "detail": ""}),
        (STEP_REVIEW, {"tool": "search_the_web", "query": "analyst"}),
        (STEP_REVIEW, {"tool": "search_catalog", "query": 7}),
        (STEP_REVIEW, {"tool": "search_catalog", "query": "analyst", "limit": 50}),
        (STEP_CHOOSE, {"answer": "role", "number": "2", "quote": "", "detail": ""}),
        (STEP_CHOOSE, {"answer": "role", "number": True, "quote": "", "detail": ""}),
        (STEP_CHOOSE, {"answer": "role", "number": 2, "quote": None, "detail": ""}),
        (STEP_CHOOSE, {"answer": "role", "number": 2, "quote": ""}),
        (STEP_CHOOSE, {"answer": "role", "number": 2, "quote": "", "detail": "", "headers": {}}),
    ],
)
def test_only_a_turn_of_its_step_is_recorded(tmp_path: Path, clock, step: str, turn: dict):
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    with pytest.raises((KeyError, TypeError, ValueError)):
        keep(recorder, "k1", step, turn)
    assert recorder.requests == 0 and not recorder.partial_folder.exists()


# --- Reading a recording back --------------------------------------------------------------


@pytest.fixture
def recorded(tmp_path: Path, clock) -> Path:
    """A complete recording of two answers and one search request."""
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(recorder, "k1")
    keep(recorder, "k2", turn=recording.invalid_turn("the model refused to answer"))
    keep(recorder, "k3", STEP_REVIEW, recording.search_turn("search_catalog", "analyst"))
    recorder.finish()
    return recorder.folder


def test_the_entries_are_read_back_by_key(recorded: Path):
    entries = recording.read_entries(recorded, recording.read_manifest(recorded))
    assert sorted(entries) == ["k1", "k2", "k3"]
    assert entries["k2"]["turn"]["detail"] == "the model refused to answer"
    assert recording.review_turn(entries["k3"]["turn"], "") == SearchRequest(
        SearchTool.CATALOG, "analyst"
    )


def test_text_survives_the_files_whatever_it_holds(tmp_path: Path, clock):
    # A quote may hold a line separator of another kind; an entry is still one line.
    quote = 'line\u2028separator \u00e9 "quoted" back\\slash\ttab\nnewline'
    recorder = Recorder(settings(tmp_path), MODEL, FINGERPRINT)
    keep(recorder, "k1", turn=recording.answer_turn("role", 1, quote))
    recorder.finish()
    assert (recorder.folder / CHOOSE_FILE).read_bytes().count(b"\n") == 1
    assert "\u00e9".encode() in (recorder.folder / CHOOSE_FILE).read_bytes()  # non-ASCII kept
    entries = recording.read_entries(recorder.folder, recording.read_manifest(recorder.folder))
    assert entries["k1"]["turn"]["quote"] == quote


def _rewrite(path: Path, change) -> None:
    entries = lines(path)
    change(entries)
    text = "".join(json.dumps(entry) + "\n" for entry in entries)
    path.write_text(text, encoding="utf-8")


def _damage_missing_file(folder: Path) -> None:
    (folder / REVIEW_FILE).unlink()


def _damage_not_json(folder: Path) -> None:
    with (folder / CHOOSE_FILE).open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")


def _damage_cut_short(folder: Path) -> None:
    _rewrite(folder / CHOOSE_FILE, lambda entries: entries.pop())


def _damage_wrong_file(folder: Path) -> None:
    (folder / REVIEW_FILE).write_text(
        (folder / CHOOSE_FILE).read_text(encoding="utf-8"), encoding="utf-8"
    )


def _damage_repeated_key(folder: Path) -> None:
    _rewrite(folder / CHOOSE_FILE, lambda entries: entries.append(entries[0]))


def _damage_unknown_answer(folder: Path) -> None:
    _rewrite(folder / CHOOSE_FILE, lambda entries: entries[0]["turn"].update(answer="maybe"))


def _damage_no_date(folder: Path) -> None:
    _rewrite(folder / CHOOSE_FILE, lambda entries: entries[0].pop("date"))


def _damage_not_an_object(folder: Path) -> None:
    (folder / CHOOSE_FILE).write_text('"k1"\n"k2"\n', encoding="utf-8")


@pytest.mark.parametrize(
    ("damage", "said"),
    [
        (_damage_missing_file, "review.jsonl cannot be read"),
        (_damage_not_json, "choose.jsonl, line 3 is no entry of a recording"),
        (_damage_cut_short, "holds 2 entries and its manifest counts 3"),
        (_damage_wrong_file, "review.jsonl, line 1 is no entry of a recording"),
        (_damage_repeated_key, "choose.jsonl, line 3 holds a key the recording already has"),
        (_damage_unknown_answer, "choose.jsonl, line 1 is no entry of a recording"),
        (_damage_no_date, "choose.jsonl, line 1 is no entry of a recording"),
        (_damage_not_an_object, "choose.jsonl, line 1 is no entry of a recording"),
    ],
)
def test_a_damaged_recording_is_refused_naming_the_place(recorded: Path, damage, said: str):
    manifest = recording.read_manifest(recorded)
    damage(recorded)
    with pytest.raises(RecordingError) as raised:
        recording.read_entries(recorded, manifest)
    assert said in str(raised.value)
    assert str(recorded) in str(raised.value)
