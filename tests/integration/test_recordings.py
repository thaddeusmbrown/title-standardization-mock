"""The recordings under recordings/: what is committed and what it may hold.

DESIGN.md 2.5 ("Recordings are never written by hand") and 2.8 ("No
secrets. ... No keys in recordings."); docs/build-decisions.md Q02 (one live
recording, made with the offline embedder under the model alias as written),
Q29 (the layout, and what an entry stores: "never headers or the key"), Q32 b
(no recording made by a test is ever committed).

These tests read the files as they lie in the folder and write nothing there.
The run that replays the committed recording is in
tests/integration/test_recordings_replay.py.

Decided here, where the record leaves the detail open:

- A key-shaped string is the prefix Q02 and Q29 name, in any letter case. It
  is looked for in the bytes of every file under recordings/, whatever kind
  of file it is and whether or not a complete recording owns it, so the
  folder a live run that stopped leaves behind is scanned too.
- "Never headers or the key" is also checked by the shape of what is there:
  every line of a recording holds the twelve fields the recorder writes and
  no other.
- A test names its model "test-double" (Q32 b), so a recording under that
  name in recordings/ is one a test wrote into the repository by mistake.
- A manifest is not compared with the data or the settings in force. It
  explains a miss and is no second validity rule (Q29): a recording is good
  for a run if it holds the run's requests, and the replay run tests that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from titlestd import recording
from titlestd.recording import CHOOSE_FILE, MANIFEST_FILE, PARTIAL_SUFFIX, REVIEW_FILE
from titlestd.settings import EXAMPLE_LLM_MODEL, Settings

# The shape of a key, in two pieces: no tracked file holds a key-shaped string (Q02).
KEY_PREFIX = "sk-" + "ant-"
# What the recorder writes for one model request (Q29, Q33).
ENTRY_FIELDS = {
    "key",
    "key_version",
    "step",
    "family_code",
    "cleaned_title",
    "model",
    "reported_model",
    "date",
    "turn",
    "stop_reason",
    "input_tokens",
    "output_tokens",
}
TEST_MODEL = "test-double"


def files_with_a_key(folder: Path) -> list[Path]:
    """The files under a folder that hold a key-shaped string."""
    shape = KEY_PREFIX.encode()
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and shape in path.read_bytes().lower()
    )


@pytest.fixture(scope="module")
def recordings(repo_root: Path) -> Path:
    return repo_root / Settings().recording_dir


def test_the_scan_finds_a_key_shaped_string_wherever_it_stands(tmp_path):
    folder = tmp_path / TEST_MODEL / "offline"
    clean = {
        folder / CHOOSE_FILE: json.dumps(
            {"turn": {"answer": "no_match", "number": None, "quote": "treasury analyst"}}
        ).encode()
    }
    planted = {
        # In what a model said, in the folder of a run that stopped, in a file that is no text.
        folder / REVIEW_FILE: json.dumps(
            {"turn": {"tool": "search_catalog", "query": f"analyst {KEY_PREFIX}api03-0000"}}
        ).encode(),
        folder.with_name("offline" + PARTIAL_SUFFIX) / MANIFEST_FILE: json.dumps(
            {"x-api-key": KEY_PREFIX + "0000"}
        ).encode(),
        tmp_path / "left behind" / "core": b"\x00\xff" + KEY_PREFIX.upper().encode() + b"\xfe",
    }
    for path, content in (clean | planted).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert files_with_a_key(tmp_path) == sorted(planted)
    assert files_with_a_key(folder) == [folder / REVIEW_FILE]


@pytest.mark.replay
def test_no_recording_holds_a_key_shaped_string(recordings):
    # 2.8, Q29
    assert list(recordings.rglob(CHOOSE_FILE)), "the scan needs the committed recording"
    assert files_with_a_key(recordings) == []


@pytest.mark.replay
def test_every_recording_is_a_whole_run_and_holds_only_what_the_recorder_writes(recordings):
    folders = sorted(path.parent for path in recordings.rglob(MANIFEST_FILE))
    # Q02, Q31: the one live recording, with the default embedder, under the
    # model alias exactly as written.
    assert recordings / EXAMPLE_LLM_MODEL / Settings().embedder in folders
    for folder in folders:
        # Read as replay reads it: complete, every line an entry, as many as the manifest counts.
        manifest = recording.read_manifest(folder)
        entries = recording.read_entries(folder, manifest)
        # Q29: recordings/<llm_model>/<embedder>/, the model as it was configured.
        assert folder == recordings / manifest["model"] / manifest["embedder"]
        assert sorted(path.name for path in folder.iterdir()) == [
            CHOOSE_FILE,
            MANIFEST_FILE,
            REVIEW_FILE,
        ]
        for name in (CHOOSE_FILE, REVIEW_FILE):
            text = (folder / name).read_text(encoding="utf-8")
            written = [json.loads(line) for line in text.splitlines()]
            # One line per model request, sorted by key when the run ended.
            assert [entry["key"] for entry in written] == sorted(entry["key"] for entry in written)
            for entry in written:
                assert set(entry) == ENTRY_FIELDS  # no header, no credential, nothing else
                assert entry["model"] == manifest["model"]
        assert all(len(key) == 64 for key in entries)
    # No recording made by a test is ever committed (Q32 b).
    assert not (recordings / TEST_MODEL).exists()
