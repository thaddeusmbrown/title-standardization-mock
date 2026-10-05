"""The committed recording, replayed: "python3 -m titlestd run --llm replay".

One replay run is shared by every test that only reads its results. The
fixture replay_run starts it once, as the command a reader of a fresh clone
would type, on a bare interpreter from the repository root, under the import
guard of tests/conftest.py and with no key in the environment (Q32 a). It
writes into a temporary folder, so it never meets the files the default run
leaves under out/.

A miss stops a replay with status 1 (2.5, Q29), so a run that ends with status
0 had none (Q02, Q32). One more run, in this process, notes the key of every
request it makes: they are the keys of the recording, all of them and no
other.

Decided here, where the record leaves the detail open:

- The tests state rules, not the numbers of this recording. The owner may
  record again (Q02), and a recording made by the program on the default
  data, seed and settings must pass unchanged.
- No model is named on the command line: replay finds the one recording made
  with the embedder in force and takes the model from its manifest (Q31).
"""

from __future__ import annotations

import dataclasses
import sqlite3
import time
from pathlib import Path

import pytest

from titlestd import (
    datafiles,
    pipeline,
    recording,
    store,
    switch,
)
from titlestd.contracts import (
    LABEL_LIVE,
    LABEL_STAND_IN,
    InputData,
    Llm,
    Models,
    recorded_label,
)
from titlestd.live import API_KEY_VARIABLE
from titlestd.settings import Settings

# Every test here makes or reads a whole run on the mock data with the
# committed recording (S14).
pytestmark = [pytest.mark.replay, pytest.mark.e2e]

# After the import guard: the run command as "python3 -m titlestd run --llm
# replay" makes it, writing into the folder given, then what the guard saw.
GUARDED_REPLAY = """
from titlestd.cli import main

status = main(["run", "--llm", "replay", "--out-dir", sys.argv[1]])
print(sorted(set(attempted_imports)), file=sys.stderr)
raise SystemExit(status)
"""


@dataclasses.dataclass(frozen=True)
class Run:
    """What the replay run printed, how long it took and the folder it wrote into."""

    returncode: int
    stdout: str
    stderr: str
    seconds: float
    folder: Path

    def state(self, query: str, *arguments: object) -> list[tuple]:
        connection = sqlite3.connect(self.folder / store.STATE_FILE)
        try:
            return list(connection.execute(query, arguments))
        finally:
            connection.close()


@pytest.fixture(scope="module")
def replay_run(bare_python, import_guard, tmp_path_factory) -> Run:
    """The replay of the committed recording, made once for this file."""
    folder = tmp_path_factory.mktemp("replay")
    with pytest.MonkeyPatch.context() as environment:
        environment.delenv(API_KEY_VARIABLE, raising=False)  # replay needs no key (2.5)
        started = time.monotonic()
        result = bare_python(import_guard + GUARDED_REPLAY, str(folder))
        seconds = time.monotonic() - started
    return Run(result.returncode, result.stdout, result.stderr, seconds, folder)


@pytest.fixture(scope="module")
def committed(repo_root) -> tuple[dict, dict]:
    """The manifest and the entries of the committed recording, read as replay reads them."""
    settings = Settings(recording_dir=repo_root / "recordings")
    (model,) = recording.recorded_models(settings)
    folder = recording.recording_folder(settings, model)
    manifest = recording.read_manifest(folder)
    return manifest, recording.read_entries(folder, manifest)


@pytest.fixture(scope="module")
def labels(committed) -> set[str]:
    """The labels a replayed answer may carry: the day of each recorded call
    and the configured model (Q40)."""
    manifest, entries = committed
    return {recorded_label(entry["date"], manifest["model"]) for entry in entries.values()}


@pytest.fixture(scope="module")
def data(repo_root) -> InputData:
    return datafiles.read_inputs(repo_root / "data")


# --- the run itself (2.5, Q02, Q29, Q32) -----------------------------------------------


def test_the_committed_recording_replays_with_no_miss_under_the_import_guard(replay_run):
    # A miss is status 1 and no output file; status 0 is a run with none.
    assert replay_run.returncode == 0, replay_run.stderr
    # Q32 (a): no anthropic, httpx2, model2vec or numpy, nor anything else from
    # outside the standard library, was even looked for.
    assert replay_run.stderr == "[]\n"
    for name in (*pipeline.OUTPUT_FILES, store.STATE_FILE):
        assert (replay_run.folder / name).is_file(), name
    assert replay_run.seconds < 60  # 2.1: the whole idea in under a minute


class Asking:
    """The replaying chooser and reviewer, with a note of the key of every
    request the run makes of them (Q33)."""

    def __init__(self, llm: Llm, model: str) -> None:
        self._llm, self._model = llm, model
        self.keys: set[str] = set()

    def choose(self, record, roles):
        request = recording.choose_request(record, roles)
        self.keys.add(recording.request_key(recording.STEP_CHOOSE, self._model, request))
        return self._llm.chooser.choose(record, roles)

    def next_turn(self, transcript):
        request = recording.review_request(transcript)
        self.keys.add(recording.request_key(recording.STEP_REVIEW, self._model, request))
        return self._llm.reviewer.next_turn(transcript)


def test_the_run_makes_the_requests_of_the_recording_and_no_other(
    replay_run, committed, data, repo_root, tmp_path
):
    manifest, entries = committed
    settings = Settings(
        llm="replay",
        data_dir=repo_root / "data",
        recording_dir=repo_root / "recordings",
        out_dir=tmp_path,
    )
    fingerprint = pipeline.data_fingerprint(settings.data_dir)
    models = switch.build_models(settings, data.catalog.ladder, data.abbreviations, fingerprint)
    asking = Asking(models.llm, manifest["model"])
    pipeline.run_pipeline(settings, models=Models(models.embedder, Llm(asking, asking)))
    # Every request was found, or the run had stopped; and the recording holds
    # no answer this run does not ask for: it is the recording of this run.
    assert asking.keys == set(entries)
    assert len(entries) == manifest["requests"]
    # The run of the fixture again, in this process: the same bytes (Q06).
    for name in pipeline.OUTPUT_FILES:
        assert (tmp_path / name).read_bytes() == (replay_run.folder / name).read_bytes(), name


def test_every_answer_carries_the_day_and_the_model_of_the_recording(replay_run, committed, labels):
    # 2.5, Q40: "recorded <date>, <model>", and neither of the other two labels.
    manifest, _ = committed
    assert labels and not labels & {LABEL_STAND_IN, LABEL_LIVE}
    sources = {source for (source,) in replay_run.state("SELECT source FROM model_answers")}
    assert sources and sources <= labels
    assert {label for (label,) in replay_run.state("SELECT label FROM answer_labels")} == sources
    (footer,) = [
        line for line in replay_run.stdout.splitlines() if line.startswith("LLM and agent answers")
    ]
    assert footer.startswith("LLM and agent answers: recorded ")
    assert footer.endswith(f", {manifest['model']}. Embedder: offline.")  # the model as configured
