"""A whole run recorded and replayed, on the mini data set (Q32 b).

Live mode runs through the pipeline with a scripted client under the model ID
"test-double". The client goes in through live.build_llm(..., client=...) and
the models through pipeline.run_pipeline(..., models=...), the two seams of
S09; every answer is written by the real recorder into a temporary folder.
Replay of that recording is then an ordinary run, built by the switch like any
other: it gives the same decisions and files apart from the answer label, and
with one input changed it stops at the first miss with a non-zero exit and no
output file.

Nothing here reaches the network, reads a key or needs the SDK, so the tests
run in every environment of the gate. No recording made here is committed:
each is written by the program, about a model that does not exist, under
pytest's temporary folder.

Decided here, where the record leaves the detail open:

- What the scripted client says. It answers by rule from what a request
  shows (see ScriptedModel), and the rules are chosen so that one run on the
  mini data set records every kind of turn: a role, "no match",
  "insufficient data" and a refusal from the LLM fallback; both searches, a
  pick and an escalation from the agent. The agent's pick rests on a search
  of the approved titles, which grow between the waves, so replay only finds
  the agent's last turn if it runs the searches again and they return what
  they returned in the live run (Q20).
- "The same decisions and files, apart from the answer label": the results
  of the two runs are equal once the label is taken out of them; load.csv,
  reverse.csv and exceptions.csv are the same bytes; review_queue.csv is the
  same in every column but answer_source; the state file holds the same
  model answers under another label and the same roles shown.
- "One changed input": three are tried, each alone. Another seed misses at
  the first request. Another job description misses at that title, in Wave
  2, after Wave 1 was replayed. Another true role changes no record at all:
  the simulated SME approves another role in Wave 1, the approved titles a
  later search returns differ, and the miss comes at the agent's turn after
  that search. It shows why the manifest's fingerprint covers the truth
  file (S09).
- "No output files" are the four of pipeline.OUTPUT_FILES. The run that
  misses writes into a folder in which the same replay, on unchanged data,
  has just left all four, and none is there afterwards. The state file is no
  output file and stays, as titlestd.pipeline decided.
- The exit status of a replay is that of the real command on a bare
  interpreter (python -S -E -m titlestd run --llm replay ...): 0 for the
  recorded run, 1 for a miss.
- The day an answer was recorded on is fixed by replacing recording.utc_now,
  so the recorded label is the same on every day the tests run.

The last test is of a defect this branch found in titlestd.live and fixed
there (rule 9 of the build): a quote or a search query that is not valid
Unicode stopped a live run when the recorder tried to write it.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from titlestd import (
    datafiles,
    live,
    offline_embedder,
    people,
    pipeline,
    recording,
    release,
    report,
    store,
)
from titlestd.contracts import (
    EMAIL_MASK,
    LABEL_LIVE,
    LABEL_STAND_IN,
    Models,
    Reason,
    RunResult,
    TableRow,
)
from titlestd.live import LiveError
from titlestd.recording import CHOOSE_FILE, MANIFEST_FILE, PARTIAL_SUFFIX, REVIEW_FILE
from titlestd.settings import Settings

MODEL = "test-double"  # Q32 b: an obviously fake model ID
REPORTED = "test-double-0001"  # what the scripted model says it is
RECORDED_ON = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
RECORDED = "recorded 2026-01-05, test-double"

# The lines of a numbered list that carry a number: "3. Sales Engineer (family ...".
NUMBERED = re.compile(r"^(\d+)\. ([^(:]+)", re.MULTILINE)
# The model answers a run recorded, without the label they carry.
ANSWERS = (
    "SELECT stage, family_code, cleaned_title, step, answer, number, quote, detail, role_id, "
    "route, reasons FROM model_answers ORDER BY 1, 2, 3, 4"
)
SHOWN = "SELECT * FROM shown_roles ORDER BY 1, 2, 3, 4, 5, 6, 7"


# --- A scripted client ------------------------------------------------------------------


class ScriptedModel:
    """What stands in for the hosted model: an object with the SDK client's
    messages.create (S09) that answers each request by rule, from nothing but
    what the request shows, and keeps the requests it was sent.

    As the LLM fallback it refuses a record that shows a masked email address.
    Otherwise it picks the first listed role whose name shares a word with the
    title; where none does it answers "no match", or "insufficient data" when
    the record has no description. It quotes the title.

    As the agent it searches the catalog and then the approved titles for the
    title. It then picks the role of the most similar approved title when
    that role carries a number, and escalates when it carries none.

    fail_at is the number of the request that gets no answer, for a run that
    stops; 0 for none.
    """

    def __init__(self, fail_at: int = 0) -> None:
        self.messages = self
        self.requests: list[dict] = []
        self._fail_at = fail_at

    def create(self, **arguments) -> SimpleNamespace:
        # Through JSON and back, as the SDK would send it: plain data only.
        request = json.loads(json.dumps(arguments))
        self.requests.append(request)
        if len(self.requests) == self._fail_at:
            raise ConnectionError("connection reset")
        messages = request["messages"]
        record, _, roles = messages[0]["content"].partition("\n\nRoles:\n")
        title = record.splitlines()[0].removeprefix("Title: ")
        if "tools" not in request:
            return self._choose(record, roles, title)
        if len(messages) == 1:
            return self.search("search_catalog", title)
        if len(messages) == 3:
            return self.search("search_approved_titles", title)
        nearest = NUMBERED.match(messages[-1]["content"][0]["content"])
        if nearest:
            return self.answer("role", int(nearest[1]), title)
        return self.answer("escalate", None, "")

    def _choose(self, record: str, roles: str, title: str) -> SimpleNamespace:
        if EMAIL_MASK in record:
            return _said([], "refusal")
        for number, name in NUMBERED.findall(roles):
            if set(name.lower().split()) & set(title.split()):
                return self.answer("role", int(number), title)
        described = not record.endswith("Description:")
        return self.answer("no_match" if described else "insufficient_data", None, title)

    def answer(self, kind: str, number: int | None, quote: str) -> SimpleNamespace:
        fields = {"answer": kind, "number": number, "quote": quote}
        return _said([SimpleNamespace(type="text", text=json.dumps(fields))], "end_turn")

    def search(self, tool: str, query: str) -> SimpleNamespace:
        call = SimpleNamespace(
            type="tool_use", id=f"toolu_{len(self.requests):04d}", name=tool, input={"query": query}
        )
        return _said([call], "tool_use")


def _said(content: list, stop_reason: str) -> SimpleNamespace:
    """A message with what the live adapter reads of a response."""
    usage = SimpleNamespace(input_tokens=250, output_tokens=30)
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=REPORTED, usage=usage)


# --- The two runs -----------------------------------------------------------------------


def record_a_run(settings: Settings, model: ScriptedModel) -> RunResult:
    """A whole live run: the pipeline, the live adapter on the scripted client
    and the real recorder, put together through the seams of S09."""
    fingerprint = pipeline.data_fingerprint(settings.data_dir)
    llm = live.build_llm(settings, fingerprint, client=model)
    models = Models(offline_embedder.build_embedder(settings), llm)
    return pipeline.run_pipeline(settings, models=models)


@dataclasses.dataclass(frozen=True)
class Recorded:
    """A live run that completed: its settings, its result and its client."""

    settings: Settings
    result: RunResult
    model: ScriptedModel

    @property
    def folder(self) -> Path:
        """The folder of the recording the run made."""
        return self.settings.recording_dir / MODEL / "offline"

    def replay_settings(self, out_dir: Path) -> Settings:
        """The settings that replay the recording: no model is named (Q31)."""
        return dataclasses.replace(self.settings, llm="replay", llm_model=None, out_dir=out_dir)

    def replay_flags(self, out_dir: Path) -> list[str]:
        """The same on the command line, the data folder left to the caller."""
        flags = {
            "--llm": "replay",
            "--recording-dir": self.settings.recording_dir,
            "--out-dir": out_dir,
            "--wave1-size": self.settings.wave1_size,
            "--golden-size": self.settings.golden_size,
        }
        return [str(word) for flag in flags.items() for word in flag]


@pytest.fixture(autouse=True)
def recorded_on(monkeypatch) -> None:
    monkeypatch.setattr(recording, "utc_now", lambda: RECORDED_ON)


@pytest.fixture(autouse=True)
def no_key(monkeypatch) -> None:
    """Nothing here may use a key, whatever the shell that runs the tests holds."""
    monkeypatch.delenv(live.API_KEY_VARIABLE, raising=False)


@pytest.fixture
def recorded(mini_settings: Settings) -> Recorded:
    settings = dataclasses.replace(mini_settings, llm="anthropic", llm_model=MODEL)
    model = ScriptedModel()
    return Recorded(settings, record_a_run(settings, model), model)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def state(out_dir: Path, query: str) -> list[tuple]:
    """What a query selects in the state file a run left in a folder."""
    connection = sqlite3.connect(out_dir / store.STATE_FILE)
    try:
        return list(connection.execute(query))
    finally:
        connection.close()


def queue_file(out_dir: Path) -> list[dict[str, str]]:
    return datafiles.read_table(out_dir / release.REVIEW_QUEUE_FILE, release.REVIEW_QUEUE_COLUMNS)


def unlabelled(result: RunResult) -> RunResult:
    """The result of a run with the answer label taken out wherever it stands."""
    queue = tuple(dataclasses.replace(entry, answer_source="") for entry in result.queue)
    return dataclasses.replace(result, queue=queue, answer_labels=())


def files_in(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


# --- Recording a whole run --------------------------------------------------------------


def test_a_live_run_writes_its_recording_through_the_real_recorder(recorded, mini_dir):
    # Q29: one line per model request, and the manifest written last.
    assert [path.name for path in recorded.folder.parent.iterdir()] == ["offline"]
    assert sorted(files_in(recorded.folder)) == [CHOOSE_FILE, MANIFEST_FILE, REVIEW_FILE]
    manifest = recording.read_manifest(recorded.folder)
    chosen, reviewed = lines(recorded.folder / CHOOSE_FILE), lines(recorded.folder / REVIEW_FILE)
    assert manifest["requests"] == len(recorded.model.requests) == len(chosen) + len(reviewed)
    assert (manifest["model"], manifest["embedder"]) == (MODEL, "offline")
    assert (manifest["seed"], manifest["wave1_size"], manifest["golden_size"]) == (20260101, 4, 3)
    # The fingerprint is the one the pipeline hands the switch in a real run (S09).
    assert manifest["data_fingerprint"] == pipeline.data_fingerprint(mini_dir)
    for entry in (*chosen, *reviewed):
        assert (entry["model"], entry["reported_model"]) == (MODEL, REPORTED)
        assert entry["date"] == "2026-01-05"
    assert recorded.result.answer_labels == (LABEL_LIVE,)


def test_the_scripted_run_records_every_kind_of_turn(recorded):
    # The premise of the tests below: a round trip of one kind of answer would
    # show little. Here both steps, both waves and both machine paths are in it.
    chosen, reviewed = lines(recorded.folder / CHOOSE_FILE), lines(recorded.folder / REVIEW_FILE)
    assert {entry["turn"]["answer"] for entry in chosen} == {
        "role",
        "no_match",
        "insufficient_data",
        "invalid",
    }
    assert {entry["turn"].get("tool") or entry["turn"]["answer"] for entry in reviewed} == {
        "search_catalog",
        "search_approved_titles",
        "role",
        "escalate",
    }
    asked = state(recorded.settings.out_dir, "SELECT DISTINCT stage, step FROM model_answers")
    assert {("Wave 1", "review"), ("Wave 2", "review"), ("close", "choose")} <= set(asked)
    rows = {decision.row for decision in recorded.result.decisions}
    assert {TableRow.MODELS_AGREE, TableRow.AGENT_SETTLED, TableRow.REVIEW_QUEUE} <= rows
    reasons = {reason for entry in recorded.result.queue for reason in entry.reasons}
    assert {Reason.NO_MATCH, Reason.INSUFFICIENT_DATA, Reason.INVALID_ANSWER} <= reasons


# --- Replaying it -----------------------------------------------------------------------


def test_replay_gives_the_same_decisions_and_files_apart_from_the_answer_label(recorded, tmp_path):
    again = recorded.replay_settings(tmp_path / "replayed")
    asked = len(recorded.model.requests)
    replayed = pipeline.run_pipeline(again)  # an ordinary run: the switch builds replay
    assert len(recorded.model.requests) == asked  # replay asked the model nothing
    assert replayed.decisions == recorded.result.decisions
    assert unlabelled(replayed) == unlabelled(recorded.result)
    first, second = recorded.settings.out_dir, again.out_dir
    for name in (release.LOAD_FILE, release.REVERSE_FILE, release.EXCEPTIONS_FILE):
        assert (second / name).read_bytes() == (first / name).read_bytes(), name
    assert [{**line, "answer_source": ""} for line in queue_file(second)] == [
        {**line, "answer_source": ""} for line in queue_file(first)
    ]
    # The same answers to the same questions, and the same roles shown: the
    # searches ran again and found what they had found (Q20).
    assert state(second, ANSWERS) == state(first, ANSWERS)
    assert state(second, SHOWN) == state(first, SHOWN)
    found = state(second, "SELECT DISTINCT shown_in FROM shown_roles")
    assert ("search_catalog",) in found and ("search_approved_titles",) in found


def test_every_answer_carries_one_of_the_three_labels(recorded, mini_settings, tmp_path):
    # 2.5, Q40: "stand-in", "live" or "recorded <date>, <model>".
    stand_in = dataclasses.replace(mini_settings, out_dir=tmp_path / "stand-in")
    again = recorded.replay_settings(tmp_path / "replayed")
    runs = [
        (stand_in, pipeline.run_pipeline(stand_in), LABEL_STAND_IN),
        (recorded.settings, recorded.result, LABEL_LIVE),
        (again, pipeline.run_pipeline(again), RECORDED),
    ]
    for settings, result, label in runs:
        assert result.answer_labels == (label,)
        sources = state(settings.out_dir, "SELECT DISTINCT source FROM model_answers")
        assert sources == [(label,)]
        # In the review-queue file: on every title a model answered for, and on no other.
        shown = {line["answer_source"] for line in queue_file(settings.out_dir)}
        assert shown == {"", label}
        footer = f"LLM and agent answers: {label}. Embedder: offline."
        assert footer in report.render(result, settings).splitlines()


def test_the_replay_command_needs_only_the_recording(recorded, bare_cli, mini_dir, tmp_path):
    # The real command on a bare interpreter: no SDK, no key and no model named.
    out_dir = tmp_path / "replayed"
    result = bare_cli("run", "--data-dir", str(mini_dir), *recorded.replay_flags(out_dir))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    # The table of the live run, apart from the line that says where answers came from.
    expected = report.render(recorded.result, recorded.settings).splitlines()
    printed = result.stdout.splitlines()[:-1]  # the last line names the folder
    assert [line for line in printed if line not in expected] == [
        f"LLM and agent answers: {RECORDED}. Embedder: offline."
    ]
    assert len(printed) == len(expected)
    for name in (release.LOAD_FILE, release.REVERSE_FILE, release.EXCEPTIONS_FILE):
        assert (out_dir / name).read_bytes() == (recorded.settings.out_dir / name).read_bytes()
    assert (out_dir / release.REVIEW_QUEUE_FILE).is_file()


def a_copy_of(
    mini_dir: Path, folder: Path, file_name: str = "", old: bytes = b"", new: bytes = b""
) -> Path:
    """A copy of the mini data set, with one text of one file changed if a file is named."""
    data_dir = Path(shutil.copytree(mini_dir, folder))
    if file_name:
        content = (data_dir / file_name).read_bytes()
        assert content.count(old) == 1, "the text to change stands once in the file"
        (data_dir / file_name).write_bytes(content.replace(old, new))
    return data_dir


@pytest.mark.parametrize(
    ("flags", "edit", "miss", "difference"),
    [
        (
            ("--seed", "7"),
            (),
            "step choose for the title 'financial analyst' in family 'FIN'",
            "seed: recorded 20260101, now 7",
        ),
        (
            (),
            (
                datafiles.EMPLOYEES_FILE,
                b"Manages cash positions and bank accounts.",
                b"Manages cash positions.",
            ),
            "step choose for the title 'treasury analyst' in family 'FIN'",
            "data_fingerprint: recorded ",
        ),
        (
            (),
            (people.TRUTH_FILE, b"Fin Analyst,FIN,FIN-01", b"Fin Analyst,FIN,FIN-06"),
            "step review for the title 'treasury analyst' in family 'FIN'",
            "data_fingerprint: recorded ",
        ),
    ],
    ids=["another seed", "another job description", "another true role"],
)
def test_one_changed_input_is_a_miss_a_non_zero_exit_and_no_output_files(
    recorded, bare_cli, mini_dir, tmp_path, flags, edit, miss, difference
):
    # 2.5: "In replay mode a miss is an error; never fall back silently." Q29:
    # the run stops at once, and the message names the step and the title and
    # says how the manifest differs from the settings in force.
    out_dir = tmp_path / "replayed"
    replay = recorded.replay_flags(out_dir)
    unchanged = a_copy_of(mini_dir, tmp_path / "unchanged")
    changed = a_copy_of(mini_dir, tmp_path / "changed", *edit)
    # First the run as it was recorded, on a copy of the data: it leaves its files.
    assert bare_cli("run", "--data-dir", str(unchanged), *replay).returncode == 0
    assert all((out_dir / name).is_file() for name in pipeline.OUTPUT_FILES)
    before = files_in(recorded.folder)

    result = bare_cli("run", "--data-dir", str(changed), *replay, *flags)

    assert result.returncode == 1
    assert result.stdout == ""  # no table of a run that did not finish
    assert result.stderr.startswith(f"titlestd: replay miss at {miss}: ")
    assert f"Its manifest differs from the settings in force: {difference}" in result.stderr
    assert not [name for name in pipeline.OUTPUT_FILES if (out_dir / name).exists()]
    assert files_in(recorded.folder) == before  # replay writes no recording


# --- A live run that stops --------------------------------------------------------------


def test_a_live_run_that_stops_keeps_its_answers_and_writes_no_output_file(recorded, tmp_path):
    # Q39: an error of the API stops the run, keeps the answers already
    # recorded and writes no load file. Q29: the previous recording is
    # replaced only when a run completes.
    out_dir = recorded.settings.out_dir
    assert all((out_dir / name).is_file() for name in pipeline.OUTPUT_FILES)
    before = files_in(recorded.folder)
    partial = recorded.folder.with_name(recorded.folder.name + PARTIAL_SUFFIX)
    with pytest.raises(LiveError) as raised:
        record_a_run(recorded.settings, ScriptedModel(fail_at=10))
    assert f"the answers recorded so far (9) are kept in {partial}" in str(raised.value)
    assert not [name for name in pipeline.OUTPUT_FILES if (out_dir / name).exists()]
    kept = lines(partial / CHOOSE_FILE) + lines(partial / REVIEW_FILE)
    assert len(kept) == 9 and not (partial / MANIFEST_FILE).exists()
    assert files_in(recorded.folder) == before
    # The recording of the run that completed still replays, whole.
    replayed = pipeline.run_pipeline(recorded.replay_settings(tmp_path / "replayed"))
    assert unlabelled(replayed) == unlabelled(recorded.result)


# --- Model text that no recording can hold ----------------------------------------------

# Half of a surrogate pair. A JSON text can carry it as an escape, and the SDK
# hands it on in a tool request; UTF-8 cannot encode it, so no file can hold it.
LONE_SURROGATE = "\ud800"


class Garbling(ScriptedModel):
    """The scripted model, with a lone surrogate in every quote it gives or in
    every search it asks for."""

    def __init__(self, where: str) -> None:
        super().__init__()
        self._where = where

    def answer(self, kind: str, number: int | None, quote: str) -> SimpleNamespace:
        return super().answer(kind, number, quote + LONE_SURROGATE * (self._where == "quote"))

    def search(self, tool: str, query: str) -> SimpleNamespace:
        return super().search(tool, query + LONE_SURROGATE * (self._where == "query"))


@pytest.mark.parametrize(
    ("where", "step", "detail"),
    [
        ("quote", "choose", "the quote is not valid Unicode text"),
        ("query", "review", "the query of a search is not valid Unicode text"),
    ],
)
def test_text_that_is_not_valid_unicode_is_an_invalid_answer_that_replay_repeats(
    mini_settings, tmp_path, where, step, detail
):
    # Q39, S10: an answer that fails validation sends the title to a person
    # and is recorded, so replay repeats it. Before, such a quote or query
    # stopped the live run with a UnicodeEncodeError when it was recorded.
    settings = dataclasses.replace(mini_settings, llm="anthropic", llm_model=MODEL)
    model = Garbling(where)
    recorded = Recorded(settings, record_a_run(settings, model), model)
    given = state(
        settings.out_dir, f"SELECT answer, detail FROM model_answers WHERE step = '{step}'"
    )
    assert ("invalid", detail) in given and {answer for answer, _ in given} == {"invalid"}
    reasons = {reason for entry in recorded.result.queue for reason in entry.reasons}
    assert Reason.INVALID_ANSWER in reasons
    # No search was run with a query that is no text: the turn ended the case.
    assert not state(settings.out_dir, "SELECT 1 FROM shown_roles WHERE shown_in LIKE 'search%'")
    # The recording is complete and is valid UTF-8, and it replays.
    turns = [
        entry["turn"]
        for name in (CHOOSE_FILE, REVIEW_FILE)
        for entry in lines(recorded.folder / name)
    ]
    assert recording.invalid_turn(detail) in turns
    replayed = pipeline.run_pipeline(recorded.replay_settings(tmp_path / "replayed"))
    assert unlabelled(replayed) == unlabelled(recorded.result)
