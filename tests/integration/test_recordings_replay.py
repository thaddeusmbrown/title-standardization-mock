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

The replay run then passes the invariants the default run passes in
tests/integration/test_pipeline_default_run.py: the rows partition the titles
and the employees, the status of a path follows from its counts and the footer
agrees with the files (Q16, Q23), the files cover every employee once (Q05),
and the rules of DESIGN.md 2.7 hold on the files and the state. In the
default run T2 and T4 need a model that is made to misbehave. Here the answers
are a hosted model's own, so both are read off the state: where its quote is
not in the record, or its number names no role it was shown, the title went
to a person.

Decided here, where the record leaves the detail open:

- The tests state rules, not the numbers of this recording. The owner may
  record again (Q02), and a recording made by the program on the default
  data, seed and settings must pass unchanged. So nothing here says how many
  titles a path holds or how often the model was right, and no table is
  frozen for the replay run. One thing is taken for granted: that a machine
  path settled some title, without which T2 and T3 would hold of nothing.
- The invariants are written out in this file. A test file does not import
  another (pytest runs in importlib mode, and each file's helpers are its
  own), and the default run's tests also hold numbers of the stand-ins that
  a hosted model does not repeat.
- No model is named on the command line: replay finds the one recording made
  with the embedder in force and takes the model from its manifest (Q31).
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
import time
from pathlib import Path

import pytest

from titlestd import (
    datafiles,
    fallback,
    pipeline,
    prepare,
    recording,
    release,
    store,
    switch,
)
from titlestd.catalog import (
    family_codes,
    generic_role,
    get_role,
    is_catalog_title,
    level_for_grade,
    standard_title,
)
from titlestd.contracts import (
    AUDIT_SAMPLE_SIZE,
    LABEL_LIVE,
    LABEL_STAND_IN,
    InputData,
    Llm,
    MachinePath,
    Models,
    TableRow,
    TitleKey,
    TitleRow,
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

ROW_NAMES = [row.value for row in TableRow]
PATH_ROWS = [path.value for path in MachinePath]
# The lines of an audit of 60 titles, as 2.4 states them: confirmed with at
# most 2 wrong, stopped with more than 12.
CONFIRMED_UP_TO, STOPPED_ABOVE = 2, 12
# The answers whose quote is checked against the record (Q15).
QUOTE_CHECKED = {"choose": ("role", "no_match", "insufficient_data"), "review": ("role",)}
# The status of the titles of a row that is no machine path (Q16).
ROW_STATUS = {
    "look-up hit": "Approved",
    "approved in Wave 1": "Approved",
    "review queue": "Approved",
    "generic family role": "Provisional",
    "left on legacy title": "",
}
# The titles on which the compliance check of 1.7 fired, in any scoring of a run.
COMPLIANCE_FIRED = (
    "SELECT DISTINCT family_code, cleaned_title FROM scorings "
    "WHERE reasons LIKE '%compliance-flagged role%'"
)


@dataclasses.dataclass(frozen=True)
class Run:
    """What the replay run printed, how long it took and the folder it wrote into."""

    returncode: int
    stdout: str
    stderr: str
    seconds: float
    folder: Path

    def table(self, name: str, columns) -> list[dict[str, str]]:
        return datafiles.read_table(self.folder / name, columns)

    @property
    def load(self) -> list[dict[str, str]]:
        return self.table(release.LOAD_FILE, release.LOAD_COLUMNS)

    @property
    def reverse(self) -> list[dict[str, str]]:
        return self.table(release.REVERSE_FILE, release.REVERSE_COLUMNS)

    @property
    def exceptions(self) -> list[dict[str, str]]:
        return self.table(release.EXCEPTIONS_FILE, release.EXCEPTIONS_COLUMNS)

    @property
    def queue(self) -> dict[TitleKey, dict[str, str]]:
        lines = self.table(release.REVIEW_QUEUE_FILE, release.REVIEW_QUEUE_COLUMNS)
        return {TitleKey(line["job_family_code"], line["cleaned_title"]): line for line in lines}

    def state(self, query: str, *arguments: object) -> list[tuple]:
        connection = sqlite3.connect(self.folder / store.STATE_FILE)
        try:
            return list(connection.execute(query, arguments))
        finally:
            connection.close()

    def keys(self, query: str, *arguments: object) -> set[TitleKey]:
        """The titles a query of the state file selects, by family code and cleaned title."""
        return {TitleKey(*row) for row in self.state(query, *arguments)}


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


@pytest.fixture(scope="module")
def title_rows(data) -> dict[TitleKey, TitleRow]:
    rows = prepare.prepare(data.employees, data.abbreviations, family_codes(data.catalog))
    return {row.key: row for row in rows}


def printed_table(text: str) -> dict[str, list[str]]:
    """The cells of each row of the printed table, by the name in its first column."""
    found = {}
    for line in text.splitlines():
        for name in (*ROW_NAMES, "total"):
            if line.startswith(name + "  "):
                found[name] = re.split(r"\s{2,}", line.strip())[1:]
    return found


def printed_numbers(text: str, pattern: str) -> list[int]:
    """The whole numbers that the groups of the pattern find on the one printed
    line it matches in full."""
    (found,) = [match for line in text.splitlines() if (match := re.fullmatch(pattern, line))]
    return [int(number) for number in found.groups()]


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


# --- the table and the files (2.6, Q05, Q16, Q23) ---------------------------------------


def test_the_rows_partition_the_titles_and_the_employees(replay_run, data, title_rows):
    # Q16: each title is in exactly one row; titles and employees add up to the totals.
    decided = replay_run.state("SELECT family_code, cleaned_title, table_row FROM decisions")
    assert sorted(TitleKey(*row[:2]) for row in decided) == sorted(title_rows)
    titles = dict.fromkeys(ROW_NAMES, 0)
    employees = dict.fromkeys(ROW_NAMES, 0)
    for family, title, table_row in decided:
        titles[table_row] += 1
        employees[table_row] += title_rows[TitleKey(family, title)].headcount
    printed = printed_table(replay_run.stdout)
    for name in ROW_NAMES:
        assert printed[name][:2] == [str(titles[name]), str(employees[name])], name
        share = float(printed[name][2].rstrip("%"))
        assert abs(share - 100 * employees[name] / len(data.employees)) <= 0.05, name
    assert printed["total"] == [str(len(title_rows)), str(len(data.employees)), "100.0%"]
    assert sum(employees.values()) == len(data.employees)


def test_the_status_of_a_path_follows_from_its_audited_and_wrong_counts(replay_run):
    # 2.4, Q23: by the same lines as in the default run, whatever the model answered.
    printed = printed_table(replay_run.stdout)
    audits = replay_run.state("SELECT path, path_size, audited, wrong, outcome FROM audits")
    assert [row[0] for row in audits] == PATH_ROWS
    for path, size, audited, wrong, outcome in audits:
        assert audited == min(size, AUDIT_SAMPLE_SIZE)
        if size <= AUDIT_SAMPLE_SIZE:
            expected = "all labeled"
        elif wrong <= CONFIRMED_UP_TO:
            expected = "Confirmed"
        elif wrong > STOPPED_ABOVE:
            expected = "Stopped"
        else:
            expected = "Provisional"
        assert outcome == expected
        assert printed[path][3:5] == [str(audited), str(wrong)]
        assert not size or printed[path][5].startswith(expected)
        # Wrong means the role the path accepted differs from the SME's label (2.4).
        differing = replay_run.state(
            "SELECT count(*) FROM audit_samples s "
            "JOIN human_decisions h USING (family_code, cleaned_title) "
            "JOIN model_answers a USING (family_code, cleaned_title) "
            "WHERE s.path = ? AND h.occasion = ? AND a.stage = 'Wave 2' "
            "AND a.route = 'accept' AND a.role_id IS NOT h.role_id",
            path,
            pipeline.OCCASION_AUDIT,
        )
        assert differing == [(wrong,)]
    # Audited and wrong are filled on the path rows only. On the other rows
    # the status is that of the row's titles, where it holds any (Q16).
    for name, status in ROW_STATUS.items():
        held = int(printed[name][0])
        assert printed[name][3:] == ([status] if status and held else []), name


def test_load_file_and_exceptions_cover_every_employee_exactly_once(replay_run, data):
    # Q05: load.csv plus the released = 0 exceptions; the reverse file holds
    # exactly the employees of the load file.
    loaded = [line["employee_id"] for line in replay_run.load]
    legacy = [line["employee_id"] for line in replay_run.exceptions if line["released"] == "0"]
    assert sorted(loaded + legacy) == sorted(employee.employee_id for employee in data.employees)
    assert [line["employee_id"] for line in replay_run.reverse] == loaded == sorted(loaded)
    # 2.4: no grade gives the title with no level word, and the employee is listed.
    grade = {employee.employee_id: employee.grade for employee in data.employees}
    no_grade = {each for each in loaded if level_for_grade(data.catalog, grade[each]) is None}
    for line in replay_run.load:
        unlevelled = line["employee_id"] in no_grade
        assert (line["no_grade"] == "1") == (line["level"] == "") == unlevelled
    listed = [line for line in replay_run.exceptions if line["released"] == "1"]
    assert {line["employee_id"] for line in listed} == no_grade
    assert {line["reason"] for line in listed} <= {"no grade"}


def test_the_footer_agrees_with_the_files_and_the_state(replay_run, data, title_rows):
    text = replay_run.stdout
    # Coverage by status is by employee, includes those left on a legacy title
    # and adds up to every employee (Q05, Q16).
    by_status = dict.fromkeys(("Approved", "Confirmed", "Provisional"), 0)
    for line in replay_run.load:
        by_status[line["status"]] += 1
    legacy = [line for line in replay_run.exceptions if line["released"] == "0"]
    by_status["left on legacy title"] = len(legacy)
    assert sum(by_status.values()) == len(data.employees)
    (coverage,) = [line for line in text.splitlines() if line.startswith("Coverage by status")]
    for name, count in by_status.items():
        assert re.search(rf"{name} (\d+) \(", coverage)[1] == str(count), name
    # Sent to a person: the Wave 1 list and the review queue, over all titles.
    wave1 = replay_run.keys("SELECT family_code, cleaned_title FROM title_rows WHERE on_wave1_list")
    sent, total, on_list, in_queue, audited, golden = printed_numbers(
        text,
        r"Sent to a person: (\d+) of (\d+) titles \([\d.]+%\): (\d+) on the Wave 1 list, "
        r"(\d+) in the review queue\. Blind labels besides: (\d+) in the audits, "
        r"(\d+) on the golden set\.",
    )
    assert (on_list, in_queue) == (len(wave1), len(replay_run.queue))
    assert sent == len(wave1 | set(replay_run.queue)) and total == len(title_rows)
    assert [(audited,)] == replay_run.state("SELECT sum(audited) FROM audits")
    assert [(golden,)] == replay_run.state("SELECT count(*) FROM title_rows WHERE in_golden_set")
    # The golden set at each stage, as the state holds it (Q08).
    scores = replay_run.state("SELECT * FROM golden_scores ORDER BY rowid")
    assert [score[0] for score in scores] == ["Wave 1", "Wave 2", "close"]
    for stage, size, ml_right, llm_right, no_answer in scores:
        line = rf"  {stage}: ML first choice (\d+)/(\d+), LLM pick (\d+)/(\d+) \((\d+) with .*"
        assert printed_numbers(text, line) == [ml_right, size, llm_right, size, no_answer]
        assert size == golden and no_answer <= size - llm_right
    # The two counts under the table (Q05).
    no_grade = [line for line in replay_run.exceptions if line["released"] == "1"]
    counts = printed_numbers(
        text, r"Exceptions list: (\d+) employees left on a legacy title; (\d+) employees .*"
    )
    assert counts == [len(legacy), len(no_grade)]


def test_every_released_title_says_who_decided_it_and_how_far_to_rely_on_it(replay_run, title_rows):
    # 1.9, Q05: the load file against the final decisions, and those against
    # what people and the machine paths said when they were asked.
    row_of = {
        employee_id: key for key, row in title_rows.items() for employee_id in row.employee_ids
    }
    decided = {
        TitleKey(family, title): rest
        for family, title, *rest in replay_run.state("SELECT * FROM decisions")
    }
    people = set(
        replay_run.state(
            "SELECT family_code, cleaned_title, role_id, decided_by FROM human_decisions "
            "WHERE answer = 'role' AND simulated"
        )
    )
    accepted = {
        TitleKey(family, title): role_id
        for family, title, role_id in replay_run.state(
            "SELECT family_code, cleaned_title, role_id FROM model_answers "
            "WHERE stage = 'Wave 2' AND route = 'accept'"
        )
    }
    audited = replay_run.keys("SELECT family_code, cleaned_title FROM audit_samples")
    for line in replay_run.load:
        key = row_of[line["employee_id"]]
        table_row, source, role_id, status, decided_by, simulated, error_rate, legacy = decided[key]
        assert legacy is None and table_row != "left on legacy title"
        assert (line["role_id"], line["status"], line["path"]) == (role_id, status, table_row)
        assert (line["decided_by"], line["simulated"]) == (decided_by, str(simulated))
        if status == "Approved":  # a named person decided it
            assert source in ("person", "look-up") and decided_by
            assert source == "look-up" or (simulated and (*key, role_id, decided_by) in people)
        else:  # no person read it: its machine path's own answer, or code's generic role
            assert source in ("machine", "code") and not decided_by and not simulated
            assert source == "code" or (accepted[key] == role_id and key not in audited)
        # An error rate only where Provisional comes from an audited path.
        by_a_path = status == "Provisional" and source == "machine"
        assert (line["error_rate"] != "") == by_a_path == (error_rate is not None)


def test_the_review_queue_file_lists_every_title_no_machine_settled(replay_run, title_rows, labels):
    # Q16, S12: the queue as it stood after Wave 2, with what happened to it.
    queue = replay_run.queue
    routed = replay_run.keys(
        "SELECT family_code, cleaned_title FROM scorings "
        "WHERE stage = 'Wave 2' AND route = 'person'"
    )
    unscored = {key for key, row in title_rows.items() if not row.model_eligible}
    assert set(queue) == routed | unscored and routed.isdisjoint(unscored)
    outcomes = {
        TitleKey(family, title): table_row
        for family, title, table_row in replay_run.state(
            "SELECT family_code, cleaned_title, table_row FROM decisions"
        )
    }
    for key, line in queue.items():
        assert line["headcount"] == str(title_rows[key].headcount)
        assert line["reasons"] and line["outcome"] == outcomes[key]
        assert line["simulated"] == "1"  # people are simulated, and the file says so (Q27)
        # The answer label, on every title a model answered for and on no other (Q40).
        shown = set(line["answer_source"].split("; ")) - {""}
        assert shown <= labels and bool(shown) == (key in routed)
        decided = line["decision"] not in ("could not decide", "no suitable role")
        assert (line["outcome"] == "review queue") == decided


# --- the rules of 2.7, on the run's files and state --------------------------------------


def test_t1_no_title_outside_the_catalog_reaches_the_load_file(replay_run, data):
    catalog = data.catalog
    grade = {employee.employee_id: employee.grade for employee in data.employees}
    assert replay_run.load
    for line in replay_run.load:
        role = get_role(catalog, line["role_id"])
        assert role is not None, line
        assert is_catalog_title(catalog, line["long_title"], line["short_title"]), line
        # The titles are the role's own, at the level HR's table gives the grade (1.3, 1.4).
        title = standard_title(catalog, role, grade[line["employee_id"]])
        assert (line["long_title"], line["short_title"]) == (title.long_title, title.short_title)


def test_t2_a_quote_that_is_not_in_the_record_sends_the_title_to_a_person(replay_run, title_rows):
    # 1.7, on every answer of the recording, at every stage. The check is made
    # again here, on the record block as the model was shown it (Q15).
    failed = set()
    for stage, family, title, step, answer, quote, route, reasons in replay_run.state(
        "SELECT stage, family_code, cleaned_title, step, answer, quote, route, reasons "
        "FROM model_answers"
    ):
        key = TitleKey(family, title)
        block = fallback.render_record(title_rows[key].record)
        fails = answer in QUOTE_CHECKED[step] and not fallback.quote_in_record(quote, block)
        assert ("quote not in record" in reasons.split("; ")) == fails, (stage, key, step)
        if fails:
            assert route == "person", (stage, key, step)
            if stage == "Wave 2":
                failed.add(key)
    # Where a title's release was at stake, it is in the review queue with the
    # reason, and in no machine path.
    queue = replay_run.queue
    for key in failed:
        assert "quote not in record" in queue[key]["reasons"].split("; "), key
    in_a_path = replay_run.keys(
        "SELECT family_code, cleaned_title FROM decisions WHERE source = 'machine' "
        "OR table_row IN ('models agree', 'agent-settled')"
    )
    assert in_a_path and failed.isdisjoint(in_a_path)


def test_t3_a_compliance_flagged_role_is_never_settled_by_a_machine(replay_run, data):
    flagged = {role.role_id for role in data.catalog.roles if role.compliance_flag}
    assert flagged
    top_three: dict[TitleKey, set[str]] = {}
    for family, title, role_id in replay_run.state(
        "SELECT family_code, cleaned_title, role_id FROM ml_shortlist "
        "WHERE stage = 'Wave 2' AND place <= 3"
    ):
        top_three.setdefault(TitleKey(family, title), set()).add(role_id)
    # No title a machine path settled has a flagged role, or one among the ML
    # model's top three (1.7).
    settled = replay_run.state(
        "SELECT family_code, cleaned_title, role_id FROM model_answers "
        "WHERE stage = 'Wave 2' AND route = 'accept'"
    )
    assert settled
    for family, title, role_id in settled:
        assert role_id not in flagged
        assert top_three[TitleKey(family, title)].isdisjoint(flagged)
    # What went out with no person reading it carries no flagged role.
    by_machine = [line for line in replay_run.load if line["status"] != "Approved"]
    assert not [line for line in by_machine if line["role_id"] in flagged]
    # The titles that do touch a flagged role all went to the review queue.
    touching = {key for key, roles in top_three.items() if roles & flagged}
    touching |= replay_run.keys(
        "SELECT family_code, cleaned_title FROM model_answers WHERE stage = 'Wave 2' "
        f"AND role_id IN ({', '.join('?' * len(flagged))})",
        *sorted(flagged),
    )
    queue = replay_run.queue
    assert touching and touching <= set(queue)
    for key in touching:
        assert "compliance-flagged role" in queue[key]["reasons"], key


def test_t4_the_agent_picks_only_a_numbered_role_of_the_recorded_family(replay_run, data):
    # Every number the agent was offered, in a case or in a search result,
    # stands for one role of the title's own family (Q10).
    numbered: dict[tuple, dict[int, str]] = {}
    for stage, family, title, number, role_id in replay_run.state(
        "SELECT stage, family_code, cleaned_title, number, role_id FROM shown_roles "
        "WHERE step = 'review' AND number IS NOT NULL"
    ):
        assert get_role(data.catalog, role_id).family_code == family
        offered = numbered.setdefault((stage, family, title), {})
        assert offered.setdefault(number, role_id) == role_id  # a number never changes
    # A pick is the role its number was offered with. A number that was not
    # offered names no role, and the title goes to a person.
    for stage, family, title, number, role_id, route, reasons in replay_run.state(
        "SELECT stage, family_code, cleaned_title, number, role_id, route, reasons "
        "FROM model_answers WHERE step = 'review' AND answer = 'role'"
    ):
        offered = numbered.get((stage, family, title), {})
        assert role_id == offered.get(number), (stage, family, title)
        if role_id is None:
            assert route == "person" and "role not in catalog" in reasons.split("; ")


def test_t5_no_model_is_ever_shown_a_generic_family_role(replay_run, data):
    generic = {role.role_id for role in data.catalog.roles if role.is_generic}
    assert generic
    # Never on a shortlist, in a numbered list, in a case or in a search result.
    listed = {role_id for (role_id,) in replay_run.state("SELECT role_id FROM ml_shortlist")}
    shown = {role_id for (role_id,) in replay_run.state("SELECT role_id FROM shown_roles")}
    assert listed and shown and (listed | shown).isdisjoint(generic)
    # No answer of a model and no decision of a person names one.
    for table in ("model_answers", "human_decisions"):
        named = {role_id for (role_id,) in replay_run.state(f"SELECT role_id FROM {table}")}
        assert named.isdisjoint(generic), table
    # Only code assigns one (1.2): where the load file has it, nobody chose it.
    with_generic = [line for line in replay_run.load if line["role_id"] in generic]
    assert {(line["path"], line["decided_by"]) for line in with_generic} <= {
        ("generic family role", "")
    }


def test_t6_golden_set_titles_never_enter_the_look_up_table_or_the_examples(replay_run):
    golden = replay_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE in_golden_set"
    )
    assert len(golden) == Settings().golden_size
    looked_up = replay_run.keys("SELECT family_code, cleaned_title FROM lookup")
    examples = replay_run.keys("SELECT family_code, cleaned_title FROM examples")
    assert looked_up and looked_up.isdisjoint(golden)
    assert examples and examples.isdisjoint(golden)
    # Everything else a person decided was learned, and nothing a machine path
    # released without one (1.6 step 10).
    by_person = replay_run.keys(
        "SELECT family_code, cleaned_title FROM decisions WHERE source IN ('person', 'look-up')"
    )
    assert looked_up == by_person - golden


def test_t7_the_generic_role_is_never_past_its_cap_or_on_a_barred_title(
    replay_run, data, title_rows
):
    settings = Settings()
    generic = {
        TitleKey(family, title): role_id
        for family, title, role_id in replay_run.state(
            "SELECT family_code, cleaned_title, role_id FROM decisions "
            "WHERE table_row = 'generic family role'"
        )
    }
    # Never past the cap: 2% of the employees (1.8), whole titles only (Q25).
    held = sum(title_rows[key].headcount for key in generic)
    assert held <= settings.generic_cap_share * len(data.employees)
    # The Wave 1 list, worked out again from the headcounts (Q24).
    widest = sorted(title_rows, key=lambda key: (-title_rows[key].headcount, key))
    wave1 = set(widest[: settings.wave1_size])
    fired = replay_run.keys(COMPLIANCE_FIRED)
    queue = replay_run.queue
    for key, role_id in generic.items():
        assert key not in wave1  # never a Wave 1 title
        assert key not in fired  # never a flagged title, in any scoring of it (Q25)
        assert queue[key]["decision"] == "could not decide"  # a title nobody could decide (1.8)
        assert title_rows[key].model_eligible
        assert role_id == generic_role(data.catalog, key.family_code).role_id
    statuses = replay_run.state(
        "SELECT DISTINCT source, status, decided_by, simulated, error_rate FROM decisions "
        "WHERE table_row = 'generic family role'"
    )
    assert statuses in ([], [("code", "Provisional", "", 0, None)])


def test_t8_applying_the_reverse_file_to_the_load_file_restores_every_legacy_title(
    replay_run, data
):
    restored = release.apply_reverse(replay_run.load, replay_run.reverse)
    received = {employee.employee_id: employee.job_title for employee in data.employees}
    assert list(restored) == [line["employee_id"] for line in replay_run.load]
    assert restored
    for employee_id, title in restored.items():
        assert title == received[employee_id]  # as received, byte for byte
