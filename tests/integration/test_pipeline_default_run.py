"""The default run, end to end: "python3 -m titlestd run" on the mock data.

One run is shared by every test that only reads its results: the fixture
default_run starts it once per session, on a bare interpreter from the
repository root and under the import guard of tests/conftest.py, and keeps
what it printed and a copy of the files it wrote under out/. The eight rules of
docs/design.md 2.7 are checked on those files and on the state file.

A few more runs on the mock data, each made once and in this process, show
what the default seed does not: a close with more leftovers than the cap holds
(Q01) and the other two outcomes of an audit (2.4).

The tests state rules, not the numbers of the default run: the run shape may
be tuned by regenerating data/ (Q23).
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from titlestd import audit, datafiles, pipeline, prepare, release, report, store
from titlestd.catalog import (
    family_codes,
    generic_role,
    get_role,
    is_catalog_title,
    level_for_grade,
    standard_title,
)
from titlestd.close import generic_cap
from titlestd.contracts import (
    AUDIT_SAMPLE_SIZE,
    LOOKUP_DECIDED_BY,
    AuditOutcome,
    InputData,
    RunResult,
    TableRow,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings

ROW_NAMES = [row.value for row in TableRow]
PATH_ROWS = ("models agree", "agent-settled")

# After the import guard: the run command exactly as "python3 -m titlestd run"
# makes it, then what the guard saw, on standard error.
GUARDED_RUN = """
from titlestd.cli import main

status = main(["run"])
print(sorted(set(attempted_imports)), file=sys.stderr)
raise SystemExit(status)
"""


@dataclasses.dataclass(frozen=True)
class Run:
    """What one default run printed, how long it took and a copy of what it wrote."""

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


def run_by_default(bare_python, import_guard: str, repo_root: Path, keep: Path) -> Run:
    started = time.monotonic()
    result = bare_python(import_guard + GUARDED_RUN)
    seconds = time.monotonic() - started
    if result.returncode == 0:
        shutil.copytree(repo_root / "out", keep, dirs_exist_ok=True)
    return Run(result.returncode, result.stdout, result.stderr, seconds, keep)


@pytest.fixture(scope="session")
def default_run(bare_python, import_guard, repo_root, tmp_path_factory) -> Run:
    """The default run, made once. Its files are copied, so that nothing a later
    run writes under out/ changes what the tests read."""
    return run_by_default(bare_python, import_guard, repo_root, tmp_path_factory.mktemp("first"))


@pytest.fixture(scope="session")
def data(repo_root) -> InputData:
    return datafiles.read_inputs(repo_root / "data")


@pytest.fixture(scope="session")
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


# --- the run itself (2.1, 2.5, Q32 a, Q38) -------------------------------------------


def test_the_default_run_succeeds_on_a_bare_interpreter_under_the_import_guard(default_run):
    # Q32 (a): no anthropic, httpx2, model2vec or numpy, nor anything else from
    # outside the standard library, was even looked for.
    assert default_run.returncode == 0, default_run.stderr
    assert default_run.stderr == "[]\n"


def test_the_default_run_finishes_well_under_a_minute(default_run):
    # 2.1: "Show the whole idea in under a minute". Q38: more than 60 seconds fails.
    assert default_run.seconds < 60


def test_it_prints_one_table_and_writes_the_files_under_out(default_run, repo_root):
    lines = default_run.stdout.splitlines()
    assert lines[0] == "Job-title standardization mock: set-up, Wave 1, Wave 2 and the close"
    assert lines[-1] == (
        "Files written to the folder out: load.csv, reverse.csv, exceptions.csv, "
        "review_queue.csv, state.sqlite."
    )
    assert set(printed_table(default_run.stdout)) == {*ROW_NAMES, "total"}
    for name in (*pipeline.OUTPUT_FILES, store.STATE_FILE):
        assert (repo_root / "out" / name).is_file(), name
    assert "Model answers: stand-in. Embedder: offline." in lines
    assert any(line.startswith("People were simulated") for line in lines)


def test_a_second_run_in_another_process_gives_the_same_bytes(
    default_run, bare_python, import_guard, repo_root, tmp_path
):
    # Q06, Q23: the printed table and the output files, byte for byte.
    second = run_by_default(bare_python, import_guard, repo_root, tmp_path)
    assert second.returncode == 0, second.stderr
    assert second.stdout == default_run.stdout
    for name in pipeline.OUTPUT_FILES:
        assert (second.folder / name).read_bytes() == (default_run.folder / name).read_bytes(), name
    # The state holds the same rows too, whatever SQLite does with the file.
    dumps = []
    for folder in (default_run.folder, second.folder):
        connection = sqlite3.connect(folder / store.STATE_FILE)
        dumps.append(list(connection.iterdump()))
        connection.close()
    assert dumps[0] == dumps[1] and len(dumps[0]) > 1000


# --- the table and its footer (2.6, Q16, Q23) ------------------------------------------


def test_the_rows_partition_the_titles_and_the_employees(default_run, data, title_rows):
    # Q16: each title is in exactly one row; titles and employees add up to
    # the totals and the shares to 100%.
    decided = default_run.state("SELECT family_code, cleaned_title, table_row FROM decisions")
    assert sorted(TitleKey(*row[:2]) for row in decided) == sorted(title_rows)
    titles = dict.fromkeys(ROW_NAMES, 0)
    employees = dict.fromkeys(ROW_NAMES, 0)
    for family, title, table_row in decided:
        titles[table_row] += 1
        employees[table_row] += title_rows[TitleKey(family, title)].headcount
    printed = printed_table(default_run.stdout)
    for name in ROW_NAMES:
        assert printed[name][:2] == [str(titles[name]), str(employees[name])], name
        share = float(printed[name][2].rstrip("%"))
        assert abs(share - 100 * employees[name] / len(data.employees)) <= 0.05, name
    assert printed["total"] == [str(len(title_rows)), str(len(data.employees)), "100.0%"]
    assert sum(titles.values()) == len(title_rows)
    assert sum(employees.values()) == len(data.employees)
    shares = sum(float(printed[name][2].rstrip("%")) for name in ROW_NAMES)
    assert abs(shares - 100) <= 0.05 * len(ROW_NAMES)


def test_the_default_run_has_the_shape_the_record_asks_for(default_run):
    # Q23, on the default seed: all seven rows hold titles; the models-agree
    # path holds more than 60, so a sampled audit really runs, and it is not
    # stopped. The agent-settled path need not exceed 60: a path of 60 titles
    # or fewer is labeled in full and gets no status (2.4).
    printed = printed_table(default_run.stdout)
    for name in ROW_NAMES:
        assert int(printed[name][0]) > 0, name
    audits = {row[0]: row[1:] for row in default_run.state("SELECT * FROM audits")}
    size, audited, _, outcome = audits["models agree"]
    assert size > AUDIT_SAMPLE_SIZE and audited == AUDIT_SAMPLE_SIZE
    assert outcome in (AuditOutcome.CONFIRMED.value, AuditOutcome.PROVISIONAL.value)
    size, audited, _, outcome = audits["agent-settled"]
    assert audited == min(size, AUDIT_SAMPLE_SIZE) and outcome != AuditOutcome.STOPPED.value
    # The leftover mechanism of Q01 is what fills the last two rows.
    undecided = [
        line for line in default_run.queue.values() if line["decision"] == "could not decide"
    ]
    assert {line["outcome"] for line in undecided} == {
        "generic family role",
        "left on legacy title",
    }


def test_the_status_of_a_path_follows_from_its_audited_and_wrong_counts(default_run):
    printed = printed_table(default_run.stdout)
    audits = default_run.state("SELECT path, path_size, audited, wrong, outcome FROM audits")
    assert [row[0] for row in audits] == list(PATH_ROWS)
    for path, size, audited, wrong, outcome in audits:
        assert default_run.state("SELECT count(*) FROM audit_samples WHERE path = ?", path) == [
            (audited,)
        ]
        if size <= AUDIT_SAMPLE_SIZE:
            expected = "all labeled"
        elif wrong <= audit.confirm_limit(audited):
            expected = "Confirmed"
        elif wrong > audit.stop_limit(audited):
            expected = "Stopped"
        else:
            expected = "Provisional"
        assert outcome == expected
        assert printed[path][3:5] == [str(audited), str(wrong)]
        assert printed[path][5].startswith(expected)
        # Wrong means the machine's role differs from the SME's label (2.4).
        differing = default_run.state(
            "SELECT count(*) FROM audit_samples s "
            "JOIN human_decisions h USING (family_code, cleaned_title) "
            "JOIN model_answers a USING (family_code, cleaned_title) "
            "WHERE s.path = ? AND h.occasion = 'audit label' AND a.stage = 'Wave 2' "
            "AND a.step = ? AND a.role_id IS NOT h.role_id",
            path,
            "choose" if path == "models agree" else "review",
        )
        assert differing == [(wrong,)]
    for name in set(ROW_NAMES) - set(PATH_ROWS):
        assert len(printed[name]) <= 4  # audited and wrong are filled on the path rows only
    assert printed["left on legacy title"][3:] == []
    assert printed["generic family role"][3:] == ["Provisional"]
    for name in ("look-up hit", "approved in Wave 1", "review queue"):
        assert printed[name][3:] == ["Approved"]


def test_the_footer_agrees_with_the_files_and_the_state(default_run, data, title_rows):
    text = default_run.stdout
    # Coverage by status is by employee and includes those left on a legacy title.
    by_status = dict.fromkeys(("Approved", "Confirmed", "Provisional"), 0)
    for line in default_run.load:
        by_status[line["status"]] += 1
    by_status["left on legacy title"] = len(data.employees) - len(default_run.load)
    (coverage,) = [line for line in text.splitlines() if line.startswith("Coverage by status")]
    for name, count in by_status.items():
        share = float(re.search(rf"{name} ([\d.]+)%", coverage).group(1))
        assert abs(share - 100 * count / len(data.employees)) <= 0.05, name
    # Sent to a person: the Wave 1 list and the review queue, over all titles.
    wave1 = default_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE on_wave1_list"
    )
    sent, total, on_list, in_queue, audited, golden = printed_numbers(
        text,
        r"Sent to a person: (\d+) of (\d+) titles \([\d.]+%\): (\d+) on the Wave 1 list, "
        r"(\d+) in the review queue\. Blind labels besides: (\d+) in the audits, "
        r"(\d+) on the golden set\.",
    )
    assert (on_list, in_queue) == (len(wave1), len(default_run.queue))
    assert sent == len(wave1 | set(default_run.queue)) and total == len(title_rows)
    assert audited == default_run.state("SELECT sum(audited) FROM audits")[0][0]
    assert golden == 40
    # The golden set at each stage, as the state holds it.
    scores = default_run.state("SELECT * FROM golden_scores ORDER BY rowid")
    assert [score[0] for score in scores] == ["Wave 1", "Wave 2", "close"]
    for stage, size, ml_right, llm_right, no_answer in scores:
        line = rf"  {stage}: ML first choice (\d+)/(\d+), LLM pick (\d+)/(\d+) \((\d+) with .*"
        assert printed_numbers(text, line) == [ml_right, size, llm_right, size, no_answer]
        assert size == 40 and ml_right <= size and no_answer <= size - llm_right
    # The two counts under the table (Q05).
    legacy = [line for line in default_run.exceptions if line["released"] == "0"]
    no_grade = [line for line in default_run.exceptions if line["released"] == "1"]
    counts = printed_numbers(
        text, r"Exceptions list: (\d+) employees left on a legacy title; (\d+) employees .*"
    )
    assert counts == [len(legacy), len(no_grade)]
    assert int(printed_table(text)["left on legacy title"][1]) == len(legacy)


def test_the_golden_set_is_scored_against_the_sme_s_labels(default_run):
    # Q08: the frame, the labels and what counts as right.
    golden = default_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE in_golden_set"
    )
    assert len(golden) == 40
    outside = default_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE on_wave1_list "
        "OR has_person_name OR NOT family_usable"
    ) | default_run.keys("SELECT family_code, cleaned_title FROM lookup WHERE joined_at = 'set-up'")
    assert golden.isdisjoint(outside)
    labels = default_run.keys(
        "SELECT family_code, cleaned_title FROM human_decisions WHERE occasion = 'golden-set label'"
    )
    assert labels == golden
    for stage, ml_right, llm_right in default_run.state(
        "SELECT stage, ml_right, llm_right FROM golden_scores"
    ):
        scored = default_run.state(
            "SELECT s.ml_first_choice, a.role_id, h.role_id FROM scorings s "
            "JOIN human_decisions h USING (family_code, cleaned_title) "
            "LEFT JOIN model_answers a ON a.stage = s.stage AND a.step = 'choose' "
            "AND a.family_code = s.family_code AND a.cleaned_title = s.cleaned_title "
            "WHERE s.stage = ? AND h.occasion = 'golden-set label'",
            stage,
        )
        assert len(scored) == 40
        assert sum(first == label for first, _, label in scored) == ml_right
        assert sum(pick == label for _, pick, label in scored) == llm_right
    # Re-scored at every wave, on answers of its own: the examples grow in between.
    asked = default_run.state(
        "SELECT stage, count(*) FROM model_answers WHERE step = 'choose' "
        "AND (family_code, cleaned_title) IN (SELECT family_code, cleaned_title FROM title_rows "
        "WHERE in_golden_set) GROUP BY stage"
    )
    assert dict(asked) == {"Wave 1": 40, "Wave 2": 40, "close": 40}
    # Never the agent in the two passes that are made for scoring only.
    assert not default_run.state(
        "SELECT 1 FROM model_answers WHERE step = 'review' AND stage = 'close'"
    )


# --- the files (1.6 step 9, Q05, Q27, S12) -----------------------------------------------


def test_load_file_and_exceptions_cover_every_employee_exactly_once(default_run, data):
    # Q05: load.csv plus the released = 0 exceptions.
    loaded = [line["employee_id"] for line in default_run.load]
    legacy = [line["employee_id"] for line in default_run.exceptions if line["released"] == "0"]
    assert sorted(loaded + legacy) == sorted(employee.employee_id for employee in data.employees)
    assert len(set(loaded + legacy)) == len(data.employees)
    assert loaded == sorted(loaded)  # sorted by employee_id
    # The reverse file holds exactly the employees of the load file.
    assert [line["employee_id"] for line in default_run.reverse] == loaded
    # An employee is on the exceptions list once.
    listed = [line["employee_id"] for line in default_run.exceptions]
    assert len(listed) == len(set(listed)) and listed == sorted(listed)


def test_the_load_file_says_what_was_decided_by_whom_and_how_far_to_rely_on_it(
    default_run, title_rows
):
    row_of = {
        employee_id: key for key, row in title_rows.items() for employee_id in row.employee_ids
    }
    decisions = {
        TitleKey(family, title): rest
        for family, title, *rest in default_run.state("SELECT * FROM decisions")
    }
    statuses = set()
    for line in default_run.load:
        table_row, source, role_id, status, decided_by, simulated, error_rate, legacy = decisions[
            row_of[line["employee_id"]]
        ]
        assert legacy is None and table_row != "left on legacy title"
        assert (line["role_id"], line["status"], line["path"]) == (role_id, status, table_row)
        assert (line["decided_by"], line["simulated"]) == (decided_by, str(simulated))
        statuses.add(status)
        # 1.9: Approved is "a named person decided it".
        if status == "Approved":
            assert source in ("person", "look-up") and decided_by
            person = source == "person"
            assert simulated == person and decided_by.endswith("(simulated)") == person
            assert person or decided_by == LOOKUP_DECIDED_BY
        else:
            assert source in ("machine", "code") and not decided_by and not simulated
        # An error rate only where Provisional comes from an audited path.
        by_a_path = status == "Provisional" and source == "machine"
        assert (line["error_rate"] != "") == by_a_path == (error_rate is not None)
    assert statuses <= {"Approved", "Confirmed", "Provisional"} and "Approved" in statuses


def test_an_employee_with_no_grade_gets_the_title_with_no_level_word_and_is_listed(
    default_run, data
):
    # 2.4, Q05
    grade = {employee.employee_id: employee.grade for employee in data.employees}
    released = {line["employee_id"]: line for line in default_run.load}
    no_grade = {employee_id for employee_id in released if grade[employee_id] is None}
    assert no_grade
    for employee_id, line in released.items():
        assert (line["no_grade"] == "1") == (line["level"] == "") == (employee_id in no_grade)
    listed = {
        line["employee_id"]: line for line in default_run.exceptions if line["released"] == "1"
    }
    assert set(listed) == no_grade
    assert {line["reason"] for line in listed.values()} == {"no grade"}
    # One with no grade who stays on a legacy title has the released = 0 line only.
    reasons = {line["reason"] for line in default_run.exceptions if line["released"] == "0"}
    assert reasons and "no grade" not in reasons


def test_the_exceptions_list_gives_the_legacy_title_and_the_reason(default_run, data, title_rows):
    employees = {employee.employee_id: employee for employee in data.employees}
    row_of = {
        employee_id: key for key, row in title_rows.items() for employee_id in row.employee_ids
    }
    reasons = {
        TitleKey(family, title): reason
        for family, title, reason in default_run.state(
            "SELECT family_code, cleaned_title, legacy_reason FROM decisions "
            "WHERE table_row = 'left on legacy title'"
        )
    }
    legacy = [line for line in default_run.exceptions if line["released"] == "0"]
    assert legacy
    for line in default_run.exceptions:
        employee = employees[line["employee_id"]]
        assert line["legacy_job_title"] == employee.job_title
        assert line["job_family_code"] == employee.job_family_code
    for line in legacy:
        assert line["reason"] == reasons[row_of[line["employee_id"]]]
    assert {row_of[line["employee_id"]] for line in legacy} == set(reasons)


def test_the_review_queue_file_lists_every_title_that_entered_the_queue(default_run, title_rows):
    # Q16, S12: the queue as it stood after Wave 2, with what happened to it.
    queue = default_run.queue
    routed = default_run.keys(
        "SELECT family_code, cleaned_title FROM scorings "
        "WHERE stage = 'Wave 2' AND route = 'person'"
    )
    unscored = {key for key, row in title_rows.items() if not row.model_eligible}
    assert set(queue) == routed | unscored and routed.isdisjoint(unscored) and unscored
    outcomes = {
        TitleKey(family, title): table_row
        for family, title, table_row in default_run.state(
            "SELECT family_code, cleaned_title, table_row FROM decisions"
        )
    }
    for key, line in queue.items():
        assert line["headcount"] == str(title_rows[key].headcount)
        assert line["reasons"] and line["outcome"] == outcomes[key]
        # People are simulated, and the file says so on every decision (Q27).
        assert line["simulated"] == "1" and line["decided_by"] == "Reviewer (simulated)"
        if key in unscored:
            assert line["answer_source"] == "" and line["ml_first_choice"] == ""
        else:
            assert line["answer_source"] == "stand-in" and line["ml_first_choice"]
        decided = line["decision"] not in ("could not decide", "no suitable role")
        assert (line["outcome"] == "review queue") == decided
    assert {line["outcome"] for line in queue.values()} == {
        "review queue",
        "generic family role",
        "left on legacy title",
    }
    # Every title has a truth row in the mock data, so nothing is marked (Q03).
    assert "no suitable role" not in {line["decision"] for line in queue.values()}


# --- the eight rules of 2.7, on the run's files and state ----------------------------------


def test_t1_no_title_outside_the_catalog_reaches_the_load_file(default_run, data):
    catalog = data.catalog
    grade = {employee.employee_id: employee.grade for employee in data.employees}
    assert len(default_run.load) > 1400
    for line in default_run.load:
        role = get_role(catalog, line["role_id"])
        assert role is not None, line
        assert is_catalog_title(catalog, line["long_title"], line["short_title"]), line
        # The titles are the role's own, at the level HR's table gives the grade (1.3, 1.4).
        title = standard_title(catalog, role, grade[line["employee_id"]])
        assert (line["long_title"], line["short_title"]) == (title.long_title, title.short_title)
        assert line["level"] == (level_for_grade(catalog, grade[line["employee_id"]]) or "")


def test_t3_a_compliance_flagged_role_is_never_settled_by_a_machine(default_run, data):
    flagged = {role.role_id for role in data.catalog.roles if role.compliance_flag}
    assert len(flagged) == 3
    # No title in a machine path, audited or not, has a flagged role or one
    # among the ML model's top three (1.7).
    in_a_path = default_run.state(
        "SELECT s.family_code, s.cleaned_title, a.role_id FROM scorings s "
        "JOIN model_answers a USING (stage, family_code, cleaned_title) "
        "WHERE s.stage = 'Wave 2' AND s.route IN ('models agree', 'agent-settled') "
        "AND a.route = 'accept'"
    )
    assert len(in_a_path) > 200
    top_three = {}
    for family, title, role_id in default_run.state(
        "SELECT family_code, cleaned_title, role_id FROM ml_shortlist "
        "WHERE stage = 'Wave 2' AND place <= 3"
    ):
        top_three.setdefault(TitleKey(family, title), set()).add(role_id)
    for family, title, role_id in in_a_path:
        assert role_id not in flagged
        assert top_three[TitleKey(family, title)].isdisjoint(flagged)
    # What a machine released without a person carries no flagged role.
    by_machine = [line for line in default_run.load if line["status"] != "Approved"]
    assert by_machine and not [line for line in by_machine if line["role_id"] in flagged]
    # The titles that do touch a flagged role all went to the review queue.
    touching = {key for key, roles in top_three.items() if roles & flagged}
    touching |= default_run.keys(
        "SELECT family_code, cleaned_title FROM model_answers WHERE stage = 'Wave 2' "
        f"AND role_id IN ({', '.join('?' * len(flagged))})",
        *sorted(flagged),
    )
    assert len(touching) > 10 and touching <= set(default_run.queue)
    for key in touching:
        assert "compliance-flagged role" in default_run.queue[key]["reasons"]
    # People do give flagged roles: those rows are Approved.
    assert {line["status"] for line in default_run.load if line["role_id"] in flagged} == {
        "Approved"
    }


def test_t5_no_model_is_ever_shown_a_generic_family_role(default_run, data):
    generic = {role.role_id for role in data.catalog.roles if role.is_generic}
    assert len(generic) == 8
    # Never on a shortlist.
    listed = {role_id for (role_id,) in default_run.state("SELECT role_id FROM ml_shortlist")}
    assert len(listed) == 48 and listed.isdisjoint(generic)
    # Never in a numbered list or a search result, at any stage.
    shown = default_run.state("SELECT DISTINCT stage, shown_in, role_id FROM shown_roles")
    assert {role_id for _, _, role_id in shown}.isdisjoint(generic)
    places = {(stage, shown_in) for stage, shown_in, _ in shown}
    assert places >= {
        ("Wave 1", "numbered list"),
        ("Wave 2", "numbered list"),
        ("close", "numbered list"),
        ("Wave 2", "case"),
        ("Wave 2", "search_catalog"),
        ("Wave 2", "search_approved_titles"),
    }
    # Only code assigns one (1.2): where the load file has it, nobody chose it.
    with_generic = [line for line in default_run.load if line["role_id"] in generic]
    assert with_generic
    assert {(line["path"], line["decided_by"]) for line in with_generic} == {
        ("generic family role", "")
    }
    people_gave = {role for (role,) in default_run.state("SELECT role_id FROM human_decisions")}
    assert people_gave.isdisjoint(generic)


def test_t6_golden_set_titles_never_enter_the_look_up_table_or_the_examples(default_run):
    golden = default_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE in_golden_set"
    )
    # People did decide golden titles after labeling them: in an audit and in
    # the review queue (Q07, Q21). None of it was learned, at any stage.
    decided = default_run.keys(
        "SELECT family_code, cleaned_title FROM human_decisions "
        "WHERE occasion IN ('audit label', 'review-queue decision') AND answer = 'role'"
    )
    assert len(golden & decided) >= 5
    for table in ("lookup", "examples"):
        held = default_run.keys(f"SELECT family_code, cleaned_title FROM {table}")
        assert len(held) > 100 and held.isdisjoint(golden), table
        stages = {stage for (stage,) in default_run.state(f"SELECT joined_at FROM {table}")}
        assert stages == {"set-up", "Wave 1", "Wave 2", "close"}
    # For release they are ordinary long-tail titles (Q07).
    rows = default_run.state(
        "SELECT DISTINCT table_row FROM decisions d JOIN title_rows t "
        "USING (family_code, cleaned_title) WHERE t.in_golden_set"
    )
    assert {"models agree", "review queue"} <= {row for (row,) in rows}


def test_only_what_a_person_decided_is_learned(default_run):
    # 1.6 step 10: a title the agent settled does not join until a person approves it.
    looked_up = default_run.keys("SELECT family_code, cleaned_title FROM lookup")
    by_person = default_run.keys(
        "SELECT family_code, cleaned_title FROM decisions WHERE source IN ('person', 'look-up')"
    )
    golden = default_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE in_golden_set"
    )
    assert looked_up == by_person - golden
    unaudited = default_run.keys(
        "SELECT family_code, cleaned_title FROM decisions WHERE source = 'machine'"
    )
    assert unaudited and looked_up.isdisjoint(unaudited)
    # A title with a person's name joins the look-up table and never the examples (Q14).
    examples = default_run.keys("SELECT family_code, cleaned_title FROM examples")
    named = default_run.keys(
        "SELECT family_code, cleaned_title FROM title_rows WHERE has_person_name"
    )
    assert len(named) == 5 and examples == looked_up - named


def check_the_generic_role(run_queue, decisions, title_rows, data, settings) -> int:
    """T7 on one run; returns the number of employees given a generic role."""
    catalog = data.catalog
    cap = generic_cap(len(data.employees), settings)
    generic = {key: role for key, (row, role) in decisions.items() if row == "generic family role"}
    assert generic
    held = sum(title_rows[key].headcount for key in generic)
    assert held <= cap  # never past the cap
    # The Wave 1 list, worked out again from the headcounts (Q24).
    widest = sorted(title_rows, key=lambda key: (-title_rows[key].headcount, key))
    wave1 = set(widest[: settings.wave1_size])
    for key, role_id in generic.items():
        assert key not in wave1  # never a Wave 1 title
        line = run_queue[key]
        assert line["decision"] == "could not decide"  # a title nobody could decide (1.8)
        assert "compliance-flagged role" not in line["reasons"]  # never a flagged title
        assert title_rows[key].model_eligible  # a model scored it, and it has a family
        assert role_id == generic_role(catalog, key.family_code).role_id
    return held


def test_t7_the_generic_role_is_never_past_its_cap_or_on_a_barred_title(
    default_run, data, title_rows
):
    decisions = {
        TitleKey(family, title): (row, role)
        for family, title, row, role in default_run.state(
            "SELECT family_code, cleaned_title, table_row, role_id FROM decisions"
        )
    }
    check_the_generic_role(default_run.queue, decisions, title_rows, data, Settings())
    statuses = default_run.state(
        "SELECT DISTINCT source, status, decided_by, simulated, error_rate FROM decisions "
        "WHERE table_row = 'generic family role'"
    )
    assert statuses == [("code", "Provisional", "", 0, None)]
    assert default_run.state(
        "SELECT value FROM thresholds WHERE name = 'generic_cap_employees'"
    ) == [("30",)]


def test_t8_applying_the_reverse_file_to_the_load_file_restores_every_legacy_title(
    default_run, data
):
    restored = release.apply_reverse(default_run.load, default_run.reverse)
    received = {employee.employee_id: employee.job_title for employee in data.employees}
    assert list(restored) == [line["employee_id"] for line in default_run.load]
    assert len(restored) > 1400
    for employee_id, title in restored.items():
        assert title == received[employee_id]  # as received, byte for byte
    # The titles really are messy: most do not read like the Long Title given.
    assert (
        sum(line["long_title"] != restored[line["employee_id"]] for line in default_run.load) > 1000
    )


# --- runs the default seed does not show (Q01, 2.4, Q21, Q23) -----------------------------


@pytest.fixture(scope="module")
def run_with(repo_root, tmp_path_factory):
    """Run the pipeline on the mock data in this process, once per setting."""
    made: dict[tuple, tuple[Settings, RunResult]] = {}

    def run(**changes: object) -> tuple[Settings, RunResult]:
        name = tuple(sorted(changes.items()))
        if name not in made:
            out_dir = tmp_path_factory.mktemp("run")
            settings = Settings(data_dir=repo_root / "data", out_dir=out_dir, **changes)
            made[name] = (settings, pipeline.run_pipeline(settings))
        return made[name]

    return run


def headcounts(result: RunResult) -> dict[TitleKey, int]:
    return {row.key: row.headcount for row in result.rows}


def test_at_a_high_undecided_rate_the_generic_role_stops_at_its_cap(run_with, data, title_rows):
    # Q01: one run at a high rate, for the cap overflow and the two bars (Q25).
    settings, result = run_with(undecided_rate=0.8)
    queue = {
        TitleKey(line["job_family_code"], line["cleaned_title"]): line
        for line in datafiles.read_table(
            settings.out_dir / release.REVIEW_QUEUE_FILE, release.REVIEW_QUEUE_COLUMNS
        )
    }
    decisions = {each.key: (each.row.value, each.role_id) for each in result.decisions}
    held = check_the_generic_role(queue, decisions, title_rows, data, settings)
    cap = generic_cap(len(data.employees), settings)
    size = headcounts(result)
    legacy = {each.key: each.legacy_reason.value for each in result.decisions if each.legacy_reason}
    over = [key for key, reason in legacy.items() if reason == "over the generic cap"]
    assert over, "the test needs more eligible leftovers than the cap holds"
    # Whole titles in ascending headcount: the smallest left out would not have fitted.
    generic = [each.key for each in result.decisions if each.row is TableRow.GENERIC]
    assert held + min(size[key] for key in over) > cap
    assert max(size[key] for key in generic) <= min(size[key] for key in over)
    # The two bars: a flagged title and one no model scored stay on legacy titles.
    assert "undecided widely held or flagged title" in legacy.values()
    assert "undecided title with a person's name or no usable family" in legacy.values()
    for key, reason in legacy.items():
        line = queue[key]
        assert line["decision"] == "could not decide" and line["outcome"] == "left on legacy title"
        flagged = "compliance-flagged role" in line["reasons"]
        unscored = not title_rows[key].model_eligible
        assert (reason == "undecided widely held or flagged title") == flagged
        assert (reason == "undecided title with a person's name or no usable family") == (
            unscored and not flagged
        )
        assert (reason == "over the generic cap") == (not flagged and not unscored)
    # Every row still adds up, and the queue did not change: only who could decide.
    assert sum(size[each.key] for each in result.decisions) == len(data.employees)
    default = run_with()[1]
    assert {entry.row.key for entry in result.queue} == {e.row.key for e in default.queue}


def test_the_in_process_run_is_the_default_run(run_with, default_run):
    # The runs below differ from the default run in one setting only.
    settings, result = run_with()
    for name in pipeline.OUTPUT_FILES:
        assert (settings.out_dir / name).read_bytes() == (default_run.folder / name).read_bytes()
    assert len(result.rows) == 400


def path_rows(result: RunResult) -> dict[str, list]:
    """Per machine path: its audit and the decisions counted in its row of the table."""
    found = {}
    for each in result.audits:
        row = [d for d in result.decisions if d.row.value == each.path.value]
        found[each.path.value] = [each, row]
    return found


def test_a_path_that_misses_the_bar_is_released_as_provisional_with_its_error_rate(run_with):
    # 1.9, 2.4: 3 to 12 wrong of 60. More SME errors count against the machine (Q27).
    settings, result = run_with(sme_error_rate=0.15)
    audited, row = path_rows(result)["models agree"]
    assert audited.outcome is AuditOutcome.PROVISIONAL
    assert audit.confirm_limit(60) < audited.wrong <= audit.stop_limit(60)
    assert len(row) == audited.path_size > AUDIT_SAMPLE_SIZE
    load = datafiles.read_table(settings.out_dir / release.LOAD_FILE, release.LOAD_COLUMNS)
    released = [line for line in load if line["path"] == "models agree"]
    rates = {line["error_rate"] for line in released if line["status"] == "Provisional"}
    assert rates == {f"{audited.wrong / 60:.4f}"}
    # The audited titles take the SME's label and are Approved, right or wrong (Q21).
    by_status = dict.fromkeys(("Approved", "Provisional"), 0)
    for decision in row:
        by_status[decision.status.value] += 1
        assert (decision.key in audited.audited_keys) == (decision.status.value == "Approved")
    assert by_status == {"Approved": 60, "Provisional": audited.path_size - 60}
    assert {line["status"] for line in released} == {"Approved", "Provisional"}
    assert not [line for line in load if line["status"] == "Confirmed"]


@pytest.mark.parametrize("release_stopped_paths", [False, True])
def test_a_stopped_path_stays_on_legacy_titles_unless_the_setting_releases_it(
    run_with, release_stopped_paths
):
    # 2.4, Q16, Q21: more than 12 wrong of 60.
    settings, result = run_with(sme_error_rate=0.3, release_stopped_paths=release_stopped_paths)
    audited, row = path_rows(result)["models agree"]
    assert audited.outcome is AuditOutcome.STOPPED and audited.wrong > audit.stop_limit(60)
    stopped = [
        each
        for each in result.decisions
        if each.legacy_reason and each.legacy_reason.value == "stopped path"
    ]
    unaudited = sum(
        each.path_size - each.audited
        for each in result.audits
        if each.outcome is AuditOutcome.STOPPED
    )
    load = datafiles.read_table(settings.out_dir / release.LOAD_FILE, release.LOAD_COLUMNS)
    in_row = [line for line in load if line["path"] == "models agree"]
    printed = printed_table(report.render(result, settings))["models agree"]
    # The 60 audited titles are released as Approved either way.
    assert sum(decision.status.value == "Approved" for decision in row) == 60
    assert {line["status"] for line in in_row} >= {"Approved"}
    if release_stopped_paths:
        assert not stopped and len(row) == audited.path_size
        provisional = [line for line in in_row if line["status"] == "Provisional"]
        assert provisional and {line["error_rate"] for line in provisional} == {
            f"{audited.wrong / 60:.4f}"
        }
        assert {line["decided_by"] for line in provisional} == {""}
        assert printed[5].startswith("Stopped, released as provisional, error rate ")
    else:
        # The path's row keeps its audited titles; the rest is counted as legacy.
        assert len(row) == 60 and {line["status"] for line in in_row} == {"Approved"}
        assert len(stopped) == unaudited >= audited.path_size - 60
        exceptions = datafiles.read_table(
            settings.out_dir / release.EXCEPTIONS_FILE, release.EXCEPTIONS_COLUMNS
        )
        size = headcounts(result)
        assert sum(line["reason"] == "stopped path" for line in exceptions) == sum(
            size[decision.key] for decision in stopped
        )
        assert printed[5] == f"Stopped, {audited.path_size - 60} titles left on legacy titles"
    # Nothing the machine alone decided is called Confirmed in a stopped path.
    assert not [d for d in row if d.status.value == "Confirmed"]
