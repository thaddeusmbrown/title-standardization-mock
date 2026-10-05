"""The printed table and its footer, rendered from hand-built results (2.6, Q16)."""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from titlestd import report
from titlestd.contracts import (
    LOOKUP_DECIDED_BY,
    AuditOutcome,
    AuditResult,
    Decision,
    DecisionSource,
    ExceptionReason,
    GoldenScore,
    MachinePath,
    QueueEntry,
    Reason,
    Record,
    RunResult,
    Stage,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
    recorded_label,
)
from titlestd.settings import Settings

SETTINGS = Settings()


def title_row(title: str, employees: int) -> TitleRow:
    record = Record(title, "SAL", "Sales", "")
    return TitleRow(record, tuple(f"{title}-{at}" for at in range(employees)), (title,))


def decided(row: TitleRow, table_row: TableRow) -> Decision:
    """A decision of the kind the table row holds."""
    if table_row is TableRow.LEGACY:
        return Decision(
            row.key, table_row, DecisionSource.NONE, legacy_reason=ExceptionReason.STOPPED_PATH
        )
    if table_row is TableRow.GENERIC:
        return Decision(row.key, table_row, DecisionSource.CODE, "SAL-GEN", Status.PROVISIONAL)
    if table_row is TableRow.LOOKUP:
        return Decision(
            row.key, table_row, DecisionSource.LOOKUP, "SAL-01", Status.APPROVED, LOOKUP_DECIDED_BY
        )
    return Decision(
        row.key, table_row, DecisionSource.PERSON, "SAL-01", Status.APPROVED, "SME 1", True
    )


def a_result(**changes) -> RunResult:
    """Ten titles on 40 employees, at least one in every row of the table."""
    plan = [
        ("seeded", 3, TableRow.LOOKUP),
        ("widely held", 12, TableRow.WAVE1),
        ("agreed and audited", 5, TableRow.MODELS_AGREE),
        ("agreed", 4, TableRow.MODELS_AGREE),
        ("settled", 3, TableRow.AGENT_SETTLED),
        ("queued", 6, TableRow.REVIEW_QUEUE),
        ("queued too", 1, TableRow.REVIEW_QUEUE),
        ("leftover", 1, TableRow.GENERIC),
        ("barred", 2, TableRow.LEGACY),
        ("stopped", 3, TableRow.LEGACY),
    ]
    rows = tuple(sorted((title_row(title, count) for title, count, _ in plan), key=_key))
    by_title = {row.record.cleaned_title: row for row in rows}
    decisions = {title: decided(by_title[title], table_row) for title, _, table_row in plan}
    decisions["agreed"] = Decision(
        by_title["agreed"].key,
        TableRow.MODELS_AGREE,
        DecisionSource.MACHINE,
        "SAL-01",
        Status.CONFIRMED,
    )
    queue = tuple(
        QueueEntry(by_title[title], (Reason.NO_MATCH,), None, None, None, "stand-in", None, outcome)
        for title, outcome in (
            ("barred", TableRow.LEGACY),
            ("leftover", TableRow.GENERIC),
            ("queued", TableRow.REVIEW_QUEUE),
            ("queued too", TableRow.REVIEW_QUEUE),
        )
    )
    result = RunResult(
        rows=rows,
        employee_count=40,
        no_grade_employees=2,
        wave1_keys=(by_title["widely held"].key,),
        decisions=tuple(decisions[row.record.cleaned_title] for row in rows),
        audits=(
            audit(MachinePath.MODELS_AGREE, 2, 1, 0, AuditOutcome.ALL_LABELED),
            audit(MachinePath.AGENT_SETTLED, 1, 1, 1, AuditOutcome.ALL_LABELED),
        ),
        golden_keys=(by_title["agreed"].key, by_title["queued"].key),
        golden_scores=(
            GoldenScore(Stage.WAVE1, 2, 1, 2, 0),
            GoldenScore(Stage.WAVE2, 2, 2, 1, 1),
            GoldenScore(Stage.CLOSE, 2, 2, 0, 2),
        ),
        queue=queue,
        answer_labels=("stand-in",),
    )
    return dataclasses.replace(result, **changes)


def _key(row: TitleRow) -> TitleKey:
    return row.key


def audit(
    path: MachinePath, size: int, audited: int, wrong: int, outcome: AuditOutcome
) -> AuditResult:
    keys = tuple(TitleKey("SAL", f"audited {at}") for at in range(audited))
    return AuditResult(path, size, keys, wrong, outcome)


def table(text: str) -> dict[str, list[str]]:
    """The cells of each table row, by the name in its first column."""
    names = [*(row.value for row in TableRow), report.TOTAL]
    found = {}
    for line in text.splitlines():
        for name in names:
            if line.startswith(name + " "):
                found[name] = re.split(r"\s{2,}", line.strip())
    return found


def line_starting(text: str, start: str) -> str:
    (line,) = [line for line in text.splitlines() if line.startswith(start)]
    return line


# --- the table ------------------------------------------------------------------


def test_the_table_has_the_seven_rows_in_order_and_the_six_columns():
    text = report.render(a_result(), SETTINGS)
    lines = text.splitlines()
    header = next(at for at, line in enumerate(lines) if line.lstrip().startswith("titles"))
    assert re.split(r"\s{2,}", lines[header].strip()) == [
        "titles",
        "employees",
        "share of employees",
        "audited",
        "wrong",
        "status",
    ]
    assert set(lines[header + 1]) == {"-", " "}
    names = [line.split("  ")[0] for line in lines[header + 2 : header + 9]]
    assert names == [
        "look-up hit",
        "approved in Wave 1",
        "models agree",
        "agent-settled",
        "review queue",
        "generic family role",
        "left on legacy title",
    ]
    assert lines[header + 10].startswith("total")


def test_each_row_counts_its_titles_and_their_employees():
    rows = table(report.render(a_result(), SETTINGS))
    assert rows["look-up hit"][1:4] == ["1", "3", "7.5%"]
    assert rows["approved in Wave 1"][1:4] == ["1", "12", "30.0%"]
    assert rows["models agree"][1:4] == ["2", "9", "22.5%"]
    assert rows["agent-settled"][1:4] == ["1", "3", "7.5%"]
    assert rows["review queue"][1:4] == ["2", "7", "17.5%"]
    assert rows["generic family role"][1:4] == ["1", "1", "2.5%"]
    assert rows["left on legacy title"][1:4] == ["2", "5", "12.5%"]
    assert rows["total"] == ["total", "10", "40", "100.0%"]


def test_the_rows_add_up_to_the_totals_and_the_shares_to_100_percent():
    # Q16: every title is in exactly one row.
    rows = table(report.render(a_result(), SETTINGS))
    body = [rows[row.value] for row in TableRow]
    assert sum(int(cells[1]) for cells in body) == int(rows["total"][1]) == 10
    assert sum(int(cells[2]) for cells in body) == int(rows["total"][2]) == 40
    # Here every share is a whole number of tenths, so the printed ones add up too.
    tenths = sum(int(cells[3].rstrip("%").replace(".", "")) for cells in body)
    assert tenths == 1000


def test_audited_and_wrong_are_filled_only_on_the_two_path_rows():
    rows = table(report.render(a_result(), SETTINGS))
    assert rows["models agree"][4:] == ["1", "0", "all labeled"]
    assert rows["agent-settled"][4:] == ["1", "1", "all labeled"]
    # The other rows: the status of their titles, and none for legacy.
    assert rows["look-up hit"][4:] == ["Approved"]
    assert rows["approved in Wave 1"][4:] == ["Approved"]
    assert rows["review queue"][4:] == ["Approved"]
    assert rows["generic family role"][4:] == ["Provisional"]
    assert rows["left on legacy title"][4:] == []


@pytest.mark.parametrize(
    ("outcome", "wrong", "release", "status"),
    [
        (AuditOutcome.CONFIRMED, 2, False, "Confirmed"),
        (AuditOutcome.PROVISIONAL, 3, False, "Provisional, error rate 5.0%"),
        (AuditOutcome.PROVISIONAL, 12, True, "Provisional, error rate 20.0%"),
        (AuditOutcome.STOPPED, 13, False, "Stopped, 150 titles left on legacy titles"),
        (AuditOutcome.STOPPED, 15, True, "Stopped, released as provisional, error rate 25.0%"),
    ],
)
def test_the_status_of_a_path_row_is_its_audit_outcome(outcome, wrong, release, status):
    audits = (
        audit(MachinePath.MODELS_AGREE, 210, 60, wrong, outcome),
        audit(MachinePath.AGENT_SETTLED, 0, 0, 0, AuditOutcome.ALL_LABELED),
    )
    settings = Settings(release_stopped_paths=release)
    rows = table(report.render(a_result(audits=audits), settings))
    assert rows["models agree"][4:] == ["60", str(wrong), status]
    # A path with no title was audited in nothing and has no status.
    assert rows["agent-settled"][4:] == ["0", "0"]


def test_a_row_with_no_title_shows_no_status():
    def elsewhere(decision: Decision) -> Decision:
        if decision.row is TableRow.LOOKUP:
            return dataclasses.replace(decision, row=TableRow.WAVE1)
        if decision.row is TableRow.GENERIC:
            return decided(TitleRow(Record(*decision.key[::-1], "", ""), (), ()), TableRow.LEGACY)
        return decision

    result = a_result()
    moved = tuple(elsewhere(decision) for decision in result.decisions)
    rows = table(report.render(dataclasses.replace(result, decisions=moved), SETTINGS))
    assert rows["look-up hit"] == ["look-up hit", "0", "0", "0.0%"]
    assert rows["generic family role"] == ["generic family role", "0", "0", "0.0%"]
    assert rows["approved in Wave 1"][1:] == ["2", "15", "37.5%", "Approved"]


# --- shares -----------------------------------------------------------------------


def test_a_share_is_rounded_half_up_to_one_decimal_on_its_own():
    assert report._percent(1, 3) == "33.3%" and report._percent(2, 3) == "66.7%"
    assert report._percent(13, 1500) == "0.9%" and report._percent(11, 1500) == "0.7%"
    # Exactly half a tenth goes up, which binary floating point would not promise.
    assert report._percent(1, 2000) == "0.1%" and report._percent(5, 2000) == "0.3%"
    assert report._percent(0, 40) == "0.0%" and report._percent(40, 40) == "100.0%"
    assert report._percent(0, 0) == "0.0%"


def test_the_same_count_shows_the_same_share_in_the_table_and_in_the_footer():
    # 13 of 1,500 employees are on a legacy title: 0.9% in both places.
    result = a_result(employee_count=1500)
    big = title_row("widely held", 1500 - 40 + 12 - 8)  # the legacy rows grow from 5 to 13
    rows = tuple(big if row.record.cleaned_title == "widely held" else row for row in result.rows)
    legacy = [d.key for d in result.decisions if d.row is TableRow.LEGACY]
    sized = {legacy[0]: 6, legacy[1]: 7}
    rows = tuple(
        title_row(row.record.cleaned_title, sized[row.key]) if row.key in sized else row
        for row in rows
    )
    assert sum(row.headcount for row in rows) == 1500
    text = report.render(dataclasses.replace(result, rows=rows), SETTINGS)
    assert table(text)["left on legacy title"][1:4] == ["2", "13", "0.9%"]
    assert line_starting(text, "Coverage by status").endswith("left on legacy title 0.9%")


# --- the footer -------------------------------------------------------------------


def test_coverage_by_status_is_by_employee_and_adds_up_to_100_percent():
    text = report.render(a_result(), SETTINGS)
    # Approved 3 + 12 + 5 + 3 + 7 = 30, Confirmed 4, Provisional 1, legacy 5, of 40.
    assert line_starting(text, "Coverage by status") == (
        "Coverage by status, by employee: Approved 75.0%, Confirmed 10.0%, "
        "Provisional 2.5%, left on legacy title 12.5%"
    )


def test_sent_to_a_person_is_the_wave_1_list_and_the_review_queue_over_all_titles():
    line = line_starting(report.render(a_result(), SETTINGS), "Sent to a person")
    assert line == (
        "Sent to a person: 5 of 10 titles (50.0%): 1 on the Wave 1 list, 4 in the review "
        "queue. Blind labels besides: 2 in the audits, 2 on the golden set."
    )


def test_a_title_on_the_wave_1_list_and_in_the_queue_is_sent_to_a_person_once():
    result = a_result()
    both = dataclasses.replace(result, wave1_keys=(result.queue[0].row.key,))
    assert "Sent to a person: 4 of 10 titles (40.0%)" in report.render(both, SETTINGS)


def test_the_golden_set_is_shown_for_each_stage_with_the_titles_given_no_role():
    lines = report.render(a_result(), SETTINGS).splitlines()
    start = lines.index("Golden set, right by the SME's label:")
    assert lines[start + 1 : start + 4] == [
        "  Wave 1: ML first choice 1/2, LLM pick 2/2 (0 with no role named)",
        "  Wave 2: ML first choice 2/2, LLM pick 1/2 (1 with no role named)",
        "  close: ML first choice 2/2, LLM pick 0/2 (2 with no role named)",
    ]


def test_the_two_counts_of_the_exceptions_list_are_printed():
    # Q05: employees left on a legacy title, and employees with no grade.
    assert line_starting(report.render(a_result(), SETTINGS), "Exceptions list") == (
        "Exceptions list: 5 employees left on a legacy title; 2 employees with no grade "
        "given a title with no level word."
    )


@pytest.mark.parametrize(
    ("labels", "shown"),
    [
        (("stand-in",), "stand-in"),
        (("live",), "live"),
        (
            (recorded_label("2026-10-05", "claude-haiku-4-5"),),
            "recorded 2026-10-05, claude-haiku-4-5",
        ),
        (
            tuple(recorded_label(day, "claude-haiku-4-5") for day in ("2026-10-07", "2026-10-05")),
            "recorded 2026-10-05 to 2026-10-07, claude-haiku-4-5",
        ),
        (
            ("live", recorded_label("2026-10-05", "b"), recorded_label("2026-10-06", "a")),
            "live; recorded 2026-10-06, a; recorded 2026-10-05, b",
        ),
        ((), "none, no model was asked"),
    ],
)
def test_one_line_says_where_the_answers_came_from_and_names_the_embedder(labels, shown):
    settings = Settings(embedder="local")
    text = report.render(a_result(answer_labels=labels), settings)
    assert line_starting(text, "Model answers") == f"Model answers: {shown}. Embedder: local."


def test_the_footer_says_people_were_simulated():
    text = report.render(a_result(), SETTINGS)
    assert line_starting(text, "People were simulated").startswith(
        "People were simulated from the truth file"
    )


# --- the text as a whole --------------------------------------------------------


def test_the_same_result_renders_the_same_text_with_no_time_in_it():
    first = report.render(a_result(), SETTINGS)
    assert first == report.render(a_result(), SETTINGS)
    assert not first.endswith("\n") and "\n\n\n" not in first
    assert all(line == line.rstrip() for line in first.splitlines())
    # Q06: no output may hold the wall-clock time.
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}", first)


def test_the_report_reads_no_data_file():
    # Everything printed comes from the RunResult and the settings (S11).
    tree = ast.parse(Path(report.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Name):
            assert node.id != "open"
    assert imported == {
        "__future__",
        "collections.abc",
        "titlestd.contracts",
        "titlestd.settings",
    }
