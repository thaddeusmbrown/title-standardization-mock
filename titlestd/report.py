"""The printed table and its footer.

Owner: feat/pipeline-report-cli (stage 4).
Governed by: docs/design.md 2.6; docs/build-decisions.md Q05 (the counts
printed under the table), Q08 (golden-set lines), Q16 (rows, columns, footer),
Q27 (people were simulated), Q40 (the answer-source line).

Rows, in the order of contracts.TableRow; each title row is in exactly one, so
titles and employees sum to the totals and shares to 100%.
Columns: titles, employees, share of employees, audited, wrong, status.
"Audited" and "wrong" are filled only on the two machine-path rows; "status"
on a path row is the audit outcome (Q16).
Footer: coverage by status, by employee (Approved, Confirmed, Provisional,
left on legacy title); the share of titles sent to a person (the Wave 1 list,
RunResult.wave1_keys, plus the titles routed to the review queue), with blind
labels as a separate count; golden-set accuracy of the ML first choice and of
the LLM's pick, per stage; one line saying whether answers came from stand-ins,
a recording or a live model, naming the embedder; and that people were
simulated. Under the table, the two counts of Q05: employees left on a legacy
title (the headcount of the legacy row) and employees with no grade
(RunResult.no_grade_employees).

Everything printed comes from the RunResult and the settings: this module
reads no data file.

The text holds no wall-clock time, so two runs print the same bytes (Q06).

Decided here, where the record is silent:
- Shares are printed with one decimal and rounded so that the printed shares
  of a column add up to exactly 100.0% (largest remainder, in whole-number
  arithmetic): the table's share column, and the coverage by status.
- The status cell. A path row shows its audit outcome: "Confirmed";
  "Provisional, error rate 5.0%"; "Stopped, 150 titles left on legacy
  titles", or "Stopped, released as provisional, error rate 25.0%" when
  settings.release_stopped_paths is on; "all labeled" for a path of 60 titles
  or fewer. A path with no title shows nothing. Any other row shows the
  status its titles carry, and nothing when it holds none.
- "Sent to a person" counts each title once: the Wave 1 list as drawn (a
  seed look-up hit on it is a person's earlier decision, Q13) and every
  title in the review queue. The percentage is of all titles.
- Blind labels are counted per occasion, so a golden-set title that an audit
  also drew is counted in both numbers.
- The answer-source line lists the distinct answer labels, separated by
  "; ". Recorded labels of one model on several days are shown as one, with
  the first and the last day (Q40). A run in which no model answered says so.
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import (
    AuditOutcome,
    AuditResult,
    RunResult,
    Status,
    TableRow,
    recorded_label,
)
from titlestd.settings import Settings

HEADER = ("", "titles", "employees", "share of employees", "audited", "wrong", "status")
TOTAL = "total"
LEGACY = TableRow.LEGACY.value
_RECORDED = "recorded "

# The status of a row's titles, where it is not an audit outcome (Q16).
_ROW_STATUS = {
    TableRow.LOOKUP: Status.APPROVED,
    TableRow.WAVE1: Status.APPROVED,
    TableRow.REVIEW_QUEUE: Status.APPROVED,
    TableRow.GENERIC: Status.PROVISIONAL,
}


def render(result: RunResult, settings: Settings) -> str:
    """The table and footer of one run, as the text the run command prints."""
    headcount = {row.key: row.headcount for row in result.rows}
    titles = dict.fromkeys(TableRow, 0)
    employees = dict.fromkeys(TableRow, 0)
    by_status = dict.fromkeys((*Status, None), 0)
    for decision in result.decisions:
        titles[decision.row] += 1
        employees[decision.row] += headcount[decision.key]
        by_status[decision.status] += headcount[decision.key]
    audits = {audit.path.value: audit for audit in result.audits}

    lines = [HEADER]
    shares = percentages(list(employees.values()))
    for table_row, share in zip(TableRow, shares, strict=True):
        audit = audits.get(table_row.value)
        if audit is not None:
            audited, wrong = str(audit.audited), str(audit.wrong)
            status = _path_status(audit, settings)
        else:
            audited = wrong = ""
            row_status = _ROW_STATUS.get(table_row)
            status = row_status.value if row_status and titles[table_row] else ""
        lines.append(
            (
                table_row.value,
                str(titles[table_row]),
                str(employees[table_row]),
                share,
                audited,
                wrong,
                status,
            )
        )
    total = (TOTAL, str(len(result.decisions)), str(result.employee_count), "100.0%", "", "", "")

    coverage = zip(
        (*(status.value for status in Status), LEGACY),
        percentages(list(by_status.values())),
        strict=True,
    )
    sent = {*result.wave1_keys, *(entry.row.key for entry in result.queue)}
    audited_titles = sum(audit.audited for audit in result.audits)
    text = [
        "Job-title standardization mock: set-up, Wave 1, Wave 2 and the close",
        "",
        *_table(lines, total),
        "",
        "Coverage by status, by employee: "
        + ", ".join(f"{name} {share}" for name, share in coverage),
        f"Sent to a person: {len(sent)} of {len(result.decisions)} titles "
        f"({_percent(len(sent), len(result.decisions))}): {len(result.wave1_keys)} on the "
        f"Wave 1 list, {len(result.queue)} in the review queue. "
        f"Blind labels besides: {audited_titles} in the audits, "
        f"{len(result.golden_keys)} on the golden set.",
        "Golden set, right by the SME's label:",
    ]
    for score in result.golden_scores:
        text.append(
            f"  {score.stage.value}: ML first choice {score.ml_right}/{score.size}, "
            f"LLM pick {score.llm_right}/{score.size} ({score.llm_no_answer} with no role named)"
        )
    text += [
        f"Exceptions list: {employees[TableRow.LEGACY]} employees left on a legacy title; "
        f"{result.no_grade_employees} employees with no grade given a title with no level word.",
        f"Model answers: {_sources(result.answer_labels)}. Embedder: {settings.embedder}.",
        "People were simulated from the truth file: every decision a person made in this "
        "run is a simulated one.",
    ]
    return "\n".join(text)


def percentages(counts: Sequence[int]) -> list[str]:
    """Each count as a share of their sum, with one decimal, so that the printed
    shares add up to exactly 100.0% (largest remainder)."""
    total = sum(counts)
    if total == 0:
        return ["0.0%" for _ in counts]
    tenths = [count * 1000 // total for count in counts]
    lost = [count * 1000 % total for count in counts]
    by_loss = sorted(range(len(counts)), key=lambda at: (-lost[at], at))
    for at in by_loss[: 1000 - sum(tenths)]:
        tenths[at] += 1
    return [f"{each // 10}.{each % 10}%" for each in tenths]


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "0.0%"


def _path_status(audit: AuditResult, settings: Settings) -> str:
    """The status cell of a machine-path row: the audit outcome (Q16)."""
    if audit.path_size == 0:
        return ""
    rate = f"error rate {_percent(audit.wrong, audit.audited)}"
    if audit.outcome is AuditOutcome.PROVISIONAL:
        return f"{audit.outcome.value}, {rate}"
    if audit.outcome is AuditOutcome.STOPPED:
        if settings.release_stopped_paths:
            return f"{audit.outcome.value}, released as provisional, {rate}"
        moved = audit.path_size - audit.audited
        return f"{audit.outcome.value}, {moved} titles left on legacy titles"
    return audit.outcome.value


def _table(lines: Sequence[Sequence[str]], total: Sequence[str]) -> list[str]:
    """The rows as text: the first and last column to the left, numbers to the right."""
    widths = [max(len(line[at]) for line in (*lines, total)) for at in range(len(HEADER))]
    rule = tuple("-" * width for width in widths)

    def laid_out(line: Sequence[str]) -> str:
        cells = [line[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(line[1:-1], widths[1:-1], strict=True)]
        return "  ".join((*cells, line[-1])).rstrip()

    header, *body = lines
    return [laid_out(line) for line in (header, rule, *body, (*rule[:-1], ""), total)]


def _sources(labels: Sequence[str]) -> str:
    """The distinct answer labels, recorded ones of one model as a range of days (Q40)."""
    shown: list[str] = []
    days: dict[str, list[str]] = {}
    for label in labels:
        date, comma, model = label.removeprefix(_RECORDED).partition(", ")
        if label.startswith(_RECORDED) and comma:
            days.setdefault(model, []).append(date)
        else:
            shown.append(label)
    for model in sorted(days):
        first, last = min(days[model]), max(days[model])
        shown.append(recorded_label(first if first == last else f"{first} to {last}", model))
    return "; ".join(shown) if shown else "none, no model was asked"
