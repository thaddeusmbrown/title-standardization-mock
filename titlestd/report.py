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
"""

from __future__ import annotations

from titlestd.contracts import RunResult
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.report is built by feat/pipeline-report-cli"


def render(result: RunResult, settings: Settings) -> str:
    """The table and footer of one run, as the text the run command prints."""
    raise NotImplementedError(_NOT_BUILT)
