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
Footer: coverage by status, by employee (Approved, Confirmed, Provisional, not
released); the share of titles sent to a person, with blind labels as a
separate count; golden-set accuracy of the ML first choice and of the LLM's
pick, per stage; one line saying whether answers came from stand-ins, a
recording or a live model, naming the embedder; and that people were simulated.

The text holds no wall-clock time, so two runs print the same bytes (Q06).
"""

from __future__ import annotations

from titlestd.contracts import RunResult
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.report is built by feat/pipeline-report-cli"


def render(result: RunResult, settings: Settings) -> str:
    """The table and footer of one run, as the text the run command prints."""
    raise NotImplementedError(_NOT_BUILT)
