"""The pipeline: set-up, Wave 1, Wave 2 and the close, in one run.

Owner: feat/pipeline-report-cli (stage 4).
Governed by: docs/design.md 1.6 (the flow), 1.10 (waves), 2.2 (what is in the
mock), 2.6 (what a run shows); docs/build-decisions.md Q04 (the ML-alone
route), Q06 (fresh state on every run), Q07 and Q08 (the golden set), Q13
(look-up first), Q16 (the table is a partition), Q18 (processing order; score
a whole wave, then learn), Q21 (audited titles), Q22 (Wave 1 scoring is one
batch before any approval), Q24 (the Wave 1 list), Q37 (what set-up does).
Section 2.7 tests, end to end: all eight, on the run's files and state.
Other tests this branch owns, listed with their entries in "How the build is
cut" of docs/build-decisions.md: the run shape and its tuning (Q23), the frozen
table (Q23, Q38), the invariants of Q05 and Q16, two runs with the same bytes
(Q06), a run at a high undecided rate (Q01), the default run under the import
guard (Q32 a), the 60-second limit (Q38), spies showing that no model is
called for a title with a person's name or no usable family (Q14, Q26), a seed
hit on the Wave 1 list (Q13) and the ML-alone route switched on (Q04).

Order of a run:
  set-up  read and check the data; fix the settings; prepare the title rows;
          load the seed look-up; pick the Wave 1 list (the top
          settings.wave1_size rows by headcount, then family code, then cleaned
          title); draw the golden set.
  Wave 1  look-up; the machine proposes for the Wave 1 list in one batch; an
          HR SME approves every one; the golden set is labeled and scored; the
          agent only recommends. Then the approvals are learned.
  Wave 2  the long tail is scored for release; the agent's answer stands; both
          machine paths are audited; everything else goes to the review queue.
  close   people clear the queue; leftovers take a generic family role under
          the cap; the files are written and the table printed.

Titles are processed sorted by key. This module and titlestd.mockdata are the
only ones that may import titlestd.people; a model call is handed only its
arguments (Q22).

run_pipeline(settings, models=...) is the seam for tests (S09): a test hands
in the models it wants, for example a misbehaving chooser, spying providers or
a live adapter built on a scripted client. Nothing else differs.
"""

from __future__ import annotations

from pathlib import Path

from titlestd.contracts import Models, RunResult
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.pipeline is built by feat/pipeline-report-cli"


def data_fingerprint(data_dir: Path) -> str:
    """The fingerprint of the seven data files, as a recording's manifest holds it.

    datafiles.input_fingerprint combined with people.truth_fingerprint (S09).
    The pipeline hands it to the switch; a test that builds a provider itself
    calls this, so both use the same value.
    """
    raise NotImplementedError(_NOT_BUILT)


def run_pipeline(settings: Settings, models: Models | None = None) -> RunResult:
    """Run set-up, Wave 1, Wave 2 and the close; write the state and output files.

    models is None in a real run: the pipeline builds them with
    titlestd.switch.build_models for the settings in force. A test passes its
    own, and the switch is then not called. Either way Models.llm.finish is
    called once, after everything else succeeded (Q29).
    """
    raise NotImplementedError(_NOT_BUILT)


def run_command(settings: Settings) -> int:
    """The run and record commands: run_pipeline, then print the table rendered
    by titlestd.report. Returns the exit status."""
    raise NotImplementedError(_NOT_BUILT)
