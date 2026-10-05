"""The mock data generator: writes the seven data files of docs/design.md 2.3.

Owner: feat/mock-data (stage 2, after prepare and catalog-naming, because the
generator must use the real cleaner and naming rule). It also owns the
committed files in data/.
Governed by: docs/design.md 2.1 (sizes), 2.3 (files and deliberate mess);
docs/build-decisions.md Q06 (own fixed seed, committed output, byte-for-byte
regeneration), Q09 (raw titles; one true role per cleaned row), Q13 (seed
titles sit in the long tail), Q23 (the difficulty knobs, and the flagged roles
in three families), Q24 (the 40th and 41st rows differ), Q26 (rows with no
usable family code), Q35 (the counts), Q36 (the grade table and the worked
example).

The generator has its own fixed seed, separate from the run seed, so changing
the seed setting never changes the data. Its difficulty knobs are named
constants at the top of this module, never settings of a run. Names, locations
and requisition patterns come from titlestd.reference; every email uses
reference.EMAIL_DOMAIN. All randomness goes through titlestd.seeding.

The run shape of Q23 cannot be measured in stage 2, where the matcher, the
rules, the agent and the pipeline do not exist yet. This branch is done when
its data tests pass (Q03, Q06, Q09, Q14, Q23 for the flagged roles, Q24, Q35)
and the knobs are exposed. feat/pipeline-report-cli measures the shape in
stage 4 and may then change those constants and regenerate data/, in at most
three rounds ("How the build is cut" in docs/build-decisions.md).

It writes six files with titlestd.datafiles and the truth file with
titlestd.people.write_truth; apart from the pipeline it is the only module
that may import titlestd.people.
"""

from __future__ import annotations

from pathlib import Path

from titlestd.settings import Settings

_NOT_BUILT = "titlestd.mockdata is built by feat/mock-data"


def generate(data_dir: Path) -> None:
    """Write the seven data files into data_dir, the same bytes every time."""
    raise NotImplementedError(_NOT_BUILT)


def generate_command(settings: Settings) -> int:
    """The generate-data command: generate(settings.data_dir). Returns the exit status."""
    raise NotImplementedError(_NOT_BUILT)
