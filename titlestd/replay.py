"""llm = replay: recorded answers, and nothing else.

Owner: feat/live-replay (stage 4).
Governed by: docs/design.md 2.5 ("In replay mode a miss is an error; never
fall back silently"; the label "recorded <date>, <model>");
docs/build-decisions.md Q18 (the manifest explains a miss), Q20 (replay looks
each agent turn up and re-runs the searches for real), Q29 (a missing or
incomplete recording is refused before any work), Q31 (the model comes from
the manifest), Q40 (the label's date and model).

build_llm refuses a missing or incomplete recording before any work is done.
Which recording it opens (Q31): with settings.llm_model set, the folder of
that model and the embedder in force. With it unset, the one folder
<recording_dir>/<model>/<embedder in force> that exists; none, or more than
one, is an error at start-up that lists the models found and names
--llm-model. The model is then read from that recording's manifest; if
settings.llm_model is set and differs from it, that is an error at start-up
too.

A miss stops the run at once with a non-zero exit and no output files; the
message names the step and the title and prints the differences between the
manifest and the settings in force. The manifest explains a miss; it is not a
second validity rule.

Replay never imports the SDK, never reads ANTHROPIC_API_KEY and never calls
the network. This module must not import titlestd.people.
"""

from __future__ import annotations

from titlestd.contracts import Llm
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.replay is built by feat/live-replay"


def build_llm(settings: Settings, data_fingerprint: str) -> Llm:
    """The replaying chooser and reviewer.

    Every answer carries contracts.recorded_label(date, model) with the UTC
    day stored in its recording and the configured model string.
    """
    raise NotImplementedError(_NOT_BUILT)
