"""llm = anthropic: the live hosted model, recording every answer.

Owner: feat/live-replay (stage 4).
Governed by: docs/design.md 2.5 (the only secret; lazy import; record each
answer; label "live"); docs/build-decisions.md Q02 (smoke run, request cap,
at most two full runs), Q15 (structured output), Q20 (one request per agent
turn), Q29 (where a recording is written), Q31 (the model string and the
request parameters left out), Q32 (how it is tested), Q39 (failures and the
credential).

The key is read only from the environment variable ANTHROPIC_API_KEY, never
from a file, and only when llm = anthropic. build_llm first checks that the
variable is set and not empty and stops with a clear message if it is not;
only then does it import the SDK (the optional extra "live") and build the
client with no arguments. A missing extra stops with a TitlestdError whose
message holds contracts.run_with_extra(EXTRA), the one command text of S09.

build_llm(..., client=...) is the seam for tests (Q32 b and c, S09): a given
client is used as it is, the key is not checked and the SDK is not imported,
so the record-then-replay tests run where the SDK is not installed.

Requests send no temperature, top_p or top_k, no thinking and no effort
parameter, and never a forced tool choice (Q31). An API or network error after
the SDK's own retries stops the run, keeps the answers already recorded and
writes no load file. A refusal, a cut-off or an answer that fails validation
becomes an INVALID answer for that title, and is recorded so replay repeats it.
More than settings.request_cap requests stops the run cleanly. The stand-in is
never substituted (Q39).

Prompts are built from the shared renderers in titlestd.fallback and
titlestd.agent, so a prompt is the same string in live and replay mode.
This module must not import titlestd.people.
"""

from __future__ import annotations

from typing import Any

from titlestd.contracts import Llm
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.live is built by feat/live-replay"

EXTRA = "live"
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"


def build_llm(settings: Settings, data_fingerprint: str, client: Any | None = None) -> Llm:
    """The live chooser and reviewer for settings.llm_model.

    Every answer carries contracts.LABEL_LIVE and is recorded. Llm.finish
    completes the recording: it sorts the entries, writes the manifest last and
    only then replaces the previous recording (Q29).

    client is None in a real run: the key is checked, the SDK imported and its
    client built with no arguments (Q39). A test passes its own: a scripted
    object with the SDK client's messages.create, or a real SDK client on a
    mock transport (Q32). Everything else, the recorder included, is the same.
    """
    raise NotImplementedError(_NOT_BUILT)
