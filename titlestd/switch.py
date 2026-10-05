"""The model switch: which code answers embed, choose and review.

Owner: feat/model-switch-offline (stage 1).
Governed by: docs/design.md 2.5; docs/build-decisions.md Q30 and Q31 (no
silent fall-back, a missing extra names its install command), Q32 (lazy
imports), Q40 (answer labels).

build_models reads the two settings and returns the three calls. Each provider
is one function in its own module, imported here only when its setting is
chosen, so the default run imports nothing outside the standard library:

  embedder = offline    titlestd.offline_embedder.build_embedder(settings)
  embedder = local      titlestd.local_embedder.build_embedder(settings)
  llm = offline         titlestd.standin.build_llm(settings, ladder, abbreviations)
  llm = replay          titlestd.replay.build_llm(settings, data_fingerprint)
  llm = anthropic       titlestd.live.build_llm(settings, data_fingerprint)

These names are fixed: the later branches fill the provider modules in, and
this file does not change when they do.

No model module (this one, offline_embedder, standin, local_embedder, live,
replay, recording) may import titlestd.people: stand-in models never read the
truth file (2.4, Q22). A provider gets the level ladder and the abbreviations,
never the catalog, so it cannot reach a generic family role either.
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import Abbreviation, LadderLevel, Models
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.switch is built by feat/model-switch-offline"


def build_models(
    settings: Settings,
    ladder: Sequence[LadderLevel],
    abbreviations: Sequence[Abbreviation],
    data_fingerprint: str = "",
) -> Models:
    """The embedder and the LLM for the settings in force.

    ladder and abbreviations serve the stand-ins (level words and abbreviation
    expansion, Q17). data_fingerprint identifies the data files of the run; the
    replay and live providers put it in, or compare it with, a recording's
    manifest (Q18, Q29).
    """
    raise NotImplementedError(_NOT_BUILT)
