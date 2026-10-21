"""The model switch: which code answers embed, choose and review.

Owner: feat/model-switch-offline (stage 1).
Governed by: DESIGN.md 2.5; docs/build-decisions.md Q30 and Q31 (no
silent fall-back), Q32 (lazy imports), Q40 (answer labels), S09 (the provider
table, and the command a provider prints when its extra is missing: the
provider raises it with contracts.run_with_extra, the switch lets it through).

build_models reads the two settings and returns the three calls. Each provider
is one function in its own module, imported here only when its setting is
chosen, so the default run imports nothing outside the standard library:

  embedder = offline    titlestd.offline_embedder.build_embedder(settings)
  embedder = local      titlestd.local_embedder.build_embedder(settings)
  llm = offline         titlestd.standin.build_llm(settings, ladder, abbreviations)
  llm = replay          titlestd.replay.build_llm(settings, data_fingerprint)
  llm = anthropic       titlestd.live.build_llm(settings, data_fingerprint)

These names are fixed (S09): a provider may change behind its function, and
this file does not change when it does.

No model module (this one, offline_embedder, standin, local_embedder, live,
replay, recording) may import titlestd.people: stand-in models never read the
truth file (2.4, Q22). A provider gets the level ladder and the abbreviations,
never the catalog, so it cannot reach a generic family role either.

Decided here, where the record is silent:
- The embedder is built before the LLM. An embedder writes nothing and
  spends nothing, so a missing extra or model stops the run before the live
  provider opens a recording or calls the hosted model.
- The switch never substitutes one provider for another (Q30, Q31, Q39):
  whatever a provider raises passes through unchanged, and a setting value
  with no provider is an error, not a reason to use the offline one.
"""

from __future__ import annotations

from collections.abc import Sequence

from titlestd.contracts import Abbreviation, Embedder, LadderLevel, Llm, Models
from titlestd.settings import Settings, SettingsError


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
    embedder = _build_embedder(settings)
    llm = _build_llm(settings, ladder, abbreviations, data_fingerprint)
    return Models(embedder=embedder, llm=llm)


def _build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "offline":
        from titlestd import offline_embedder

        return offline_embedder.build_embedder(settings)
    if settings.embedder == "local":
        from titlestd import local_embedder

        return local_embedder.build_embedder(settings)
    raise SettingsError(f"setting embedder has no provider for {settings.embedder!r}")


def _build_llm(
    settings: Settings,
    ladder: Sequence[LadderLevel],
    abbreviations: Sequence[Abbreviation],
    data_fingerprint: str,
) -> Llm:
    if settings.llm == "offline":
        from titlestd import standin

        return standin.build_llm(settings, ladder, abbreviations)
    if settings.llm == "replay":
        from titlestd import replay

        return replay.build_llm(settings, data_fingerprint)
    if settings.llm == "anthropic":
        from titlestd import live

        return live.build_llm(settings, data_fingerprint)
    raise SettingsError(f"setting llm has no provider for {settings.llm!r}")
