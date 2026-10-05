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
<recording_dir>/<model>/<embedder in force> that exists. More than one is an
error at start-up that lists the models found and names --llm-model. None is
an error at start-up too; it says which embedders the recordings under
recording_dir were made with, because naming a model cannot help where the
embedder in force has no recording. The model is then read from that
recording's manifest; if settings.llm_model is set and differs from it, that
is an error at start-up as well.

A miss stops the run at once with a non-zero exit and no output files; the
message names the step and the title and prints the differences between the
manifest and the settings in force. The manifest explains a miss; it is not a
second validity rule.

Replay never imports the SDK, never reads ANTHROPIC_API_KEY and never calls
the network. This module must not import titlestd.people.

Decided here, where the record leaves the detail open:

- Replay builds the request live mode would send, with the functions of
  titlestd.recording, and looks its key up. It never imports titlestd.live.
- A turn of the agent is looked up like an answer of the LLM fallback. A
  recorded search request is handed back to the shared loop, which runs
  the search again; what it returns is part of the next turn's key.
- The whole recording is read and checked at start-up, so a damaged file
  stops the run before any work, like a missing one.
- A manifest that differs from the settings in force (another seed, other
  data, another prompt version) stops nothing at start-up. If every request
  of the run is in the recording, the run is the recorded one.
- An entry that the run never asks for is not an error.
- The miss itself is this module's part of "a non-zero exit and no output
  files": a ReplayMiss, which is a TitlestdError, raised from choose or
  next_turn. The command line turns it into the exit status, and the
  pipeline writes its files only after the last model call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from titlestd import recording
from titlestd.contracts import (
    AgentTranscript,
    ChooseAnswer,
    Llm,
    NumberedRole,
    Record,
    ReviewAnswer,
    SearchRequest,
    TitlestdError,
    recorded_label,
)
from titlestd.settings import Settings


class ReplayMiss(TitlestdError):
    """The recording holds no answer for a request of this run (2.5)."""


class Replay:
    """The replaying chooser and reviewer: every turn is looked up by its key."""

    def __init__(
        self,
        folder: Path,
        model: str,
        entries: Mapping[str, Mapping[str, Any]],
        differences: Sequence[str],
    ) -> None:
        self._folder = folder
        self._model = model
        self._entries = entries
        self._differences = differences

    def choose(self, record: Record, roles: Sequence[NumberedRole]) -> ChooseAnswer:
        request = recording.choose_request(record, roles)
        entry = self._entry(recording.STEP_CHOOSE, request, record)
        return recording.choose_answer(entry["turn"], self._label(entry))

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        request = recording.review_request(transcript)
        entry = self._entry(recording.STEP_REVIEW, request, transcript.case.record)
        return recording.review_turn(entry["turn"], self._label(entry))

    def _label(self, entry: Mapping[str, Any]) -> str:
        return recorded_label(entry["date"], self._model)

    def _entry(self, step: str, request: Mapping[str, Any], record: Record) -> Mapping[str, Any]:
        key = recording.request_key(step, self._model, request)
        entry = self._entries.get(key)
        if entry is None:
            raise ReplayMiss(self._miss(step, record, key))
        return entry

    def _miss(self, step: str, record: Record, key: str) -> str:
        if self._differences:
            why = "Its manifest differs from the settings in force: " + "; ".join(self._differences)
        else:
            why = (
                "Its manifest matches the settings in force, so the request itself differs "
                "from the recorded one: the record, the roles shown or a search result"
            )
        return (
            f"replay miss at step {step} for the title {record.cleaned_title!r} in family "
            f"{record.family_code!r}: the recording in {self._folder} holds no answer for this "
            f"request (key {key}). {why}. Replay never falls back to another model; make a "
            "new recording with the record command, or run with the settings of this one."
        )


def _folder(settings: Settings) -> Path:
    """The folder of the recording to replay (Q31)."""
    if settings.llm_model is not None:
        return recording.recording_folder(settings, settings.llm_model)
    models = recording.recorded_models(settings)
    if len(models) == 1:
        return recording.recording_folder(settings, models[0])
    where = f"for embedder = {settings.embedder} under {settings.recording_dir}"
    if models:
        raise recording.RecordingError(
            f"llm = replay found more than one recording {where} (models found: "
            f"{', '.join(models)}). Name the model of the recording to replay with "
            "--llm-model; recordings are made with the record command."
        )
    # No folder for the embedder in force, so every recording is another embedder's.
    others = recording.recorded_embedders(settings)
    if others:
        made_with = " and ".join(f"embedder = {name}" for name in others)
        raise recording.RecordingError(
            f"llm = replay found no recording {where}. The recordings there were made with "
            f"{made_with}: replay with that setting, or make a recording for embedder = "
            f"{settings.embedder} with the record command."
        )
    raise recording.RecordingError(
        f"llm = replay found no recording {where}, nor one for another embedder. "
        "Recordings are made with the record command."
    )


def build_llm(settings: Settings, data_fingerprint: str) -> Llm:
    """The replaying chooser and reviewer.

    Every answer carries contracts.recorded_label(date, model) with the UTC
    day stored in its recording and the configured model string.
    """
    folder = _folder(settings)
    manifest = recording.read_manifest(folder)
    model = manifest["model"]
    if settings.llm_model is not None and settings.llm_model != model:
        raise recording.RecordingError(
            f"the recording in {folder} was made with the model {model!r}, says its manifest, "
            f"and llm_model is {settings.llm_model!r}. Replay takes the model from the "
            "recording: leave --llm-model out, or name that model."
        )
    entries = recording.read_entries(folder, manifest)
    current = recording.describe_run(settings, model, data_fingerprint)
    replay = Replay(folder, model, entries, recording.manifest_differences(manifest, current))
    return Llm(chooser=replay, reviewer=replay)
