"""Recordings: the key of a model request, the files, the manifest.

Owner: feat/live-replay (stage 4).
Governed by: docs/design.md 2.5 ("record each answer in a file keyed by a hash
of the step, model ID, prompt and request"; "Recordings are never written by
hand"), 2.8 ("No keys in recordings"); docs/build-decisions.md Q18 (the
manifest), Q29 (layout), Q33 (the key), Q40 (date and model of the label).

The key (Q33) is the SHA-256 hex of UTF-8 JSON (sorted keys, compact
separators, non-ASCII kept, strings NFC-normalised) of: key_version; step
("choose" or "review"); the model exactly as written in settings.llm_model;
the system prompt text; the tool definitions; the output schema; max_tokens
and any other request parameter the program sets; and the normalised turns
(role and text, tool name and input, tool result text). Tool-use IDs, dates,
request IDs and the SDK version are left out. One key per model turn. The
structure is built by this program before any SDK is imported.

Layout (Q29): <recording_dir>/<llm_model>/<embedder>/ holds choose.jsonl and
review.jsonl, one line per model request, sorted by key when the run ends, and
manifest.json, written last. An entry stores the key, key version, step, a
title-row hint, the configured model, the model ID the API reported (as
information only), the UTC date, the parsed answer or tool request, the stop
reason and token counts. It never stores headers or a credential.

This module must work without the SDK installed and must not import
titlestd.people.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from titlestd.settings import Settings

_NOT_BUILT = "titlestd.recording is built by feat/live-replay"

KEY_VERSION = 1
STEP_CHOOSE = "choose"
STEP_REVIEW = "review"
CHOOSE_FILE = "choose.jsonl"
REVIEW_FILE = "review.jsonl"
MANIFEST_FILE = "manifest.json"


def request_key(step: str, model: str, request: Mapping[str, Any]) -> str:
    """The key of one model request (Q33)."""
    raise NotImplementedError(_NOT_BUILT)


def recording_folder(settings: Settings, model: str) -> Path:
    """The folder of the recording for a model and the embedder in force (Q29)."""
    raise NotImplementedError(_NOT_BUILT)


def write_manifest(folder: Path, manifest: Mapping[str, Any]) -> None:
    """Write manifest.json: model, embedder, seed, data fingerprint, prompt and
    key version, UTC start and end, request count, complete flag (Q29)."""
    raise NotImplementedError(_NOT_BUILT)


def read_manifest(folder: Path) -> dict[str, Any]:
    """Read manifest.json. Raises TitlestdError when the recording is missing
    or not complete."""
    raise NotImplementedError(_NOT_BUILT)
