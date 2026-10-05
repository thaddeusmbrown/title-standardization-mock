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

Decided here, where the record leaves the detail open:

- This module holds everything live mode and replay share, the requests
  included: the two system prompts, the two tools, the two output schemas
  and the functions that build a request from a record or a transcript.
  Replay must build the request live mode built to find its key, and the
  switch imports a provider only when its setting is chosen (S09), so
  neither provider imports the other.
- A request is this program's own structure, a mapping of "system",
  "max_tokens", "tools", "tool_choice", "output_schema" and "turns". A turn
  is {"role", "text"}, {"role", "tool", "input"} or {"role", "tool_result"}.
  Its key hashes that mapping beside key_version, step and model.
  titlestd.live turns the same mapping into the SDK's arguments, so what is
  keyed is what is sent.
- The two system prompts are written out in full as plain text, and every
  request asks for at most MAX_TOKENS (1024) tokens: an answer is a number
  and a quote of at most 25 words.
- The prompt of choose is the record block of fallback.render_record, an
  empty line, "Roles:" and the list of fallback.render_roles. The agent's
  case, which agent.render_case writes, opens with the same lines.
- An agent turn is rebuilt from the transcript alone: the case, then per
  search the request and the text of agent.render_step. Words the model
  wrote beside a tool request are not kept, so a turn's key depends on
  nothing but what shared code holds.
- The choose schema lists the numbers 1 to contracts.SHORTLIST_SIZE as an
  enum, the review schema takes any integer (Q15, Q20). Both allow null.
- PROMPT_VERSION is raised when a prompt, a tool or a schema is reworded.
  It is told in the manifest to explain a miss; the key is what decides.
- A stored turn mirrors the contract types: {"answer", "number", "quote",
  "detail"} for an answer, {"tool", "query"} for a search request.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from titlestd import agent, fallback
from titlestd.contracts import (
    SHORTLIST_SIZE,
    AgentTranscript,
    ChooseAnswer,
    ChooseKind,
    NumberedRole,
    Record,
    ReviewAnswer,
    ReviewKind,
    SearchRequest,
    SearchTool,
)
from titlestd.settings import Settings

KEY_VERSION = 1
STEP_CHOOSE = "choose"
STEP_REVIEW = "review"
CHOOSE_FILE = "choose.jsonl"
REVIEW_FILE = "review.jsonl"
MANIFEST_FILE = "manifest.json"

STEP_FILES = {STEP_CHOOSE: CHOOSE_FILE, STEP_REVIEW: REVIEW_FILE}


# --- The requests (Q15, Q20, Q31, S16) ----------------------------------------

# Raise when a prompt, a tool or a schema below is reworded.
PROMPT_VERSION = 1
MAX_TOKENS = 1024

# The prompts are written out as a model reads them. What they repeat of shared
# code (the masks, the limits of a quote, the search cap, the tool names, the
# marks and the reasons of a case) is held to it by a test of
# tests/unit/test_recording_requests.py.
CHOOSE_SYSTEM = """\
You map one job title of a large employer onto a standard role from a catalog that HR approved.

The user message holds:
- the record of the job title: the cleaned title, the code of the job family the employer
  recorded for it, the department and the job description. Names and email addresses are masked
  as [NAME] and [EMAIL]. A field may be empty.
- a numbered list of roles, each with its name, the code of its job family and a one-line
  definition.

Decide which listed role is the same job as the record describes. The level of a job (junior,
senior, lead, a numeral such as II) is set elsewhere from the pay grade: ignore it.

Answer with one JSON object:
- "answer": "role" when one listed role fits the record; "no_match" when the record makes clear
  what the job is and no listed role fits; "insufficient_data" when the record says too little
  to tell what the job is.
- "number": the number of the role you chose, exactly as listed; null unless "answer" is "role".
- "quote": the words of the record that your answer rests on, copied character for character
  from the title or the description: whole words, at least 3 characters and at most 25 words.
  A quote is required with every answer; when you choose no role, quote the title.

You can only choose a role by its number. Never write a role name or a title of your own."""

REVIEW_SYSTEM = """\
You review one job title that an earlier automatic step could not settle. Either map it onto a
standard role from a catalog that HR approved, or pass it to a person.

The first user message holds:
- the record of the job title: the cleaned title, the code of the job family the employer
  recorded for it, the department and the job description. Names and email addresses are masked
  as [NAME] and [EMAIL]. A field may be empty.
- the roles shortlisted for it, each with its name, the code of its job family and a one-line
  definition. "[the LLM's pick]" marks the role an earlier model chose, and "[the ML model's
  first choice]" the role a similarity match ranked first.
- why the title was referred to you: "models disagree" (the two earlier choices differ), "role
  outside recorded family" (the earlier model chose a role of another family) or "no match" (the
  earlier model found no listed role that fits).

You may only pick a role of the record's own job family. Such a role is listed with a number; a
role of another family is listed with "-" and cannot be picked.

Two searches are available. You may run at most 4 searches in all, one at a time:
- search_catalog finds roles in the whole catalog that are similar to a query.
- search_approved_titles finds job titles that a person has already mapped to a role.
A search can show roles that were not on the shortlist. A new role of the record's own family
gets the next free number, and you may pick it. Search when the shortlist does not settle the
case; you may also answer at once.

When you have decided, call no tool and answer with one JSON object:
- "answer": "role" when one numbered role is the same job as the record describes; "escalate"
  when none is, or the record says too little, so that a person decides.
- "number": the number of the role you picked; null when you escalate.
- "quote": with "role", the words of the record that your pick rests on, copied character for
  character from the title or the description: whole words, at least 3 characters and at most
  25 words. With "escalate", an empty string.

The level of a job (junior, senior, lead, a numeral such as II) is set elsewhere from the pay
grade: ignore it. Never write a role name or a title of your own."""


def _answer_schema(kinds: Sequence[str], number: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": list(kinds)},
            "number": {"anyOf": [dict(number), {"type": "null"}]},
            "quote": {"type": "string"},
        },
        "required": ["answer", "number", "quote"],
        "additionalProperties": False,
    }


# What a model may answer: every kind but INVALID, which only code gives (S10).
CHOOSE_KINDS = (
    ChooseKind.ROLE.value,
    ChooseKind.NO_MATCH.value,
    ChooseKind.INSUFFICIENT_DATA.value,
)
REVIEW_KINDS = (ReviewKind.ROLE.value, ReviewKind.ESCALATE.value)
# Hand-written, not derived from a library (Q33). The schemas the hosted model
# takes know no minimum or maximum, so 1 to 10 is an enum (Q15) and the agent's
# number a plain integer whose range shared code checks (Q20).
CHOOSE_SCHEMA = _answer_schema(
    CHOOSE_KINDS, {"type": "integer", "enum": list(range(1, SHORTLIST_SIZE + 1))}
)
REVIEW_SCHEMA = _answer_schema(REVIEW_KINDS, {"type": "integer"})


def _search_tool(name: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Words to search for, for example the job title or a phrase of the "
                        "job description."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


SEARCH_TOOLS = (
    _search_tool(
        SearchTool.CATALOG.value,
        "Search the whole catalog of standard roles, in every job family, for the roles most "
        "similar to a free-text query. Each result is a role: its name, the code of its job "
        "family and its definition. A role of the record's own family is listed with the number "
        'to pick it by; a role of another family is listed with "-" and cannot be picked.',
    ),
    _search_tool(
        SearchTool.APPROVED_TITLES.value,
        "Search the job titles that a person has already mapped to a role for the titles most "
        "similar to a free-text query. Each result is a title, the code of its job family and "
        "the role it was approved for. That role is listed with the number to pick it by when "
        'it is of the record\'s own family, and with "-" when it is not.',
    ),
)
# One search a turn at most; none once the searches are used up (Q20). Never a
# forced tool choice (Q31).
ONE_SEARCH_A_TURN = {"type": "auto", "disable_parallel_tool_use": True}
NO_MORE_SEARCHES = {"type": "none"}


def choose_prompt(record: Record, roles: Sequence[NumberedRole]) -> str:
    """What the LLM fallback is shown: the record block and the numbered roles (S16)."""
    return "\n".join([fallback.render_record(record), "", "Roles:", fallback.render_roles(roles)])


def choose_request(record: Record, roles: Sequence[NumberedRole]) -> dict[str, Any]:
    """The request of one choose call: one user turn, a structured answer, no tool."""
    return {
        "system": CHOOSE_SYSTEM,
        "max_tokens": MAX_TOKENS,
        "tools": [],
        "tool_choice": None,
        "output_schema": CHOOSE_SCHEMA,
        "turns": [{"role": "user", "text": choose_prompt(record, roles)}],
    }


def review_request(transcript: AgentTranscript) -> dict[str, Any]:
    """The request of one agent turn: the case, then every search so far with
    its result (Q20). After the last search a case may run, the tools are
    withdrawn and the turn can only be the final answer."""
    turns: list[dict[str, Any]] = [{"role": "user", "text": agent.render_case(transcript.case)}]
    for step in transcript.steps:
        tool = SearchTool(step.request.tool).value
        turns.append({"role": "assistant", "tool": tool, "input": {"query": step.request.query}})
        turns.append({"role": "user", "tool_result": agent.render_step(step)})
    return {
        "system": REVIEW_SYSTEM,
        "max_tokens": MAX_TOKENS,
        "tools": list(SEARCH_TOOLS),
        "tool_choice": ONE_SEARCH_A_TURN if transcript.searches_left else NO_MORE_SEARCHES,
        "output_schema": REVIEW_SCHEMA,
        "turns": turns,
    }


# --- The key (Q33) ------------------------------------------------------------


def request_key(step: str, model: str, request: Mapping[str, Any]) -> str:
    """The key of one model request (Q33)."""
    if step not in STEP_FILES:
        raise ValueError(f"a step is {STEP_CHOOSE!r} or {STEP_REVIEW!r}, not {step!r}")
    keyed = {"key_version": KEY_VERSION, "step": step, "model": model}
    if keyed.keys() & request.keys():
        raise ValueError(f"a request cannot hold {sorted(keyed.keys() & request.keys())}")
    text = json.dumps(
        _nfc({**keyed, **request}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nfc(value: Any) -> Any:
    """The value with every string in it, a mapping's names included, NFC-normalised."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {_nfc(name): _nfc(each) for name, each in value.items()}
    if isinstance(value, list | tuple):
        return [_nfc(each) for each in value]
    return value


# --- A stored turn: the parsed answer or the tool request (Q29) -----------------


def answer_turn(answer: str, number: int | None, quote: str, detail: str = "") -> dict[str, Any]:
    return {"answer": answer, "number": number, "quote": quote, "detail": detail}


def invalid_turn(detail: str) -> dict[str, Any]:
    """No usable answer, with the reason (S10). "invalid" is a kind of both steps."""
    return answer_turn(ChooseKind.INVALID.value, None, "", detail)


def search_turn(tool: str, query: str) -> dict[str, Any]:
    return {"tool": tool, "query": query}


def choose_answer(turn: Mapping[str, Any], source: str) -> ChooseAnswer:
    """A stored choose turn as the answer it is, under the label given (Q40)."""
    return ChooseAnswer(
        ChooseKind(turn["answer"]), turn["number"], turn["quote"], source, turn["detail"]
    )


def review_turn(turn: Mapping[str, Any], source: str) -> SearchRequest | ReviewAnswer:
    """A stored agent turn: one search request, or the final answer under the label given."""
    if "tool" in turn:
        return SearchRequest(SearchTool(turn["tool"]), turn["query"])
    return ReviewAnswer(
        ReviewKind(turn["answer"]), turn["number"], turn["quote"], source, turn["detail"]
    )


# --- The files (Q29) ----------------------------------------------------------

_NOT_BUILT = "titlestd.recording is built by feat/live-replay"


def recording_folder(settings: Settings, model: str) -> Path:
    """The folder of the recording for a model and the embedder in force (Q29):
    settings.recording_dir / model / settings.embedder.

    Live mode passes settings.llm_model. Replay passes it too when it is set;
    when it is not, replay first finds the model as Q31 says (see
    titlestd.replay).
    """
    raise NotImplementedError(_NOT_BUILT)


def write_manifest(folder: Path, manifest: Mapping[str, Any]) -> None:
    """Write manifest.json: model, embedder, seed, data fingerprint, prompt and
    key version, UTC start and end, request count, complete flag (Q29)."""
    raise NotImplementedError(_NOT_BUILT)


def read_manifest(folder: Path) -> dict[str, Any]:
    """Read manifest.json. Raises TitlestdError when the recording is missing
    or not complete."""
    raise NotImplementedError(_NOT_BUILT)
