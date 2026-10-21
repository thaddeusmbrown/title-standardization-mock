"""llm = anthropic: the live hosted model, recording every answer.

Owner: feat/live-replay (stage 4).
Governed by: DESIGN.md 2.5 (the only secret; lazy import; record each
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

Decided here, where the record leaves the detail open:

- The requests themselves (prompts, tools, schemas, turns) are built in
  titlestd.recording, which replay shares. This module turns one into the
  arguments of client.messages.create, sends it and reads the response.
- What is sent: model, max_tokens, system, messages and output_config.format
  (json_schema) on every request; on an agent turn also the two strict tools
  and tool_choice, auto with disable_parallel_tool_use, or none once the
  four searches are used. Nothing else.
- A search already run goes back as a tool_use block and its tool_result.
  Their ids are made here ("search_1", "search_2", ...), because a turn is
  rebuilt from the transcript and the ids the API gave are not kept (Q33).
- One answer per key in a run. A request whose key this run has already
  recorded is not sent again: the recorded turn is returned. So a recording
  holds one line per request made, replay is a plain look-up, and an
  identical prompt cannot get two answers. The request cap counts requests
  sent.
- No request is made at start-up. A rejected key or an unknown model stops
  the run on the first request, and nothing has been written by then: the
  recorder makes its folder when the first answer is recorded.
- Which failures stop the run: whatever client.messages.create raises. The
  SDK is not imported when a client is handed in, so its error classes are
  not named here; the message reads the status_code the SDK's status errors
  carry (401 the key, 404 the model) and takes an error without one for a
  request that got no answer. The error's own type and words are repeated.
- What is read of a response: stop_reason, content (blocks with type, and
  text or name and input), model, usage.input_tokens, usage.output_tokens
  and, on a refusal, stop_details.category when there is one. A scripted
  client answers with objects that have these.
- What fails validation (S10): a stop reason other than end_turn, or
  tool_use on an agent turn; no text, text that is no JSON object, fields
  other than answer, number and quote; an answer that is not of the step's
  kinds; a quote that is no string; a number that is no whole number;
  "role" with no number; a tool request that is not one search with a text
  query, or more than one in a turn. A number beside any other answer than
  "role" is dropped. The range of a number and the quote itself are not
  judged here: shared code maps the number and checks the quote (Q15).
- A quote or a search query that is not valid Unicode fails validation too.
  A JSON answer can carry a lone surrogate as an escape, and UTF-8 cannot
  encode one: the recorder could not write the answer, nor could the key of
  the next agent turn be made. Such a turn is invalid like any other, so it
  is recorded and the title goes to a person, where the run used to stop
  with a UnicodeEncodeError (found and fixed by feat/recordings-replay-e2e,
  rule 9 of the build).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from titlestd import recording
from titlestd.contracts import (
    LABEL_LIVE,
    AgentTranscript,
    ChooseAnswer,
    ChooseKind,
    Llm,
    NumberedRole,
    Record,
    ReviewAnswer,
    SearchRequest,
    SearchTool,
    TitlestdError,
    run_with_extra,
)
from titlestd.settings import EXAMPLE_LLM_MODEL, Settings, SettingsError

EXTRA = "live"
API_KEY_VARIABLE = "ANTHROPIC_API_KEY"

_ANSWER_FIELDS = {"answer", "number", "quote"}


class LiveError(TitlestdError):
    """Live mode cannot start, or the run must stop: no key, a request that
    failed, the request cap."""


def sdk_arguments(model: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """The keyword arguments of client.messages.create for one request of
    titlestd.recording (Q20, Q31)."""
    arguments: dict[str, Any] = {
        "model": model,
        "max_tokens": request["max_tokens"],
        "system": request["system"],
        "messages": _messages(request["turns"]),
        "output_config": {"format": {"type": "json_schema", "schema": request["output_schema"]}},
    }
    if request["tools"]:
        arguments["tools"] = request["tools"]
        arguments["tool_choice"] = request["tool_choice"]
    return arguments


def _messages(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    searches = 0
    for turn in turns:
        content: Any
        if "text" in turn:
            content = turn["text"]
        elif "tool" in turn:
            searches += 1
            content = [
                {
                    "type": "tool_use",
                    "id": f"search_{searches}",
                    "name": turn["tool"],
                    "input": turn["input"],
                }
            ]
        else:
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": f"search_{searches}",
                    "content": turn["tool_result"],
                }
            ]
        messages.append({"role": turn["role"], "content": content})
    return messages


# --- Reading a response (Q20, Q39, S10) ---------------------------------------


def read_choose(response: Any) -> dict[str, Any]:
    """The stored turn of a choose response: the answer, or why there is none."""
    problem = _no_answer(response)
    if not problem and response.stop_reason == "tool_use":
        problem = "the model asked for a tool, and the LLM fallback has none"
    return recording.invalid_turn(problem) if problem else _answer(response, recording.CHOOSE_KINDS)


def read_review(response: Any) -> dict[str, Any]:
    """The stored turn of an agent response: one search request, the final
    answer, or why there is neither."""
    problem = _no_answer(response)
    if problem:
        return recording.invalid_turn(problem)
    if response.stop_reason == "tool_use":
        return _search(response)
    return _answer(response, recording.REVIEW_KINDS)


def _no_answer(response: Any) -> str:
    """Why the response holds nothing to read; empty when the model ended its turn."""
    stop = response.stop_reason
    if stop == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        return "the model refused to answer" + (f" ({category})" if category else "")
    if stop == "max_tokens":
        return f"the answer was cut off at max_tokens = {recording.MAX_TOKENS}"
    if stop not in ("end_turn", "tool_use"):
        return f"the model stopped for the reason {stop!r}"
    return ""


def _search(response: Any) -> dict[str, Any]:
    calls = [block for block in response.content if block.type == "tool_use"]
    if len(calls) != 1:
        return recording.invalid_turn(
            f"the turn holds {len(calls)} tool requests; a turn is one search or the final answer"
        )
    name, given = calls[0].name, calls[0].input
    if name not in tuple(SearchTool) or not isinstance(given, Mapping):
        return recording.invalid_turn(f"the model asked for {name!r}, which is no search")
    if set(given) != {"query"} or not isinstance(given["query"], str):
        return recording.invalid_turn(f"a search takes one text query, not {dict(given)!r}")
    if not _is_unicode(given["query"]):
        return recording.invalid_turn("the query of a search is not valid Unicode text")
    return recording.search_turn(name, given["query"])


def _answer(response: Any, kinds: Sequence[str]) -> dict[str, Any]:
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        return recording.invalid_turn("the answer holds no text")
    try:
        fields = json.loads(text)
    except ValueError:
        return recording.invalid_turn("the answer is not JSON")
    if not isinstance(fields, dict) or set(fields) != _ANSWER_FIELDS:
        return recording.invalid_turn("the answer is not an object of answer, number and quote")
    answer, number, quote = fields["answer"], fields["number"], fields["quote"]
    if answer not in kinds:
        return recording.invalid_turn(f"the answer {answer!r} is not one of {', '.join(kinds)}")
    if not isinstance(quote, str):
        return recording.invalid_turn("the quote is not text")
    if not _is_unicode(quote):
        return recording.invalid_turn("the quote is not valid Unicode text")
    if number is not None and (isinstance(number, bool) or not isinstance(number, int)):
        return recording.invalid_turn(f"the number {number!r} is not a whole number")
    if answer != ChooseKind.ROLE.value:  # the same word in both steps
        number = None
    elif number is None:
        return recording.invalid_turn('the answer "role" names no number')
    return recording.answer_turn(answer, number, quote)


def _is_unicode(text: str) -> bool:
    """False for text that UTF-8 cannot encode: it holds a lone surrogate,
    which a JSON answer can carry as an escape. No recording could hold it."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


# --- The provider -------------------------------------------------------------


class LiveLlm:
    """The live chooser and reviewer: one request per new key, every answer recorded."""

    def __init__(self, client: Any, settings: Settings, recorder: recording.Recorder) -> None:
        self._client = client
        self._model = settings.llm_model
        self._cap = settings.request_cap
        self._recorder = recorder

    def choose(self, record: Record, roles: Sequence[NumberedRole]) -> ChooseAnswer:
        request = recording.choose_request(record, roles)
        turn = self._turn(recording.STEP_CHOOSE, request, record)
        return recording.choose_answer(turn, LABEL_LIVE)

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        request = recording.review_request(transcript)
        turn = self._turn(recording.STEP_REVIEW, request, transcript.case.record)
        return recording.review_turn(turn, LABEL_LIVE)

    def _turn(self, step: str, request: Mapping[str, Any], record: Record) -> Mapping[str, Any]:
        key = recording.request_key(step, self._model, request)
        turn = self._recorder.turn(key)
        if turn is not None:
            return turn
        response = self._send(sdk_arguments(self._model, request))
        read = read_choose if step == recording.STEP_CHOOSE else read_review
        turn = read(response)
        self._recorder.record(
            key,
            step,
            record,
            turn,
            reported_model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return turn

    def _send(self, arguments: Mapping[str, Any]) -> Any:
        if self._recorder.requests >= self._cap:
            raise self._stop(
                f"the request cap is reached: this run has made {self._cap} model requests "
                "(setting request_cap)"
            )
        try:
            return self._client.messages.create(**arguments)
        except Exception as error:
            failure = f"{self._failure(error)}: {type(error).__name__}: {error}"
            raise self._stop(failure) from error

    def _failure(self, error: Exception) -> str:
        status = getattr(error, "status_code", None)
        if status == 401:
            return f"the hosted API rejected the key in {API_KEY_VARIABLE} (status 401)"
        if status == 404:
            return (
                f"the hosted API does not know the model {self._model!r}, or the key may not "
                "use it (status 404)"
            )
        if status is None:
            return "a request to the hosted API got no answer, after the SDK's own retries"
        retried = (
            ", after the SDK's own retries" if status in (408, 409, 429) or status >= 500 else ""
        )
        return f"the hosted API answered a request with status {status}{retried}"

    def _stop(self, why: str) -> LiveError:
        made = self._recorder.requests
        kept = (
            f"the answers recorded so far ({made}) are kept in {self._recorder.partial_folder}"
            if made
            else "no answer was recorded and nothing was written"
        )
        return LiveError(f"{why}. The run stops and nothing stands in for the model; {kept}.")


def _sdk_client() -> Any:
    """The SDK's client, built with no arguments, after the key was checked (Q39)."""
    if not os.environ.get(API_KEY_VARIABLE, "").strip():
        raise LiveError(
            f"llm = anthropic needs the environment variable {API_KEY_VARIABLE}, which is not "
            "set or is empty. Export it in the shell, or keep it in a file outside version "
            "control and pass that file with uv run --env-file; the program reads the key "
            "from nowhere else."
        )
    try:
        import anthropic
    except ImportError as error:
        raise TitlestdError(
            f"llm = anthropic needs the anthropic library, which cannot be imported ({error}). "
            f'It comes with the optional extra "{EXTRA}"; run the program with that extra, '
            f"for example: {run_with_extra(EXTRA)} record --llm-model {EXAMPLE_LLM_MODEL}"
        ) from error
    return anthropic.Anthropic()


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
    if settings.llm_model is None:
        raise SettingsError(
            "llm = anthropic needs llm_model, which has no default; "
            f"name the model, for example --llm-model {EXAMPLE_LLM_MODEL}"
        )
    recorder = recording.Recorder(settings, settings.llm_model, data_fingerprint)
    if client is None:
        client = _sdk_client()
    live = LiveLlm(client, settings, recorder)
    return Llm(chooser=live, reviewer=live, finish=recorder.finish)
