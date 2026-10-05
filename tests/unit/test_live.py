"""llm = anthropic, with a scripted client in place of the SDK's (Q32 b, Q39, S09).

No test here needs the SDK, the network or a key. The client handed to
build_llm is a scripted object with the SDK client's messages.create; it
answers with objects that have what the adapter reads of a response. Where
the adapter builds the SDK's client itself, a stub of the anthropic module
stands where the lazy import looks, in sys.modules, or blocks the import
there. The real SDK on a mock transport has its tests in
tests/integration/test_live_replay.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from titlestd import agent, live, recording
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    LABEL_LIVE,
    AgentCase,
    AgentTranscript,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    TitlestdError,
    run_with_extra,
)
from titlestd.live import API_KEY_VARIABLE, LiveError
from titlestd.recording import CHOOSE_FILE, MANIFEST_FILE, REVIEW_FILE
from titlestd.settings import Settings, SettingsError

MODEL = "test-double"
REPORTED = "test-double-0001"
FINGERPRINT = "0f" * 32
# The shape of a key, in two pieces: no tracked file holds a key-shaped string (Q02).
KEY_PREFIX = "sk-" + "ant-"
NOT_A_KEY = KEY_PREFIX + "not-a-real-key-000"

RECORD = Record(
    "senior account manager",
    "SAL",
    "Enterprise Sales",
    "Owns the largest customer accounts and negotiates renewals.",
)
OTHER = Record("treasury analyst", "FIN", "Treasury", "Manages cash positions.")
ACCOUNT_MANAGER = CatalogRole(
    "SAL-01", "SAL", "Account Manager", "Acct Mgr", "Manages a portfolio of customer accounts."
)
TREASURY_ANALYST = CatalogRole(
    "FIN-06", "FIN", "Treasury Analyst", "Treasury Analyst", "Manages cash positions."
)
ROLES = (NumberedRole(1, TREASURY_ANALYST), NumberedRole(2, ACCOUNT_MANAGER))
CASE = AgentCase(
    record=RECORD,
    roles=(NumberedRole(None, TREASURY_ANALYST), NumberedRole(1, ACCOUNT_MANAGER)),
    llm_answer=ChooseAnswer(ChooseKind.ROLE, 1, "account manager", LABEL_LIVE),
    llm_role_id="FIN-06",
    ml_first_choice="SAL-01",
    reasons=(Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH),
)
STEP = SearchStep(
    SearchRequest(SearchTool.CATALOG, "account manager"),
    (SearchHit(NumberedRole(1, ACCOUNT_MANAGER)), SearchHit(NumberedRole(None, TREASURY_ANALYST))),
)


def transcript(searches: int = 0) -> AgentTranscript:
    return AgentTranscript(CASE, (STEP,) * searches)


# --- What a scripted client answers with -------------------------------------------


def response(*blocks: SimpleNamespace, stop: str | None = "end_turn", **more) -> SimpleNamespace:
    """A response with what the adapter reads of one (see titlestd.live)."""
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        model=REPORTED,
        usage=SimpleNamespace(input_tokens=300, output_tokens=20),
        **more,
    )


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, given: object, block_id: str = "toolu_01") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=given)


def answer(kind: str, number: int | None, quote: str) -> SimpleNamespace:
    return response(text(json.dumps({"answer": kind, "number": number, "quote": quote})))


def search(name: str, query: str) -> SimpleNamespace:
    return response(tool_use(name, {"query": query}), stop="tool_use")


class ScriptedClient:
    """The SDK client's messages.create, answered from a list, in order. Something
    that is an exception is raised in place of an answer."""

    def __init__(self, *script: object) -> None:
        self.messages = self
        self.script = list(script)
        self.requests: list[dict] = []

    def create(self, **arguments) -> SimpleNamespace:
        self.requests.append(arguments)
        turn = self.script.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


class StatusError(Exception):
    """An error with the status code of a response, as the SDK's status errors have."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(llm="anthropic", llm_model=MODEL, recording_dir=tmp_path / "recordings")


@pytest.fixture(autouse=True)
def no_key_and_no_sdk(monkeypatch):
    """Whatever the machine has: no key, and an SDK that cannot be imported. A
    test that passes its own client must not notice either."""
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    monkeypatch.setitem(sys.modules, "anthropic", None)


def partial(settings: Settings) -> Path:
    return settings.recording_dir / MODEL / "offline.partial"


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- The arguments of a request (Q20, Q31) -------------------------------------------


def test_a_choose_request_is_sent_with_these_arguments_and_no_other():
    arguments = live.sdk_arguments(MODEL, recording.choose_request(RECORD, ROLES))
    assert arguments == {
        "model": MODEL,
        "max_tokens": recording.MAX_TOKENS,
        "system": recording.CHOOSE_SYSTEM,
        "messages": [{"role": "user", "content": recording.choose_prompt(RECORD, ROLES)}],
        "output_config": {"format": {"type": "json_schema", "schema": recording.CHOOSE_SCHEMA}},
    }


def test_an_agent_turn_is_sent_with_the_tools_and_a_tool_choice_that_forces_nothing():
    arguments = live.sdk_arguments(MODEL, recording.review_request(transcript()))
    assert arguments == {
        "model": MODEL,
        "max_tokens": recording.MAX_TOKENS,
        "system": recording.REVIEW_SYSTEM,
        "messages": [{"role": "user", "content": agent.render_case(CASE)}],
        "output_config": {"format": {"type": "json_schema", "schema": recording.REVIEW_SCHEMA}},
        "tools": list(recording.SEARCH_TOOLS),
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
    }


@pytest.mark.parametrize("searches", range(AGENT_MAX_SEARCHES + 1))
def test_no_request_holds_what_q31_leaves_out(searches: int):
    for request in (
        recording.choose_request(RECORD, ROLES),
        recording.review_request(transcript(searches)),
    ):
        arguments = live.sdk_arguments("claude-haiku-4-5", request)
        assert arguments["model"] == "claude-haiku-4-5"  # the alias as written, no date
        assert set(arguments) <= {
            "model",
            "max_tokens",
            "system",
            "messages",
            "output_config",
            "tools",
            "tool_choice",
        }
        assert set(arguments["output_config"]) == {"format"}  # no effort
        assert arguments.get("tool_choice", {}).get("type") in (None, "auto", "none")


def test_a_search_that_ran_goes_back_as_a_tool_request_and_its_result():
    arguments = live.sdk_arguments(MODEL, recording.review_request(transcript(2)))
    assert arguments["messages"] == [
        {"role": "user", "content": agent.render_case(CASE)},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "search_1",
                    "name": "search_catalog",
                    "input": {"query": "account manager"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "search_1",
                    "content": agent.render_step(STEP),
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "search_2",
                    "name": "search_catalog",
                    "input": {"query": "account manager"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "search_2",
                    "content": agent.render_step(STEP),
                }
            ],
        },
    ]


def test_after_the_last_search_the_tools_are_withdrawn_not_forced():
    arguments = live.sdk_arguments(MODEL, recording.review_request(transcript(AGENT_MAX_SEARCHES)))
    assert arguments["tool_choice"] == {"type": "none"}
    assert len(arguments["messages"]) == 1 + 2 * AGENT_MAX_SEARCHES
    assert arguments["tools"] == list(recording.SEARCH_TOOLS)


# --- Reading a choose response (Q15, Q39, S10) ----------------------------------------


@pytest.mark.parametrize(
    ("given", "turn"),
    [
        (("role", 2, "account manager"), ("role", 2, "account manager")),
        (
            ("no_match", None, "senior account manager"),
            ("no_match", None, "senior account manager"),
        ),
        (("insufficient_data", None, "bdr"), ("insufficient_data", None, "bdr")),
        (("role", 2, ""), ("role", 2, "")),  # the shared quote check fails it, not this
        # A number beside another answer than "role" is dropped.
        (("no_match", 3, "bdr"), ("no_match", None, "bdr")),
        (("insufficient_data", 3, "bdr"), ("insufficient_data", None, "bdr")),
    ],
)
def test_an_answer_of_the_schema_is_read_as_it_is(given, turn):
    assert live.read_choose(answer(*given)) == recording.answer_turn(*turn)


@pytest.mark.parametrize("number", [0, 11, -3, 10**6])
def test_a_number_out_of_range_is_left_to_the_catalog_check(number: int):
    # Q15: an out-of-range number fails the catalog check and goes to a person.
    assert live.read_choose(answer("role", number, "account manager")) == recording.answer_turn(
        "role", number, "account manager"
    )


INVALID_ANSWERS = [
    (response(text("{}"), stop="refusal"), "the model refused to answer"),
    (
        response(text(""), stop="refusal", stop_details=SimpleNamespace(category="cyber")),
        "the model refused to answer (cyber)",
    ),
    (
        response(text(""), stop="refusal", stop_details=SimpleNamespace(category=None)),
        "the model refused to answer",
    ),
    (response(text(""), stop="refusal", stop_details=None), "the model refused to answer"),
    (
        response(text('{"answer": "role", "number": 2, "quo'), stop="max_tokens"),
        "the answer was cut off at max_tokens = 1024",
    ),
    (response(stop="pause_turn"), "the model stopped for the reason 'pause_turn'"),
    (
        response(text("{}"), stop="stop_sequence"),
        "the model stopped for the reason 'stop_sequence'",
    ),
    (response(text("{}"), stop=None), "the model stopped for the reason None"),
    (response(), "the answer holds no text"),
    (response(text("The role is number 2.")), "the answer is not JSON"),
    (response(text("")), "the answer is not JSON"),
    (response(text("[1, 2]")), "the answer is not an object of answer, number and quote"),
    (response(text('"role"')), "the answer is not an object of answer, number and quote"),
    (
        response(text('{"answer": "role", "number": 2}')),
        "the answer is not an object of answer, number and quote",
    ),
    (
        response(text('{"answer": "role", "number": 2, "quote": "q", "role_id": "SAL-01"}')),
        "the answer is not an object of answer, number and quote",
    ),
    (answer("Account Manager", None, "q"), "the answer 'Account Manager' is not one of"),
    (answer("invalid", None, "q"), "the answer 'invalid' is not one of"),
    (answer("role", None, "account manager"), 'the answer "role" names no number'),
    (
        response(text('{"answer": "role", "number": "2", "quote": "q"}')),
        "the number '2' is not a whole number",
    ),
    (
        response(text('{"answer": "role", "number": 2.0, "quote": "q"}')),
        "the number 2.0 is not a whole number",
    ),
    (
        response(text('{"answer": "role", "number": true, "quote": "q"}')),
        "the number True is not a whole number",
    ),
    (response(text('{"answer": "role", "number": 2, "quote": null}')), "the quote is not text"),
    (response(text('{"answer": "role", "number": 2, "quote": ["q"]}')), "the quote is not text"),
]


@pytest.mark.parametrize(("given", "detail"), INVALID_ANSWERS)
def test_a_choose_response_with_no_usable_answer_is_invalid_with_the_reason(given, detail: str):
    turn = live.read_choose(given)
    assert (turn["answer"], turn["number"], turn["quote"]) == ("invalid", None, "")
    assert turn["detail"].startswith(detail)
    assert recording.choose_answer(turn, LABEL_LIVE).answer is ChooseKind.INVALID


def test_the_agent_s_kinds_are_no_answer_of_choose():
    assert live.read_choose(answer("escalate", None, ""))["answer"] == "invalid"
    asked = live.read_choose(search("search_catalog", "account manager"))
    assert asked == recording.invalid_turn(
        "the model asked for a tool, and the LLM fallback has none"
    )


# --- Reading an agent response (Q20, S10) --------------------------------------------------


@pytest.mark.parametrize("name", [tool.value for tool in SearchTool])
def test_one_tool_request_is_one_search(name: str):
    assert live.read_review(search(name, "account manager")) == {
        "tool": name,
        "query": "account manager",
    }


def test_words_beside_a_tool_request_are_not_kept():
    given = response(
        text("I will look the title up."),
        tool_use("search_catalog", {"query": "account manager"}),
        stop="tool_use",
    )
    assert live.read_review(given) == recording.search_turn("search_catalog", "account manager")


@pytest.mark.parametrize(
    ("given", "turn"),
    [
        (("role", 1, "account manager"), ("role", 1, "account manager")),
        (("role", 14, "account manager"), ("role", 14, "account manager")),  # shared code maps it
        (("escalate", None, ""), ("escalate", None, "")),
        (("escalate", 1, "account manager"), ("escalate", None, "account manager")),
    ],
)
def test_the_final_answer_of_the_agent_is_read_as_it_is(given, turn):
    assert live.read_review(answer(*given)) == recording.answer_turn(*turn)


@pytest.mark.parametrize(
    ("given", "detail"),
    [
        *INVALID_ANSWERS,
        (
            response(
                tool_use("search_catalog", {"query": "a"}, "toolu_01"),
                tool_use("search_approved_titles", {"query": "a"}, "toolu_02"),
                stop="tool_use",
            ),
            "the turn holds 2 tool requests; a turn is one search or the final answer",
        ),
        (response(text("{}"), stop="tool_use"), "the turn holds 0 tool requests"),
        (
            response(tool_use("search_the_web", {"query": "a"}), stop="tool_use"),
            "the model asked for 'search_the_web', which is no search",
        ),
        (
            response(tool_use("search_catalog", {}), stop="tool_use"),
            "a search takes one text query, not {}",
        ),
        (
            response(tool_use("search_catalog", {"query": 7}), stop="tool_use"),
            "a search takes one text query, not {'query': 7}",
        ),
        (
            response(tool_use("search_catalog", {"query": "a", "family": "SAL"}), stop="tool_use"),
            "a search takes one text query, not",
        ),
        (
            response(tool_use("search_catalog", "account manager"), stop="tool_use"),
            "the model asked for 'search_catalog', which is no search",
        ),
        (answer("no_match", None, "q"), "the answer 'no_match' is not one of role, escalate"),
    ],
)
def test_an_agent_response_with_no_usable_turn_is_invalid_with_the_reason(given, detail: str):
    turn = live.read_review(given)
    assert (turn["answer"], turn["number"], turn["quote"]) == ("invalid", None, "")
    assert turn["detail"].startswith(detail)
    made = recording.review_turn(turn, LABEL_LIVE)
    assert isinstance(made, ReviewAnswer) and made.answer is ReviewKind.INVALID


# --- The provider with a scripted client (Q32 b, Q40) ----------------------------------------


def test_choose_sends_the_request_and_labels_the_answer_live(settings: Settings):
    client = ScriptedClient(answer("role", 2, "account manager"))
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    assert llm.chooser.choose(RECORD, ROLES) == ChooseAnswer(
        ChooseKind.ROLE, 2, "account manager", LABEL_LIVE
    )
    assert client.requests == [live.sdk_arguments(MODEL, recording.choose_request(RECORD, ROLES))]
    assert LABEL_LIVE == "live"


def test_each_agent_turn_is_one_request(settings: Settings):
    client = ScriptedClient(
        search("search_catalog", "account manager"), answer("role", 1, "account manager")
    )
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    assert llm.reviewer is llm.chooser
    assert llm.reviewer.next_turn(transcript()) == SearchRequest(
        SearchTool.CATALOG, "account manager"
    )
    assert llm.reviewer.next_turn(transcript(1)) == ReviewAnswer(
        ReviewKind.ROLE, 1, "account manager", LABEL_LIVE
    )
    assert client.requests == [
        live.sdk_arguments(MODEL, recording.review_request(transcript())),
        live.sdk_arguments(MODEL, recording.review_request(transcript(1))),
    ]


def test_every_answer_is_recorded_as_it_is_made(settings: Settings):
    client = ScriptedClient(
        answer("role", 2, "account manager"), search("search_catalog", "account manager")
    )
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    assert not settings.recording_dir.exists()  # nothing is written before the first answer
    llm.chooser.choose(RECORD, ROLES)
    (entry,) = lines(partial(settings) / CHOOSE_FILE)
    request = recording.choose_request(RECORD, ROLES)
    assert entry["key"] == recording.request_key("choose", MODEL, request)
    assert (entry["model"], entry["reported_model"]) == (MODEL, REPORTED)
    assert (entry["family_code"], entry["cleaned_title"]) == ("SAL", "senior account manager")
    assert entry["turn"] == recording.answer_turn("role", 2, "account manager")
    assert (entry["stop_reason"], entry["input_tokens"], entry["output_tokens"]) == (
        "end_turn",
        300,
        20,
    )
    llm.reviewer.next_turn(transcript())
    (turn,) = lines(partial(settings) / REVIEW_FILE)
    assert turn["key"] == recording.request_key(
        "review", MODEL, recording.review_request(transcript())
    )
    assert (turn["turn"], turn["stop_reason"]) == (
        recording.search_turn("search_catalog", "account manager"),
        "tool_use",
    )


def test_finish_completes_the_recording(settings: Settings):
    client = ScriptedClient(answer("role", 2, "account manager"))
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    llm.chooser.choose(RECORD, ROLES)
    assert llm.finish() is None
    folder = settings.recording_dir / MODEL / "offline"
    assert sorted(path.name for path in folder.iterdir()) == [
        CHOOSE_FILE,
        MANIFEST_FILE,
        REVIEW_FILE,
    ]
    manifest = recording.read_manifest(folder)
    assert (manifest["model"], manifest["data_fingerprint"]) == (MODEL, FINGERPRINT)
    assert (manifest["requests"], manifest["complete"]) == (1, True)
    assert not partial(settings).exists()


def test_an_invalid_answer_is_an_answer_and_is_recorded(settings: Settings):
    # Q39: a refusal, a cut-off or a failed validation goes to a person with
    # the reason, and is recorded so replay repeats it.
    client = ScriptedClient(
        response(text(""), stop="refusal"),
        response(text('{"answer": "ro'), stop="max_tokens"),
        response(text("no JSON")),
    )
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    refused = llm.chooser.choose(RECORD, ROLES)
    assert refused == ChooseAnswer(
        ChooseKind.INVALID, None, "", LABEL_LIVE, "the model refused to answer"
    )
    cut_off = llm.reviewer.next_turn(transcript())
    assert cut_off == ReviewAnswer(
        ReviewKind.INVALID, None, "", LABEL_LIVE, "the answer was cut off at max_tokens = 1024"
    )
    assert llm.chooser.choose(OTHER, ROLES).detail == "the answer is not JSON"
    recorded = [entry["turn"]["detail"] for entry in lines(partial(settings) / CHOOSE_FILE)]
    assert recorded == ["the model refused to answer", "the answer is not JSON"]
    assert [entry["stop_reason"] for entry in lines(partial(settings) / REVIEW_FILE)] == [
        "max_tokens"
    ]
    assert client.script == []  # one request each: nothing was asked a second time


def test_a_request_already_answered_in_the_run_is_not_sent_again(settings: Settings):
    client = ScriptedClient(
        answer("role", 2, "account manager"), search("search_catalog", "account manager")
    )
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    first = llm.chooser.choose(RECORD, ROLES)
    assert llm.chooser.choose(RECORD, ROLES) == first
    turn = llm.reviewer.next_turn(transcript())
    assert llm.reviewer.next_turn(transcript()) == turn
    assert len(client.requests) == 2
    assert len(lines(partial(settings) / CHOOSE_FILE)) == 1
    assert len(lines(partial(settings) / REVIEW_FILE)) == 1


def test_another_record_or_another_list_is_another_request(settings: Settings):
    client = ScriptedClient(*[answer("role", 1, "account manager")] * 3)
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    llm.chooser.choose(RECORD, ROLES)
    llm.chooser.choose(OTHER, ROLES)
    llm.chooser.choose(RECORD, ROLES[::-1])
    assert len(client.requests) == 3


# --- Failures that stop the run (Q02, Q39) ----------------------------------------------------


def test_the_request_cap_stops_the_run_and_keeps_what_was_recorded(tmp_path: Path):
    settings = Settings(
        llm="anthropic", llm_model=MODEL, recording_dir=tmp_path / "recordings", request_cap=2
    )
    client = ScriptedClient(*[answer("role", 1, "account manager")] * 3)
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    llm.chooser.choose(RECORD, ROLES)
    llm.chooser.choose(OTHER, ROLES)
    llm.chooser.choose(RECORD, ROLES)  # answered before: no request, so not over the cap
    with pytest.raises(LiveError) as raised:
        llm.reviewer.next_turn(transcript())
    message = str(raised.value)
    assert "the request cap is reached" in message and "request_cap" in message
    assert "2 model requests" in message
    assert f"the answers recorded so far (2) are kept in {partial(settings)}" in message
    assert len(client.requests) == 2  # the third was never sent
    assert len(lines(partial(settings) / CHOOSE_FILE)) == 2
    assert not (settings.recording_dir / MODEL / "offline").exists()
    assert isinstance(raised.value, TitlestdError)


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (
            StatusError(401, "invalid x-api-key"),
            "the hosted API rejected the key in ANTHROPIC_API_KEY (status 401): "
            "StatusError: invalid x-api-key.",
        ),
        (
            StatusError(404, "model: test-double"),
            "the hosted API does not know the model 'test-double', or the key may not use it "
            "(status 404): StatusError: model: test-double.",
        ),
        (
            StatusError(400, "tools.0: unknown field"),
            "the hosted API answered a request with status 400: StatusError: tools.0: unknown",
        ),
        (
            StatusError(429, "rate limited"),
            "the hosted API answered a request with status 429, after the SDK's own retries: ",
        ),
        (
            StatusError(529, "overloaded"),
            "the hosted API answered a request with status 529, after the SDK's own retries: ",
        ),
        (
            ConnectionError("no route to host"),
            "a request to the hosted API got no answer, after the SDK's own retries: "
            "ConnectionError: no route to host.",
        ),
    ],
)
def test_a_failed_first_request_stops_the_run_and_nothing_is_written(
    settings: Settings, error: Exception, said: str
):
    client = ScriptedClient(error)
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    with pytest.raises(LiveError) as raised:
        llm.chooser.choose(RECORD, ROLES)
    message = str(raised.value)
    assert message.startswith(said)
    assert message.endswith(
        "The run stops and nothing stands in for the model; "
        "no answer was recorded and nothing was written."
    )
    assert raised.value.__cause__ is error
    assert not settings.recording_dir.exists()


def test_a_request_that_fails_later_keeps_the_answers_already_recorded(settings: Settings):
    done = settings.recording_dir / MODEL / "offline"
    earlier = live.build_llm(settings, FINGERPRINT, client=ScriptedClient(answer("role", 1, "q")))
    earlier.chooser.choose(OTHER, ROLES)
    earlier.finish()
    before = {path.name: path.read_bytes() for path in done.iterdir()}

    client = ScriptedClient(
        answer("role", 2, "account manager"),
        search("search_catalog", "account manager"),
        StatusError(500, "internal error"),
    )
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    llm.chooser.choose(RECORD, ROLES)
    llm.reviewer.next_turn(transcript())
    with pytest.raises(LiveError) as raised:
        llm.reviewer.next_turn(transcript(1))
    assert "status 500, after the SDK's own retries" in str(raised.value)
    assert f"the answers recorded so far (2) are kept in {partial(settings)}" in str(raised.value)
    assert len(lines(partial(settings) / CHOOSE_FILE)) == 1
    assert len(lines(partial(settings) / REVIEW_FILE)) == 1
    assert not (partial(settings) / MANIFEST_FILE).exists()
    # The recording of the earlier run is as it was: the pipeline never calls finish.
    assert {path.name: path.read_bytes() for path in done.iterdir()} == before


def test_nothing_stands_in_for_a_failed_request(settings: Settings):
    # Q39: the stand-in is never substituted. The failure is not an answer.
    client = ScriptedClient(ConnectionError("down"), ConnectionError("down"))
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    with pytest.raises(LiveError):
        llm.chooser.choose(RECORD, ROLES)
    with pytest.raises(LiveError):
        llm.reviewer.next_turn(transcript())
    assert len(client.requests) == 2 and not settings.recording_dir.exists()
    # The provider has no way to: it does not know the stand-in's module.
    assert "standin" not in Path(live.__file__).read_text(encoding="utf-8")
    assert "standin" not in Path(recording.__file__).read_text(encoding="utf-8").replace(
        "standin_", ""
    )


def test_an_interrupted_run_is_not_reported_as_a_failed_request(settings: Settings):
    llm = live.build_llm(settings, FINGERPRINT, client=ScriptedClient(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        llm.chooser.choose(RECORD, ROLES)


def test_no_credential_reaches_a_recording(settings: Settings, monkeypatch):
    # 2.8: no keys in recordings. With a key in the environment, nothing of it
    # is in any file a run writes.
    monkeypatch.setenv(API_KEY_VARIABLE, NOT_A_KEY)
    client = ScriptedClient(
        answer("role", 2, "account manager"), search("search_catalog", "account manager")
    )
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    llm.chooser.choose(RECORD, ROLES)
    llm.reviewer.next_turn(transcript())
    llm.finish()
    written = [path for path in settings.recording_dir.rglob("*") if path.is_file()]
    assert len(written) == 3
    for path in written:
        assert KEY_PREFIX not in path.read_text(encoding="utf-8"), path


# --- Start-up: the model, the key, the SDK (Q31, Q39, S09) -------------------------------------


def test_live_mode_without_a_model_is_refused_naming_the_example(tmp_path: Path):
    # Q31: no default, and the message names claude-haiku-4-5.
    with pytest.raises(SettingsError, match="--llm-model claude-haiku-4-5"):
        Settings(llm="anthropic")
    # The provider says the same of settings that chose another llm.
    with pytest.raises(SettingsError, match="--llm-model claude-haiku-4-5") as raised:
        live.build_llm(Settings(recording_dir=tmp_path), FINGERPRINT, client=ScriptedClient())
    assert "llm_model, which has no default" in str(raised.value)
    assert list(tmp_path.iterdir()) == []


def test_a_model_that_cannot_name_a_recording_is_refused_at_start_up(tmp_path: Path):
    settings = Settings(llm="anthropic", llm_model="models/latest", recording_dir=tmp_path)
    with pytest.raises(recording.RecordingError, match="one plain folder name"):
        live.build_llm(settings, FINGERPRINT, client=ScriptedClient())


@pytest.mark.parametrize("value", [None, "", "   "])
def test_without_the_key_live_mode_stops_before_the_sdk_is_looked_for(
    settings: Settings, monkeypatch, value
):
    if value is not None:
        monkeypatch.setenv(API_KEY_VARIABLE, value)
    # The fixture blocks the import: were the SDK looked for first, the message
    # would be the one for a missing extra.
    with pytest.raises(LiveError) as raised:
        live.build_llm(settings, FINGERPRINT)
    message = str(raised.value)
    assert message.startswith(
        "llm = anthropic needs the environment variable ANTHROPIC_API_KEY, which is not set "
        "or is empty."
    )
    assert "uv run --env-file" in message and "outside version control" in message
    assert run_with_extra("live") not in message
    assert not settings.recording_dir.exists()


def test_a_missing_extra_stops_with_the_command_of_s09(settings: Settings, monkeypatch):
    monkeypatch.setenv(API_KEY_VARIABLE, NOT_A_KEY)
    with pytest.raises(TitlestdError) as raised:
        live.build_llm(settings, FINGERPRINT)
    message = str(raised.value)
    assert "needs the anthropic library, which cannot be imported" in message
    assert 'the optional extra "live"' in message
    assert f"{run_with_extra(live.EXTRA)} record --llm-model claude-haiku-4-5" in message
    assert message.count("uv run --extra live python -m titlestd") == 1
    assert isinstance(raised.value.__cause__, ImportError)
    assert NOT_A_KEY not in message
    assert not settings.recording_dir.exists()


@pytest.fixture
def stub_sdk(monkeypatch) -> list[tuple]:
    """A stub of the anthropic module where the lazy import looks. Returns how
    its client was built: the arguments of every Anthropic(...)."""
    built: list[tuple] = []

    class Anthropic(ScriptedClient):
        def __init__(self, *args, **kwargs) -> None:
            built.append((args, kwargs))
            super().__init__(answer("role", 2, "account manager"))

    stub = ModuleType("anthropic")
    stub.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    return built


def test_with_the_key_set_the_sdk_s_client_is_built_with_no_arguments(
    settings: Settings, monkeypatch, stub_sdk
):
    # 2.5, Q39: the SDK reads the key from the environment by itself.
    monkeypatch.setenv(API_KEY_VARIABLE, NOT_A_KEY)
    llm = live.build_llm(settings, FINGERPRINT)
    assert stub_sdk == [((), {})]
    assert llm.chooser.choose(RECORD, ROLES).source == LABEL_LIVE
    assert not settings.recording_dir.joinpath(MODEL, "offline").exists()  # until finish


def test_the_sdk_is_looked_for_only_when_live_mode_is_built(settings: Settings, stub_sdk):
    # The stub was put in place long after titlestd.live was imported.
    with pytest.raises(LiveError, match=API_KEY_VARIABLE):
        live.build_llm(settings, FINGERPRINT)
    assert stub_sdk == []


def test_a_client_that_is_handed_in_needs_no_key_and_no_sdk(settings: Settings, stub_sdk):
    # S09: the seam. The key is not checked and the SDK is not imported.
    client = ScriptedClient(answer("role", 2, "account manager"))
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    assert llm.chooser.choose(RECORD, ROLES).number == 2
    assert stub_sdk == [] and len(client.requests) == 1


def test_the_key_is_read_nowhere_but_in_the_start_up_check():
    # 2.5: the only secret, read from the environment. The module never opens
    # a file and names the variable once.
    source = Path(live.__file__).read_text(encoding="utf-8")
    code = source.split('"""', 2)[2]
    assert code.count("os.environ") == 1 and "getenv" not in code
    assert "open(" not in code and "read_text" not in code and "dotenv" not in source
    assert live.API_KEY_VARIABLE == "ANTHROPIC_API_KEY"
