"""The requests of live mode and replay, and their key (Q15, Q20, Q31, Q33, S16).

A request is the program's own structure, built by titlestd.recording from a
record or a transcript with the shared renderers. Its key is a hash of that
structure. No SDK is involved: the module works where it is not installed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unicodedata

import pytest

from titlestd import agent, fallback, recording
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    EMAIL_MASK,
    NAME_MASK,
    QUOTE_MAX_WORDS,
    QUOTE_MIN_CHARS,
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
)
from titlestd.recording import STEP_CHOOSE, STEP_REVIEW, request_key

MODEL = "test-double"

RECORD = Record(
    "senior account manager",
    "SAL",
    "Enterprise Sales",
    "Owns the largest customer accounts and negotiates renewals.",
)
ACCOUNT_MANAGER = CatalogRole(
    "SAL-01", "SAL", "Account Manager", "Acct Mgr", "Manages a portfolio of customer accounts."
)
SALES_ENGINEER = CatalogRole(
    "SAL-04", "SAL", "Sales Engineer", "Sales Eng", "Explains the technical side of products."
)
TREASURY_ANALYST = CatalogRole(
    "FIN-06", "FIN", "Treasury Analyst", "Treasury Analyst", "Manages cash positions."
)
GENERIC = CatalogRole("SAL-GEN", "SAL", "Sales", "Sales", "Generic role.", is_generic=True)
ROLES = (
    NumberedRole(1, TREASURY_ANALYST),
    NumberedRole(2, ACCOUNT_MANAGER),
    NumberedRole(3, SALES_ENGINEER),
)
LLM_ANSWER = ChooseAnswer(ChooseKind.ROLE, 1, "account manager", "test label")
CASE = AgentCase(
    record=RECORD,
    roles=(
        NumberedRole(None, TREASURY_ANALYST),
        NumberedRole(1, ACCOUNT_MANAGER),
        NumberedRole(2, SALES_ENGINEER),
    ),
    llm_answer=LLM_ANSWER,
    llm_role_id="FIN-06",
    ml_first_choice="SAL-01",
    reasons=(Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH),
)
CATALOG_STEP = SearchStep(
    SearchRequest(SearchTool.CATALOG, "account manager"),
    (SearchHit(NumberedRole(1, ACCOUNT_MANAGER)), SearchHit(NumberedRole(None, TREASURY_ANALYST))),
)
TITLES_STEP = SearchStep(
    SearchRequest(SearchTool.APPROVED_TITLES, "senior account manager"),
    (SearchHit(NumberedRole(1, ACCOUNT_MANAGER), "account manager ii", "SAL"),),
)


def transcript(*steps: SearchStep) -> AgentTranscript:
    return AgentTranscript(CASE, steps)


# --- The request of choose (Q15, S16) -------------------------------------------


def test_a_choose_request_is_one_user_turn_a_schema_and_no_tool():
    request = recording.choose_request(RECORD, ROLES)
    assert request == {
        "system": recording.CHOOSE_SYSTEM,
        "max_tokens": recording.MAX_TOKENS,
        "tools": [],
        "tool_choice": None,
        "output_schema": recording.CHOOSE_SCHEMA,
        "turns": [{"role": "user", "text": recording.choose_prompt(RECORD, ROLES)}],
    }


def test_the_choose_prompt_is_the_record_block_and_the_numbered_roles():
    # The shared renderers, and nothing of this module's own but the heading (S16).
    assert recording.choose_prompt(RECORD, ROLES) == (
        "Title: senior account manager\n"
        "Family: SAL\n"
        "Department: Enterprise Sales\n"
        "Description: Owns the largest customer accounts and negotiates renewals.\n"
        "\n"
        "Roles:\n"
        "1. Treasury Analyst (family FIN): Manages cash positions.\n"
        "2. Account Manager (family SAL): Manages a portfolio of customer accounts.\n"
        "3. Sales Engineer (family SAL): Explains the technical side of products."
    )


def test_the_prompt_holds_the_block_the_quote_is_checked_against():
    # Q15: the same string serves the prompt and the check.
    block = fallback.render_record(RECORD)
    assert recording.choose_prompt(RECORD, ROLES).startswith(block + "\n")
    assert fallback.quote_in_record("negotiates renewals", block)


def test_a_prompt_shows_no_role_id():
    shown = recording.choose_prompt(RECORD, ROLES) + agent.render_case(CASE)
    for role in (ACCOUNT_MANAGER, SALES_ENGINEER, TREASURY_ANALYST):
        assert role.role_id not in shown


@pytest.mark.parametrize("number", [1, None])
def test_no_request_can_show_a_generic_family_role(number):
    # 2.7: no model is ever shown a generic family role.
    with pytest.raises(ValueError, match="generic family role SAL-GEN"):
        recording.choose_request(RECORD, (*ROLES, NumberedRole(number, GENERIC)))
    case = dataclasses.replace(CASE, roles=(*CASE.roles, NumberedRole(number, GENERIC)))
    with pytest.raises(ValueError, match="generic family role SAL-GEN"):
        recording.review_request(AgentTranscript(case))
    step = SearchStep(CATALOG_STEP.request, (SearchHit(NumberedRole(number, GENERIC)),))
    with pytest.raises(ValueError, match="generic family role SAL-GEN"):
        recording.review_request(transcript(step))


def test_the_choose_schema_lists_one_to_ten_as_an_enum():
    # Q15: the schemas know no minimum or maximum, so 1 to 10 is an enum.
    assert recording.CHOOSE_SCHEMA == {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["role", "no_match", "insufficient_data"]},
            "number": {
                "anyOf": [
                    {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    {"type": "null"},
                ]
            },
            "quote": {"type": "string"},
        },
        "required": ["answer", "number", "quote"],
        "additionalProperties": False,
    }


def test_the_review_schema_takes_a_plain_integer():
    # Q20: shared code checks the range of the agent's number.
    assert recording.REVIEW_SCHEMA == {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["role", "escalate"]},
            "number": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "quote": {"type": "string"},
        },
        "required": ["answer", "number", "quote"],
        "additionalProperties": False,
    }


def _nodes(schema):
    yield schema
    for value in schema.values() if isinstance(schema, dict) else schema:
        if isinstance(value, dict | list):
            yield from _nodes(value)


@pytest.mark.parametrize(
    "schema",
    [
        recording.CHOOSE_SCHEMA,
        recording.REVIEW_SCHEMA,
        *(tool["input_schema"] for tool in recording.SEARCH_TOOLS),
    ],
)
def test_no_schema_uses_what_the_hosted_model_does_not_take(schema):
    # Library facts: every object closed and with its fields required; no
    # minimum, maximum, minLength or maxLength.
    for node in _nodes(schema):
        if not isinstance(node, dict):
            continue
        assert not {"minimum", "maximum", "minLength", "maxLength"} & set(node)
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])


# --- The request of an agent turn (Q10, Q20) ------------------------------------


def test_the_first_agent_turn_shows_the_case_and_offers_the_two_searches():
    request = recording.review_request(transcript())
    assert request == {
        "system": recording.REVIEW_SYSTEM,
        "max_tokens": recording.MAX_TOKENS,
        "tools": list(recording.SEARCH_TOOLS),
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        "output_schema": recording.REVIEW_SCHEMA,
        "turns": [{"role": "user", "text": agent.render_case(CASE)}],
    }


def test_the_two_tools_are_the_two_searches_strict_with_one_text_query():
    assert [tool["name"] for tool in recording.SEARCH_TOOLS] == [
        "search_catalog",
        "search_approved_titles",
    ]
    assert [tool["name"] for tool in recording.SEARCH_TOOLS] == [tool.value for tool in SearchTool]
    for tool in recording.SEARCH_TOOLS:
        assert set(tool) == {"name", "description", "strict", "input_schema"}
        assert tool["strict"] is True
        assert tool["input_schema"]["required"] == ["query"]
        assert tool["input_schema"]["properties"]["query"]["type"] == "string"
        assert len(tool["description"]) > 100


def test_each_search_is_a_tool_request_and_its_result_text():
    request = recording.review_request(transcript(CATALOG_STEP, TITLES_STEP))
    assert request["turns"] == [
        {"role": "user", "text": agent.render_case(CASE)},
        {"role": "assistant", "tool": "search_catalog", "input": {"query": "account manager"}},
        {"role": "user", "tool_result": agent.render_step(CATALOG_STEP)},
        {
            "role": "assistant",
            "tool": "search_approved_titles",
            "input": {"query": "senior account manager"},
        },
        {"role": "user", "tool_result": agent.render_step(TITLES_STEP)},
    ]
    assert request["turns"][4]["tool_result"] == (
        '1. Account Manager: approved for the title "account manager ii" (family SAL)'
    )


@pytest.mark.parametrize("searches", range(AGENT_MAX_SEARCHES + 1))
def test_the_tools_are_withdrawn_once_the_searches_are_used(searches: int):
    # Q20: tool_choice none once four searches are used; never a forced choice (Q31).
    request = recording.review_request(transcript(*[CATALOG_STEP] * searches))
    if searches < AGENT_MAX_SEARCHES:
        assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    else:
        assert request["tool_choice"] == {"type": "none"}
    assert request["tools"] == list(recording.SEARCH_TOOLS)  # the turns still name them


def test_the_prompts_repeat_what_shared_code_holds():
    # The prompts are written out in full; each thing they repeat is held here
    # to the constant that shared code enforces.
    choose, review = (
        " ".join(prompt.split()) for prompt in (recording.CHOOSE_SYSTEM, recording.REVIEW_SYSTEM)
    )
    for prompt in (choose, review):
        assert f"masked as {NAME_MASK} and {EMAIL_MASK}." in prompt
        assert (
            f"at least {QUOTE_MIN_CHARS} characters and at most {QUOTE_MAX_WORDS} words"
        ) in prompt
    for kind in recording.CHOOSE_KINDS:
        assert f'"{kind}"' in choose
    for kind in recording.REVIEW_KINDS:
        assert f'"{kind}"' in review
    assert f"at most {AGENT_MAX_SEARCHES} searches in all" in review
    for tool in SearchTool:
        assert f"- {tool.value} finds " in review
    assert f'"{agent.LLM_PICK_MARK}" marks' in review
    assert f'"{agent.ML_FIRST_CHOICE_MARK}" the role' in review
    # The three reasons for which the rules send a title to the agent (1.7).
    for reason in (Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH, Reason.NO_MATCH):
        assert f'"{reason.value}" (' in review


def test_the_llm_is_told_it_may_choose_a_role_of_any_family():
    # 1.7: a role outside the recorded family is for the rules to catch and
    # the agent to settle, so the LLM must not rule one out by itself. The
    # agent is told the opposite: it may only pick a role of the family (1.4).
    choose, review = (
        " ".join(prompt.split()) for prompt in (recording.CHOOSE_SYSTEM, recording.REVIEW_SYSTEM)
    )
    assert "Any listed role may be chosen, whichever job family it belongs to." in choose
    assert "You may only pick a role of the record's own job family." in review


def test_the_prompts_are_plain_text_with_no_line_left_over():
    for prompt in (recording.CHOOSE_SYSTEM, recording.REVIEW_SYSTEM):
        assert prompt == prompt.strip() and "{" not in prompt and "\t" not in prompt
        assert max(map(len, prompt.splitlines())) <= 100
        assert prompt.isascii()


def test_a_request_sets_nothing_but_these_six_things():
    # Q31: no temperature, top_p or top_k, no thinking, no effort.
    names = {"system", "max_tokens", "tools", "tool_choice", "output_schema", "turns"}
    assert set(recording.choose_request(RECORD, ROLES)) == names
    assert set(recording.review_request(transcript(CATALOG_STEP))) == names


# --- The key (Q33) ----------------------------------------------------------------

SMALL = {
    "system": "Répondez.",
    "max_tokens": 8,
    "tools": [],
    "tool_choice": None,
    "output_schema": {"type": "object"},
    "turns": [{"role": "user", "text": "café"}],
}
# SMALL beside key_version, step and model: sorted names, compact separators,
# non-ASCII kept. Written out by hand.
SMALL_AS_KEYED = (
    '{"key_version":1,"max_tokens":8,"model":"test-double","output_schema":{"type":"object"},'
    '"step":"choose","system":"Répondez.","tool_choice":null,"tools":[],'
    '"turns":[{"role":"user","text":"café"}]}'
)


def test_the_key_is_the_sha_256_of_the_canonical_json():
    expected = hashlib.sha256(SMALL_AS_KEYED.encode("utf-8")).hexdigest()
    assert request_key(STEP_CHOOSE, MODEL, SMALL) == expected
    assert len(expected) == 64


def test_the_key_does_not_depend_on_the_order_of_the_names():
    backwards = dict(reversed(SMALL.items()))
    assert list(backwards) != list(SMALL)
    assert request_key(STEP_CHOOSE, MODEL, backwards) == request_key(STEP_CHOOSE, MODEL, SMALL)


def test_strings_are_nfc_normalised_before_hashing():
    decomposed = json.loads(unicodedata.normalize("NFD", json.dumps(SMALL, ensure_ascii=False)))
    assert decomposed != SMALL and decomposed["turns"][0]["text"] == "cafe\u0301"
    assert request_key(STEP_CHOOSE, MODEL, decomposed) == request_key(STEP_CHOOSE, MODEL, SMALL)
    # The names of a mapping and the model are strings too.
    named = {**SMALL, "output_schema": {"e\u0301tat": 1}}
    same = {**SMALL, "output_schema": {"\u00e9tat": 1}}
    assert request_key(STEP_CHOOSE, MODEL, named) == request_key(STEP_CHOOSE, MODEL, same)
    assert request_key(STEP_CHOOSE, "mode\u0300le", SMALL) == request_key(
        STEP_CHOOSE, "mod\u00e8le", SMALL
    )


def test_a_tuple_and_a_list_are_keyed_alike():
    as_tuples = {**SMALL, "tools": (), "turns": tuple(SMALL["turns"])}
    assert request_key(STEP_CHOOSE, MODEL, as_tuples) == request_key(STEP_CHOOSE, MODEL, SMALL)


REVIEW = recording.review_request(transcript(CATALOG_STEP))


def _changed(path: tuple, value) -> dict:
    """REVIEW with one value replaced, at the end of a path of names and positions."""
    changed = json.loads(json.dumps(REVIEW))
    node = changed
    for name in path[:-1]:
        node = node[name]
    assert node[path[-1]] != value
    node[path[-1]] = value
    return changed


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("system",), recording.REVIEW_SYSTEM + " "),
        (("max_tokens",), recording.MAX_TOKENS + 1),
        (("tools", 0, "description"), "Search."),
        (("tools", 1, "strict"), False),
        (("tool_choice",), {"type": "none"}),
        (("tool_choice", "disable_parallel_tool_use"), False),
        (("output_schema", "properties", "answer", "enum"), ["role"]),
        (("turns", 0, "role"), "assistant"),
        (("turns", 0, "text"), "Title: account manager"),
        (("turns", 1, "tool"), "search_approved_titles"),
        (("turns", 1, "input"), {"query": "Account Manager"}),
        (("turns", 2, "tool_result"), "No results."),
    ],
    ids=lambda value: ".".join(map(str, value)) if isinstance(value, tuple) else None,
)
def test_everything_the_program_sets_is_in_the_key(path, value):
    assert request_key(STEP_REVIEW, MODEL, _changed(path, value)) != request_key(
        STEP_REVIEW, MODEL, REVIEW
    )


def test_the_step_the_model_and_the_key_version_are_in_the_key(monkeypatch):
    key = request_key(STEP_REVIEW, MODEL, REVIEW)
    assert request_key(STEP_CHOOSE, MODEL, REVIEW) != key
    # The model as configured: an alias and its dated id are two models here (Q33).
    assert request_key(STEP_REVIEW, "claude-haiku-4-5", REVIEW) != request_key(
        STEP_REVIEW, "claude-haiku-4-5-20251001", REVIEW
    )
    monkeypatch.setattr(recording, "KEY_VERSION", recording.KEY_VERSION + 1)
    assert request_key(STEP_REVIEW, MODEL, REVIEW) != key


def test_one_key_per_agent_turn():
    # Q33: up to five keys for a case, one for each transcript it can be asked with.
    turns = [transcript(*[CATALOG_STEP] * n) for n in range(AGENT_MAX_SEARCHES + 1)]
    keys = {request_key(STEP_REVIEW, MODEL, recording.review_request(turn)) for turn in turns}
    assert len(keys) == AGENT_MAX_SEARCHES + 1


def test_the_key_of_a_turn_depends_on_what_the_search_returned():
    # Q20: replay runs the searches again; another result is another key.
    other = SearchStep(CATALOG_STEP.request, CATALOG_STEP.hits[:1])
    assert request_key(STEP_REVIEW, MODEL, recording.review_request(transcript(other))) != (
        request_key(STEP_REVIEW, MODEL, REVIEW)
    )


def test_nothing_but_the_request_is_keyed():
    # Q33: no tool-use id, date or request id is in the structure at all.
    text = json.dumps(REVIEW)
    assert '"id"' not in text and "tool_use_id" not in text and "date" not in text
    assert request_key(STEP_REVIEW, MODEL, REVIEW) == request_key(
        STEP_REVIEW, MODEL, recording.review_request(transcript(CATALOG_STEP))
    )


@pytest.mark.parametrize("step", ["", "embed", "CHOOSE", None])
def test_a_step_is_choose_or_review(step):
    with pytest.raises(ValueError, match="a step is 'choose' or 'review'"):
        request_key(step, MODEL, SMALL)


@pytest.mark.parametrize("name", ["key_version", "step", "model"])
def test_a_request_cannot_hide_what_is_keyed_beside_it(name: str):
    with pytest.raises(ValueError, match=name):
        request_key(STEP_CHOOSE, MODEL, {**SMALL, name: "other"})


def test_a_request_must_be_plain_json():
    with pytest.raises(ValueError):
        request_key(STEP_CHOOSE, MODEL, {**SMALL, "max_tokens": float("nan")})
    with pytest.raises(TypeError):
        request_key(STEP_CHOOSE, MODEL, {**SMALL, "tools": {1, 2}})


# The keys of two real requests under the example model. They move when a
# prompt, a tool, a schema or a shared renderer is reworded, and every recording
# made before is then a miss: raise PROMPT_VERSION with them, and record again.
PINNED_CHOOSE_KEY = "d9f1a0c603c01c1902d8e8c84ebf00b3763577fc586fc13352a8ac39a120aa69"
PINNED_REVIEW_KEY = "70db86097a8c62b54177ef6da71f1938decd31f35f53d3e4df81ee274490de42"


def test_the_keys_of_the_real_requests_are_pinned():
    assert (recording.PROMPT_VERSION, recording.KEY_VERSION) == (1, 1)
    choose = recording.choose_request(RECORD, ROLES)
    assert request_key(STEP_CHOOSE, "claude-haiku-4-5", choose) == PINNED_CHOOSE_KEY
    assert request_key(STEP_REVIEW, "claude-haiku-4-5", REVIEW) == PINNED_REVIEW_KEY


# --- A stored turn ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn", "answer"),
    [
        (
            recording.answer_turn("role", 3, "account manager"),
            ChooseAnswer(ChooseKind.ROLE, 3, "account manager", "a label"),
        ),
        (
            recording.answer_turn("no_match", None, "bdr"),
            ChooseAnswer(ChooseKind.NO_MATCH, None, "bdr", "a label"),
        ),
        (
            recording.invalid_turn("the model refused to answer"),
            ChooseAnswer(ChooseKind.INVALID, None, "", "a label", "the model refused to answer"),
        ),
    ],
)
def test_a_stored_choose_turn_is_the_answer_under_the_label_given(turn, answer):
    assert recording.choose_answer(turn, "a label") == answer
    assert recording.choose_answer(json.loads(json.dumps(turn)), "a label") == answer


@pytest.mark.parametrize(
    ("turn", "made"),
    [
        (
            recording.search_turn("search_catalog", "account manager"),
            SearchRequest(SearchTool.CATALOG, "account manager"),
        ),
        (
            recording.search_turn("search_approved_titles", ""),
            SearchRequest(SearchTool.APPROVED_TITLES, ""),
        ),
        (
            recording.answer_turn("role", 12, "renewals"),
            ReviewAnswer(ReviewKind.ROLE, 12, "renewals", "a label"),
        ),
        (
            recording.answer_turn("escalate", None, ""),
            ReviewAnswer(ReviewKind.ESCALATE, None, "", "a label"),
        ),
        (
            recording.invalid_turn("cut off"),
            ReviewAnswer(ReviewKind.INVALID, None, "", "a label", "cut off"),
        ),
    ],
)
def test_a_stored_agent_turn_is_a_search_request_or_the_final_answer(turn, made):
    assert recording.review_turn(json.loads(json.dumps(turn)), "a label") == made
    assert isinstance(recording.review_turn(turn, "a label"), type(made))
