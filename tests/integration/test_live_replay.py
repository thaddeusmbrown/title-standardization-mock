"""Live mode and replay on the mini data set, with the real modules (Q32 b and c).

Every title row a model may see goes the way a run takes it: the real readers
and prepare step, the seed look-up, the ML match on the offline embedder, the
seeded numbering, choose, the rules of 1.7, the agent's case and the shared
loop with its two searches, and the agent's answer back through the rules. The
pipeline is built beside this branch, so the steps are put together here; it
is test-only code and does not bind the pipeline.

What answers is a hosted model in miniature: it is handed one Messages
request, as the data the API would receive, and gives back one message. It
answers what the stand-in answered when it was shown the same texts, so a
live run must come to the stand-in's decisions and can be compared with them.
It stands behind two clients:

- tier (b): a scripted object with the SDK client's messages.create, handed
  to live.build_llm (S09). It needs no SDK, so it runs in every environment,
  once more in a bare interpreter under the import guard.
- tier (c): the real SDK client on a mock transport. It runs where the "live"
  extra is installed, which the gate's first run has, and skips elsewhere.

Nothing here reaches the network or needs a key. Recordings are made by the
program's recorder in a temporary folder and are never committed.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from titlestd import agent, catalog, fallback, live, prepare, recording, replay, rules, switch
from titlestd.contracts import (
    LABEL_LIVE,
    LABEL_STAND_IN,
    AgentTranscript,
    Catalog,
    ChooseAnswer,
    ChooseKind,
    Embedder,
    InputData,
    Llm,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
    SearchRequest,
    SearchStep,
    TitleKey,
    TitleRow,
    TitlestdError,
    Vector,
    run_with_extra,
)
from titlestd.live import API_KEY_VARIABLE, LiveError
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher
from titlestd.recording import CHOOSE_FILE, MANIFEST_FILE, REVIEW_FILE, RecordingError
from titlestd.replay import ReplayMiss
from titlestd.settings import Settings

MODEL = "test-double"  # Q32 b: an obviously fake model ID
REPORTED = "test-double-0001"  # what the model in miniature says it is
FINGERPRINT = "0f" * 32
# The shape of a key, in two pieces: no tracked file holds a key-shaped string (Q02).
KEY_PREFIX = "sk-" + "ant-"
NOT_A_KEY = KEY_PREFIX + "not-a-real-key-000"
RECORDED_ON = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)
RECORDED = "recorded 2026-01-05, test-double"

# Titles the rules send to the agent, which the mini data set has none of: two
# recorded in the wrong family (1.7), and one on which the two methods differ.
TREASURY_IN_SALES = TitleRow(
    Record("treasury analyst", "SAL", "Treasury", "Manages cash positions and bank accounts."),
    ("X001",),
    ("Treasury Analyst",),
)
ENGINEER_IN_FINANCE = TitleRow(
    Record(
        "sales engineer",
        "FIN",
        "Solutions",
        "Explains the technical side of products and runs demonstrations during a sale.",
    ),
    ("X002",),
    ("Sales Engineer",),
)
SUCCESS_REPRESENTATIVE = TitleRow(
    Record("customer success representative", "SAL", "Customer Success", ""),
    ("X004",),
    ("Customer Success Rep",),
)
HAND_MADE = (TREASURY_IN_SALES, ENGINEER_IN_FINANCE, SUCCESS_REPRESENTATIVE)


# --- A run up to and through the agent ------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What a run decided for one title row."""

    llm_answer: ChooseAnswer
    referral: RuleOutcome
    agent_answer: ReviewAnswer
    agent_role_id: str | None
    searches: tuple[SearchStep, ...]
    settled: RuleOutcome

    @property
    def labels(self) -> set[str]:
        return {self.llm_answer.source, self.agent_answer.source}

    def unlabelled(self) -> Outcome:
        return dataclasses.replace(
            self,
            llm_answer=dataclasses.replace(self.llm_answer, source=""),
            agent_answer=dataclasses.replace(self.agent_answer, source=""),
        )


def unlabelled(outcomes: dict[TitleKey, Outcome]) -> dict[TitleKey, Outcome]:
    return {key: outcome.unlabelled() for key, outcome in outcomes.items()}


def labels(outcomes: dict[TitleKey, Outcome]) -> set[str]:
    return set().union(*(outcome.labels for outcome in outcomes.values()))


@dataclasses.dataclass(frozen=True)
class World:
    """The mini data set as a run meets it, before any wave."""

    catalog: Catalog
    rows: tuple[TitleRow, ...]
    knowledge: Knowledge
    embedder: Embedder
    standin: Llm


def run(
    llm: Llm,
    world: World,
    settings: Settings,
    knowledge: Knowledge | None = None,
    embedder: Embedder | None = None,
) -> dict[TitleKey, Outcome]:
    """The steps of 1.6 from the ML match to the agent's checked answer, for
    every row a model may see. The agent is asked about each, referred or not."""
    knowledge = knowledge or world.knowledge
    embedder = embedder or world.embedder
    cat = world.catalog
    matches = Matcher(cat, embedder).match(world.rows, knowledge)
    tools = agent.SearchTools(cat, knowledge, embedder, settings)
    outcomes = {}
    for row in world.rows:
        if not row.model_eligible:
            continue
        match, family = matches[row.key], row.record.family_code
        shortlist = [catalog.get_role(cat, each.role_id) for each in match.shortlist]
        shown = fallback.number_roles(shortlist, row.key, settings.seed)
        block = fallback.render_record(row.record)
        answer = llm.chooser.choose(row.record, shown)
        referral = rules.check_llm(answer, shown, block, match, family, cat)
        case = agent.build_case(row, shown, answer, referral.role_id, match, referral.reasons)
        result = agent.review(case, tools, llm.reviewer)
        numbered = result.transcript.numbered_roles
        settled = rules.check_agent(result.answer, numbered, block, match, family, cat)
        outcomes[row.key] = Outcome(
            answer, referral, result.answer, result.role_id, result.transcript.steps, settled
        )
    return outcomes


@pytest.fixture(scope="module")
def world(mini_inputs: InputData) -> World:
    cat = mini_inputs.catalog
    rows = prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(cat)
    )
    knowledge = Knowledge()
    load_seed(knowledge, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    models = switch.build_models(Settings(), cat.ladder, mini_inputs.abbreviations)
    every = tuple(sorted([*rows, *HAND_MADE], key=lambda row: row.key))
    return World(cat, every, knowledge, models.embedder, models.llm)


# --- A hosted model in miniature ------------------------------------------------------


def shown_to_choose(record: Record, roles: Sequence[NumberedRole]) -> tuple:
    """What the LLM fallback is shown, put together from the shared renderers (S16)."""
    return (fallback.render_record(record) + "\n\nRoles:\n" + fallback.render_roles(roles),)


def shown_to_review(transcript: AgentTranscript) -> tuple:
    """What the agent is shown before a turn: the case, then each search with
    its request and its result text (Q10, Q20)."""
    searches = [
        (step.request.tool.value, step.request.query, agent.render_step(step))
        for step in transcript.steps
    ]
    return (agent.render_case(transcript.case), *searches)


def shown_in(body: dict) -> tuple:
    """The same, read back from the messages of a request to the hosted API."""
    first, *rest = body["messages"]
    assert first["role"] == "user" and isinstance(first["content"], str)
    assert len(rest) % 2 == 0
    searches = []
    for asked, result in zip(rest[::2], rest[1::2], strict=True):
        (call,), (found,) = asked["content"], result["content"]
        assert (asked["role"], call["type"]) == ("assistant", "tool_use")
        assert (result["role"], found["type"]) == ("user", "tool_result")
        assert found["tool_use_id"] == call["id"]  # each result answers its own request
        assert set(call["input"]) == {"query"}
        searches.append((call["name"], call["input"]["query"], found["content"]))
    assert len({asked["content"][0]["id"] for asked in rest[::2]}) == len(searches)
    return (first["content"], *searches)


class Scripting:
    """The stand-in, keeping each of its turns under what it was shown."""

    def __init__(self, standin: Llm) -> None:
        self._standin = standin
        self.turns: dict[tuple, ChooseAnswer | SearchRequest | ReviewAnswer] = {}

    def choose(self, record: Record, roles: Sequence[NumberedRole]) -> ChooseAnswer:
        answer = self._standin.chooser.choose(record, roles)
        self.turns[shown_to_choose(record, roles)] = answer
        return answer

    def next_turn(self, transcript: AgentTranscript) -> SearchRequest | ReviewAnswer:
        turn = self._standin.reviewer.next_turn(transcript)
        self.turns[shown_to_review(transcript)] = turn
        return turn


def message(content: list[dict], stop_reason: str, **more) -> dict:
    """One message as the hosted API returns it."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": REPORTED,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 250, "output_tokens": 30},
        **more,
    }


class ModelInMiniature:
    """Answers a Messages request with the turn the stand-in gave when it was
    shown the same texts. A request that shows anything else is a KeyError."""

    def __init__(self, turns: dict[tuple, ChooseAnswer | SearchRequest | ReviewAnswer]) -> None:
        self._turns = turns
        self.bodies: list[dict] = []

    def respond(self, body: dict) -> dict:
        self.bodies.append(body)
        turn = self._turns[shown_in(body)]
        if isinstance(turn, SearchRequest):
            assert body["tool_choice"]["type"] == "auto"  # tools were on offer
            call = {
                "type": "tool_use",
                "id": f"toolu_{len(self.bodies):04d}",
                "name": turn.tool.value,
                "input": {"query": turn.query},
            }
            return message([call], "tool_use")
        fields = {"answer": turn.answer.value, "number": turn.number, "quote": turn.quote}
        return message([{"type": "text", "text": json.dumps(fields)}], "end_turn")


def as_response(data: dict) -> SimpleNamespace:
    """A message as an object with what the adapter reads of a response."""
    blocks = [SimpleNamespace(**block) for block in data["content"]]
    usage = SimpleNamespace(**data["usage"])
    return SimpleNamespace(**{**data, "content": blocks, "usage": usage})


class ScriptedClient:
    """Tier (b): a scripted object with the SDK client's messages.create (S09)."""

    def __init__(self, respond) -> None:
        self.messages = self
        self._respond = respond

    def create(self, **arguments) -> SimpleNamespace:
        # Through JSON and back, as the SDK would send it: plain data only.
        return as_response(self._respond(json.loads(json.dumps(arguments))))


@pytest.fixture(scope="module")
def standin_run(world: World) -> tuple[dict[TitleKey, Outcome], dict]:
    """The run with the stand-in itself, and what it answered to what."""
    scripting = Scripting(world.standin)
    outcomes = run(Llm(chooser=scripting, reviewer=scripting), world, Settings())
    return outcomes, scripting.turns


@pytest.fixture
def miniature(standin_run) -> ModelInMiniature:
    return ModelInMiniature(standin_run[1])


@pytest.fixture(autouse=True)
def recorded_on(monkeypatch) -> None:
    monkeypatch.setattr(recording, "utc_now", lambda: RECORDED_ON)


@pytest.fixture(autouse=True)
def no_key(monkeypatch) -> None:
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(llm="anthropic", llm_model=MODEL, recording_dir=tmp_path / "recordings")


def replay_settings(settings: Settings, **changes) -> Settings:
    return Settings(llm="replay", recording_dir=settings.recording_dir, **changes)


def record_a_run(world: World, settings: Settings, model: ModelInMiniature) -> dict:
    """A whole live run with the scripted client, through the real recorder."""
    llm = live.build_llm(settings, FINGERPRINT, client=ScriptedClient(model.respond))
    outcomes = run(llm, world, settings)
    llm.finish()
    return outcomes


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- The premises: what the stand-in decides on these rows ------------------------------


def test_the_rows_cover_what_a_recording_must_hold(standin_run):
    outcomes, turns = standin_run
    assert len(outcomes) == 16  # 13 mini rows a model may see and the three made by hand
    assert labels(outcomes) == {LABEL_STAND_IN}
    kinds = {outcome.llm_answer.answer for outcome in outcomes.values()}
    assert kinds == {ChooseKind.ROLE, ChooseKind.NO_MATCH, ChooseKind.INSUFFICIENT_DATA}
    assert {outcome.agent_answer.answer for outcome in outcomes.values()} == {
        ReviewKind.ROLE,
        ReviewKind.ESCALATE,
    }
    # Referred to the agent by the rules, and settled or escalated by it.
    assert outcomes[TREASURY_IN_SALES.key].referral == RuleOutcome(
        Route.AGENT, (Reason.FAMILY_MISMATCH,), "FIN-06"
    )
    assert outcomes[TREASURY_IN_SALES.key].settled == RuleOutcome(Route.ACCEPT, (), "SAL-03")
    assert outcomes[SUCCESS_REPRESENTATIVE.key].referral.reasons == (Reason.MODELS_DISAGREE,)
    assert outcomes[ENGINEER_IN_FINANCE.key].settled.reasons == (Reason.AGENT_ESCALATED,)
    # One choose answer per row, and per row two searches and the final answer.
    assert all(len(outcome.searches) == 2 for outcome in outcomes.values())
    assert len(turns) == 16 + 16 * 3


# --- Tier (b): live mode with a scripted client -------------------------------------------


def test_live_mode_comes_to_the_scripted_decisions_and_labels_them_live(
    world, settings, miniature, standin_run
):
    llm = live.build_llm(settings, FINGERPRINT, client=ScriptedClient(miniature.respond))
    outcomes = run(llm, world, settings)
    # Nothing is lost between the model's words and the rules: the adapter's
    # prompts are the shared renderers' texts, and its answers the model's.
    assert unlabelled(outcomes) == unlabelled(standin_run[0])
    assert labels(outcomes) == {LABEL_LIVE}
    assert len(miniature.bodies) == 16 + 16 * 3  # one request per turn (Q20)


def test_every_request_is_recorded_under_its_key_with_what_the_model_said(
    world, settings, miniature
):
    outcomes = record_a_run(world, settings, miniature)
    folder = settings.recording_dir / MODEL / "offline"
    assert sorted(path.name for path in folder.parent.iterdir()) == ["offline"]
    manifest = recording.read_manifest(folder)
    assert manifest["requests"] == len(miniature.bodies) == 64
    assert (manifest["model"], manifest["embedder"], manifest["seed"]) == (
        MODEL,
        "offline",
        settings.seed,
    )
    assert manifest["data_fingerprint"] == FINGERPRINT
    chosen, reviewed = lines(folder / CHOOSE_FILE), lines(folder / REVIEW_FILE)
    assert (len(chosen), len(reviewed)) == (16, 48)
    for entries in (chosen, reviewed):
        assert [entry["key"] for entry in entries] == sorted(entry["key"] for entry in entries)
        assert {entry["date"] for entry in entries} == {"2026-01-05"}
        assert {(entry["model"], entry["reported_model"]) for entry in entries} == {
            (MODEL, REPORTED)
        }
    # Each title's choose entry holds the answer the run was given for it.
    by_title = {(entry["family_code"], entry["cleaned_title"]): entry for entry in chosen}
    for key, outcome in outcomes.items():
        answer = recording.choose_answer(by_title[tuple(key)]["turn"], LABEL_LIVE)
        assert answer == outcome.llm_answer
    assert sum("tool" in entry["turn"] for entry in reviewed) == 32  # the search requests


def test_replay_gives_the_same_decisions_under_the_recorded_label(
    world, settings, miniature, standin_run
):
    recorded = record_a_run(world, settings, miniature)
    asked = len(miniature.bodies)
    again = replay_settings(settings)
    replayed = run(replay.build_llm(again, FINGERPRINT), world, again)
    assert unlabelled(replayed) == unlabelled(recorded)
    assert len(miniature.bodies) == asked  # replay asked the model nothing
    # 2.5: every answer is labelled "stand-in", "live" or "recorded <date>, <model>".
    assert labels(standin_run[0]) == {"stand-in"}
    assert labels(recorded) == {"live"}
    assert labels(replayed) == {RECORDED}


class SpyingEmbedder:
    """The real embedder, with a record of every text it was given."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.texts += texts
        return self._embedder.embed(texts)


def test_replay_runs_the_searches_again(world, settings, miniature):
    # Q20: replay looks each turn up by key and re-runs the searches for real.
    recorded = record_a_run(world, settings, miniature)
    again = replay_settings(settings)
    spy = SpyingEmbedder(world.embedder)
    replayed = run(replay.build_llm(again, FINGERPRINT), world, again, embedder=spy)
    # The query of a search is the title alone, which nothing else embeds for
    # a row that has a description.
    assert TREASURY_IN_SALES.record.cleaned_title in spy.texts
    for key, outcome in replayed.items():
        assert outcome.searches == recorded[key].searches
        assert all(step.hits for step in outcome.searches)


def test_a_changed_seed_is_a_miss_that_says_so(world, settings, miniature):
    # Q18: the shuffled order is part of the prompt, so of the key.
    record_a_run(world, settings, miniature)
    other = replay_settings(settings, seed=7)
    llm = replay.build_llm(other, FINGERPRINT)  # the manifest refuses nothing by itself
    with pytest.raises(ReplayMiss) as raised:
        run(llm, world, other)
    message = str(raised.value)
    assert "replay miss at step choose for the title " in message
    assert "seed: recorded 20260101, now 7" in message
    assert isinstance(raised.value, TitlestdError)


def test_one_more_approved_title_is_a_miss_at_the_turn_after_the_search(
    world, settings, miniature, mini_inputs
):
    record_a_run(world, settings, miniature)
    again = replay_settings(settings)
    more = Knowledge()
    load_seed(more, mini_inputs.seed_lookup, world.rows, mini_inputs.abbreviations)
    more.learn(TREASURY_IN_SALES, "SAL-03")
    with pytest.raises(ReplayMiss) as raised:
        run(replay.build_llm(again, FINGERPRINT), world, again, knowledge=more)
    message = str(raised.value)
    assert "replay miss at step " in message
    # Nothing of the settings changed: the manifest cannot explain it, and says so.
    assert "Its manifest matches the settings in force" in message


def test_a_manifest_that_differs_is_no_second_validity_rule(world, settings, miniature):
    # Q29: the same prompts are found whatever the manifest says of the data.
    recorded = record_a_run(world, settings, miniature)
    again = replay_settings(settings, wave1_size=4, golden_size=3)
    replayed = run(replay.build_llm(again, "ab" * 32), world, again)
    assert unlabelled(replayed) == unlabelled(recorded)


def test_a_run_that_fails_keeps_its_answers_and_the_previous_recording(world, settings, miniature):
    recorded = record_a_run(world, settings, miniature)
    folder = settings.recording_dir / MODEL / "offline"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}

    def fails_at_the_tenth(body: dict) -> dict:
        if len(miniature.bodies) - 64 == 9:
            raise ConnectionError("connection reset")
        return miniature.respond(body)

    llm = live.build_llm(settings, FINGERPRINT, client=ScriptedClient(fails_at_the_tenth))
    with pytest.raises(LiveError) as raised:
        run(llm, world, settings)
    partial = folder.with_name("offline.partial")
    assert f"the answers recorded so far (9) are kept in {partial}" in str(raised.value)
    kept = lines(partial / CHOOSE_FILE) + lines(partial / REVIEW_FILE)
    assert len(kept) == 9 and not (partial / MANIFEST_FILE).exists()
    # The pipeline does not call finish after a failure: the recording stands.
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    again = replay_settings(settings)
    replayed = run(replay.build_llm(again, FINGERPRINT), world, again)
    assert unlabelled(replayed) == unlabelled(recorded)


def test_a_refusal_goes_to_a_person_and_replay_repeats_it(world, settings, miniature):
    # Q39, S10: recorded like any answer. The rules send it to a person.
    refused_title = "Title: treasury analyst\nFamily: FIN"

    def refuses_one(body: dict) -> dict:
        if "tools" not in body and body["messages"][0]["content"].startswith(refused_title):
            details = {"type": "refusal", "category": None, "explanation": None}
            return message([], "refusal", stop_details=details)
        return miniature.respond(body)

    llm = live.build_llm(settings, FINGERPRINT, client=ScriptedClient(refuses_one))
    key = TitleKey("FIN", "treasury analyst")
    (row,) = [row for row in world.rows if row.key == key]
    match = Matcher(world.catalog, world.embedder).match([row], world.knowledge)[key]
    shortlist = [catalog.get_role(world.catalog, each.role_id) for each in match.shortlist]
    shown = fallback.number_roles(shortlist, key, settings.seed)
    block = fallback.render_record(row.record)

    answer = llm.chooser.choose(row.record, shown)
    assert answer == ChooseAnswer(
        ChooseKind.INVALID, None, "", LABEL_LIVE, "the model refused to answer"
    )
    outcome = rules.check_llm(answer, shown, block, match, "FIN", world.catalog)
    assert outcome == RuleOutcome(Route.PERSON, (Reason.INVALID_ANSWER,), None)
    llm.finish()

    again = replay_settings(settings)
    repeated = replay.build_llm(again, FINGERPRINT).chooser.choose(row.record, shown)
    assert repeated == dataclasses.replace(answer, source=RECORDED)


# --- Tier (b) where nothing is installed: a bare interpreter under the import guard ----------

# Run after the import guard: a live run with a client scripted by rule, through
# the real recorder, then the same run replayed. Reports both and what the
# guard saw. Arguments: the data folder and a folder for the recording.
ROUND_TRIP = """
import json
from pathlib import Path
from types import SimpleNamespace

from titlestd import agent, catalog, datafiles, fallback, live, prepare, replay, rules
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher
from titlestd.offline_embedder import build_embedder
from titlestd.settings import Settings

data_dir, recording_dir = Path(sys.argv[1]), Path(sys.argv[2])
inputs = datafiles.read_inputs(data_dir)
cat = inputs.catalog
rows = prepare.prepare(inputs.employees, inputs.abbreviations, catalog.family_codes(cat))
known = Knowledge()
load_seed(known, inputs.seed_lookup, rows, inputs.abbreviations)


def said(content, stop_reason):
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, model="test-double-0001", usage=usage
    )


class Client:
    # As the LLM: the first listed role, quoting the title. As the agent: one
    # catalog search for the title, then an escalation.
    def __init__(self):
        self.messages, self.requests = self, 0

    def create(self, **arguments):
        self.requests += 1
        title = arguments["messages"][0]["content"].splitlines()[0].removeprefix("Title: ")
        if "tools" not in arguments:
            fields = {"answer": "role", "number": 1, "quote": title}
        elif len(arguments["messages"]) == 1:
            call = SimpleNamespace(
                type="tool_use", id="toolu_01", name="search_catalog", input={"query": title}
            )
            return said([call], "tool_use")
        else:
            fields = {"answer": "escalate", "number": None, "quote": ""}
        return said([SimpleNamespace(type="text", text=json.dumps(fields))], "end_turn")


def run(llm, settings):
    embedder = build_embedder(settings)
    matches = Matcher(cat, embedder).match(rows, known)
    tools = agent.SearchTools(cat, known, embedder, settings)
    decided = []
    for row in rows:
        if not row.model_eligible:
            continue
        match = matches[row.key]
        shortlist = [catalog.get_role(cat, each.role_id) for each in match.shortlist]
        shown = fallback.number_roles(shortlist, row.key, settings.seed)
        block = fallback.render_record(row.record)
        answer = llm.chooser.choose(row.record, shown)
        referral = rules.check_llm(answer, shown, block, match, row.record.family_code, cat)
        case = agent.build_case(row, shown, answer, referral.role_id, match, referral.reasons)
        result = agent.review(case, tools, llm.reviewer)
        decided.append({
            "title": row.key.cleaned_title,
            "answer": [answer.answer, answer.number, answer.quote],
            "referral": [referral.route, list(referral.reasons), referral.role_id],
            "agent": [result.answer.answer, len(result.transcript.steps)],
            "labels": sorted({answer.source, result.answer.source}),
        })
    return decided


recording_settings = Settings(llm="anthropic", llm_model="test-double", recording_dir=recording_dir)
client = Client()
llm = live.build_llm(recording_settings, "fingerprint", client=client)
recorded = run(llm, recording_settings)
llm.finish()
replay_settings = Settings(llm="replay", recording_dir=recording_dir)
replayed = run(replay.build_llm(replay_settings, "fingerprint"), replay_settings)
print(json.dumps({
    "recorded": recorded,
    "replayed": replayed,
    "requests": client.requests,
    "attempted": sorted(set(attempted_imports)),
    "outside": sorted(
        {name.partition(".")[0] for name in sys.modules}
        - set(sys.stdlib_module_names)
        - {"titlestd", "__main__"}
    ),
}))
"""


def test_record_and_replay_need_nothing_installed(bare_python, import_guard, mini_dir, tmp_path):
    result = bare_python(import_guard + ROUND_TRIP, str(mini_dir), str(tmp_path / "recordings"))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    # 2.5, S09: with a client handed in the SDK is not even looked for, and
    # replay never looks for it.
    assert report["attempted"] == [] and report["outside"] == []
    assert len(report["recorded"]) == 13 and report["requests"] == 13 * 3
    for recorded, replayed in zip(report["recorded"], report["replayed"], strict=True):
        assert recorded["labels"] == ["live"]
        assert replayed["labels"][0].startswith("recorded 20")
        assert replayed["labels"][0].endswith(", test-double")
        assert {**replayed, "labels": None} == {**recorded, "labels": None}
    assert {decided["agent"][0] for decided in report["recorded"]} == {"escalate"}
    assert (tmp_path / "recordings" / "test-double" / "offline" / MANIFEST_FILE).is_file()


# Run after the import guard: build live mode the way a real run does, with no
# client handed in, and report how it stopped and what the guard saw.
START_UP = """
import json

from titlestd import live
from titlestd.settings import Settings

settings = Settings(llm="anthropic", llm_model="claude-haiku-4-5", recording_dir=sys.argv[1])
try:
    live.build_llm(settings, "fingerprint")
except Exception as error:
    print(json.dumps({
        "error": [kind.__name__ for kind in type(error).__mro__],
        "message": str(error),
        "attempted": attempted_imports,
    }))
"""


def test_without_the_extra_live_mode_stops_with_the_command_of_s09(
    bare_python, import_guard, tmp_path, monkeypatch
):
    monkeypatch.setenv(API_KEY_VARIABLE, NOT_A_KEY)
    result = bare_python(import_guard + START_UP, str(tmp_path / "recordings"))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert "TitlestdError" in report["error"]
    assert "needs the anthropic library, which cannot be imported" in report["message"]
    assert f"{run_with_extra('live')} record --llm-model claude-haiku-4-5" in report["message"]
    assert NOT_A_KEY not in result.stdout + result.stderr
    # The SDK was looked for once, here, and nothing else from outside.
    assert report["attempted"] == ["anthropic"]
    assert not (tmp_path / "recordings").exists()


def test_without_the_key_live_mode_stops_before_it_looks_for_the_sdk(
    bare_python, import_guard, tmp_path
):
    result = bare_python(import_guard + START_UP, str(tmp_path / "recordings"))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["error"][:2] == ["LiveError", "TitlestdError"]
    assert "ANTHROPIC_API_KEY, which is not set or is empty" in report["message"]
    assert report["attempted"] == []  # Q39: the key first, only then the SDK


def test_replay_of_no_recording_stops_at_start_up_through_the_switch(mini_inputs, tmp_path):
    chosen = Settings(llm="replay", recording_dir=tmp_path / "recordings")
    with pytest.raises(RecordingError, match="found no recording") as raised:
        switch.build_models(chosen, mini_inputs.catalog.ladder, mini_inputs.abbreviations, "f")
    assert "--llm-model" in str(raised.value)


def test_live_mode_without_a_key_stops_at_start_up_through_the_switch(mini_inputs, settings):
    with pytest.raises(LiveError, match=API_KEY_VARIABLE):
        switch.build_models(settings, mini_inputs.catalog.ladder, mini_inputs.abbreviations, "f")
    assert not settings.recording_dir.exists()


# --- Tier (c): the real SDK client on a mock transport ------------------------------------------


class Sdk:
    """The SDK and its HTTP library, where the "live" extra is installed."""

    def __init__(self) -> None:
        self.anthropic = pytest.importorskip("anthropic")
        self.httpx2 = pytest.importorskip("httpx2")
        self.requests: list = []

    def client(self, respond):
        """A real client whose requests are answered here, never by the network.
        respond takes the request body and returns a message, or a pair of an
        HTTP status and an error body."""

        def handler(request):
            self.requests.append(request)
            answer = respond(json.loads(request.content))
            status, data = answer if isinstance(answer, tuple) else (200, answer)
            return self.httpx2.Response(status, json=data)

        return self.anthropic.Anthropic(
            api_key=NOT_A_KEY,
            max_retries=0,  # a test does not wait for the back-off of a retry
            http_client=self.anthropic.DefaultHttpxClient(
                transport=self.httpx2.MockTransport(handler)
            ),
        )


@pytest.fixture
def sdk() -> Sdk:
    return Sdk()


def test_a_real_run_builds_the_sdk_s_own_client_from_the_environment(sdk, settings, monkeypatch):
    # 2.5, Q39: with no client handed in, the SDK is imported and its client
    # built with no arguments; it reads the key itself and keeps its defaults,
    # the two retries among them. Building it sends nothing.
    built: list = []

    class Spying(sdk.anthropic.Anthropic):
        def __init__(self, *args, **kwargs) -> None:
            built.append((args, kwargs))
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(sdk.anthropic, "Anthropic", Spying)
    monkeypatch.setenv(API_KEY_VARIABLE, NOT_A_KEY)
    llm = live.build_llm(settings, FINGERPRINT)
    arguments, client = built
    assert arguments == ((), {})
    assert (client.api_key, client.max_retries) == (NOT_A_KEY, 2)
    assert llm.reviewer is llm.chooser
    assert not settings.recording_dir.exists()


def test_the_sdk_sends_exactly_what_the_scripted_client_is_given(
    sdk, world, settings, miniature, standin_run, tmp_path
):
    llm = live.build_llm(settings, FINGERPRINT, client=sdk.client(miniature.respond))
    outcomes = run(llm, world, settings)
    llm.finish()
    assert unlabelled(outcomes) == unlabelled(standin_run[0])
    assert labels(outcomes) == {LABEL_LIVE}

    # The same run with the scripted object: the SDK adds nothing to a request
    # and takes nothing away, so tier (b) tests what would be sent.
    other = Settings(llm="anthropic", llm_model=MODEL, recording_dir=tmp_path / "scripted")
    scripted = ModelInMiniature(standin_run[1])
    record_a_run(world, other, scripted)
    assert miniature.bodies == scripted.bodies
    for name in (CHOOSE_FILE, REVIEW_FILE):
        with_sdk = (settings.recording_dir / MODEL / "offline" / name).read_text(encoding="utf-8")
        assert with_sdk == (other.recording_dir / MODEL / "offline" / name).read_text(
            encoding="utf-8"
        )

    again = replay_settings(settings)
    replayed = run(replay.build_llm(again, FINGERPRINT), world, again)
    assert unlabelled(replayed) == unlabelled(outcomes) and labels(replayed) == {RECORDED}


def test_the_requests_have_the_shape_the_record_fixes(sdk, world, tmp_path, standin_run):
    # Q20, Q31, Library facts. The model is the alias exactly as written.
    chosen = Settings(
        llm="anthropic", llm_model="claude-haiku-4-5", recording_dir=tmp_path / "recordings"
    )
    model = ModelInMiniature(standin_run[1])
    llm = live.build_llm(chosen, FINGERPRINT, client=sdk.client(model.respond))
    run(llm, world, chosen)

    assert len(sdk.requests) == len(model.bodies) == 64
    for request in sdk.requests:
        assert (request.method, request.url.path) == ("POST", "/v1/messages")
    chooses = [body for body in model.bodies if "tools" not in body]
    reviews = [body for body in model.bodies if "tools" in body]
    assert (len(chooses), len(reviews)) == (16, 48)
    for body in model.bodies:
        assert body["model"] == "claude-haiku-4-5"
        assert body["max_tokens"] == recording.MAX_TOKENS
        assert set(body["output_config"]) == {"format"}
        assert body["output_config"]["format"]["type"] == "json_schema"
        # Q31: no sampling parameter, no thinking, no effort, nothing streamed.
        assert set(body) <= {
            "model",
            "max_tokens",
            "system",
            "messages",
            "output_config",
            "tools",
            "tool_choice",
        }
    for body in chooses:
        assert set(body) == {"model", "max_tokens", "system", "messages", "output_config"}
        assert body["system"] == recording.CHOOSE_SYSTEM
        assert body["output_config"]["format"]["schema"] == recording.CHOOSE_SCHEMA
        assert [each["role"] for each in body["messages"]] == ["user"]
    for body in reviews:
        assert body["system"] == recording.REVIEW_SYSTEM
        assert body["output_config"]["format"]["schema"] == recording.REVIEW_SCHEMA
        assert [tool["name"] for tool in body["tools"]] == [
            "search_catalog",
            "search_approved_titles",
        ]
        for tool in body["tools"]:
            assert tool["strict"] is True
            assert tool["input_schema"]["additionalProperties"] is False
        # One tool call a turn at most, and never a forced one.
        assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        roles = [each["role"] for each in body["messages"]]
        assert roles == ["user", "assistant", "user", "assistant", "user"][: len(roles)]


def test_after_four_searches_the_sdk_sends_tool_choice_none(sdk, world, settings, standin_run):
    # Q20: at most five requests per case, the fifth with the tools withdrawn.
    def searches_on(body: dict) -> dict:
        if body["tool_choice"] == {"type": "none"}:
            fields = {"answer": "escalate", "number": None, "quote": ""}
            return message([{"type": "text", "text": json.dumps(fields)}], "end_turn")
        number = len(body["messages"]) // 2 + 1
        call = {"type": "tool_use", "id": f"toolu_{number}", "name": "search_catalog"}
        return message([{**call, "input": {"query": f"analyst {number}"}}], "tool_use")

    llm = live.build_llm(settings, FINGERPRINT, client=sdk.client(searches_on))
    row = TREASURY_IN_SALES
    match = Matcher(world.catalog, world.embedder).match([row], world.knowledge)[row.key]
    shortlist = [catalog.get_role(world.catalog, each.role_id) for each in match.shortlist]
    shown = fallback.number_roles(shortlist, row.key, settings.seed)
    llm_answer = standin_run[0][row.key].llm_answer
    case = agent.build_case(row, shown, llm_answer, "FIN-06", match, (Reason.FAMILY_MISMATCH,))
    tools = agent.SearchTools(world.catalog, world.knowledge, world.embedder, settings)

    result = agent.review(case, tools, llm.reviewer)
    assert result.answer == ReviewAnswer(ReviewKind.ESCALATE, None, "", LABEL_LIVE)
    assert [step.request.query for step in result.transcript.steps] == [
        "analyst 1",
        "analyst 2",
        "analyst 3",
        "analyst 4",
    ]
    bodies = [json.loads(request.content) for request in sdk.requests]
    assert len(bodies) == 5
    assert [body["tool_choice"]["type"] for body in bodies] == ["auto"] * 4 + ["none"]
    assert [len(body["messages"]) for body in bodies] == [1, 3, 5, 7, 9]
    assert all(len(body["tools"]) == 2 for body in bodies)


def test_no_key_reaches_a_recording(sdk, world, settings, miniature):
    llm = live.build_llm(settings, FINGERPRINT, client=sdk.client(miniature.respond))
    run(llm, world, settings)
    llm.finish()
    # The key was in play: the SDK sent it with every request.
    assert {request.headers["x-api-key"] for request in sdk.requests} == {NOT_A_KEY}
    written = [path for path in settings.recording_dir.rglob("*") if path.is_file()]
    assert sorted(path.name for path in written) == [CHOOSE_FILE, MANIFEST_FILE, REVIEW_FILE]
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert KEY_PREFIX not in text and "x-api-key" not in text and "authorization" not in text
    # An entry holds its twelve fields and no header, request id or message id.
    for entry in lines(settings.recording_dir / MODEL / "offline" / REVIEW_FILE):
        assert len(entry) == 12 and "msg_test" not in json.dumps(entry)


def test_what_the_api_reports_as_the_model_is_kept_for_information_only(sdk, world, tmp_path):
    # Owner decision 8: the alias reports a dated ID; the key and the label
    # use the configured string.
    chosen = Settings(
        llm="anthropic", llm_model="claude-haiku-4-5", recording_dir=tmp_path / "recordings"
    )
    fields = {"answer": "no_match", "number": None, "quote": "treasury analyst"}
    said = message([{"type": "text", "text": json.dumps(fields)}], "end_turn")
    said["model"] = "claude-haiku-4-5-20251001"
    llm = live.build_llm(chosen, FINGERPRINT, client=sdk.client(lambda body: said))
    record, roles = TREASURY_IN_SALES.record, ()
    assert llm.chooser.choose(record, roles).answer is ChooseKind.NO_MATCH
    llm.finish()
    folder = chosen.recording_dir / "claude-haiku-4-5" / "offline"
    (entry,) = lines(folder / CHOOSE_FILE)
    assert (entry["model"], entry["reported_model"]) == (
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
    )
    assert entry["key"] == recording.request_key(
        "choose", "claude-haiku-4-5", recording.choose_request(record, roles)
    )
    again = Settings(llm="replay", recording_dir=chosen.recording_dir)
    repeated = replay.build_llm(again, FINGERPRINT).chooser.choose(record, roles)
    assert repeated.source == "recorded 2026-01-05, claude-haiku-4-5"


@pytest.mark.parametrize(
    ("stop_reason", "more", "detail"),
    [
        (
            "refusal",
            {"stop_details": {"type": "refusal", "category": "cyber", "explanation": "No."}},
            "the model refused to answer (cyber)",
        ),
        ("refusal", {"stop_details": None}, "the model refused to answer"),
        ("refusal", {}, "the model refused to answer"),
        ("max_tokens", {}, "the answer was cut off at max_tokens = 1024"),
        ("pause_turn", {}, "the model stopped for the reason 'pause_turn'"),
    ],
)
def test_a_response_of_the_sdk_that_holds_no_answer_is_invalid(
    sdk, settings, stop_reason, more, detail
):
    cut = [{"type": "text", "text": '{"answer": "role", "numb'}]
    client = sdk.client(lambda body: message(cut, stop_reason, **more))
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    answer = llm.chooser.choose(TREASURY_IN_SALES.record, ())
    assert answer == ChooseAnswer(ChooseKind.INVALID, None, "", LABEL_LIVE, detail)
    (entry,) = lines(settings.recording_dir / MODEL / "offline.partial" / CHOOSE_FILE)
    assert (entry["stop_reason"], entry["turn"]["detail"]) == (stop_reason, detail)


def error_body(kind: str, text: str) -> dict:
    return {"type": "error", "error": {"type": kind, "message": text}}


@pytest.mark.parametrize(
    ("status", "kind", "said"),
    [
        (401, "authentication_error", "rejected the key in ANTHROPIC_API_KEY (status 401): Auth"),
        (404, "not_found_error", "does not know the model 'test-double', or the key may not"),
        (400, "invalid_request_error", "answered a request with status 400: BadRequestError"),
        (429, "rate_limit_error", "with status 429, after the SDK's own retries: RateLimitError"),
        (500, "api_error", "with status 500, after the SDK's own retries: InternalServerError"),
        (529, "overloaded_error", "with status 529, after the SDK's own retries: "),
    ],
)
def test_an_error_status_of_the_api_stops_the_run(sdk, settings, status, kind, said):
    client = sdk.client(lambda body: (status, error_body(kind, "refused by the test")))
    llm = live.build_llm(settings, FINGERPRINT, client=client)
    with pytest.raises(LiveError) as raised:
        llm.chooser.choose(TREASURY_IN_SALES.record, ())
    message_text = str(raised.value)
    assert said in message_text and "refused by the test" in message_text
    assert "no answer was recorded and nothing was written" in message_text
    assert isinstance(raised.value.__cause__, sdk.anthropic.APIStatusError)
    assert NOT_A_KEY not in message_text
    assert not settings.recording_dir.exists()


def test_a_network_failure_stops_the_run(sdk, settings):
    def unreachable(body: dict):
        raise sdk.httpx2.ConnectError("no route to host")

    llm = live.build_llm(settings, FINGERPRINT, client=sdk.client(unreachable))
    with pytest.raises(LiveError) as raised:
        llm.reviewer.next_turn(AgentTranscript(_a_case(TREASURY_IN_SALES)))
    assert "a request to the hosted API got no answer, after the SDK's own retries: " in str(
        raised.value
    )
    assert isinstance(raised.value.__cause__, sdk.anthropic.APIConnectionError)
    assert not settings.recording_dir.exists()


def _a_case(row: TitleRow):
    answer = ChooseAnswer(ChooseKind.NO_MATCH, None, row.record.cleaned_title, LABEL_LIVE)
    return agent.AgentCase(row.record, (), answer, None, None, (Reason.NO_MATCH,))
