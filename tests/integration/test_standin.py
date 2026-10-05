"""The stand-ins on the mini data set, and "stand-in models never read the truth file".

docs/design.md 2.4; docs/build-decisions.md Q15, Q17, Q19 and the behavioural
test (a) of Q22: the stand-in embed, choose and review return identical output
for identical arguments, including with the truth file absent.

The modules that will call the stand-ins are built later: the nearest-neighbour
match (titlestd.matcher), the shuffle, the record block and the quote check
(titlestd.fallback), the agent's loop and searches (titlestd.agent). This file
does their part in miniature, by the rules the decisions record gives them, on
the real readers, the real seeding and the real offline embedder. No expected
answer here comes from the truth file: each is written out by hand. The truth
file is only taken away or rewritten, with the scaffold's reader and writer.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from titlestd import datafiles, people, seeding, switch
from titlestd.contracts import (
    LABEL_STAND_IN,
    QUOTE_MAX_WORDS,
    QUOTE_MIN_CHARS,
    SHORTLIST_SIZE,
    AgentCase,
    AgentTranscript,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    InputData,
    Models,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    TitleRow,
    Vector,
)
from titlestd.settings import Settings

# What the stand-in LLM should answer for each model-eligible mini row: the
# kind of answer, the role and the quote, worked out by hand from Q17.
EXPECTED = {
    ("FIN", "accounts payable clerk"): ("role", "FIN-05", "accounts payable clerk"),
    ("FIN", "financial analyst"): ("role", "FIN-01", "financial analyst"),
    ("FIN", "internal auditor"): ("role", "FIN-04", "internal auditor"),
    ("FIN", "payroll specialist"): ("role", "FIN-03", "payroll specialist"),
    ("FIN", "senior financial analyst"): ("role", "FIN-01", "financial analyst"),
    ("FIN", "special projects"): ("no_match", None, "special projects"),
    ("FIN", "treasury analyst"): ("role", "FIN-06", "treasury analyst"),
    ("SAL", "account manager ii"): ("role", "SAL-01", "account manager"),
    ("SAL", "bdr"): ("insufficient_data", None, "bdr"),
    ("SAL", "sales engineer"): ("role", "SAL-04", "sales engineer"),
    ("SAL", "sales operations analyst"): ("role", "SAL-03", "sales operations analyst"),
    ("SAL", "sales representative"): ("role", "SAL-02", "sales representative"),
    ("SAL", "senior account manager"): ("role", "SAL-01", "account manager"),
}

# Titles a person has already decided, as the agent's second search finds them:
# the mini seed look-up, cleaned, and one Wave 1 approval.
APPROVED = (
    ("accounts payable clerk", "FIN", "FIN-05"),
    ("sales engineer", "SAL", "SAL-04"),
    ("senior account manager", "SAL", "SAL-01"),
)

# A family mismatch (1.7): a finance title recorded in the sales family.
MISFILED = TitleRow(
    record=Record(
        "financial analyst",
        "SAL",
        "Sales Operations",
        "Builds budgets and forecasts for the sales teams.",
    ),
    employee_ids=("X001",),
    raw_titles=("Financial Analyst",),
)


# --- The callers of the stand-ins, in miniature -----------------------------------


def dot(a: Vector, b: Vector) -> float:
    return math.fsum(weight * b.get(index, 0.0) for index, weight in a.items())


def role_text(role: CatalogRole) -> str:
    return f"{role.role_name}. {role.definition}"


def record_text(record: Record) -> str:
    return f"{record.cleaned_title} {record.description}".strip()


def nearest(models: Models, query: str, texts: Sequence[str]) -> list[int]:
    """The positions of texts, nearest to the query first, ties by position."""
    target, *vectors = models.embedder.embed([query, *texts])
    return sorted(range(len(texts)), key=lambda at: (-dot(target, vectors[at]), at))


def visible_roles(data: InputData) -> list[CatalogRole]:
    """The roles a model may be shown, by role_id: never a generic family role."""
    return sorted(
        (role for role in data.catalog.roles if not role.is_generic), key=lambda role: role.role_id
    )


def shown_to_the_llm(
    models: Models, row: TitleRow, roles: Sequence[CatalogRole], seed: int
) -> list[NumberedRole]:
    """The ten nearest roles, sorted by role_id, shuffled per title, numbered (Q18)."""
    order = nearest(models, record_text(row.record), [role_text(role) for role in roles])
    shortlist = sorted((roles[at] for at in order[:SHORTLIST_SIZE]), key=lambda r: r.role_id)
    seeding.rng(seed, seeding.STREAM_SHUFFLE, *row.key).shuffle(shortlist)
    return [NumberedRole(number, role) for number, role in enumerate(shortlist, start=1)]


def record_block(record: Record) -> str:
    """Each field verbatim on a labelled line of its own (S16)."""
    return (
        f"Title: {record.cleaned_title}\nFamily: {record.family_code}\n"
        f"Department: {record.department}\nDescription: {record.description}"
    )


def quote_in_record(quote: str, block: str) -> bool:
    """The quote check of Q15."""
    quote, block = (" ".join(text.casefold().split()) for text in (quote, block))
    if len(quote) < QUOTE_MIN_CHARS or not any(character.isalpha() for character in quote):
        return False
    if len(quote.split()) > QUOTE_MAX_WORDS:
        return False
    return re.search(rf"(?<!\w){re.escape(quote)}(?!\w)", block) is not None


@dataclasses.dataclass
class AgentRun:
    transcripts: list[AgentTranscript]  # what each turn was given
    turns: list[SearchRequest | ReviewAnswer]  # what each turn returned

    @property
    def answer(self) -> ReviewAnswer:
        return self.turns[-1]

    @property
    def role_id(self) -> str | None:
        by_number = {shown.number: shown.role for shown in self.transcripts[-1].numbered_roles}
        return by_number[self.answer.number].role_id if self.answer.number else None


def run_agent(
    models: Models,
    row: TitleRow,
    llm_roles: Sequence[NumberedRole],
    llm_answer: ChooseAnswer,
    roles: Sequence[CatalogRole],
    search_k: int,
) -> AgentRun:
    """The loop of Q20 and the two searches and the numbering of Q10."""
    record = row.record
    by_id = {role.role_id: role for role in roles}
    numbers: dict[str, int] = {}

    def show(role: CatalogRole) -> NumberedRole:
        if role.family_code != record.family_code:
            return NumberedRole(None, role)
        return NumberedRole(numbers.setdefault(role.role_id, len(numbers) + 1), role)

    llm_role_id = next((n.role.role_id for n in llm_roles if n.number == llm_answer.number), None)
    case = AgentCase(
        record=record,
        roles=tuple(show(shown.role) for shown in llm_roles),
        llm_answer=llm_answer,
        llm_role_id=llm_role_id,
        ml_first_choice=None,
        reasons=(Reason.MODELS_DISAGREE,),
    )
    transcript = AgentTranscript(case)
    run = AgentRun([], [])
    while True:
        turn = models.llm.reviewer.next_turn(transcript)
        run.transcripts.append(transcript)
        run.turns.append(turn)
        if isinstance(turn, ReviewAnswer):
            return run
        assert transcript.searches_left > 0, "a search after the cap"
        if turn.tool is SearchTool.CATALOG:
            found = nearest(models, turn.query, [role_text(role) for role in roles])[:search_k]
            hits = tuple(SearchHit(show(roles[at])) for at in found)
        else:
            found = nearest(models, turn.query, [title for title, _, _ in APPROVED])[:search_k]
            hits = tuple(
                SearchHit(show(by_id[APPROVED[at][2]]), APPROVED[at][0], APPROVED[at][1])
                for at in found
            )
        transcript = AgentTranscript(case, (*transcript.steps, SearchStep(turn, hits)))


@dataclasses.dataclass
class Everything:
    """Every call made to the three stand-ins for the mini rows, and its result."""

    texts: list[str]
    vectors: list[Vector]
    records: dict[tuple[str, str], Record]
    chosen: dict[tuple[str, str], tuple[list[NumberedRole], ChooseAnswer]]
    agents: dict[tuple[str, str], AgentRun]

    def plain(self) -> dict:
        """Arguments and results as plain data, the same after a trip through JSON."""
        choose_calls, choose_results = [], []
        for key, (shown, answer) in self.chosen.items():
            choose_calls.append(
                {
                    "record": dataclasses.asdict(self.records[key]),
                    "roles": [dataclasses.asdict(numbered) for numbered in shown],
                }
            )
            choose_results.append(dataclasses.asdict(answer))
        review_calls, review_results = [], []
        for run in self.agents.values():
            review_calls += [dataclasses.asdict(transcript) for transcript in run.transcripts]
            review_results += [dataclasses.asdict(turn) for turn in run.turns]
        plain = {
            "arguments": {"embed": self.texts, "choose": choose_calls, "review": review_calls},
            "results": {
                "embed": [sorted(vector.items()) for vector in self.vectors],
                "choose": choose_results,
                "review": review_results,
            },
        }
        return json.loads(json.dumps(plain))


def run_everything(data_dir: Path, rows: Sequence[TitleRow]) -> Everything:
    """Read the six input files in data_dir, build the default models with the
    switch and put every model-eligible row through embed, choose and review."""
    data = datafiles.read_inputs(data_dir)
    settings = Settings(data_dir=data_dir)
    models = switch.build_models(
        settings, data.catalog.ladder, data.abbreviations, datafiles.input_fingerprint(data_dir)
    )
    roles = visible_roles(data)
    eligible = [row for row in rows if row.model_eligible]
    texts = [role_text(role) for role in roles] + [record_text(row.record) for row in eligible]
    everything = Everything(texts, models.embedder.embed(texts), {}, {}, {})
    for row in eligible:
        shown = shown_to_the_llm(models, row, roles, settings.seed)
        answer = models.llm.chooser.choose(row.record, shown)
        everything.records[tuple(row.key)] = row.record
        everything.chosen[tuple(row.key)] = (shown, answer)
        everything.agents[tuple(row.key)] = run_agent(
            models, row, shown, answer, roles, settings.agent_search_k
        )
    return everything


@pytest.fixture(scope="module")
def everything(mini_dir: Path, mini_rows: tuple[TitleRow, ...]) -> Everything:
    return run_everything(mini_dir, [*mini_rows, MISFILED])


# --- The stand-in LLM over the mini rows (Q17) -------------------------------------


def test_the_table_covers_every_model_eligible_row(mini_rows, everything):
    eligible = [tuple(row.key) for row in mini_rows if row.model_eligible]
    assert sorted(EXPECTED) == eligible
    assert sorted(everything.chosen) == sorted([*eligible, tuple(MISFILED.key)])


@pytest.mark.parametrize("key", sorted(EXPECTED), ids=" / ".join)
def test_the_stand_in_llm_answers_each_mini_row_as_worked_out_by_hand(everything, key):
    shown, answer = everything.chosen[key]
    kind, role_id, quote = EXPECTED[key]
    assert answer.answer == kind
    picked = next((n.role.role_id for n in shown if n.number == answer.number), None)
    assert (picked, answer.quote) == (role_id, quote)
    assert answer.source == LABEL_STAND_IN


def test_the_ten_roles_were_shuffled_so_the_answers_did_not_come_from_the_order(everything):
    numbers = {answer.number for _, answer in everything.chosen.values() if answer.number}
    assert len(numbers) >= 4
    for shown, _ in everything.chosen.values():
        assert len(shown) == SHORTLIST_SIZE
        assert not any(numbered.role.is_generic for numbered in shown)


def test_every_quote_passes_the_shared_quote_check(everything):
    # No fault injection (Q17): the stand-in never fails the check by itself.
    for key, (_, answer) in everything.chosen.items():
        assert quote_in_record(answer.quote, record_block(everything.records[key])), key
    for key, run in everything.agents.items():
        if run.answer.answer is ReviewKind.ROLE:
            assert quote_in_record(run.answer.quote, record_block(everything.records[key])), key


def test_the_three_kinds_of_answer_all_occur(everything):
    kinds = {answer.answer for _, answer in everything.chosen.values()}
    assert kinds == {ChooseKind.ROLE, ChooseKind.NO_MATCH, ChooseKind.INSUFFICIENT_DATA}


def test_the_two_opinions_are_different_calculations(mini_inputs: InputData):
    # Q12: the embedder must not be the word-overlap stand-in again, or the
    # agreement of 1.1 would be one method agreeing with itself. Two words
    # that share most of their letters are close for the embedder and have
    # nothing in common for the stand-in.
    models = switch.build_models(Settings(), mini_inputs.catalog.ladder, mini_inputs.abbreviations)
    words = models.llm.chooser.content_words
    assert words("accountant") & words("accounting") == set()
    one, other = models.embedder.embed(["accountant", "accounting"])
    assert dot(one, other) > 0.5


# --- The stand-in agent over the mini rows (Q19) -----------------------------------


def test_the_agent_runs_its_script_through_the_capped_searches(everything):
    for key, run in everything.agents.items():
        *requests, answer = run.turns
        assert isinstance(answer, ReviewAnswer)
        assert requests[:2] == [
            SearchRequest(SearchTool.CATALOG, key[1]),
            SearchRequest(SearchTool.APPROVED_TITLES, key[1]),
        ]
        # Ten shortlisted roles of twelve always show two of the family: no third search.
        assert len(requests) == 2, key


def test_the_agent_confirms_what_the_llm_picked_on_every_plain_row(everything):
    for key, (kind, role_id, quote) in EXPECTED.items():
        run = everything.agents[key]
        if kind == "role":
            assert (run.answer.answer, run.role_id, run.answer.quote) == (
                ReviewKind.ROLE,
                role_id,
                quote,
            ), key


def test_the_agent_escalates_what_shares_no_word_with_any_role(everything):
    for key in (("FIN", "special projects"), ("SAL", "bdr")):
        assert everything.agents[key].answer == ReviewAnswer(
            ReviewKind.ESCALATE, None, "", LABEL_STAND_IN
        )


def test_on_a_family_mismatch_the_agent_corrects_to_a_role_of_the_recorded_family(everything):
    key = tuple(MISFILED.key)
    shown, llm_answer = everything.chosen[key]
    llm_role = next(n.role for n in shown if n.number == llm_answer.number)
    assert (llm_role.role_id, llm_role.family_code) == ("FIN-01", "FIN")  # outside the family
    run = everything.agents[key]
    assert (run.answer.answer, run.role_id, run.answer.quote) == (
        ReviewKind.ROLE,
        "SAL-03",
        "analyst",
    )


def test_the_agent_only_ever_picks_a_numbered_role_of_the_recorded_family(everything):
    for key, run in everything.agents.items():
        final = run.transcripts[-1]
        assert all(
            shown.role.family_code == key[0] and not shown.role.is_generic
            for shown in final.numbered_roles
        )
        if run.answer.answer is ReviewKind.ROLE:
            assert run.answer.number in {shown.number for shown in final.numbered_roles}


# --- Behavioural test (a) of Q22 ----------------------------------------------------


def copy_of_the_inputs(mini_dir: Path, target: Path) -> Path:
    """The six input files, and not the truth file."""
    target.mkdir()
    for name in datafiles.INPUT_FILES:
        shutil.copyfile(mini_dir / name, target / name)
    return target


def test_identical_arguments_give_identical_output(mini_dir, mini_rows, everything):
    again = run_everything(mini_dir, [*mini_rows, MISFILED])
    assert again.plain() == everything.plain()
    assert again.vectors == everything.vectors


def test_the_output_is_the_same_with_the_truth_file_absent(
    mini_dir, mini_rows, everything, tmp_path, monkeypatch
):
    bare = copy_of_the_inputs(mini_dir, tmp_path / "data")
    assert (mini_dir / people.TRUTH_FILE).is_file() and not (bare / people.TRUTH_FILE).exists()
    monkeypatch.chdir(tmp_path)  # so that a relative "data" folder has no truth file either
    without = run_everything(bare, [*mini_rows, MISFILED])
    assert without.plain() == everything.plain()
    assert sorted(path.name for path in bare.iterdir()) == sorted(datafiles.INPUT_FILES)


def test_the_output_is_the_same_whatever_the_truth_file_says(
    mini_dir, mini_rows, everything, tmp_path, monkeypatch
):
    # Every title's true role replaced by another: a stand-in that peeked
    # would now answer differently.
    other = copy_of_the_inputs(mini_dir, tmp_path / "data")
    truth = people.read_truth(mini_dir)
    distinct = sorted({row.true_role_id for row in truth})
    another = dict(zip(distinct, distinct[1:] + distinct[:1], strict=True))
    assert len(distinct) > 1 and all(another[role] != role for role in distinct)
    people.write_truth(
        other, [dataclasses.replace(row, true_role_id=another[row.true_role_id]) for row in truth]
    )
    assert people.read_truth(other) != truth
    monkeypatch.chdir(tmp_path)
    assert run_everything(other, [*mini_rows, MISFILED]).plain() == everything.plain()


# Run in another process: rebuild the recorded arguments from JSON, make the
# same calls on freshly built stand-ins, and report the results with every file
# the process opened that is not Python source.
REPLAY_THE_CALLS = """
import dataclasses, json, os, sys

opened = []


def watch(event, arguments):
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        path = arguments[0] if isinstance(arguments[0], str) else arguments[0].decode()
        if ".py" not in os.path.basename(path):
            opened.append(path)
    elif event in ("sqlite3.connect", "socket.connect", "subprocess.Popen"):
        opened.append(event)


sys.addaudithook(watch)

from titlestd import switch
from titlestd.contracts import (
    Abbreviation, AgentCase, AgentTranscript, CatalogRole, ChooseAnswer, ChooseKind, LadderLevel,
    NumberedRole, Reason, Record, SearchHit, SearchRequest, SearchStep, SearchTool,
)
from titlestd.settings import Settings

given = json.load(sys.stdin)


def numbered(plain):
    return NumberedRole(plain["number"], CatalogRole(**plain["role"]))


def transcript(plain):
    case = plain["case"]
    llm = case["llm_answer"]
    return AgentTranscript(
        AgentCase(
            Record(**case["record"]),
            tuple(numbered(role) for role in case["roles"]),
            ChooseAnswer(ChooseKind(llm["answer"]), llm["number"], llm["quote"], llm["source"]),
            case["llm_role_id"],
            case["ml_first_choice"],
            tuple(Reason(reason) for reason in case["reasons"]),
        ),
        tuple(
            SearchStep(
                SearchRequest(SearchTool(step["request"]["tool"]), step["request"]["query"]),
                tuple(
                    SearchHit(
                        numbered(hit["role"]), hit["approved_title"], hit["title_family_code"]
                    )
                    for hit in step["hits"]
                ),
            )
            for step in plain["steps"]
        ),
    )


models = switch.build_models(
    Settings(data_dir=given["data_dir"]),
    [LadderLevel(**level) for level in given["ladder"]],
    [Abbreviation(**abbreviation) for abbreviation in given["abbreviations"]],
)
calls = given["arguments"]
results = {
    "embed": [sorted(vector.items()) for vector in models.embedder.embed(calls["embed"])],
    "choose": [
        dataclasses.asdict(
            models.llm.chooser.choose(
                Record(**call["record"]), [numbered(role) for role in call["roles"]]
            )
        )
        for call in calls["choose"]
    ],
    "review": [
        dataclasses.asdict(models.llm.reviewer.next_turn(transcript(call)))
        for call in calls["review"]
    ],
}
models.llm.finish()
print(json.dumps({"results": results, "opened": opened}))
"""


@pytest.mark.parametrize(
    ("hash_seed", "truth"), [("0", "present"), ("1", "absent"), ("12345", "absent")]
)
def test_another_process_gives_the_same_output_and_opens_no_file(
    repo_root, mini_dir, mini_inputs, everything, tmp_path, hash_seed, truth
):
    # The process works in the data folder itself, once with the truth file
    # beside it and twice without. It is handed its arguments and nothing else.
    data_dir = mini_dir if truth == "present" else copy_of_the_inputs(mini_dir, tmp_path / "data")
    assert (data_dir / people.TRUTH_FILE).exists() == (truth == "present")
    plain = everything.plain()
    given = {
        "data_dir": str(data_dir),
        "ladder": [dataclasses.asdict(level) for level in mini_inputs.catalog.ladder],
        "abbreviations": [dataclasses.asdict(each) for each in mini_inputs.abbreviations],
        "arguments": plain["arguments"],
    }
    result = subprocess.run(
        [sys.executable, "-c", REPLAY_THE_CALLS],
        input=json.dumps(given),
        cwd=data_dir,
        env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["opened"] == []  # not the truth file, and no other file either
    assert report["results"] == plain["results"]
    assert len(plain["arguments"]["review"]) == len(plain["results"]["review"]) >= 3 * 14
