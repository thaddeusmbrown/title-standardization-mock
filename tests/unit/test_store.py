"""The state file: what each function of titlestd.store writes and reads back.

Everything here is built by hand from the plain types of titlestd.contracts,
so the tests say what a table holds without running a pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from titlestd import store
from titlestd.contracts import (
    AgentCase,
    AgentResult,
    AgentTranscript,
    AuditOutcome,
    AuditResult,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    Decision,
    DecisionSource,
    ExceptionReason,
    GoldenScore,
    HumanAnswer,
    HumanDecision,
    MachinePath,
    MatchResult,
    NumberedRole,
    QueueEntry,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
    RunResult,
    ScoredRole,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    Stage,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
)
from titlestd.lookup import Knowledge
from titlestd.settings import Settings

SAL_1 = CatalogRole("SAL-01", "SAL", "Account Manager", "Acct Mgr", "Manages accounts.")
SAL_2 = CatalogRole("SAL-02", "SAL", "Sales Representative", "Sales Rep", "Sells products.")
FIN_1 = CatalogRole("FIN-01", "FIN", "Financial Analyst", "Fin Analyst", "Builds budgets.")


def title_row(title: str, family: str = "SAL", employees: int = 1, **flags: bool) -> TitleRow:
    record = Record(title, family, "Enterprise Sales", "Looks after customer accounts.")
    ids = tuple(f"E{at}-{title}" for at in range(employees))
    return TitleRow(record, ids, (title.upper(),), **flags)


def read(connection: sqlite3.Connection, query: str) -> list[tuple]:
    return list(connection.execute(query))


@pytest.fixture
def connection(tmp_path: Path):
    opened = store.create_store(tmp_path / "out" / store.STATE_FILE)
    yield opened
    opened.close()


def test_create_store_makes_the_folder_and_an_empty_schema(tmp_path):
    path = tmp_path / "not" / "there" / "yet" / store.STATE_FILE
    connection = store.create_store(path)
    try:
        tables = {name for (name,) in read(connection, "SELECT name FROM sqlite_master")}
        assert {"settings", "thresholds", "title_rows", "lookup", "examples"} <= tables
        assert {"scorings", "ml_shortlist", "model_answers", "shown_roles"} <= tables
        assert {"human_decisions", "golden_scores", "audits", "decisions"} <= tables
        for table in sorted(name for name in tables if not name.startswith("sqlite_")):
            assert read(connection, f"SELECT count(*) FROM {table}") == [(0,)], table
    finally:
        connection.close()
    assert path.is_file()


def test_create_store_deletes_what_an_earlier_run_left(tmp_path):
    # Q06: the state is recreated at the start of every run.
    path = tmp_path / store.STATE_FILE
    first = store.create_store(path)
    store.save_people(
        first,
        "Wave 1 approval",
        [HumanDecision(TitleKey("SAL", "bdr"), HumanAnswer.ROLE, "SAL-05", "SME 1", True)],
    )
    first.close()
    second = store.create_store(path)
    try:
        assert read(second, "SELECT count(*) FROM human_decisions") == [(0,)]
    finally:
        second.close()


def test_the_state_file_is_named_as_the_record_says():
    assert store.STATE_FILE == "state.sqlite"  # Q06, Q38


def test_save_setup_records_settings_thresholds_rows_and_the_two_lists(connection):
    settings = Settings(seed=7, llm_model=None, release_stopped_paths=True)
    rows = [
        title_row("account manager", employees=3),
        title_row("bdr"),
        title_row("clerk [NAME]", has_person_name=True),
        title_row("staff accountant", family="", family_usable=False),
    ]
    store.save_setup(
        connection,
        settings,
        {"audit_confirm_limit": 2, "release_stopped_paths": True},
        rows,
        wave1_keys=[rows[0].key],
        golden_keys=[rows[1].key],
    )
    saved = dict(read(connection, "SELECT name, value FROM settings"))
    assert set(saved) == set(settings.as_dict())
    assert saved["seed"] == "7" and saved["embedder"] == "offline" and saved["out_dir"] == "out"
    # None is written empty and a flag as 1 or 0.
    assert saved["llm_model"] == "" and saved["release_stopped_paths"] == "1"
    assert saved["ml_alone"] == "0"
    assert dict(read(connection, "SELECT name, value FROM thresholds")) == {
        "audit_confirm_limit": "2",
        "release_stopped_paths": "1",
    }
    assert read(
        connection,
        "SELECT family_code, cleaned_title, headcount, has_person_name, family_usable, "
        "on_wave1_list, in_golden_set FROM title_rows ORDER BY family_code, cleaned_title",
    ) == [
        ("", "staff accountant", 1, 0, 0, 0, 0),
        ("SAL", "account manager", 3, 0, 1, 1, 0),
        ("SAL", "bdr", 1, 0, 1, 0, 1),
        ("SAL", "clerk [NAME]", 1, 1, 1, 0, 0),
    ]
    assert read(connection, "SELECT department, description FROM title_rows LIMIT 1") == [
        ("Enterprise Sales", "Looks after customer accounts.")
    ]


def test_save_knowledge_adds_what_is_new_and_keeps_the_stage_a_title_joined_at(connection):
    knowledge = Knowledge()
    knowledge.learn(title_row("sales engineer"), "SAL-04")
    store.save_knowledge(connection, Stage.SETUP, knowledge)
    knowledge.learn(title_row("account manager"), "SAL-01")
    # A title with a person's name joins the look-up table and never the examples (Q14).
    knowledge.learn(title_row("clerk [NAME]", has_person_name=True), "SAL-02")
    store.save_knowledge(connection, Stage.WAVE1, knowledge)
    store.save_knowledge(connection, Stage.CLOSE, knowledge)
    assert read(connection, "SELECT * FROM lookup ORDER BY cleaned_title") == [
        ("SAL", "account manager", "SAL-01", "Wave 1"),
        ("SAL", "clerk [NAME]", "SAL-02", "Wave 1"),
        ("SAL", "sales engineer", "SAL-04", "set-up"),
    ]
    assert read(connection, "SELECT * FROM examples ORDER BY cleaned_title") == [
        ("SAL", "account manager", "SAL-01", "Wave 1"),
        ("SAL", "sales engineer", "SAL-04", "set-up"),
    ]


MATCH = MatchResult(
    (ScoredRole("SAL-02", 0.5, "sales representative"), ScoredRole("SAL-01", 0.25, "Account")),
    first_choice_examples=3,
)
SHOWN = (NumberedRole(1, SAL_1), NumberedRole(2, FIN_1), NumberedRole(3, SAL_2))


def test_save_scoring_records_the_match_the_answer_and_every_role_shown(connection):
    key = TitleKey("SAL", "account manager ii")
    answer = ChooseAnswer(ChooseKind.ROLE, 1, "account manager", "stand-in")
    outcome = RuleOutcome(Route.AGENT, (Reason.MODELS_DISAGREE,), "SAL-01")
    case = AgentCase(
        Record("account manager ii", "SAL", "", ""),
        (NumberedRole(1, SAL_1), NumberedRole(None, FIN_1), NumberedRole(2, SAL_2)),
        answer,
        "SAL-01",
        "SAL-02",
        (Reason.MODELS_DISAGREE,),
    )
    step = SearchStep(
        SearchRequest(SearchTool.APPROVED_TITLES, "account manager ii"),
        (SearchHit(NumberedRole(2, SAL_2), "sales representative", "SAL"),),
    )
    review = AgentResult(
        ReviewAnswer(ReviewKind.ROLE, 2, "manager", "stand-in"),
        "SAL-02",
        AgentTranscript(case, (step,)),
    )
    store.save_scoring(
        connection,
        Stage.WAVE2,
        key,
        match=MATCH,
        route="agent-settled",
        reasons=(Reason.MODELS_DISAGREE,),
        shown=SHOWN,
        answer=answer,
        outcome=outcome,
        review=review,
        review_outcome=RuleOutcome(Route.ACCEPT, (), "SAL-02"),
    )
    # Margin and example count are kept for every title (Q04).
    assert read(connection, "SELECT * FROM scorings") == [
        (
            "Wave 2",
            "SAL",
            "account manager ii",
            "SAL-02",
            0.25,
            3,
            "agent-settled",
            "models disagree",
        )
    ]
    assert read(connection, "SELECT place, role_id, score, matched_text FROM ml_shortlist") == [
        (1, "SAL-02", 0.5, "sales representative"),
        (2, "SAL-01", 0.25, "Account"),
    ]
    assert read(
        connection,
        "SELECT step, answer, number, quote, source, detail, role_id, route, reasons "
        "FROM model_answers ORDER BY step",
    ) == [
        (
            "choose",
            "role",
            1,
            "account manager",
            "stand-in",
            "",
            "SAL-01",
            "agent",
            "models disagree",
        ),
        ("review", "role", 2, "manager", "stand-in", "", "SAL-02", "accept", ""),
    ]
    assert read(
        connection,
        "SELECT step, shown_in, query, place, number, role_id FROM shown_roles ORDER BY rowid",
    ) == [
        ("choose", "numbered list", "", 1, 1, "SAL-01"),
        ("choose", "numbered list", "", 2, 2, "FIN-01"),
        ("choose", "numbered list", "", 3, 3, "SAL-02"),
        ("review", "case", "", 1, 1, "SAL-01"),
        ("review", "case", "", 2, None, "FIN-01"),
        ("review", "case", "", 3, 2, "SAL-02"),
        ("review", "search_approved_titles", "account manager ii", 1, 2, "SAL-02"),
    ]


def test_save_scoring_on_the_ml_alone_route_has_no_answer_and_shows_no_role(connection):
    store.save_scoring(
        connection,
        Stage.WAVE2,
        TitleKey("SAL", "sales rep"),
        match=MATCH,
        route="person",
        reasons=(Reason.ML_ALONE, Reason.COMPLIANCE_FLAG),
    )
    assert read(connection, "SELECT route, reasons FROM scorings") == [
        ("person", "ML alone; compliance-flagged role")
    ]
    assert read(connection, "SELECT count(*) FROM ml_shortlist") == [(2,)]
    assert read(connection, "SELECT count(*) FROM model_answers") == [(0,)]
    assert read(connection, "SELECT count(*) FROM shown_roles") == [(0,)]


def test_a_title_is_scored_once_per_stage(connection):
    key = TitleKey("SAL", "sales rep")
    store.save_scoring(connection, Stage.WAVE1, key, match=MATCH, route="person", reasons=())
    store.save_scoring(connection, Stage.CLOSE, key, match=MATCH, route="person", reasons=())
    with pytest.raises(sqlite3.IntegrityError):
        store.save_scoring(connection, Stage.CLOSE, key, match=MATCH, route="person", reasons=())


def test_save_people_keeps_the_occasion_the_persona_and_the_simulated_mark(connection):
    key = TitleKey("FIN", "analyst")
    store.save_people(
        connection,
        "golden-set label",
        [HumanDecision(key, HumanAnswer.ROLE, "FIN-01", "SME 2 (simulated)", True)],
    )
    store.save_people(
        connection,
        "review-queue decision",
        [
            HumanDecision(key, HumanAnswer.UNDECIDED, None, "Reviewer (simulated)", True),
            HumanDecision(
                TitleKey("FIN", "odd job"), HumanAnswer.NO_SUITABLE_ROLE, None, "R", False
            ),
        ],
    )
    assert read(connection, "SELECT * FROM human_decisions ORDER BY occasion, cleaned_title") == [
        ("golden-set label", "FIN", "analyst", "role", "FIN-01", "SME 2 (simulated)", 1),
        (
            "review-queue decision",
            "FIN",
            "analyst",
            "could not decide",
            None,
            "Reviewer (simulated)",
            1,
        ),
        ("review-queue decision", "FIN", "odd job", "no suitable role", None, "R", 0),
    ]


def test_save_run_writes_the_decisions_the_audits_the_scores_and_the_queue(connection):
    kept = title_row("account manager", employees=2)
    stopped = title_row("sales rep")
    queued = title_row("odd job")
    decision = HumanDecision(queued.key, HumanAnswer.UNDECIDED, None, "Reviewer (simulated)", True)
    result = RunResult(
        rows=(kept, queued, stopped),
        employee_count=4,
        no_grade_employees=1,
        wave1_keys=(kept.key,),
        decisions=(
            Decision(
                key=kept.key,
                row=TableRow.MODELS_AGREE,
                source=DecisionSource.MACHINE,
                role_id="SAL-01",
                status=Status.PROVISIONAL,
                error_rate=0.25,
            ),
            Decision(
                key=queued.key,
                row=TableRow.GENERIC,
                source=DecisionSource.CODE,
                role_id="SAL-GEN",
                status=Status.PROVISIONAL,
            ),
            Decision(
                key=stopped.key,
                row=TableRow.LEGACY,
                source=DecisionSource.NONE,
                legacy_reason=ExceptionReason.STOPPED_PATH,
            ),
        ),
        audits=(
            AuditResult(MachinePath.MODELS_AGREE, 2, (kept.key,), 1, AuditOutcome.PROVISIONAL),
            AuditResult(MachinePath.AGENT_SETTLED, 0, (), 0, AuditOutcome.ALL_LABELED),
        ),
        golden_keys=(queued.key,),
        golden_scores=(GoldenScore(Stage.WAVE1, 1, 1, 0, 1),),
        queue=(
            QueueEntry(
                queued,
                (Reason.NO_MATCH, Reason.AGENT_ESCALATED),
                "SAL-02",
                None,
                None,
                "stand-in",
                decision,
                TableRow.GENERIC,
            ),
        ),
        answer_labels=("live", "stand-in"),
    )
    store.save_run(connection, result)
    assert read(connection, "SELECT * FROM decisions ORDER BY cleaned_title") == [
        (
            "SAL",
            "account manager",
            "models agree",
            "machine",
            "SAL-01",
            "Provisional",
            "",
            0,
            0.25,
            None,
        ),
        (
            "SAL",
            "odd job",
            "generic family role",
            "code",
            "SAL-GEN",
            "Provisional",
            "",
            0,
            None,
            None,
        ),
        (
            "SAL",
            "sales rep",
            "left on legacy title",
            "none",
            None,
            None,
            "",
            0,
            None,
            "stopped path",
        ),
    ]
    assert read(connection, "SELECT * FROM audits ORDER BY path DESC") == [
        ("models agree", 2, 1, 1, "Provisional"),
        ("agent-settled", 0, 0, 0, "all labeled"),
    ]
    assert read(connection, "SELECT * FROM audit_samples") == [
        ("models agree", "SAL", "account manager")
    ]
    assert read(connection, "SELECT * FROM golden_scores") == [("Wave 1", 1, 1, 0, 1)]
    assert read(connection, "SELECT * FROM review_queue") == [
        ("SAL", "odd job", "no match; agent escalated", "stand-in", "generic family role")
    ]
    assert read(connection, "SELECT label FROM answer_labels ORDER BY label") == [
        ("live",),
        ("stand-in",),
    ]


def test_what_is_written_is_there_for_another_connection_at_once(tmp_path):
    # Every function commits, so a run that stops leaves the state of its last step.
    path = tmp_path / store.STATE_FILE
    writer = store.create_store(path)
    knowledge = Knowledge()
    knowledge.learn(title_row("sales engineer"), "SAL-04")
    store.save_knowledge(writer, Stage.SETUP, knowledge)
    reader = sqlite3.connect(path)
    try:
        assert read(reader, "SELECT count(*) FROM lookup") == [(1,)]
    finally:
        reader.close()
        writer.close()
