"""The shared types: fixed vocabularies, fixed numbers and small derived values."""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
from pathlib import Path

import pytest

from titlestd import contracts
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
    Example,
    ExceptionReason,
    HumanAnswer,
    Llm,
    MachinePath,
    MatchResult,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
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

# --- Vocabularies: exactly the names the design and the decisions fix ----------


def _values(vocabulary: type[enum.StrEnum]) -> list[str]:
    return [member.value for member in vocabulary]


def test_statuses_are_the_three_of_the_design():
    assert _values(Status) == ["Approved", "Confirmed", "Provisional"]


def test_there_are_two_machine_paths():
    assert _values(MachinePath) == ["models agree", "agent-settled"]


def test_table_rows_are_the_seven_of_the_design_in_order():
    assert _values(TableRow) == [
        "look-up hit",
        "approved in Wave 1",
        "models agree",
        "agent-settled",
        "review queue",
        "generic family role",
        "left on legacy title",
    ]


def test_machine_paths_are_table_rows():
    assert {path.value for path in MachinePath} <= {row.value for row in TableRow}
    assert TableRow(MachinePath.MODELS_AGREE.value) is TableRow.MODELS_AGREE
    assert TableRow(MachinePath.AGENT_SETTLED.value) is TableRow.AGENT_SETTLED


def test_answer_kinds():
    assert _values(ChooseKind) == ["role", "no_match", "insufficient_data", "invalid"]
    assert _values(ReviewKind) == ["role", "escalate", "invalid"]


def test_routes_and_reasons():
    assert _values(Route) == ["accept", "agent", "person"]
    assert _values(Reason) == [
        "role not in catalog",
        "quote not in record",
        "compliance-flagged role",
        "insufficient data",
        "models disagree",
        "role outside recorded family",
        "no match",
        "agent escalated",
        "invalid model answer",
        "person's name",
        "no usable family code",
        "ML alone",
        "no suitable role, waiting for catalog owner",
    ]


def test_people_and_outcome_vocabularies():
    assert _values(HumanAnswer) == ["role", "could not decide", "no suitable role"]
    assert _values(DecisionSource) == ["look-up", "person", "machine", "code", "none"]
    assert _values(AuditOutcome) == ["Confirmed", "Provisional", "Stopped", "all labeled"]
    assert _values(Stage) == ["set-up", "Wave 1", "Wave 2", "close"]
    assert _values(ExceptionReason) == [
        "stopped path",
        "over the generic cap",
        "undecided widely held or flagged title",
        "undecided title with a person's name or no usable family",
        "no suitable role",
        "no grade",
    ]


def test_search_tool_names():
    assert _values(SearchTool) == ["search_catalog", "search_approved_titles"]


def test_numbers_the_design_fixes():
    assert contracts.SHORTLIST_SIZE == 10
    assert contracts.ML_TOP_N == 3
    assert contracts.DESCRIPTION_WORDS == 200
    assert contracts.AGENT_MAX_SEARCHES == 4
    assert contracts.AUDIT_SAMPLE_SIZE == 60
    assert contracts.AUDIT_ACCURACY_BAR == 0.90
    assert contracts.AUDIT_PASS_CHANCE == 0.10
    assert contracts.AUDIT_STOP_SHARE == 0.20
    assert contracts.QUOTE_MIN_CHARS == 3
    assert contracts.QUOTE_MAX_WORDS == 25
    assert contracts.NAME_MASK == "[NAME]"
    assert contracts.EMAIL_MASK == "[EMAIL]"


def test_answer_labels():
    assert contracts.LABEL_STAND_IN == "stand-in"
    assert contracts.LABEL_LIVE == "live"
    assert (
        contracts.recorded_label("2026-10-05", "claude-haiku-4-5")
        == "recorded 2026-10-05, claude-haiku-4-5"
    )


# --- The module itself ---------------------------------------------------------


def _data_classes() -> list[type]:
    return [
        value
        for _, value in inspect.getmembers(contracts, inspect.isclass)
        if dataclasses.is_dataclass(value) and value.__module__ == contracts.__name__
    ]


def test_every_data_class_is_frozen_and_slotted():
    classes = _data_classes()
    assert len(classes) > 20
    for cls in classes:
        assert cls.__dataclass_params__.frozen, cls.__name__
        assert "__slots__" in vars(cls), cls.__name__


def test_contracts_imports_nothing_from_the_package():
    tree = ast.parse(Path(contracts.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [name for name in imported if name.split(".")[0] == "titlestd"]


def test_the_user_facing_error_is_an_exception():
    assert issubclass(contracts.TitlestdError, Exception)


# --- Small derived values ------------------------------------------------------


def _role(role_id: str, family: str = "SAL") -> CatalogRole:
    return CatalogRole(role_id, family, f"Role {role_id}", f"R {role_id}", "A definition.")


def test_title_keys_sort_by_family_then_title():
    keys = [TitleKey("SAL", "b"), TitleKey("FIN", "z"), TitleKey("SAL", "a"), TitleKey("", "m")]
    assert sorted(keys) == [
        TitleKey("", "m"),
        TitleKey("FIN", "z"),
        TitleKey("SAL", "a"),
        TitleKey("SAL", "b"),
    ]
    assert TitleKey("SAL", "a") == ("SAL", "a")


def test_title_row_key_headcount_and_eligibility():
    record = Record("account manager ii", "SAL", "Enterprise Sales", "")
    row = TitleRow(record, ("E001", "E002"), ("ACCT MGR II",))
    assert row.key == TitleKey("SAL", "account manager ii")
    assert row.headcount == 2
    assert row.model_eligible
    assert not dataclasses.replace(row, has_person_name=True).model_eligible
    assert not dataclasses.replace(row, family_usable=False).model_eligible


def test_an_example_is_keyed_like_its_title_row():
    record = Record("sales engineer", "SAL", "Solutions", "Runs demonstrations.")
    assert Example(record, "SAL-04").key == TitleRow(record, ("E013",), ("x",)).key


def test_match_result_first_choice_margin_and_top_three():
    shortlist = tuple(
        ScoredRole(f"R{n}", score) for n, score in enumerate([0.9, 0.75, 0.5, 0.25, 0.1], start=1)
    )
    match = MatchResult(shortlist, first_choice_examples=2)
    assert match.first_choice == "R1"
    assert match.margin == pytest.approx(0.15)
    assert match.top_three == ("R1", "R2", "R3")


def test_match_result_with_one_or_no_role():
    assert MatchResult(()).first_choice is None
    assert MatchResult(()).margin == 0.0
    assert MatchResult(()).top_three == ()
    single = MatchResult((ScoredRole("R1", 0.4),))
    assert single.first_choice == "R1"
    assert single.margin == 0.0
    assert single.top_three == ("R1",)


def _case() -> AgentCase:
    answer = ChooseAnswer(ChooseKind.ROLE, 2, "account manager", contracts.LABEL_STAND_IN)
    return AgentCase(
        record=Record("account manager ii", "SAL", "", ""),
        roles=(
            NumberedRole(1, _role("SAL-01")),
            NumberedRole(None, _role("FIN-01", "FIN")),
            NumberedRole(2, _role("SAL-02")),
        ),
        llm_answer=answer,
        llm_role_id="FIN-01",
        ml_first_choice="SAL-01",
        reasons=(Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH),
    )


def test_transcript_counts_searches_and_lists_numbered_roles_once():
    transcript = AgentTranscript(_case())
    assert transcript.searches_left == 4
    assert [role.number for role in transcript.numbered_roles] == [1, 2]

    step = SearchStep(
        SearchRequest(SearchTool.CATALOG, "account manager"),
        (
            SearchHit(NumberedRole(1, _role("SAL-01"))),  # shown before: same number
            SearchHit(NumberedRole(3, _role("SAL-03"))),
            SearchHit(NumberedRole(None, _role("FIN-02", "FIN"))),
        ),
    )
    transcript = AgentTranscript(_case(), (step,))
    assert transcript.searches_left == 3
    assert [role.role.role_id for role in transcript.numbered_roles] == [
        "SAL-01",
        "SAL-02",
        "SAL-03",
    ]
    assert AgentTranscript(_case(), (step,) * 6).searches_left == 0


def test_agent_result_confirms_only_the_llm_role():
    transcript = AgentTranscript(_case())
    answer = ReviewAnswer(ReviewKind.ROLE, 1, "account manager", contracts.LABEL_STAND_IN)
    assert not AgentResult(answer, "SAL-01", transcript).confirms_llm
    assert AgentResult(answer, "FIN-01", transcript).confirms_llm
    escalate = ReviewAnswer(ReviewKind.ESCALATE, None, "", contracts.LABEL_STAND_IN)
    assert not AgentResult(escalate, None, transcript).confirms_llm


def test_audit_result_counts_and_error_rate():
    keys = tuple(TitleKey("SAL", f"title {n}") for n in range(4))
    result = AuditResult(MachinePath.MODELS_AGREE, 10, keys, 1, AuditOutcome.PROVISIONAL)
    assert result.audited == 4
    assert result.error_rate == 0.25
    empty = AuditResult(MachinePath.AGENT_SETTLED, 0, (), 0, AuditOutcome.ALL_LABELED)
    assert empty.error_rate is None


def test_a_decision_defaults_to_no_person_and_no_role():
    decision = Decision(TitleKey("SAL", "bdr"), TableRow.LEGACY, DecisionSource.NONE)
    assert decision.role_id is None and decision.status is None
    assert decision.decided_by == "" and decision.simulated is False
    assert decision.error_rate is None and decision.legacy_reason is None


def test_llm_finish_does_nothing_by_default():
    llm = Llm(chooser=object(), reviewer=object())  # type: ignore[arg-type]
    assert llm.finish() is None
