"""The small parts of titlestd.pipeline, on hand-built values.

The run as a whole is tested in tests/integration/test_pipeline.py (the mini
data set) and tests/integration/test_pipeline_default_run.py (the mock data).
"""

from __future__ import annotations

import hashlib

import pytest

from titlestd import datafiles, people, pipeline, release, report, store
from titlestd.contracts import (
    AgentCase,
    AgentResult,
    AgentTranscript,
    ChooseAnswer,
    ChooseKind,
    DecisionSource,
    HumanAnswer,
    HumanDecision,
    MachinePath,
    MatchResult,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
    ScoredRole,
    Status,
    TableRow,
    TitleRow,
)
from titlestd.settings import Settings

ROW = TitleRow(Record("account manager ii", "SAL", "Sales", ""), ("E1",), ("ACCT MGR II",))
MATCH = MatchResult((ScoredRole("SAL-02", 0.5), ScoredRole("SAL-01", 0.25)))
ANSWER = ChooseAnswer(ChooseKind.ROLE, 1, "account manager", "recorded 2026-10-05, a-model")


def scored(outcome: RuleOutcome, review_outcome: RuleOutcome | None = None, **fields):
    review = None
    if review_outcome is not None:
        case = AgentCase(ROW.record, (), ANSWER, outcome.role_id, "SAL-02", outcome.reasons)
        source = fields.pop("review_source", "recorded 2026-10-06, a-model")
        answer = ReviewAnswer(ReviewKind.ROLE, 1, "manager", source)
        review = AgentResult(answer, review_outcome.role_id, AgentTranscript(case))
    fields.setdefault("answer", ANSWER)
    return pipeline._Scored(
        ROW, MATCH, outcome, review=review, review_outcome=review_outcome, **fields
    )


def test_the_two_models_agreeing_is_the_first_machine_path():
    each = scored(RuleOutcome(Route.ACCEPT, (), "SAL-02"))
    assert each.path is MachinePath.MODELS_AGREE
    assert (each.machine_role, each.llm_pick, each.agent_pick) == ("SAL-02", "SAL-02", None)
    assert each.reasons == () and each.route == "models agree"
    assert each.labels == ("recorded 2026-10-05, a-model",)


def test_an_agent_answer_that_passes_the_rules_is_the_second_machine_path():
    referred = RuleOutcome(Route.AGENT, (Reason.MODELS_DISAGREE,), "SAL-01")
    each = scored(referred, RuleOutcome(Route.ACCEPT, (), "SAL-03"))
    assert each.path is MachinePath.AGENT_SETTLED
    # The path's role is the agent's, a correction of the LLM's pick here.
    assert (each.machine_role, each.llm_pick, each.agent_pick) == ("SAL-03", "SAL-01", "SAL-03")
    assert each.reasons == (Reason.MODELS_DISAGREE,) and each.route == "agent-settled"
    # Q40: in replay the two answers may carry different dates.
    assert each.labels == ("recorded 2026-10-05, a-model", "recorded 2026-10-06, a-model")


def test_a_title_the_agent_does_not_settle_goes_to_a_person_with_every_reason_once():
    referred = RuleOutcome(Route.AGENT, (Reason.MODELS_DISAGREE, Reason.FAMILY_MISMATCH), "FIN-01")
    after = RuleOutcome(Route.PERSON, (Reason.FAMILY_MISMATCH, Reason.AGENT_ESCALATED))
    each = scored(referred, after, review_source=ANSWER.source)
    assert each.path is None and each.machine_role is None
    assert each.reasons == (
        Reason.MODELS_DISAGREE,
        Reason.FAMILY_MISMATCH,
        Reason.AGENT_ESCALATED,
    )
    assert each.route == "person" and (each.llm_pick, each.agent_pick) == ("FIN-01", None)
    assert each.labels == (ANSWER.source,)  # one label, named once


def test_a_person_beats_the_agent_and_a_referral_not_followed_keeps_its_route():
    to_a_person = scored(RuleOutcome(Route.PERSON, (Reason.COMPLIANCE_FLAG,), "SAL-01"))
    assert (to_a_person.path, to_a_person.route) == (None, "person")
    # A golden-set pass does not ask the agent (Q08): the referral is what is recorded.
    not_followed = scored(RuleOutcome(Route.AGENT, (Reason.NO_MATCH,)))
    assert (not_followed.path, not_followed.route) == (None, "agent")
    assert not_followed.llm_pick is None


def test_on_the_ml_alone_route_there_is_no_llm_pick_and_no_label():
    # Q04: the outcome carries the ML first choice, which is not the LLM's pick.
    alone = scored(RuleOutcome(Route.PERSON, (Reason.ML_ALONE,), "SAL-02"), answer=None)
    assert (alone.path, alone.llm_pick, alone.labels) == (None, None, ())
    assert alone.reasons == (Reason.ML_ALONE,)


@pytest.mark.parametrize(
    ("flags", "reasons"),
    [
        ({"has_person_name": True}, (Reason.PERSON_NAME,)),
        ({"family_usable": False}, (Reason.NO_USABLE_FAMILY,)),
        (
            {"has_person_name": True, "family_usable": False},
            (Reason.PERSON_NAME, Reason.NO_USABLE_FAMILY),
        ),
    ],
)
def test_a_title_no_model_may_see_says_why(flags, reasons):
    row = TitleRow(ROW.record, ROW.employee_ids, ROW.raw_titles, **flags)
    assert not row.model_eligible
    assert pipeline._unscored_reasons(row) == reasons


def test_a_role_a_person_gave_is_approved_under_the_person_s_name():
    decision = HumanDecision(ROW.key, HumanAnswer.ROLE, "SAL-01", "SME 2 (simulated)", True)
    approved = pipeline._approved(decision, TableRow.REVIEW_QUEUE)
    assert (approved.key, approved.row) == (ROW.key, TableRow.REVIEW_QUEUE)
    assert (approved.source, approved.status) == (DecisionSource.PERSON, Status.APPROVED)
    assert (approved.role_id, approved.decided_by) == ("SAL-01", "SME 2 (simulated)")
    assert approved.simulated and approved.error_rate is None and approved.legacy_reason is None


def test_the_data_fingerprint_combines_the_two_fingerprints(mini_dir):
    both = f"{datafiles.input_fingerprint(mini_dir)}\n{people.truth_fingerprint(mini_dir)}"
    assert pipeline.data_fingerprint(mini_dir) == hashlib.sha256(both.encode()).hexdigest()


def test_a_missing_data_file_is_an_error_for_the_person_running_the_program(tmp_path):
    with pytest.raises(datafiles.DataError, match="data file not found"):
        pipeline.data_fingerprint(tmp_path)
    with pytest.raises(datafiles.DataError, match="data file not found"):
        pipeline.run_pipeline(Settings(data_dir=tmp_path, out_dir=tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_the_output_files_are_the_four_the_record_names():
    # Q05, Q38, S12
    assert pipeline.OUTPUT_FILES == (
        "load.csv",
        "reverse.csv",
        "exceptions.csv",
        "review_queue.csv",
    )
    assert set(pipeline.OUTPUT_FILES) == {
        release.LOAD_FILE,
        release.REVERSE_FILE,
        release.EXCEPTIONS_FILE,
        release.REVIEW_QUEUE_FILE,
    }
    assert store.STATE_FILE not in pipeline.OUTPUT_FILES


def test_the_run_command_prints_the_rendered_table_and_then_the_files(monkeypatch, capsys):
    settings = Settings(out_dir="somewhere")
    seen = []
    result = object()

    def run_pipeline(given):
        seen.append(given)
        return result

    def render(given_result, given_settings):
        seen.append((given_result, given_settings))
        return "the table\nand its footer"

    monkeypatch.setattr(pipeline, "run_pipeline", run_pipeline)
    monkeypatch.setattr(report, "render", render)
    assert pipeline.run_command(settings) == 0
    assert seen == [settings, (result, settings)]
    assert capsys.readouterr().out == (
        "the table\nand its footer\n"
        "Files written to the folder somewhere: load.csv, reverse.csv, exceptions.csv, "
        "review_queue.csv, state.sqlite.\n"
    )
