"""The rules of 1.7 on the mini data set, with the modules of stage 1.

Every title row a model may see goes the way a run would take it: the real
prepare step, an ML match on the real offline embedder, the seeded numbering,
the stand-in LLM built by the switch, then the rules. Three tests of section
2.7 are tried on that: a quote that is not in the record sends the title to a
person, a compliance-flagged role is never settled by a machine, and the
out-of-range number of Q15, the last two with an injected answer.

The nearest-neighbour match (titlestd.matcher) and the agent's loop
(titlestd.agent) are left out of these tests: score() and run_agent() do
their part in miniature, by the rules of 1.6 step 3, Q10 and Q20. They are
test-only code. No expectation comes from the truth file, which nothing here
reads.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from titlestd import catalog, datafiles, fallback, prepare, rules, switch
from titlestd.contracts import (
    SHORTLIST_SIZE,
    AgentCase,
    AgentTranscript,
    Catalog,
    ChooseAnswer,
    ChooseKind,
    InputData,
    MatchResult,
    Models,
    NumberedRole,
    Reason,
    Record,
    ReviewAnswer,
    ReviewKind,
    Route,
    RuleOutcome,
    ScoredRole,
    SearchHit,
    SearchStep,
    SearchTool,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings

SETTINGS = Settings()
FLAGGED = "FIN-04"  # Internal Auditor, the one compliance-flagged role of the mini catalog

ACCEPT, AGENT, PERSON = Route.ACCEPT, Route.AGENT, Route.PERSON

# Where the rules send each model-eligible mini row, and why. The stand-in's
# answers are those worked out by hand from Q17, and it agrees with the ML
# first choice on every row where it names a role. What is left is the flag:
# the internal auditor's role itself, and three finance titles whose ML top
# three hold it (see test_the_rows_the_flag_stops_are_those_that_touch_it).
EXPECTED = {
    TitleKey("FIN", "accounts payable clerk"): (ACCEPT, (), "FIN-05"),
    TitleKey("FIN", "financial analyst"): (PERSON, (Reason.COMPLIANCE_FLAG,), "FIN-01"),
    TitleKey("FIN", "internal auditor"): (PERSON, (Reason.COMPLIANCE_FLAG,), "FIN-04"),
    TitleKey("FIN", "payroll specialist"): (ACCEPT, (), "FIN-03"),
    TitleKey("FIN", "senior financial analyst"): (PERSON, (Reason.COMPLIANCE_FLAG,), "FIN-01"),
    TitleKey("FIN", "special projects"): (
        PERSON,
        (Reason.COMPLIANCE_FLAG, Reason.NO_MATCH),
        None,
    ),
    TitleKey("FIN", "treasury analyst"): (ACCEPT, (), "FIN-06"),
    TitleKey("SAL", "account manager ii"): (ACCEPT, (), "SAL-01"),
    TitleKey("SAL", "bdr"): (PERSON, (Reason.INSUFFICIENT_DATA,), None),
    TitleKey("SAL", "sales engineer"): (ACCEPT, (), "SAL-04"),
    TitleKey("SAL", "sales operations analyst"): (ACCEPT, (), "SAL-03"),
    TitleKey("SAL", "sales representative"): (ACCEPT, (), "SAL-02"),
    TitleKey("SAL", "senior account manager"): (ACCEPT, (), "SAL-01"),
}
TOUCH_THE_FLAG = {
    key for key, (_, reasons, _) in EXPECTED.items() if reasons[:1] == (Reason.COMPLIANCE_FLAG,)
}

# Titles recorded in the wrong family (1.7), which the mini data set has none of.
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
ANALYST_IN_SALES = TitleRow(
    Record(
        "financial analyst",
        "SAL",
        "Sales Operations",
        "Builds budgets and forecasts for the sales teams.",
    ),
    ("X003",),
    ("Financial Analyst",),
)
# A title on which the two methods differ: by its words a customer success
# manager, by its letters nearest to the sales representative.
SUCCESS_REPRESENTATIVE = TitleRow(
    Record("customer success representative", "SAL", "Customer Success", ""),
    ("X004",),
    ("Customer Success Rep",),
)


# --- A run in miniature -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scored:
    """One title row with what the rules are given about it."""

    row: TitleRow
    match: MatchResult
    shown: tuple[NumberedRole, ...]
    block: str

    @property
    def family(self) -> str:
        return self.row.record.family_code


def ranked(models: Models, text: str, cat: Catalog) -> list[ScoredRole]:
    """The model-visible roles, nearest to the text first, ties by role_id."""
    visible = catalog.model_visible_roles(cat)
    texts = [f"{role.role_name}. {role.definition}" for role in visible]
    target, *vectors = models.embedder.embed([text, *texts])
    scored = [
        ScoredRole(
            role.role_id,
            math.fsum(weight * vector.get(index, 0.0) for index, weight in target.items()),
        )
        for role, vector in zip(visible, vectors, strict=True)
    ]
    return sorted(scored, key=lambda each: (-each.score, each.role_id))


def score(models: Models, row: TitleRow, cat: Catalog) -> Scored:
    """The ML match of 1.6 step 3, then the numbered list and the record block."""
    record = row.record
    nearest = ranked(models, f"{record.cleaned_title} {record.description}".strip(), cat)
    match = MatchResult(tuple(nearest[:SHORTLIST_SIZE]))
    shortlist = [catalog.get_role(cat, each.role_id) for each in match.shortlist]
    shown = fallback.number_roles(shortlist, row.key, SETTINGS.seed)
    return Scored(row, match, shown, fallback.render_record(record))


def check(chooser, scored: Scored, cat: Catalog) -> RuleOutcome:
    """Ask the chooser and put its answer through the rules."""
    answer = chooser.choose(scored.row.record, scored.shown)
    return rules.check_llm(answer, scored.shown, scored.block, scored.match, scored.family, cat)


def run_agent(
    models: Models, scored: Scored, llm_answer: ChooseAnswer, referral: RuleOutcome, cat: Catalog
) -> tuple[AgentTranscript, ReviewAnswer]:
    """The stand-in agent through the loop of Q20, with the numbering of Q10.

    Nobody has approved a title yet, so the approved-title search finds nothing.
    """
    record = scored.row.record
    numbers: dict[str, int] = {}

    def show(role) -> NumberedRole:
        if role.family_code != record.family_code:
            return NumberedRole(None, role)
        return NumberedRole(numbers.setdefault(role.role_id, len(numbers) + 1), role)

    case = AgentCase(
        record=record,
        roles=tuple(show(each.role) for each in scored.shown),
        llm_answer=llm_answer,
        llm_role_id=referral.role_id,
        ml_first_choice=scored.match.first_choice,
        reasons=referral.reasons,
    )
    transcript = AgentTranscript(case)
    while True:
        turn = models.llm.reviewer.next_turn(transcript)
        if isinstance(turn, ReviewAnswer):
            return transcript, turn
        assert transcript.searches_left > 0
        hits: tuple[SearchHit, ...] = ()
        if turn.tool is SearchTool.CATALOG:
            found = ranked(models, turn.query, cat)[: SETTINGS.agent_search_k]
            hits = tuple(SearchHit(show(catalog.get_role(cat, each.role_id))) for each in found)
        transcript = AgentTranscript(case, (*transcript.steps, SearchStep(turn, hits)))


class Injected:
    """A chooser whose answers are the stand-in's with one thing replaced.

    The stand-in has no fault injection (Q17), so a failed quote and an
    out-of-range number are injected here, on top of its real answers.
    """

    def __init__(self, chooser, **replaced) -> None:
        self._chooser = chooser
        self._replaced = replaced

    def choose(self, record, roles) -> ChooseAnswer:
        answer = self._chooser.choose(record, roles)
        if "number" in self._replaced and answer.answer is not ChooseKind.ROLE:
            return answer
        return dataclasses.replace(answer, **self._replaced)


@pytest.fixture(scope="module")
def models(mini_inputs: InputData) -> Models:
    return switch.build_models(SETTINGS, mini_inputs.catalog.ladder, mini_inputs.abbreviations)


@pytest.fixture(scope="module")
def chooser(models: Models):
    return models.llm.chooser


@pytest.fixture(scope="module")
def mini(mini_inputs: InputData) -> Catalog:
    return mini_inputs.catalog


def score_all(models: Models, data: InputData) -> dict[TitleKey, Scored]:
    rows = prepare.prepare(data.employees, data.abbreviations, catalog.family_codes(data.catalog))
    return {row.key: score(models, row, data.catalog) for row in rows if row.model_eligible}


@pytest.fixture(scope="module")
def scored(models: Models, mini_inputs: InputData) -> dict[TitleKey, Scored]:
    """Every model-eligible row of the mini data set, prepared and matched."""
    return score_all(models, mini_inputs)


# --- The stand-in's own answers through the rules ---------------------------------------------


def test_the_table_covers_every_row_a_model_may_see(scored, mini_rows):
    assert sorted(scored) == sorted(EXPECTED)
    assert sorted(scored) == [row.key for row in mini_rows if row.model_eligible]
    # A title with a person's name or no usable family code goes to no model
    # and so to no rule of 1.7 (Q14, Q26).
    assert len(mini_rows) - len(scored) == 2


@pytest.mark.parametrize("key", sorted(EXPECTED), ids=" / ".join)
def test_each_mini_row_goes_where_the_rules_send_it(chooser, scored, mini, key):
    assert check(chooser, scored[key], mini) == RuleOutcome(*EXPECTED[key])


def test_the_rows_the_flag_stops_are_those_that_touch_it(scored):
    # 1.7: "The proposed role, or any of the ML model's top three".
    assert len(TOUCH_THE_FLAG) == 4
    top_three = {key: each.match.top_three for key, each in scored.items()}
    assert {key for key, top in top_three.items() if FLAGGED in top} == TOUCH_THE_FLAG
    assert top_three[TitleKey("FIN", "internal auditor")][0] == FLAGGED
    assert top_three[TitleKey("FIN", "financial analyst")] == ("FIN-01", "SAL-03", "FIN-04")
    # It is the top three, not the shortlist: nearly every row is shown the role.
    listed = [
        key for key, each in scored.items() if FLAGGED in {n.role.role_id for n in each.shown}
    ]
    assert len(listed) > len(TOUCH_THE_FLAG) + 4


def test_an_accepted_row_is_one_on_which_the_two_models_agree(chooser, scored, mini):
    accepted = 0
    for each in scored.values():
        outcome = check(chooser, each, mini)
        if outcome.route is ACCEPT:
            accepted += 1
            role = catalog.get_role(mini, outcome.role_id)
            assert outcome.role_id == each.match.first_choice
            assert role.family_code == each.family
            assert not role.compliance_flag and not role.is_generic
    assert accepted == 8


def test_the_family_code_as_the_employees_file_holds_it_is_the_same_family(
    chooser, scored, mini, mini_inputs
):
    # S05: one employee of the mini data set is recorded in "FIN ", which is FIN.
    each = scored[TitleKey("FIN", "treasury analyst")]
    (employee,) = [e for e in mini_inputs.employees if e.employee_id in each.row.employee_ids]
    assert employee.job_family_code == "FIN "
    answer = chooser.choose(each.row.record, each.shown)
    assert rules.check_llm(
        answer, each.shown, each.block, each.match, employee.job_family_code, mini
    ) == RuleOutcome(ACCEPT, (), "FIN-06")
    assert rules.check_llm(answer, each.shown, each.block, each.match, "fin", mini) == RuleOutcome(
        AGENT, (Reason.FAMILY_MISMATCH,), "FIN-06"
    )


class Refusing:
    """A chooser with no usable answer, as a live backend reports a refusal (S10, Q39)."""

    def choose(self, record, roles) -> ChooseAnswer:
        return ChooseAnswer(ChooseKind.INVALID, None, "", "test", "the model refused")


def test_an_answer_that_is_not_usable_sends_every_title_to_a_person(scored, mini):
    for key, each in scored.items():
        expected = [Reason.INVALID_ANSWER]
        if key in TOUCH_THE_FLAG:
            # The ML top three are read whatever the LLM did: the close must
            # know that the title touches a flagged role (Q25).
            expected.insert(0, Reason.COMPLIANCE_FLAG)
        assert check(Refusing(), each, mini) == RuleOutcome(PERSON, tuple(expected), None), key


def test_two_models_that_disagree_send_the_title_to_the_agent_which_confirms_the_llm(models, mini):
    # 1.1: "Two methods must agree."
    each = score(models, SUCCESS_REPRESENTATIVE, mini)
    assert each.match.first_choice == "SAL-02"  # Sales Representative
    llm_answer = models.llm.chooser.choose(each.row.record, each.shown)
    referral = rules.check_llm(llm_answer, each.shown, each.block, each.match, "SAL", mini)
    assert referral == RuleOutcome(AGENT, (Reason.MODELS_DISAGREE,), "SAL-06")

    transcript, answer = run_agent(models, each, llm_answer, referral, mini)
    outcome = rules.check_agent(
        answer, transcript.numbered_roles, each.block, each.match, "SAL", mini
    )
    # Picking the LLM's role confirms it; the agent's answer is not checked for
    # agreement with the ML model, or it could never settle a disagreement.
    assert outcome == RuleOutcome(ACCEPT, (), "SAL-06")
    assert outcome.role_id != each.match.first_choice


# --- A quote that is not in the record sends the title to a person (2.7) ----------------------


def test_an_invented_quote_sends_every_title_to_a_person(chooser, scored, mini):
    misquoting = Injected(chooser, quote="Leads the regional sales organisation.")
    for key, each in scored.items():
        honest = check(chooser, each, mini)
        outcome = check(misquoting, each, mini)
        assert outcome.route is PERSON, key
        assert Reason.QUOTE_NOT_IN_RECORD in outcome.reasons
        # Nothing else changed: the same role, and the other failed checks as before.
        assert outcome.role_id == honest.role_id
        assert [reason for reason in outcome.reasons if reason in honest.reasons] == list(
            honest.reasons
        )
        assert len(outcome.reasons) == len(honest.reasons) + 1


def test_a_quote_from_another_title_s_record_is_not_evidence_for_this_one(chooser, scored, mini):
    # A sentence that does stand in a record of the data set: the clerk's.
    sentence = "Processes supplier invoices and payments."
    borrowed = Injected(chooser, quote=sentence)
    clerk = TitleKey("FIN", "accounts payable clerk")
    assert scored[clerk].row.record.description == sentence
    for key, each in scored.items():
        outcome = check(borrowed, each, mini)
        if key == clerk:
            assert outcome == RuleOutcome(ACCEPT, (), "FIN-05")
        else:
            assert outcome.route is PERSON and Reason.QUOTE_NOT_IN_RECORD in outcome.reasons


def test_the_raw_title_is_not_a_quote_once_the_title_is_cleaned(chooser, scored, mini):
    # "ACCT MGR II" is what the employees file holds; the record a model is
    # shown says "account manager ii", so the abbreviations prove nothing.
    each = scored[TitleKey("SAL", "account manager ii")]
    assert each.row.raw_titles == ("ACCT MGR II",)
    assert check(Injected(chooser, quote="ACCT MGR II"), each, mini) == RuleOutcome(
        PERSON, (Reason.QUOTE_NOT_IN_RECORD,), "SAL-01"
    )
    assert check(Injected(chooser, quote="ACCOUNT MANAGER II"), each, mini).route is ACCEPT


def test_a_misquoted_no_match_goes_to_a_person_not_to_the_agent(chooser, scored, mini_inputs):
    # With no flag in the catalog "special projects" is a plain "no match",
    # which the agent would review. With an invented quote a person does.
    unflagged = without_flags(mini_inputs.catalog)
    each = scored[TitleKey("FIN", "special projects")]
    assert check(chooser, each, unflagged) == RuleOutcome(AGENT, (Reason.NO_MATCH,), None)
    assert check(Injected(chooser, quote="nothing fits"), each, unflagged) == RuleOutcome(
        PERSON, (Reason.QUOTE_NOT_IN_RECORD, Reason.NO_MATCH), None
    )


# --- The out-of-range number (Q15) ------------------------------------------------------------


@pytest.mark.parametrize("number", [0, SHORTLIST_SIZE + 1, 12, -3, 999])
def test_a_number_outside_one_to_ten_sends_the_title_to_a_person(chooser, scored, mini, number):
    out_of_range = Injected(chooser, number=number)
    named_a_role = 0
    for key, each in scored.items():
        honest = check(chooser, each, mini)
        outcome = check(out_of_range, each, mini)
        if EXPECTED[key][2] is None:
            assert outcome == honest  # "no match" and "insufficient data" carry no number
            continue
        named_a_role += 1
        assert outcome.route is PERSON, key
        assert outcome.reasons[0] is Reason.ROLE_NOT_IN_CATALOG
        assert outcome.role_id is None
        # The role the stand-in had found is gone with the number: no agreement to report.
        assert Reason.MODELS_DISAGREE not in outcome.reasons
    assert named_a_role == 11


def test_the_stand_in_itself_never_answers_out_of_range(chooser, scored):
    for each in scored.values():
        answer = chooser.choose(each.row.record, each.shown)
        if answer.answer is ChooseKind.ROLE:
            assert 1 <= answer.number <= SHORTLIST_SIZE


# --- A compliance-flagged role is never settled by a machine (2.7) ----------------------------


def without_flags(cat: Catalog) -> Catalog:
    roles = tuple(dataclasses.replace(role, compliance_flag=False) for role in cat.roles)
    return dataclasses.replace(cat, roles=roles)


def test_no_answer_at_all_settles_a_title_that_touches_the_flagged_role(scored, mini):
    # Not only the stand-in's answer: every number the LLM could give, "no
    # match" and "insufficient data", each with a quote that stands in the record.
    touched = 0
    for key, each in scored.items():
        title = each.row.record.cleaned_title
        answers = [ChooseAnswer(ChooseKind.ROLE, number, title, "test") for number in range(1, 11)]
        answers.append(ChooseAnswer(ChooseKind.NO_MATCH, None, title, "test"))
        answers.append(ChooseAnswer(ChooseKind.INSUFFICIENT_DATA, None, title, "test"))
        for answer in answers:
            outcome = rules.check_llm(answer, each.shown, each.block, each.match, each.family, mini)
            proposed = fallback.role_for_number(answer.number, each.shown)
            touches = key in TOUCH_THE_FLAG or (proposed is not None and proposed.compliance_flag)
            assert (Reason.COMPLIANCE_FLAG in outcome.reasons) == touches
            if touches:
                touched += 1
                assert outcome.route is PERSON, (key, answer)
            # And no proposal ever comes out as a generic family role.
            assert outcome.role_id is None or not catalog.get_role(mini, outcome.role_id).is_generic
    assert touched > 4 * 12


def test_the_agent_cannot_settle_a_flagged_role_either(scored, mini):
    for key, each in scored.items():
        title = each.row.record.cleaned_title
        for number in range(1, 11):
            answer = ReviewAnswer(ReviewKind.ROLE, number, title, "test")
            outcome = rules.check_agent(
                answer, each.shown, each.block, each.match, each.family, mini
            )
            role = fallback.role_for_number(number, each.shown)
            if key in TOUCH_THE_FLAG or role.compliance_flag:
                assert outcome.route is PERSON and Reason.COMPLIANCE_FLAG in outcome.reasons
            assert outcome.route in (ACCEPT, PERSON)


def test_the_flag_in_the_catalog_file_is_what_stops_the_machine(tmp_path, models, mini_inputs):
    # The same files with the flag taken off the internal auditor: the four
    # titles it stopped now go where their other checks send them.
    roles = [dataclasses.replace(role, compliance_flag=False) for role in mini_inputs.catalog.roles]
    datafiles.write_inputs(tmp_path, mini_inputs)
    datafiles.write_catalog_roles(tmp_path, roles)
    data = datafiles.read_inputs(tmp_path)
    assert not any(role.compliance_flag for role in data.catalog.roles)
    outcomes = {
        key: check(models.llm.chooser, each, data.catalog)
        for key, each in score_all(models, data).items()
    }
    for key, (route, reasons, role_id) in EXPECTED.items():
        if key not in TOUCH_THE_FLAG:
            assert outcomes[key] == RuleOutcome(route, reasons, role_id)
    assert outcomes[TitleKey("FIN", "financial analyst")] == RuleOutcome(ACCEPT, (), "FIN-01")
    assert outcomes[TitleKey("FIN", "internal auditor")] == RuleOutcome(ACCEPT, (), "FIN-04")
    assert outcomes[TitleKey("FIN", "senior financial analyst")] == RuleOutcome(
        ACCEPT, (), "FIN-01"
    )
    assert outcomes[TitleKey("FIN", "special projects")] == RuleOutcome(
        AGENT, (Reason.NO_MATCH,), None
    )


def test_a_flag_put_on_another_role_in_the_catalog_file_stops_its_titles(
    tmp_path, models, mini_inputs
):
    roles = [
        dataclasses.replace(role, compliance_flag=role.role_id == "SAL-01")
        for role in mini_inputs.catalog.roles
    ]
    datafiles.write_inputs(tmp_path, mini_inputs)
    datafiles.write_catalog_roles(tmp_path, roles)
    data = datafiles.read_inputs(tmp_path)
    stopped = set()
    for key, each in score_all(models, data).items():
        outcome = check(models.llm.chooser, each, data.catalog)
        if Reason.COMPLIANCE_FLAG in outcome.reasons:
            assert outcome.route is PERSON
            assert "SAL-01" in (*each.match.top_three, outcome.role_id)
            stopped.add(key)
    assert {
        TitleKey("SAL", "account manager ii"),
        TitleKey("SAL", "senior account manager"),
    } < stopped
    assert not stopped & {
        TitleKey("FIN", "internal auditor"),
        TitleKey("FIN", "payroll specialist"),
    }


# --- A generic family role is never a machine's proposal (1.2) --------------------------------


def test_no_rule_lets_a_generic_role_of_the_catalog_file_through(scored, mini):
    # fallback.number_roles refuses to number one. Were a list or an ML match
    # to hold one all the same, no check accepts it or reports it as a role.
    generic_roles = [role for role in mini.roles if role.is_generic]
    assert [role.role_id for role in generic_roles] == ["SAL-GEN", "FIN-GEN"]
    for generic in generic_roles:
        each = next(one for one in scored.values() if one.family == generic.family_code)
        title = each.row.record.cleaned_title
        shown = (*each.shown, NumberedRole(11, generic))
        match = MatchResult((ScoredRole(generic.role_id, 0.99), *each.match.shortlist[:1]))
        stopped = RuleOutcome(PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)
        llm_answer = ChooseAnswer(ChooseKind.ROLE, 11, title, "test")
        assert rules.check_llm(llm_answer, shown, each.block, match, each.family, mini) == stopped
        agent_answer = ReviewAnswer(ReviewKind.ROLE, 11, title, "test")
        assert (
            rules.check_agent(agent_answer, shown, each.block, match, each.family, mini) == stopped
        )
        assert rules.check_ml_alone(match, each.family, mini) == RuleOutcome(
            PERSON, (Reason.ML_ALONE, Reason.ROLE_NOT_IN_CATALOG), None
        )


# --- To the agent, and what the rules make of its answer (1.7) --------------------------------


def test_a_title_filed_in_the_wrong_family_goes_to_the_agent_which_corrects_it(models, mini):
    each = score(models, TREASURY_IN_SALES, mini)
    llm_answer = models.llm.chooser.choose(each.row.record, each.shown)
    referral = rules.check_llm(llm_answer, each.shown, each.block, each.match, "SAL", mini)
    # Both models say Treasury Analyst, a finance role; the title is recorded in sales.
    assert referral == RuleOutcome(AGENT, (Reason.FAMILY_MISMATCH,), "FIN-06")
    assert each.match.first_choice == "FIN-06"

    transcript, answer = run_agent(models, each, llm_answer, referral, mini)
    numbered = transcript.numbered_roles
    assert numbered and all(shown.role.family_code == "SAL" for shown in numbered)
    assert (answer.answer, answer.quote) == (ReviewKind.ROLE, "analyst")
    outcome = rules.check_agent(answer, numbered, each.block, each.match, "SAL", mini)
    assert outcome == RuleOutcome(ACCEPT, (), "SAL-03")  # picking another role corrects the LLM


def test_what_the_agent_answers_is_checked_like_any_proposal(models, mini):
    each = score(models, TREASURY_IN_SALES, mini)
    llm_answer = models.llm.chooser.choose(each.row.record, each.shown)
    referral = rules.check_llm(llm_answer, each.shown, each.block, each.match, "SAL", mini)
    transcript, answer = run_agent(models, each, llm_answer, referral, mini)
    numbered = transcript.numbered_roles

    def agent(injected: ReviewAnswer) -> RuleOutcome:
        return rules.check_agent(injected, numbered, each.block, each.match, "SAL", mini)

    misquoted = dataclasses.replace(answer, quote="knows the sales pipeline")
    assert agent(misquoted) == RuleOutcome(PERSON, (Reason.QUOTE_NOT_IN_RECORD,), "SAL-03")
    not_shown = dataclasses.replace(answer, number=len(numbered) + 1)
    assert agent(not_shown) == RuleOutcome(PERSON, (Reason.ROLE_NOT_IN_CATALOG,), None)
    # The LLM's own number for the finance role means nothing in the agent's list,
    # where that role carries no number: it names a sales role or nothing.
    finance_role = fallback.role_for_number(llm_answer.number, each.shown)
    assert finance_role.role_id == "FIN-06"
    assert finance_role in [shown.role for shown in transcript.case.roles if shown.number is None]
    for number in range(-1, 13):
        outcome = agent(dataclasses.replace(answer, number=number))
        assert outcome.role_id is None or outcome.role_id.startswith("SAL-")
        assert outcome.role_id != "FIN-06"
    # And were the agent's answer read against the LLM's list, which numbers
    # that role, the family check would still send the title to a person.
    with_the_llm_s_number = dataclasses.replace(answer, number=llm_answer.number)
    assert rules.check_agent(
        with_the_llm_s_number, each.shown, each.block, each.match, "SAL", mini
    ) == RuleOutcome(PERSON, (Reason.FAMILY_MISMATCH,), "FIN-06")


def test_an_escalation_by_the_agent_goes_to_a_person(models, mini):
    each = score(models, ENGINEER_IN_FINANCE, mini)
    llm_answer = models.llm.chooser.choose(each.row.record, each.shown)
    referral = rules.check_llm(llm_answer, each.shown, each.block, each.match, "FIN", mini)
    assert referral == RuleOutcome(AGENT, (Reason.FAMILY_MISMATCH,), "SAL-04")
    # No finance role shares a word with "sales engineer": the stand-in agent escalates.
    transcript, answer = run_agent(models, each, llm_answer, referral, mini)
    assert answer.answer is ReviewKind.ESCALATE
    outcome = rules.check_agent(
        answer, transcript.numbered_roles, each.block, each.match, "FIN", mini
    )
    assert outcome == RuleOutcome(PERSON, (Reason.AGENT_ESCALATED,), None)


def test_a_person_beats_the_agent_on_a_misfiled_title_near_the_flagged_role(models, mini):
    # A family mismatch alone would go to the agent. The ML top three hold the
    # internal auditor, so a person decides, and the agent's answer could not stand.
    each = score(models, ANALYST_IN_SALES, mini)
    assert FLAGGED in each.match.top_three
    llm_answer = models.llm.chooser.choose(each.row.record, each.shown)
    referral = rules.check_llm(llm_answer, each.shown, each.block, each.match, "SAL", mini)
    assert referral == RuleOutcome(
        PERSON, (Reason.COMPLIANCE_FLAG, Reason.FAMILY_MISMATCH), "FIN-01"
    )
    transcript, answer = run_agent(models, each, llm_answer, referral, mini)
    outcome = rules.check_agent(
        answer, transcript.numbered_roles, each.block, each.match, "SAL", mini
    )
    assert outcome == RuleOutcome(PERSON, (Reason.COMPLIANCE_FLAG,), "SAL-03")
    # Without the flag the same title is the agent's to settle.
    unflagged = without_flags(mini)
    assert rules.check_llm(
        llm_answer, each.shown, each.block, each.match, "SAL", unflagged
    ) == RuleOutcome(AGENT, (Reason.FAMILY_MISMATCH,), "FIN-01")
    assert rules.check_agent(
        answer, transcript.numbered_roles, each.block, each.match, "SAL", unflagged
    ) == RuleOutcome(ACCEPT, (), "SAL-03")


# --- The ML-alone route keeps its checks (Q04) ------------------------------------------------


def test_an_ml_alone_title_goes_to_a_person_with_the_checks_that_still_run(scored, mini):
    for key, each in scored.items():
        outcome = rules.check_ml_alone(each.match, each.family, mini)
        expected = [Reason.ML_ALONE]
        if key in TOUCH_THE_FLAG:
            expected.append(Reason.COMPLIANCE_FLAG)
        # "bdr" has no description and its nearest role is the sales engineer's:
        # in the family, so nothing more to say. Every first choice is in the family.
        assert outcome == RuleOutcome(PERSON, tuple(expected), each.match.first_choice), key


def test_an_ml_alone_role_outside_the_recorded_family_is_reported(models, mini):
    each = score(models, TREASURY_IN_SALES, mini)
    assert rules.check_ml_alone(each.match, "SAL", mini) == RuleOutcome(
        PERSON, (Reason.ML_ALONE, Reason.FAMILY_MISMATCH), "FIN-06"
    )
    assert rules.check_ml_alone(each.match, "FIN", mini) == RuleOutcome(
        PERSON, (Reason.ML_ALONE,), "FIN-06"
    )
