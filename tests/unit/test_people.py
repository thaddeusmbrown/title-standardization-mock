"""Simulated people: one label per title, from the truth.

docs/design.md 2.4 and docs/build-decisions.md Q27 (the label and its error
rate), Q03 ("no suitable role") and S05 (family codes are compared after
trimming).
"""

from __future__ import annotations

import pytest

from titlestd import people, seeding
from titlestd.contracts import (
    NAME_MASK,
    Catalog,
    CatalogRole,
    HumanAnswer,
    Record,
    TitleKey,
    TitleRow,
)
from titlestd.datafiles import DataError
from titlestd.people import SimulatedPeople, TruthRow
from titlestd.settings import Settings

# The occasions on which an SME is asked about a title.
OCCASIONS = ("wave1_approval", "blind_label")

# The true role of every title row of the mini data set, written out by hand
# from tests/fixtures/mini so that the tests do not repeat the code's own join.
MINI_TRUE_ROLES = {
    TitleKey("", "staff accountant"): "FIN-02",
    TitleKey("FIN", "accounts payable clerk"): "FIN-05",
    TitleKey("FIN", "financial analyst"): "FIN-01",
    TitleKey("FIN", "internal auditor"): "FIN-04",
    TitleKey("FIN", "payroll specialist"): "FIN-03",
    TitleKey("FIN", "senior financial analyst"): "FIN-01",
    TitleKey("FIN", "special projects"): "FIN-01",
    TitleKey("FIN", "treasury analyst"): "FIN-06",
    TitleKey("SAL", "account manager ii"): "SAL-01",
    TitleKey("SAL", "bdr"): "SAL-05",
    TitleKey("SAL", f"customer success manager {NAME_MASK}"): "SAL-06",
    TitleKey("SAL", "sales engineer"): "SAL-04",
    TitleKey("SAL", "sales operations analyst"): "SAL-03",
    TitleKey("SAL", "sales representative"): "SAL-02",
    TitleKey("SAL", "senior account manager"): "SAL-01",
}


@pytest.fixture(scope="module")
def mini_truth(mini_dir) -> tuple[TruthRow, ...]:
    return people.read_truth(mini_dir)


def _people(truth, catalog, **settings) -> SimulatedPeople:
    return SimulatedPeople(truth, catalog, Settings(**settings))


def _row(family: str, title: str, *raw_titles: str) -> TitleRow:
    """A hand-made title row. Its raw title is the cleaned one in capitals unless given."""
    return TitleRow(
        record=Record(cleaned_title=title, family_code=family, department="", description=""),
        employee_ids=("E900",),
        raw_titles=raw_titles or (title.upper(),),
    )


def _many(count: int) -> tuple[list[TitleRow], list[TruthRow]]:
    """count made-up Sales titles that are all truly SAL-01, and their truth."""
    rows = [_row("SAL", f"made up title {number}") for number in range(count)]
    return rows, [TruthRow(row.raw_titles[0], "SAL", "SAL-01") for row in rows]


def _family_of(catalog: Catalog) -> dict[str, str]:
    return {role.role_id: role.family_code for role in catalog.roles}


def test_the_hand_written_truth_covers_the_mini_rows(mini_rows):
    assert [row.key for row in mini_rows] == list(MINI_TRUE_ROLES)


# --- What a decision carries (2.4, Q27) ----------------------------------------


def test_every_decision_is_simulated_and_names_who_was_asked(mini_catalog, mini_truth, mini_rows):
    simulated = _people(mini_truth, mini_catalog)
    for row in mini_rows:
        for occasion in OCCASIONS:
            decision = getattr(simulated, occasion)(row)
            assert decision.key == row.key
            assert decision.simulated is True
            assert decision.decided_by in people.PERSONA_SMES


def test_with_no_error_every_label_is_the_true_role(mini_catalog, mini_truth, mini_rows):
    simulated = _people(mini_truth, mini_catalog, sme_error_rate=0.0)
    for occasion in OCCASIONS:
        decisions = [getattr(simulated, occasion)(row) for row in mini_rows]
        assert all(decision.answer is HumanAnswer.ROLE for decision in decisions)
        assert {d.key: d.role_id for d in decisions} == MINI_TRUE_ROLES


# --- One label per title (Q27) --------------------------------------------------


def test_a_title_gets_the_same_label_on_every_occasion(mini_catalog, mini_truth, mini_rows):
    # Half the labels are wrong, so a second draw for another occasion would show.
    simulated = _people(mini_truth, mini_catalog, sme_error_rate=0.5)
    wrong = 0
    for row in mini_rows:
        roles = {getattr(simulated, occasion)(row).role_id for occasion in OCCASIONS}
        assert len(roles) == 1, row.key
        wrong += roles != {MINI_TRUE_ROLES[row.key]}
        again = {getattr(simulated, occasion)(row).role_id for occasion in OCCASIONS}
        assert again == roles
    assert 0 < wrong < len(mini_rows)  # the test saw wrong labels and right ones


def test_a_title_gets_one_label_whatever_the_order_of_calls(mini_catalog, mini_truth, mini_rows):
    def answers(calls):
        simulated = _people(mini_truth, mini_catalog, sme_error_rate=0.5)
        given = {}
        for occasion, row in calls:
            decision = getattr(simulated, occasion)(row)
            # Asked twice in one run, a title answers the same both times.
            assert given.setdefault((occasion, row.key), decision) == decision
        return given

    by_occasion = [(occasion, row) for occasion in OCCASIONS for row in mini_rows]
    by_title = [(occasion, row) for row in reversed(mini_rows) for occasion in reversed(OCCASIONS)]
    one_title_only = [(occasion, mini_rows[7]) for occasion in reversed(OCCASIONS)]

    expected = answers(by_occasion)
    assert len(expected) == len(OCCASIONS) * len(mini_rows)
    assert answers(list(reversed(by_occasion))) == expected
    assert answers(by_title + by_occasion) == expected
    assert answers(one_title_only).items() <= expected.items()


def test_labels_do_not_depend_on_the_order_of_the_truth_rows(mini_catalog, mini_truth, mini_rows):
    forward = _people(mini_truth, mini_catalog, sme_error_rate=0.5)
    backward = _people(tuple(reversed(mini_truth)), mini_catalog, sme_error_rate=0.5)
    for row in mini_rows:
        assert forward.blind_label(row) == backward.blind_label(row)


def test_a_label_depends_on_the_key_and_not_on_the_rest_of_the_row(mini_catalog):
    truth = [TruthRow("Sales Rep", "SAL", "SAL-02"), TruthRow("SALES REP.", "SAL", "SAL-02")]
    plain = _row("SAL", "sales representative", "Sales Rep")
    other = TitleRow(
        record=Record("sales representative", "SAL", "Field Sales", "Sells products."),
        employee_ids=("E001", "E002", "E003"),
        raw_titles=("SALES REP.", "Sales Rep"),
        has_person_name=True,
        family_usable=False,
    )
    for rate in (0.0, 1.0):
        simulated = _people(truth, mini_catalog, sme_error_rate=rate)
        for occasion in OCCASIONS:
            assert getattr(simulated, occasion)(plain) == getattr(simulated, occasion)(other)


def test_whether_a_label_is_wrong_is_the_titles_own_draw(mini_catalog):
    # S04: the sme_error stream, keyed on the family code, then the cleaned title.
    rows, truth = _many(400)
    settings = Settings(sme_error_rate=0.3)
    simulated = SimulatedPeople(truth, mini_catalog, settings)
    for row in rows:
        draw = seeding.unit(settings.seed, seeding.STREAM_SME_ERROR, "SAL", row.key.cleaned_title)
        assert (simulated.blind_label(row).role_id != "SAL-01") == (draw < 0.3), row.key


def test_another_seed_gives_other_labels(mini_catalog):
    rows, truth = _many(200)

    def labels(seed):
        simulated = _people(truth, mini_catalog, seed=seed, sme_error_rate=0.3)
        return [(d.role_id, d.decided_by) for d in map(simulated.blind_label, rows)]

    assert labels(20260101) == labels(20260101)
    assert labels(20260101) != labels(20260102)


def test_the_draws_are_pinned_on_every_supported_python(mini_catalog, mini_truth, mini_rows):
    # Simulated decisions feed the examples and so the later prompts (Q18): a
    # change here, or between Python versions, would invalidate a recording.
    simulated = _people(mini_truth, mini_catalog, sme_error_rate=0.5)
    labels = [simulated.blind_label(row) for row in mini_rows]
    assert [(d.role_id, d.decided_by.split()[1]) for d in labels] == [
        ("FIN-02", "3"),
        ("FIN-06", "1"),
        ("FIN-04", "2"),
        ("FIN-01", "1"),
        ("FIN-03", "3"),
        ("FIN-01", "1"),
        ("FIN-01", "1"),
        ("FIN-03", "3"),
        ("SAL-06", "1"),
        ("SAL-03", "2"),
        ("SAL-06", "1"),
        ("SAL-03", "2"),
        ("SAL-03", "1"),
        ("SAL-04", "3"),
        ("SAL-03", "2"),
    ]


# --- Which SME (decided on the branch: see the module's docstring) --------------


def test_a_title_falls_to_the_same_sme_on_every_occasion(mini_catalog):
    rows, truth = _many(300)
    simulated = _people(truth, mini_catalog)
    # Neither the error rate nor a missing truth row moves a title to another SME.
    erring = _people(truth, mini_catalog, sme_error_rate=1.0)
    without_truth = _people([], mini_catalog)
    smes = []
    for row in rows:
        sme = simulated.wave1_approval(row).decided_by
        assert simulated.blind_label(row).decided_by == sme
        assert erring.blind_label(row).decided_by == sme
        assert without_truth.wave1_approval(row).decided_by == sme
        smes.append(sme)
    assert set(smes) == set(people.PERSONA_SMES)
    assert min(smes.count(sme) for sme in people.PERSONA_SMES) > 70  # about a third each


# --- The seeded error (Q27) -----------------------------------------------------


def test_a_wrong_label_is_another_real_role_of_the_true_roles_family(
    mini_catalog, mini_truth, mini_rows
):
    simulated = _people(mini_truth, mini_catalog, sme_error_rate=1.0)
    family_of = _family_of(mini_catalog)
    generic = {role.role_id for role in mini_catalog.roles if role.is_generic}
    for row in mini_rows:
        true_role = MINI_TRUE_ROLES[row.key]
        for occasion in OCCASIONS:
            decision = getattr(simulated, occasion)(row)
            assert decision.answer is HumanAnswer.ROLE
            assert decision.role_id != true_role
            assert decision.role_id not in generic
            # The true role's family, also where the title's own code is blank.
            assert family_of[decision.role_id] == family_of[true_role]


def test_a_wrong_label_can_be_any_other_role_of_the_family_and_never_the_generic_one(
    mini_catalog,
):
    rows, truth = _many(300)
    simulated = _people(truth, mini_catalog, sme_error_rate=1.0)
    labels = [simulated.blind_label(row).role_id for row in rows]
    assert set(labels) == {"SAL-02", "SAL-03", "SAL-04", "SAL-05", "SAL-06"}
    assert min(labels.count(role_id) for role_id in set(labels)) > 30  # about a fifth each


def test_a_wrong_label_stays_in_the_true_roles_family_when_the_title_has_another_code(
    mini_catalog,
):
    # Q26: with no usable family code the truth may name a role of any family.
    rows = [_row("ZZZ", f"odd job {number}") for number in range(40)]
    truth = [TruthRow(row.raw_titles[0], "ZZZ", "SAL-03") for row in rows]
    simulated = _people(truth, mini_catalog, sme_error_rate=1.0)
    labels = {simulated.blind_label(row).role_id for row in rows}
    assert labels == {"SAL-01", "SAL-02", "SAL-04", "SAL-05", "SAL-06"}


def test_the_error_rate_is_about_the_setting(mini_catalog):
    rows, truth = _many(3000)
    default = _people(truth, mini_catalog)
    wrong = sum(default.blind_label(row).role_id != "SAL-01" for row in rows)
    assert Settings().sme_error_rate == 0.02
    assert 0.01 < wrong / len(rows) < 0.03
    never = _people(truth, mini_catalog, sme_error_rate=0.0)
    assert all(never.blind_label(row).role_id == "SAL-01" for row in rows)


def test_a_higher_error_rate_keeps_the_wrong_labels_of_a_lower_one(mini_catalog):
    rows, truth = _many(500)
    low = _people(truth, mini_catalog, sme_error_rate=0.1)
    high = _people(truth, mini_catalog, sme_error_rate=0.4)
    low_labels = {row.key: low.blind_label(row).role_id for row in rows}
    high_labels = {row.key: high.blind_label(row).role_id for row in rows}
    low_wrong = {key: role_id for key, role_id in low_labels.items() if role_id != "SAL-01"}
    assert low_wrong
    assert low_wrong.items() <= high_labels.items()


def test_in_a_family_with_one_role_the_label_is_the_true_role():
    catalog = Catalog(
        roles=(
            CatalogRole("ONE-01", "ONE", "Only Role", "Only", "The family's one role."),
            CatalogRole("ONE-GEN", "ONE", "One", "One", "Generic role.", is_generic=True),
        ),
        ladder=(),
        grade_levels=(),
    )
    row = _row("ONE", "only role")
    truth = [TruthRow("ONLY ROLE", "ONE", "ONE-01")]
    simulated = _people(truth, catalog, sme_error_rate=1.0)
    for occasion in OCCASIONS:
        decision = getattr(simulated, occasion)(row)
        assert (decision.answer, decision.role_id) == (HumanAnswer.ROLE, "ONE-01")


@pytest.mark.parametrize(
    ("true_role_id", "message"),
    [
        ("SAL-99", "not in the catalog"),
        ("SAL-GEN", "generic role"),
        ("FIN-01", "of family FIN"),
    ],
)
def test_people_are_not_built_on_a_truth_that_does_not_fit_the_catalog(
    mini_catalog, true_role_id, message
):
    # However they are built, they can then never name an unknown or a generic role.
    with pytest.raises(DataError, match=message):
        _people([TruthRow("BDR", "SAL", true_role_id)], mini_catalog)


# --- "No suitable role" (Q03) ---------------------------------------------------


@pytest.mark.parametrize(
    "rates",
    [{}, {"sme_error_rate": 0.0}, {"sme_error_rate": 1.0}],
)
def test_no_suitable_role_only_without_a_truth_row(mini_catalog, mini_truth, mini_rows, rates):
    missing = TitleKey("SAL", "bdr")
    whole = _people(mini_truth, mini_catalog, **rates)
    cut = _people([row for row in mini_truth if row.job_title != "BDR"], mini_catalog, **rates)
    for row in mini_rows:
        for occasion in OCCASIONS:
            # With a truth row: never, whatever the rates.
            assert getattr(whole, occasion)(row).answer is not HumanAnswer.NO_SUITABLE_ROLE
            decision = getattr(cut, occasion)(row)
            if row.key == missing:
                # Without one: always, on every occasion.
                assert decision.answer is HumanAnswer.NO_SUITABLE_ROLE
                assert decision.role_id is None
            else:
                assert decision == getattr(whole, occasion)(row)


def test_the_persona_who_was_asked_marks_no_suitable_role(mini_catalog):
    # The catalog owner is whom the title then waits for, not who marks it.
    simulated = _people([], mini_catalog)
    row = _row("SAL", "chief of staff")
    for occasion in OCCASIONS:
        decision = getattr(simulated, occasion)(row)
        assert decision.answer is HumanAnswer.NO_SUITABLE_ROLE
        assert decision.decided_by in people.PERSONA_SMES and decision.simulated


def test_one_raw_title_with_a_truth_row_is_enough(mini_catalog):
    row = _row("SAL", "sales representative", "SALES REP.", "Sales Rep")
    simulated = _people([TruthRow("Sales Rep", "SAL", "SAL-02")], mini_catalog, sme_error_rate=0.0)
    assert simulated.blind_label(row).role_id == "SAL-02"


def test_the_truth_of_another_title_or_family_does_not_serve(mini_catalog):
    truth = [TruthRow("Sales Rep", "SAL", "SAL-02"), TruthRow("Analyst", "FIN", "FIN-01")]
    simulated = _people(truth, mini_catalog, sme_error_rate=0.0)
    strangers = [
        _row("SAL", "sales representative", "sales rep"),  # the raw title, byte for byte (Q09)
        _row("SAL", "analyst", "Analyst"),  # the same title in another family
        _row("", "analyst", "Analyst"),
        _row("FINANCE", "analyst", "Analyst"),
    ]
    for row in strangers:
        assert simulated.blind_label(row).answer is HumanAnswer.NO_SUITABLE_ROLE, row.key
    assert simulated.blind_label(_row("FIN", "analyst", "Analyst")).role_id == "FIN-01"


# --- Family codes are compared after trimming (S05) ---------------------------------


def test_the_padded_code_of_the_mini_data_meets_its_truth(mini_catalog, mini_truth, mini_rows):
    assert TruthRow("Treasury Analyst", "FIN ", "FIN-06") in mini_truth
    (row,) = [row for row in mini_rows if row.raw_titles == ("Treasury Analyst",)]
    assert row.key.family_code == "FIN"
    simulated = _people(mini_truth, mini_catalog, sme_error_rate=0.0)
    for occasion in OCCASIONS:
        decision = getattr(simulated, occasion)(row)
        assert (decision.answer, decision.role_id) == (HumanAnswer.ROLE, "FIN-06")


@pytest.mark.parametrize(
    ("recorded", "in_key"),
    [
        ("FIN ", "FIN"),
        ("  FIN  ", "FIN"),
        ("FIN", "FIN "),  # trimmed on both sides
        ("   ", ""),  # a code made of spaces only is a blank code
        ("", ""),
        (" ZZZ", "ZZZ"),  # an unknown code is kept, and trimmed like any other
    ],
)
def test_family_codes_are_compared_after_trimming(mini_catalog, recorded, in_key):
    truth = [TruthRow("Odd Job", recorded, "FIN-02")]
    simulated = _people(truth, mini_catalog, sme_error_rate=0.0)
    row = _row(in_key, "odd job", "Odd Job")
    for occasion in OCCASIONS:
        decision = getattr(simulated, occasion)(row)
        assert (decision.answer, decision.role_id) == (HumanAnswer.ROLE, "FIN-02")
