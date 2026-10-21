"""The look-up table and the examples on the mini data set.

The rows come from the real prepare step and the real catalog, the seed
look-up from the real reader, and the decisions from the real simulated
people. DESIGN.md 1.6 steps 2 and 10; docs/build-decisions.md Q07, Q09,
Q13, Q14, Q26.

The golden set is drawn by titlestd.golden, which these tests leave out: the
golden keys here are picked by hand from the frame of Q08, long-tail titles
that miss the seed look-up, have a usable family code and carry no person's
name.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

from titlestd import catalog, datafiles, people, prepare
from titlestd.contracts import (
    NAME_MASK,
    Example,
    HumanAnswer,
    InputData,
    SeedLookupRow,
    TitleKey,
    TitleRow,
)
from titlestd.datafiles import DataError
from titlestd.lookup import Knowledge, load_seed, seed_keys
from titlestd.settings import Settings

AP_CLERK = TitleKey("FIN", "accounts payable clerk")
SALES_ENGINEER = TitleKey("SAL", "sales engineer")
NAMED = TitleKey("SAL", f"customer success manager {NAME_MASK}")
NO_FAMILY = TitleKey("", "staff accountant")

GOLDEN = (
    TitleKey("FIN", "payroll specialist"),
    TitleKey("FIN", "treasury analyst"),
    TitleKey("SAL", "sales operations analyst"),
)

# The true role of every mini title row, written out by hand from the truth
# file: what a simulated person answers when the error rate is zero.
TRUE_ROLES = {
    NO_FAMILY: "FIN-02",
    AP_CLERK: "FIN-05",
    TitleKey("FIN", "financial analyst"): "FIN-01",
    TitleKey("FIN", "internal auditor"): "FIN-04",
    TitleKey("FIN", "payroll specialist"): "FIN-03",
    TitleKey("FIN", "senior financial analyst"): "FIN-01",
    TitleKey("FIN", "special projects"): "FIN-01",
    TitleKey("FIN", "treasury analyst"): "FIN-06",
    TitleKey("SAL", "account manager ii"): "SAL-01",
    TitleKey("SAL", "bdr"): "SAL-05",
    NAMED: "SAL-06",
    SALES_ENGINEER: "SAL-04",
    TitleKey("SAL", "sales operations analyst"): "SAL-03",
    TitleKey("SAL", "sales representative"): "SAL-02",
    TitleKey("SAL", "senior account manager"): "SAL-01",
}


@pytest.fixture(scope="module")
def rows(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    """The title rows of the mini data set, from the real prepare step."""
    return prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )


@pytest.fixture
def simulated(mini_settings: Settings, mini_inputs: InputData) -> people.SimulatedPeople:
    """Simulated people who never err, so each label is the title's true role."""
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    return people.load_people(settings, mini_inputs.catalog)


def _seeded(mini_inputs: InputData, rows, golden=()) -> Knowledge:
    knowledge = Knowledge(golden)
    load_seed(knowledge, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    return knowledge


def _decide_everything(knowledge: Knowledge, rows, simulated) -> dict[TitleKey, bool]:
    """A person labels every title that misses the look-up; every label is learned."""
    learned = {}
    for row in rows:
        if knowledge.lookup(row.key) is None:
            decision = simulated.blind_label(row)
            assert decision.answer is HumanAnswer.ROLE
            learned[row.key] = knowledge.learn(row, decision.role_id)
    return learned


# --- The seed look-up (1.6 step 2, Q09, Q13) --------------------------------------


def test_the_table_of_true_roles_covers_the_mini_rows(rows):
    assert sorted(TRUE_ROLES) == [row.key for row in rows]


def test_the_seed_file_holds_raw_titles_and_its_keys_are_cleaned(mini_inputs):
    # "AP Clerk" is expanded, and "Sales Engineer #20311" loses its requisition number.
    assert [seed.job_title for seed in mini_inputs.seed_lookup] == [
        "Sales Engineer #20311",
        "AP Clerk",
    ]
    assert seed_keys(mini_inputs.seed_lookup, mini_inputs.abbreviations) == (
        AP_CLERK,
        SALES_ENGINEER,
    )


def test_the_look_up_step_has_its_two_hits_and_thirteen_misses(mini_inputs, rows):
    knowledge = _seeded(mini_inputs, rows)
    hits = {row.key: knowledge.lookup(row.key) for row in rows}
    assert {key: role for key, role in hits.items() if role} == {
        AP_CLERK: "FIN-05",
        SALES_ENGINEER: "SAL-04",
    }
    assert sum(role is None for role in hits.values()) == 13
    assert knowledge.entries() == ((AP_CLERK, "FIN-05"), (SALES_ENGINEER, "SAL-04"))


def test_every_employee_of_a_seeded_title_hits_whatever_the_spelling(mini_inputs, rows):
    knowledge = _seeded(mini_inputs, rows)
    for raw, family in [("ap clerk - NYC", "FIN "), ("SALES ENG. REQ-0042", "SAL")]:
        key = prepare.title_key(raw, family, mini_inputs.abbreviations)
        assert knowledge.lookup(key) is not None, raw


def test_seed_entries_are_examples_from_the_start(mini_inputs, rows):
    knowledge = _seeded(mini_inputs, rows)
    by_key = {row.key: row for row in rows}
    # Each example carries the record of its title row: the text the ML model
    # embeds comes from the employees file, not from the seed file (Q12).
    assert knowledge.examples() == (
        Example(by_key[AP_CLERK].record, "FIN-05"),
        Example(by_key[SALES_ENGINEER].record, "SAL-04"),
    )
    assert knowledge.examples()[0].record.description == "Processes supplier invoices and payments."
    assert knowledge.example_count("FIN-05") == knowledge.example_count("SAL-04") == 1


def test_the_golden_frame_can_leave_the_seed_keys_out_before_anything_is_loaded(mini_inputs):
    # Set-up needs the seed keys to draw the golden set, and the golden set to
    # build the knowledge: seed_keys needs neither the rows nor the knowledge.
    keys = seed_keys(mini_inputs.seed_lookup, mini_inputs.abbreviations)
    assert not set(keys) & set(GOLDEN)


def test_a_seed_row_no_employee_holds_stops_the_load(mini_dir: Path, tmp_path: Path):
    # The readers accept it: the role exists and is in the family (S01).
    data_dir = tmp_path / "data"
    shutil.copytree(mini_dir, data_dir)
    seed = [*datafiles.read_seed_lookup(data_dir), SeedLookupRow("Controller", "FIN", "FIN-02")]
    datafiles.write_seed_lookup(data_dir, seed)
    data = datafiles.read_inputs(data_dir)
    rows = prepare.prepare(data.employees, data.abbreviations, catalog.family_codes(data.catalog))
    with pytest.raises(DataError, match=r"seed_lookup\.csv: title 'Controller' in family 'FIN'"):
        load_seed(Knowledge(), data.seed_lookup, rows, data.abbreviations)


# --- Learning people's decisions (1.6 step 10) -------------------------------------


def test_people_decide_every_title_and_the_golden_set_stays_out(mini_inputs, rows, simulated):
    # 2.7: golden-set titles never enter the look-up table or the examples.
    knowledge = _seeded(mini_inputs, rows, GOLDEN)
    learned = _decide_everything(knowledge, rows, simulated)
    assert {key for key, was_learned in learned.items() if not was_learned} == set(GOLDEN)

    in_table = {key for key, _ in knowledge.entries()}
    in_examples = {example.key for example in knowledge.examples()}
    assert not in_table & set(GOLDEN)
    assert not in_examples & set(GOLDEN)
    assert all(knowledge.lookup(key) is None for key in GOLDEN)
    # Everything else was learned: fifteen rows less the three golden ones.
    assert in_table == set(TRUE_ROLES) - set(GOLDEN)
    assert dict(knowledge.entries()) == {
        key: role for key, role in TRUE_ROLES.items() if key not in GOLDEN
    }


def test_the_guard_holds_after_every_stage_and_every_kind_of_label(mini_inputs, rows, simulated):
    knowledge = _seeded(mini_inputs, rows, GOLDEN)
    golden_rows = [row for row in rows if row.key in GOLDEN]

    def assert_golden_is_out():
        assert not {key for key, _ in knowledge.entries()} & set(GOLDEN)
        assert not {example.key for example in knowledge.examples()} & set(GOLDEN)

    assert_golden_is_out()  # set-up
    # Wave 1: the golden set is labeled blind, and the Wave 1 list is approved.
    for row in golden_rows:
        assert knowledge.learn(row, simulated.blind_label(row).role_id) is False
    for row in sorted(rows, key=lambda row: (-row.headcount, row.key))[:4]:
        assert knowledge.learn(row, simulated.wave1_approval(row).role_id) is True
    assert_golden_is_out()
    # Wave 2: an audit draws golden titles like any other (Q07, Q21).
    for row in golden_rows:
        assert knowledge.learn(row, simulated.blind_label(row).role_id) is False
    assert_golden_is_out()
    # The close: a reviewer decides whatever is left, golden titles included.
    for row in rows:
        decision = simulated.queue_decision(row)
        if knowledge.lookup(row.key) is None and decision.answer is HumanAnswer.ROLE:
            assert knowledge.learn(row, decision.role_id) is (row.key not in GOLDEN)
    assert_golden_is_out()
    assert len(knowledge.entries()) >= 6


def test_the_title_with_a_persons_name_joins_the_look_up_and_never_the_examples(
    mini_inputs, rows, simulated
):
    knowledge = _seeded(mini_inputs, rows)
    learned = _decide_everything(knowledge, rows, simulated)
    assert learned[NAMED] is True
    # Under the masked key (Q14).
    assert knowledge.lookup(NAMED) == "SAL-06"
    assert NAMED not in {example.key for example in knowledge.examples()}
    assert not any(NAME_MASK in example.record.cleaned_title for example in knowledge.examples())
    assert knowledge.example_count("SAL-06") == 0
    # It is the only decided title that is no example.
    assert len(knowledge.entries()) == 15 and len(knowledge.examples()) == 14


def test_the_title_with_no_usable_family_joins_under_its_key(mini_inputs, rows, simulated):
    knowledge = _seeded(mini_inputs, rows)
    _decide_everything(knowledge, rows, simulated)
    # Keyed on the family value as received: blank stays blank (Q26).
    assert knowledge.lookup(NO_FAMILY) == "FIN-02"
    assert knowledge.lookup(TitleKey("FIN", "staff accountant")) is None
    (example,) = [example for example in knowledge.examples() if example.key == NO_FAMILY]
    assert example.role_id == "FIN-02" and example.record.family_code == ""
    assert knowledge.example_count("FIN-02") == 1


def test_the_knowledge_does_not_depend_on_the_order_of_the_decisions(mini_inputs, rows, simulated):
    forward = _seeded(mini_inputs, rows, GOLDEN)
    _decide_everything(forward, rows, simulated)
    backward = _seeded(mini_inputs, rows[::-1], GOLDEN[::-1])
    _decide_everything(backward, rows[::-1], simulated)
    assert forward.entries() == backward.entries()
    assert forward.examples() == backward.examples()
    assert forward.example_count("FIN-01") == backward.example_count("FIN-01") == 3
