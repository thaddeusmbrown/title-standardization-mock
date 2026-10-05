"""The random audit with the real modules around it.

The title rows come from the real prepare step, the blind labels from the real
simulated people, the settings from the real settings loader, and the golden
set from titlestd.golden (docs/design.md 1.9, 2.4; docs/build-decisions.md
Q03, Q07, Q16, Q21, Q23, Q27).

What puts a title on a machine path (the rules, the agent, the pipeline) is
left out of these tests, so the machine's roles are written out here.
The mini data set has 15 title rows, which shows the path that is labeled in
full. A sampled audit needs more than 60 titles, so the other tests put 200
made-up titles through the real prepare step and give each a true role.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from titlestd import audit, catalog, golden, people, prepare
from titlestd.contracts import (
    AuditOutcome,
    AuditResult,
    Catalog,
    Decision,
    DecisionSource,
    Employee,
    ExceptionReason,
    HumanAnswer,
    HumanDecision,
    InputData,
    MachinePath,
    Status,
    TableRow,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings, load_settings

# Two hand-made machine paths over the long tail of the mini data set: the role
# the machine chose for each title. It is the true role except for "special
# projects", which is truly FIN-01.
MINI_MODELS_AGREE = {
    TitleKey("FIN", "payroll specialist"): "FIN-03",
    TitleKey("FIN", "senior financial analyst"): "FIN-01",
    TitleKey("FIN", "treasury analyst"): "FIN-06",
    TitleKey("SAL", "sales operations analyst"): "SAL-03",
}
MINI_AGENT_SETTLED = {
    TitleKey("FIN", "special projects"): "FIN-05",
    TitleKey("SAL", "bdr"): "SAL-05",
}


@dataclasses.dataclass(frozen=True)
class Audited:
    """One path's audit, as a run would carry it out."""

    sample: tuple[TitleKey, ...]
    labels: dict[TitleKey, HumanDecision]
    result: AuditResult
    decisions: tuple[Decision, ...]

    def of(self, key: TitleKey) -> Decision:
        (decision,) = [decision for decision in self.decisions if decision.key == key]
        return decision

    @property
    def rest(self) -> list[Decision]:
        """The decisions on the titles nobody labeled."""
        return [decision for decision in self.decisions if decision.key not in self.labels]


def run_audit(
    path: MachinePath,
    rows: tuple[TitleRow, ...],
    machine: dict[TitleKey, str],
    simulated: people.SimulatedPeople,
    settings: Settings,
) -> Audited:
    """Draw the sample, have an SME label it blind, audit, and decide every title."""
    by_key = {row.key: row for row in rows}
    sample = audit.draw_sample(path, machine, settings.seed)
    labels = {key: simulated.blind_label(by_key[key]) for key in sample}
    result = audit.audit_path(path, machine, labels)
    return Audited(sample, labels, result, audit.path_decisions(result, machine, labels, settings))


@dataclasses.dataclass(frozen=True)
class LongTail:
    """200 made-up Sales titles, each with one true role of the Sales family."""

    rows: tuple[TitleRow, ...]
    truth: tuple[people.TruthRow, ...]
    true_roles: dict[TitleKey, str]
    sales: tuple[str, ...]  # the role ids of the family

    def another_role(self, key: TitleKey) -> str:
        """A role of the family that is not the title's true one."""
        return self.sales[(self.sales.index(self.true_roles[key]) + 1) % len(self.sales)]

    def simulated(self, roles: Catalog, **rates) -> tuple[people.SimulatedPeople, Settings]:
        """The people who know this truth, and the settings they were built with."""
        settings = Settings(**rates)
        return people.SimulatedPeople(self.truth, roles, settings), settings


@pytest.fixture(scope="module")
def long_tail(mini_inputs: InputData) -> LongTail:
    employees = [
        Employee(f"M{number:04d}", f"Made Up Title {number:03d}", "SAL", "Somewhere", 5, "")
        for number in range(200)
    ]
    rows = prepare.prepare(
        employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )
    sales = tuple(role.role_id for role in catalog.roles_in_family(mini_inputs.catalog, "SAL"))
    truth = tuple(
        people.TruthRow(employee.job_title, "SAL", sales[number % len(sales)])
        for number, employee in enumerate(employees)
    )
    true_roles = {
        prepare.title_key(row.job_title, "SAL", mini_inputs.abbreviations): row.true_role_id
        for row in truth
    }
    assert len(rows) == 200 and sorted(true_roles) == [row.key for row in rows]
    assert all(row.model_eligible for row in rows)
    return LongTail(rows, truth, true_roles, sales)


# --- A path of 60 titles or fewer, on the mini data set (2.4) ----------------------------


@pytest.fixture(scope="module")
def prepared(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    """The title rows of the mini data set, from the real prepare step."""
    return prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )


def test_a_small_path_of_the_mini_data_set_is_labeled_in_full(mini_inputs, prepared, mini_settings):
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    simulated = people.load_people(settings, mini_inputs.catalog)
    audited = run_audit(MachinePath.MODELS_AGREE, prepared, MINI_MODELS_AGREE, simulated, settings)
    assert audited.sample == tuple(sorted(MINI_MODELS_AGREE))
    assert audited.result == AuditResult(
        MachinePath.MODELS_AGREE, 4, audited.sample, 0, AuditOutcome.ALL_LABELED
    )
    assert [decision.key for decision in audited.decisions] == sorted(MINI_MODELS_AGREE)
    for decision in audited.decisions:
        assert decision.row is TableRow.MODELS_AGREE
        assert decision.source is DecisionSource.PERSON
        assert decision.status is Status.APPROVED  # 1.9: a named person decided it
        assert decision.role_id == MINI_MODELS_AGREE[decision.key]
        assert decision.decided_by in people.PERSONA_SMES
        assert decision.simulated is True  # 2.4: every simulated decision is marked
        assert decision.error_rate is None


def test_a_labeled_title_takes_the_smes_role_where_the_machine_was_wrong(
    mini_inputs, prepared, mini_settings
):
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0)
    simulated = people.load_people(settings, mini_inputs.catalog)
    audited = run_audit(
        MachinePath.AGENT_SETTLED, prepared, MINI_AGENT_SETTLED, simulated, settings
    )
    assert (audited.result.audited, audited.result.wrong) == (2, 1)
    assert audited.result.outcome is AuditOutcome.ALL_LABELED
    special = audited.of(TitleKey("FIN", "special projects"))
    assert MINI_AGENT_SETTLED[special.key] == "FIN-05"
    assert (special.role_id, special.status) == ("FIN-01", Status.APPROVED)
    bdr = audited.of(TitleKey("SAL", "bdr"))
    assert (bdr.role_id, bdr.status) == ("SAL-05", Status.APPROVED)
    assert {decision.row for decision in audited.decisions} == {TableRow.AGENT_SETTLED}


def test_at_the_default_rates_a_small_path_releases_only_roles_a_person_may_name(
    mini_inputs, prepared, mini_settings
):
    simulated = people.load_people(mini_settings, mini_inputs.catalog)
    nameable = {role.role_id for role in catalog.model_visible_roles(mini_inputs.catalog)}
    for path, machine in (
        (MachinePath.MODELS_AGREE, MINI_MODELS_AGREE),
        (MachinePath.AGENT_SETTLED, MINI_AGENT_SETTLED),
    ):
        audited = run_audit(path, prepared, machine, simulated, mini_settings)
        assert len(audited.decisions) == len(machine)
        assert all(decision.status is Status.APPROVED for decision in audited.decisions)
        assert {decision.role_id for decision in audited.decisions} <= nameable


# --- A sampled audit and its three outcomes (2.4, Q21, Q23) ------------------------------


def test_the_made_up_long_tail_is_labeled_with_its_true_roles(mini_inputs, long_tail):
    simulated, _ = long_tail.simulated(mini_inputs.catalog, sme_error_rate=0.0)
    labels = {row.key: simulated.blind_label(row).role_id for row in long_tail.rows}
    assert labels == long_tail.true_roles
    assert set(labels.values()) == set(long_tail.sales)


def test_a_machine_that_is_right_about_every_title_is_confirmed(mini_inputs, long_tail):
    simulated, settings = long_tail.simulated(mini_inputs.catalog, sme_error_rate=0.0)
    machine = dict(long_tail.true_roles)
    audited = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, settings)
    assert (audited.result.path_size, audited.result.audited, audited.result.wrong) == (200, 60, 0)
    assert audited.result.outcome is AuditOutcome.CONFIRMED
    statuses = [decision.status for decision in audited.decisions]
    assert statuses.count(Status.APPROVED) == 60  # read by an SME
    assert statuses.count(Status.CONFIRMED) == 140  # no person read it
    assert all(d.role_id == long_tail.true_roles[d.key] for d in audited.decisions)
    assert all(d.source is DecisionSource.MACHINE and not d.simulated for d in audited.rest)


@pytest.mark.parametrize(
    ("wrong", "outcome"),
    [
        (2, AuditOutcome.CONFIRMED),
        (3, AuditOutcome.PROVISIONAL),
        (12, AuditOutcome.PROVISIONAL),
        (13, AuditOutcome.STOPPED),
    ],
)
def test_the_outcome_follows_the_number_the_smes_find_wrong(mini_inputs, long_tail, wrong, outcome):
    simulated, settings = long_tail.simulated(mini_inputs.catalog, sme_error_rate=0.0)
    path = MachinePath.AGENT_SETTLED
    sample = audit.draw_sample(path, long_tail.true_roles, settings.seed)
    unread = [key for key in sorted(long_tail.true_roles) if key not in sample]
    # The machine is wrong about this many titles of the sample, and about
    # thirty that nobody will read, which the audit cannot see.
    mistaken = [*sample[:wrong], *unread[:30]]
    machine = {**long_tail.true_roles, **{key: long_tail.another_role(key) for key in mistaken}}
    audited = run_audit(path, long_tail.rows, machine, simulated, settings)

    assert audited.sample == sample
    assert (audited.result.wrong, audited.result.outcome) == (wrong, outcome)
    # Q21: a labeled title is released on the SME's role, here the true one.
    for key in sample:
        assert audited.of(key).role_id == long_tail.true_roles[key]
        assert audited.of(key).status is Status.APPROVED
    assert len(audited.rest) == 140
    if outcome is AuditOutcome.STOPPED:
        assert all(decision.row is TableRow.LEGACY for decision in audited.rest)
        assert all(
            decision.legacy_reason is ExceptionReason.STOPPED_PATH for decision in audited.rest
        )
    else:
        status = Status.CONFIRMED if outcome is AuditOutcome.CONFIRMED else Status.PROVISIONAL
        error_rate = None if outcome is AuditOutcome.CONFIRMED else wrong / 60
        for decision in audited.rest:
            assert decision.row is TableRow.AGENT_SETTLED
            assert (decision.status, decision.error_rate) == (status, error_rate)
            # No person read these: the machine's role stands, mistaken or not.
            assert decision.role_id == machine[decision.key]
        assert sum(d.role_id != long_tail.true_roles[d.key] for d in audited.rest) == 30


def test_at_the_default_error_rate_an_smes_mistake_counts_against_the_machine(
    mini_inputs, long_tail
):
    # 2.4, Q27: wrong is the machine against the SME's label, so a machine that
    # is right about every title still shows the SME's mistake. On the default
    # seed one of the 60 labels is mistaken, which the confirm line of 2 allows.
    simulated, settings = long_tail.simulated(mini_inputs.catalog)
    assert settings.sme_error_rate == 0.02
    machine = dict(long_tail.true_roles)
    audited = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, settings)
    mislabeled = [
        key for key in audited.sample if audited.labels[key].role_id != long_tail.true_roles[key]
    ]
    assert audited.result.wrong == len(mislabeled) == 1
    assert audited.result.outcome is AuditOutcome.CONFIRMED
    # The title is released on the SME's label all the same (Q21).
    assert audited.of(mislabeled[0]).role_id == audited.labels[mislabeled[0]].role_id
    assert audited.of(mislabeled[0]).role_id != machine[mislabeled[0]]


def test_a_stopped_path_keeps_its_labeled_titles_and_leaves_the_rest_on_legacy_titles(
    mini_inputs, long_tail
):
    # Every label is mistaken, so the audit finds the machine wrong 60 times.
    simulated, settings = long_tail.simulated(mini_inputs.catalog, sme_error_rate=1.0)
    machine = dict(long_tail.true_roles)
    audited = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, settings)
    assert (audited.result.wrong, audited.result.outcome) == (60, AuditOutcome.STOPPED)
    assert audited.result.error_rate == 1.0
    rows = [decision.row for decision in audited.decisions]
    assert rows.count(TableRow.MODELS_AGREE) == 60  # Q16: the path's row keeps the audited
    assert rows.count(TableRow.LEGACY) == 140
    for key in audited.sample:
        decision = audited.of(key)
        assert decision.status is Status.APPROVED
        assert decision.role_id == audited.labels[key].role_id != long_tail.true_roles[key]
    for decision in audited.rest:
        assert (decision.role_id, decision.status) == (None, None)
        assert decision.source is DecisionSource.NONE
        assert decision.legacy_reason is ExceptionReason.STOPPED_PATH


def test_the_release_setting_of_a_settings_file_releases_the_stopped_path(
    mini_inputs, long_tail, tmp_path
):
    # 2.4: "A stopped path is left on legacy titles unless a setting releases it."
    config = tmp_path / "settings.toml"
    config.write_text("sme_error_rate = 1.0\nrelease_stopped_paths = true\n", encoding="utf-8")
    settings = load_settings(config)
    simulated = people.SimulatedPeople(long_tail.truth, mini_inputs.catalog, settings)
    machine = dict(long_tail.true_roles)
    audited = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, settings)
    assert audited.result.outcome is AuditOutcome.STOPPED
    assert len(audited.rest) == 140
    for decision in audited.rest:
        assert decision.row is TableRow.MODELS_AGREE
        assert decision.status is Status.PROVISIONAL
        assert decision.error_rate == 1.0
        assert decision.role_id == machine[decision.key]
    # The setting releases the titles nobody read. The 60 an SME labeled take
    # the SME's role with status Approved, here never the machine's (Q21).
    for key in audited.sample:
        decision = audited.of(key)
        assert decision.row is TableRow.MODELS_AGREE
        assert decision.source is DecisionSource.PERSON
        assert decision.status is Status.APPROVED
        assert decision.role_id == audited.labels[key].role_id != machine[key]
        assert decision.error_rate is None
    # Without the flag the same audit leaves the rest on their legacy titles
    # and decides the labeled titles the same way.
    kept = load_settings(config, {"release_stopped_paths": False})
    held = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, kept)
    assert held.result == audited.result
    assert {decision.row for decision in held.rest} == {TableRow.LEGACY}
    assert [held.of(key) for key in held.sample] == [audited.of(key) for key in audited.sample]


def test_two_paths_are_audited_apart_and_every_title_gets_one_decision(mini_inputs, long_tail):
    # A run has two machine paths (1.2). Each is sampled, audited and given its
    # own outcome; together they decide every title once.
    simulated, settings = long_tail.simulated(mini_inputs.catalog, sme_error_rate=0.0)
    keys = sorted(long_tail.true_roles)
    agree = {key: long_tail.true_roles[key] for key in keys[:130]}
    settled = {key: long_tail.another_role(key) for key in keys[130:]}  # all wrong
    first = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, agree, simulated, settings)
    second = run_audit(MachinePath.AGENT_SETTLED, long_tail.rows, settled, simulated, settings)
    assert (first.result.path_size, first.result.audited) == (130, 60)
    assert (second.result.path_size, second.result.audited) == (70, 60)
    assert first.result.outcome is AuditOutcome.CONFIRMED
    assert second.result.outcome is AuditOutcome.STOPPED
    decisions = sorted([*first.decisions, *second.decisions], key=lambda decision: decision.key)
    assert [decision.key for decision in decisions] == keys
    rows = [decision.row for decision in decisions]
    assert rows.count(TableRow.MODELS_AGREE) == 130
    assert rows.count(TableRow.AGENT_SETTLED) == 60
    assert rows.count(TableRow.LEGACY) == 10


# --- Golden titles stay in the sampling frame (Q07, Q21) ---------------------------------


def test_golden_titles_are_drawn_and_decided_like_any_other_title_of_the_path(
    mini_inputs, long_tail
):
    simulated, settings = long_tail.simulated(mini_inputs.catalog, sme_error_rate=0.0)
    golden_keys = golden.draw_golden(long_tail.rows, (), (), settings)
    assert len(golden_keys) == 40
    machine = dict(long_tail.true_roles)
    audited = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, settings)
    # The sample is the one a path with no golden title in it would have drawn.
    assert audited.sample == audit.draw_sample(MachinePath.MODELS_AGREE, machine, settings.seed)
    drawn = [key for key in golden_keys if key in audited.sample]
    not_drawn = [key for key in golden_keys if key not in audited.sample]
    assert (len(drawn), len(not_drawn)) == (13, 27)  # on the default seed
    for key in drawn:  # labeled: its label, Approved (Q21)
        assert audited.of(key).source is DecisionSource.PERSON
        assert audited.of(key).status is Status.APPROVED
    for key in not_drawn:  # released on the machine's role with the path's status
        assert audited.of(key).source is DecisionSource.MACHINE
        assert audited.of(key).status is Status.CONFIRMED


def test_a_sample_drawn_with_the_golden_titles_held_out_gives_no_decisions(mini_inputs, long_tail):
    # Q07: golden titles stay in the sampling frame. A sample drawn from the
    # rest of the path is not the path's sample, and nothing is released on it.
    simulated, settings = long_tail.simulated(mini_inputs.catalog, sme_error_rate=0.0)
    golden_keys = set(golden.draw_golden(long_tail.rows, (), (), settings))
    machine = dict(long_tail.true_roles)
    by_key = {row.key: row for row in long_tail.rows}
    path = MachinePath.MODELS_AGREE
    rest = [key for key in machine if key not in golden_keys]
    sample = audit.draw_sample(path, rest, settings.seed)
    assert len(rest) == 160 and len(sample) == 60
    labels = {key: simulated.blind_label(by_key[key]) for key in sample}
    # audit_path is given no seed, so it cannot tell.
    result = audit.audit_path(path, machine, labels)
    assert (result.audited_keys, result.outcome) == (sample, AuditOutcome.CONFIRMED)
    with pytest.raises(ValueError, match="are not the sample that the seed 20260101 draws"):
        audit.path_decisions(result, machine, labels, settings)


# --- "No suitable role" from a real SME (Q03) --------------------------------------------


def test_a_sampled_title_with_no_truth_row_is_wrong_and_stays_on_its_legacy_title(
    mini_inputs, long_tail
):
    settings = Settings(sme_error_rate=0.0)
    machine = dict(long_tail.true_roles)
    sample = audit.draw_sample(MachinePath.MODELS_AGREE, machine, settings.seed)
    by_key = {row.key: row for row in long_tail.rows}
    orphan = sample[0]
    truth = [row for row in long_tail.truth if row.job_title not in by_key[orphan].raw_titles]
    assert len(truth) == 199
    simulated = people.SimulatedPeople(truth, mini_inputs.catalog, settings)
    audited = run_audit(MachinePath.MODELS_AGREE, long_tail.rows, machine, simulated, settings)
    assert audited.labels[orphan].answer is HumanAnswer.NO_SUITABLE_ROLE
    assert (audited.result.wrong, audited.result.outcome) == (1, AuditOutcome.CONFIRMED)
    assert audited.of(orphan) == Decision(
        orphan,
        TableRow.LEGACY,
        DecisionSource.NONE,
        legacy_reason=ExceptionReason.NO_SUITABLE_ROLE,
    )
    assert sum(decision.row is TableRow.LEGACY for decision in audited.decisions) == 1


# --- The same audit in every process, on the standard library alone (Q06) ---------------

# Run after the import guard of tests/conftest.py. The machine's roles are
# handed in through a set, whose order changes with the hash seed of the process.
AUDIT_BOTH_PATHS = """
import json
from pathlib import Path

from titlestd import audit, catalog, datafiles, people, prepare
from titlestd.contracts import Employee, MachinePath
from titlestd.settings import Settings

data = datafiles.read_inputs(Path(sys.argv[1]))
settings = Settings(sme_error_rate=0.1)
employees = [
    Employee(f"M{number:04d}", f"Made Up Title {number:03d}", "SAL", "Somewhere", 5, "")
    for number in range(200)
]
rows = prepare.prepare(employees, data.abbreviations, catalog.family_codes(data.catalog))
truth = [people.TruthRow(employee.job_title, "SAL", "SAL-01") for employee in employees]
simulated = people.SimulatedPeople(truth, data.catalog, settings)
by_key = {row.key: row for row in rows}
audits = {}
for path in MachinePath:
    machine = {key: "SAL-01" for key in set(by_key)}
    sample = audit.draw_sample(path, set(machine), settings.seed)
    labels = {key: simulated.blind_label(by_key[key]) for key in set(sample)}
    result = audit.audit_path(path, machine, labels)
    decisions = audit.path_decisions(result, machine, labels, settings)
    audits[path.value] = {
        "sample": sample,
        "audited": result.audited_keys,
        "wrong": result.wrong,
        "outcome": result.outcome.value,
        "decisions": [
            [*d.key, d.row.value, d.source.value, d.role_id, d.status, d.decided_by, d.error_rate]
            for d in decisions
        ],
    }
print(json.dumps({"audits": audits, "attempted_imports": attempted_imports}))
"""


def _audits_in_this_process(mini_inputs: InputData) -> dict:
    settings = Settings(sme_error_rate=0.1)
    employees = [
        Employee(f"M{number:04d}", f"Made Up Title {number:03d}", "SAL", "Somewhere", 5, "")
        for number in range(200)
    ]
    rows = prepare.prepare(
        employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )
    truth = [people.TruthRow(employee.job_title, "SAL", "SAL-01") for employee in employees]
    simulated = people.SimulatedPeople(truth, mini_inputs.catalog, settings)
    machine = {row.key: "SAL-01" for row in rows}
    audits = {}
    for path in MachinePath:
        audited = run_audit(path, rows, machine, simulated, settings)
        audits[path.value] = {
            "sample": audited.sample,
            "audited": audited.result.audited_keys,
            "wrong": audited.result.wrong,
            "outcome": audited.result.outcome.value,
            "decisions": [
                [
                    *d.key,
                    d.row.value,
                    d.source.value,
                    d.role_id,
                    d.status,
                    d.decided_by,
                    d.error_rate,
                ]
                for d in audited.decisions
            ],
        }
    return json.loads(json.dumps(audits))


def test_an_audit_is_the_same_in_every_process(mini_dir, mini_inputs, repo_root, import_guard):
    expected = _audits_in_this_process(mini_inputs)
    # Both paths are sampled, and the SMEs' mistakes make them provisional.
    assert {path: audit_["outcome"] for path, audit_ in expected.items()} == {
        "models agree": "Provisional",
        "agent-settled": "Provisional",
    }
    assert expected["models agree"]["sample"] != expected["agent-settled"]["sample"]
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", import_guard + AUDIT_BOTH_PATHS, str(mini_dir)],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        printed = json.loads(result.stdout)
        assert printed["audits"] == expected, hash_seed
        assert printed["attempted_imports"] == []


def test_an_audit_runs_on_a_bare_interpreter(
    mini_dir: Path, mini_inputs, bare_python, import_guard
):
    # Nothing installed and nothing outside the standard library imported (2.5).
    result = bare_python(import_guard + AUDIT_BOTH_PATHS, str(mini_dir))
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    assert printed["audits"] == _audits_in_this_process(mini_inputs)
    assert printed["attempted_imports"] == []
