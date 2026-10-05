"""The close on the mini data set, with the real prepare step, catalog and people.

The title rows come from titlestd.prepare, and who could not decide comes from
the simulated people of titlestd.people, read from the truth file. The models
and the rules are built beside this branch, so what they would hand the close
is stood in for by hand: the Wave 1 list is the four most widely held rows
(Q24), the seed look-up hits never reach the queue (Q13), and the compliance
check of 1.7 is taken to have fired on the one title of the flagged role.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from titlestd import catalog, close, people, prepare
from titlestd.contracts import (
    NAME_MASK,
    Decision,
    ExceptionReason,
    HumanAnswer,
    InputData,
    Leftover,
    TableRow,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings

# The title of the mini catalog's one compliance-flagged role, FIN-04.
FLAGGED = frozenset({TitleKey("FIN", "internal auditor")})


@pytest.fixture(scope="module")
def rows(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    return prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )


@pytest.fixture(scope="module")
def wave1(rows: tuple[TitleRow, ...]) -> frozenset[TitleKey]:
    """The four most widely held rows: headcount, then family code, then title (Q24)."""
    return frozenset(row.key for row in sorted(rows, key=lambda row: (-row.headcount, row.key))[:4])


@pytest.fixture(scope="module")
def seed_hits(mini_inputs: InputData) -> frozenset[TitleKey]:
    return frozenset(
        prepare.title_key(seed.job_title, seed.job_family_code, mini_inputs.abbreviations)
        for seed in mini_inputs.seed_lookup
    )


def _settings(data_dir: Path, **overrides) -> Settings:
    # No simulated mistakes, so a label is the truth and the tests can name it.
    return Settings(data_dir=data_dir, sme_error_rate=0.0, **overrides)


def _leftovers(rows, wave1, seed_hits, mini_inputs: InputData, settings: Settings):
    """What people leave for the close: the titles nobody gave a role."""
    simulated = people.load_people(settings, mini_inputs.catalog)
    leftovers = []
    for row in rows:
        if row.key in seed_hits:
            continue
        if row.key in wave1:
            answer = simulated.wave1_approval(row).answer
        else:
            answer = simulated.queue_decision(row).answer
        if answer is not HumanAnswer.ROLE:
            leftovers.append(
                Leftover(
                    row,
                    on_wave1_list=row.key in wave1,
                    compliance_fired=row.key in FLAGGED,
                    no_suitable_role=answer is HumanAnswer.NO_SUITABLE_ROLE,
                )
            )
    return leftovers


def _closed(rows, wave1, seed_hits, mini_inputs, settings) -> dict[TitleKey, Decision]:
    leftovers = _leftovers(rows, wave1, seed_hits, mini_inputs, settings)
    decisions = close.close_leftovers(
        leftovers, len(mini_inputs.employees), mini_inputs.catalog, settings
    )
    assert [decision.key for decision in decisions] == sorted(
        leftover.row.key for leftover in leftovers
    )
    return {decision.key: decision for decision in decisions}


def _without_truth(tmp_path: Path, mini_dir: Path, *raw_titles: str) -> Path:
    """A copy of the mini data set whose truth file has no row for the raw titles."""
    data_dir = tmp_path / "data"
    shutil.copytree(mini_dir, data_dir)
    truth = [row for row in people.read_truth(data_dir) if row.job_title not in raw_titles]
    assert len(truth) == 17 - len(raw_titles)
    people.write_truth(data_dir, truth)
    return data_dir


def test_the_close_is_tried_on_the_rows_the_prepare_step_builds(rows, mini_rows, wave1, seed_hits):
    assert rows == mini_rows
    assert sorted(wave1) == [
        TitleKey("FIN", "financial analyst"),
        TitleKey("SAL", "account manager ii"),
        TitleKey("SAL", "sales representative"),
        TitleKey("SAL", "senior account manager"),
    ]
    assert sorted(seed_hits) == [
        TitleKey("FIN", "accounts payable clerk"),
        TitleKey("SAL", "sales engineer"),
    ]


def test_when_nobody_can_decide_the_smallest_titles_take_the_generic_role_up_to_the_cap(
    rows, wave1, seed_hits, mini_inputs: InputData, mini_dir: Path
):
    # Every queue title is undecided (Q01). A quarter of 22 employees is 5.
    settings = _settings(mini_dir, undecided_rate=1.0, generic_cap_share=0.25)
    assert close.generic_cap(len(mini_inputs.employees), settings) == 5
    decisions = _closed(rows, wave1, seed_hits, mini_inputs, settings)
    # Wave 1 titles and look-up hits were decided before the close.
    assert len(decisions) == 15 - 4 - 2

    generic = {key: d.role_id for key, d in decisions.items() if d.row is TableRow.GENERIC}
    # Six titles of one employee each are eligible; family code, then title, orders them.
    assert generic == {
        TitleKey("FIN", "payroll specialist"): "FIN-GEN",
        TitleKey("FIN", "senior financial analyst"): "FIN-GEN",
        TitleKey("FIN", "special projects"): "FIN-GEN",
        TitleKey("FIN", "treasury analyst"): "FIN-GEN",
        TitleKey("SAL", "bdr"): "SAL-GEN",
    }
    by_key = {row.key: row for row in rows}
    assert sum(by_key[key].headcount for key in generic) == 5
    assert {key: d.legacy_reason for key, d in decisions.items() if d.row is TableRow.LEGACY} == {
        TitleKey("SAL", "sales operations analyst"): ExceptionReason.OVER_GENERIC_CAP,
        TitleKey("FIN", "internal auditor"): ExceptionReason.UNDECIDED_BARRED,
        TitleKey("", "staff accountant"): ExceptionReason.UNDECIDED_UNSCORED,
        TitleKey(
            "SAL", f"customer success manager {NAME_MASK}"
        ): ExceptionReason.UNDECIDED_UNSCORED,
    }
    for decision in decisions.values():
        if decision.row is TableRow.GENERIC:
            role = catalog.get_role(mini_inputs.catalog, decision.role_id)
            assert role.is_generic and role.family_code == decision.key.family_code
            assert role == catalog.generic_role(mini_inputs.catalog, decision.key.family_code)


def test_a_leftover_with_no_usable_family_stays_on_its_legacy_title(
    rows, wave1, seed_hits, mini_inputs: InputData, mini_dir: Path
):
    # Q26. The truth names a role for it, and the cap has room for everybody,
    # but with no family there is no generic role to take.
    settings = _settings(mini_dir, undecided_rate=1.0, generic_cap_share=1.0)
    decisions = _closed(rows, wave1, seed_hits, mini_inputs, settings)
    staff = decisions[TitleKey("", "staff accountant")]
    assert (staff.row, staff.role_id) == (TableRow.LEGACY, None)
    assert staff.legacy_reason is ExceptionReason.UNDECIDED_UNSCORED
    assert not catalog.is_usable_family(mini_inputs.catalog, staff.key.family_code)
    # Everything eligible is taken, and the three barred titles are not.
    assert sum(decision.row is TableRow.GENERIC for decision in decisions.values()) == 6
    assert sorted(
        key.cleaned_title for key, d in decisions.items() if d.row is TableRow.LEGACY
    ) == [f"customer success manager {NAME_MASK}", "internal auditor", "staff accountant"]


def test_a_queue_title_with_no_truth_row_is_never_generic_and_never_counts_against_the_cap(
    rows, wave1, seed_hits, mini_inputs: InputData, mini_dir: Path, tmp_path: Path
):
    # Q03: the reviewer marks it "no suitable role" whatever the undecided rate.
    data_dir = _without_truth(tmp_path, mini_dir, "Special Projects")
    settings = _settings(data_dir, undecided_rate=1.0, generic_cap_share=0.25)
    decisions = _closed(rows, wave1, seed_hits, mini_inputs, settings)
    marked = decisions[TitleKey("FIN", "special projects")]
    assert (marked.row, marked.role_id) == (TableRow.LEGACY, None)
    assert marked.legacy_reason is ExceptionReason.NO_SUITABLE_ROLE
    # With the truth row it took the fourth of five places under the cap. Now
    # the title that was over the cap has a place.
    generic = sorted(key for key, d in decisions.items() if d.row is TableRow.GENERIC)
    assert generic == [
        TitleKey("FIN", "payroll specialist"),
        TitleKey("FIN", "senior financial analyst"),
        TitleKey("FIN", "treasury analyst"),
        TitleKey("SAL", "bdr"),
        TitleKey("SAL", "sales operations analyst"),
    ]


def test_a_wave1_title_with_no_truth_row_waits_for_the_catalog_owner(
    rows, wave1, seed_hits, mini_inputs: InputData, mini_dir: Path, tmp_path: Path
):
    # The SME marks it in Wave 1 (Q03), so it comes to the close as a leftover
    # that is on the Wave 1 list. Both rules keep it off the generic role.
    data_dir = _without_truth(tmp_path, mini_dir, "ACCT MGR II")
    settings = _settings(data_dir, undecided_rate=0.0, generic_cap_share=1.0)
    decisions = _closed(rows, wave1, seed_hits, mini_inputs, settings)
    assert list(decisions) == [TitleKey("SAL", "account manager ii")]
    marked = decisions[TitleKey("SAL", "account manager ii")]
    assert (marked.row, marked.legacy_reason) == (TableRow.LEGACY, ExceptionReason.NO_SUITABLE_ROLE)


def test_at_an_undecided_rate_of_zero_there_is_nothing_to_close(
    rows, wave1, seed_hits, mini_inputs: InputData, mini_dir: Path
):
    # Q01: setting the rate to 0 turns the mechanism off.
    settings = _settings(mini_dir, undecided_rate=0.0, generic_cap_share=1.0)
    assert _closed(rows, wave1, seed_hits, mini_inputs, settings) == {}


@pytest.mark.parametrize("share", [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("rate", [0.3, 0.6, 1.0])
def test_whoever_cannot_decide_the_generic_role_stays_under_its_cap(
    rows, wave1, seed_hits, mini_inputs: InputData, mini_dir: Path, share: float, rate: float
):
    # 2.7 on the real rows, for twenty seeds: never past the cap, never to a
    # Wave 1 or flagged title, never to one no model scored.
    by_key = {row.key: row for row in rows}
    for seed in range(20):
        settings = _settings(mini_dir, seed=seed, undecided_rate=rate, generic_cap_share=share)
        cap = close.generic_cap(len(mini_inputs.employees), settings)
        decisions = _closed(rows, wave1, seed_hits, mini_inputs, settings)
        generic = [key for key, d in decisions.items() if d.row is TableRow.GENERIC]
        assert sum(by_key[key].headcount for key in generic) <= cap <= share * 22
        for key in generic:
            assert key not in wave1 and key not in FLAGGED
            assert by_key[key].model_eligible
            assert decisions[key].role_id == f"{key.family_code}-GEN"
        # Whoever is over the cap would not have fitted.
        on_generic = sum(by_key[key].headcount for key in generic)
        for key, decision in decisions.items():
            if decision.legacy_reason is ExceptionReason.OVER_GENERIC_CAP:
                assert on_generic + by_key[key].headcount > cap
