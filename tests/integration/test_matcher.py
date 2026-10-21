"""The ML match on the mini data set, with the real offline embedder.

The rows come from the real prepare step, the roles from the real catalog, the
embedder from the real switch and the decisions from the real simulated
people. DESIGN.md 1.5, 1.6 step 3, 2.4; docs/build-decisions.md Q04, Q12,
Q14, Q18, Q26; section 2.7: no model is ever shown a generic family role.

No expected first choice comes from the truth file: each is written out by hand.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from titlestd import catalog, people, prepare, switch
from titlestd.contracts import (
    NAME_MASK,
    SHORTLIST_SIZE,
    InputData,
    MatchResult,
    Models,
    Record,
    TitleKey,
    TitleRow,
    Vector,
)
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher, cosine, embed_text, ml_alone_eligible, role_text
from titlestd.settings import Settings

# The role each model-eligible mini row is nearest to, written out by hand.
NEAREST = {
    TitleKey("FIN", "accounts payable clerk"): "FIN-05",
    TitleKey("FIN", "financial analyst"): "FIN-01",
    TitleKey("FIN", "internal auditor"): "FIN-04",
    TitleKey("FIN", "payroll specialist"): "FIN-03",
    TitleKey("FIN", "senior financial analyst"): "FIN-01",
    TitleKey("FIN", "treasury analyst"): "FIN-06",
    TitleKey("SAL", "account manager ii"): "SAL-01",
    TitleKey("SAL", "sales engineer"): "SAL-04",
    TitleKey("SAL", "sales operations analyst"): "SAL-03",
    TitleKey("SAL", "sales representative"): "SAL-02",
    TitleKey("SAL", "senior account manager"): "SAL-01",
}
# An unexpanded acronym with no description, and wording shared with no role.
FAR_FROM_EVERYTHING = [TitleKey("FIN", "special projects"), TitleKey("SAL", "bdr")]
# The two rows no model may see: a person's name, and no usable family code.
NAMED = TitleKey("SAL", f"customer success manager {NAME_MASK}")
NO_FAMILY = TitleKey("", "staff accountant")
GENERIC = {"FIN-GEN", "SAL-GEN"}

# Prints one digest of the match of every mini row, seed look-up loaded.
DIGEST_SCRIPT = """
import hashlib, json
from pathlib import Path

from titlestd import catalog, datafiles, prepare, switch
from titlestd.lookup import Knowledge, load_seed
from titlestd.matcher import Matcher
from titlestd.settings import Settings

data = datafiles.read_inputs(Path("tests/fixtures/mini"))
rows = prepare.prepare(data.employees, data.abbreviations, catalog.family_codes(data.catalog))
models = switch.build_models(Settings(), data.catalog.ladder, data.abbreviations)
knowledge = Knowledge()
load_seed(knowledge, data.seed_lookup, rows, data.abbreviations)
results = Matcher(data.catalog, models.embedder).match(rows, knowledge)
plain = [
    [list(key), [[s.role_id, repr(s.score), s.matched_text] for s in result.shortlist]]
    for key, result in results.items()
]
print(hashlib.sha256(json.dumps(plain).encode()).hexdigest())
"""


class SpyingEmbedder:
    """The real embedder, with a record of every text it was given."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.texts += texts
        return self._embedder.embed(texts)


def digest(results: dict[TitleKey, MatchResult]) -> str:
    plain = [
        [list(key), [[s.role_id, repr(s.score), s.matched_text] for s in result.shortlist]]
        for key, result in results.items()
    ]
    return hashlib.sha256(json.dumps(plain).encode()).hexdigest()


def new_row(family: str, title: str, description: str = "") -> TitleRow:
    """A title that is not in the mini data set, as the prepare step would hand it on."""
    return TitleRow(Record(title, family, "Some Department", description), ("X001",), (title,))


@pytest.fixture(scope="module")
def rows(mini_inputs: InputData) -> tuple[TitleRow, ...]:
    return prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )


@pytest.fixture(scope="module")
def models(mini_inputs: InputData, mini_dir: Path) -> Models:
    """The default models, from the real switch."""
    settings = Settings(data_dir=mini_dir)
    return switch.build_models(settings, mini_inputs.catalog.ladder, mini_inputs.abbreviations)


@pytest.fixture
def spy(models: Models) -> SpyingEmbedder:
    return SpyingEmbedder(models.embedder)


@pytest.fixture(scope="module")
def results(mini_inputs, rows, models) -> dict[TitleKey, MatchResult]:
    """Every mini row matched against the catalog alone: nothing is approved yet."""
    return Matcher(mini_inputs.catalog, models.embedder).match(rows, Knowledge())


@pytest.fixture
def simulated(mini_settings: Settings, mini_inputs: InputData) -> people.SimulatedPeople:
    """Simulated people who never err and always decide."""
    settings = dataclasses.replace(mini_settings, sme_error_rate=0.0, undecided_rate=0.0)
    return people.load_people(settings, mini_inputs.catalog)


# --- Shortlist, first choice and margin (1.6 step 3) --------------------------------


def test_the_table_covers_every_row_and_only_the_eligible_ones_are_matched(rows, results):
    eligible = [row.key for row in rows if row.model_eligible]
    assert sorted([*NEAREST, *FAR_FROM_EVERYTHING]) == eligible
    assert list(results) == eligible
    assert NAMED not in results and NO_FAMILY not in results
    assert len(rows) == 15 and len(results) == 13


@pytest.mark.parametrize("key", sorted(NEAREST), ids=" / ".join)
def test_each_title_is_nearest_to_its_role_by_a_clear_margin(mini_inputs, results, key):
    result = results[key]
    assert result.first_choice == NEAREST[key]
    assert result.margin > 0.2
    assert result.first_choice_examples == 0
    # With nothing approved yet, what was matched is the role's catalog text.
    role = catalog.get_role(mini_inputs.catalog, NEAREST[key])
    assert result.shortlist[0].matched_text == f"{role.role_name}. {role.definition}"


@pytest.mark.parametrize("key", FAR_FROM_EVERYTHING, ids=" / ".join)
def test_a_title_that_resembles_no_role_has_a_low_score_and_no_margin(results, key):
    result = results[key]
    assert result.shortlist[0].score < 0.2
    assert result.margin < 0.05


def test_every_shortlist_holds_ten_different_roles_best_first(mini_inputs, results):
    visible = {role.role_id for role in catalog.model_visible_roles(mini_inputs.catalog)}
    assert len(visible) == 12
    for key, result in results.items():
        role_ids = [scored.role_id for scored in result.shortlist]
        assert len(role_ids) == len(set(role_ids)) == SHORTLIST_SIZE, key
        assert set(role_ids) <= visible, key
        ranked = [(-scored.score, scored.role_id) for scored in result.shortlist]
        assert ranked == sorted(ranked), key
        assert result.top_three == tuple(role_ids[:3])
        assert all(scored.matched_text for scored in result.shortlist)


def test_a_shortlist_of_ten_always_reaches_into_another_family(mini_inputs, results):
    # Six roles per family: the family rule of 1.7 is checked later, not here.
    family = {role.role_id: role.family_code for role in mini_inputs.catalog.roles}
    for key, result in results.items():
        families = {family[scored.role_id] for scored in result.shortlist}
        assert families == {"FIN", "SAL"}, key


def test_the_two_spellings_of_the_worked_example_meet_at_one_role(results):
    # 1.3: "ACCT MGR II" and "Senior Account Manager" are the same job.
    assert results[TitleKey("SAL", "account manager ii")].first_choice == "SAL-01"
    assert results[TitleKey("SAL", "senior account manager")].first_choice == "SAL-01"


def test_the_title_alone_is_enough_when_there_is_no_description(rows, results):
    # 2.4, missing data: "No description: use the title alone."
    (row,) = [row for row in rows if row.key == TitleKey("FIN", "payroll specialist")]
    assert row.record.description == "" and embed_text(row.record) == "payroll specialist"
    assert results[row.key].first_choice == "FIN-03"


def test_a_row_is_matched_the_same_alone_as_in_its_wave(mini_inputs, rows, models, results):
    # A whole wave is scored in one call (Q18); no row's answer leans on another's.
    matcher = Matcher(mini_inputs.catalog, models.embedder)
    for row in rows:
        alone = matcher.match([row], Knowledge())
        assert alone == ({row.key: results[row.key]} if row.model_eligible else {})


# --- No generic family role is embedded or shortlisted (2.7) -------------------------


def test_no_generic_family_role_is_embedded_or_on_a_shortlist(mini_inputs, rows, models, spy):
    generic_texts = [role_text(role) for role in mini_inputs.catalog.roles if role.is_generic]
    assert generic_texts == [
        "Sales. Generic role for the Sales family.",
        "Finance. Generic role for the Finance family.",
    ]
    # Two titles that name a family and no job: each reads like a generic role.
    family_titles = [new_row("SAL", "sales"), new_row("FIN", "finance")]
    knowledge = Knowledge()
    load_seed(knowledge, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    results = Matcher(mini_inputs.catalog, spy).match([*rows, *family_titles], knowledge)

    assert len(results) == 15
    for key, result in results.items():
        assert not GENERIC & {scored.role_id for scored in result.shortlist}, key
    assert not set(generic_texts) & set(spy.texts)
    assert not any("Generic role" in text for text in spy.texts)
    # The rule does the work: had the generic role been listed, it would have
    # been the first choice of the title that names its family.
    for row, generic_text in zip(family_titles, generic_texts, strict=True):
        title_vector, generic_vector = models.embedder.embed([embed_text(row.record), generic_text])
        assert cosine(title_vector, generic_vector) > results[row.key].shortlist[0].score + 0.1


def test_what_the_embedder_is_given_is_the_roles_the_examples_and_the_eligible_rows(
    mini_inputs, rows, spy
):
    knowledge = Knowledge()
    load_seed(knowledge, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    Matcher(mini_inputs.catalog, spy).match(rows, knowledge)
    visible = catalog.model_visible_roles(mini_inputs.catalog)
    expected = {role_text(role) for role in visible}
    expected |= {embed_text(example.record) for example in knowledge.examples()}
    expected |= {embed_text(row.record) for row in rows if row.model_eligible}
    assert set(spy.texts) == expected
    assert len(spy.texts) == len(expected) == 12 + 13  # the two seeded titles are rows too


# --- Rows no model may see (Q14, Q26) -------------------------------------------------


def test_the_named_title_and_the_title_with_no_family_never_reach_the_embedder(
    mini_inputs, rows, spy
):
    data_rows = {row.key: row for row in rows}
    hidden = [data_rows[NAMED], data_rows[NO_FAMILY]]
    assert [row.model_eligible for row in hidden] == [False, False]
    assert Matcher(mini_inputs.catalog, spy).match(hidden, Knowledge()) == {}
    assert spy.texts == []
    results = Matcher(mini_inputs.catalog, spy).match(rows, Knowledge())
    assert NAMED not in results and NO_FAMILY not in results
    assert not any(embed_text(row.record) in spy.texts for row in hidden)
    # A masked name inside a description may be embedded (1.4); the title that
    # held a name may not, in whole or in part.
    assert any(NAME_MASK in text for text in spy.texts)
    assert not any(NAMED.cleaned_title in text for text in spy.texts)


def test_decisions_on_those_two_titles_one_becomes_an_example_one_never(
    mini_inputs, rows, spy, simulated
):
    data_rows = {row.key: row for row in rows}
    knowledge = Knowledge()
    for key in (NAMED, NO_FAMILY):
        decision = simulated.queue_decision(data_rows[key])  # both go straight to a person
        assert knowledge.learn(data_rows[key], decision.role_id) is True
    assert dict(knowledge.entries()) == {NAMED: "SAL-06", NO_FAMILY: "FIN-02"}

    matcher = Matcher(mini_inputs.catalog, spy)
    later = new_row("FIN", "staff accountant ii")
    result = matcher.match([later, new_row("SAL", "customer success manager")], knowledge)
    # Q26: the decided title with no usable family is an example of the chosen role.
    assert result[later.key].first_choice == "FIN-02"
    assert result[later.key].shortlist[0].matched_text == "staff accountant"
    assert result[later.key].first_choice_examples == 1
    assert embed_text(data_rows[NO_FAMILY].record) in spy.texts
    # Q14: the decided title with a person's name is in the look-up and is no
    # example, so its text reaches the embedder in no way.
    assert knowledge.example_count("SAL-06") == 0
    assert not any(NAMED.cleaned_title in text for text in spy.texts)
    assert embed_text(data_rows[NAMED].record) not in spy.texts
    assert result[TitleKey("SAL", "customer success manager")].first_choice_examples == 0


# --- It improves with every human approval (1.5, 1.6 step 10, 2.4) --------------------


def test_an_approved_title_is_matched_by_the_next_title_like_it(mini_inputs, rows, models, results):
    matcher = Matcher(mini_inputs.catalog, models.embedder)
    knowledge = Knowledge()
    later = new_row("SAL", "bdr ii")
    before = matcher.match([later], knowledge)[later.key]
    assert before.shortlist[0].score < 0.2 and before.margin < 0.05

    # A person decides what the bare acronym means; the next title like it follows.
    (bdr,) = [row for row in rows if row.key == TitleKey("SAL", "bdr")]
    assert knowledge.learn(bdr, "SAL-05") is True
    after = matcher.match([later], knowledge)[later.key]
    assert after.first_choice == "SAL-05"
    assert after.shortlist[0].matched_text == "bdr"
    assert after.shortlist[0].score > 0.5 and after.margin > 0.4
    assert after.first_choice_examples == 1
    # What was scored before the decision stands as it was (Q18).
    assert results[bdr.key].first_choice_examples == 0


def test_seed_entries_are_matched_against_from_the_start(mini_inputs, rows, models, results):
    # Q13: seed entries are human-decided titles, so they are examples at once.
    knowledge = Knowledge()
    load_seed(knowledge, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    later = new_row("SAL", "sales engineer ii", "Runs demonstrations during a sale.")
    result = Matcher(mini_inputs.catalog, models.embedder).match([later], knowledge)[later.key]
    assert result.first_choice == "SAL-04"
    assert result.first_choice_examples == 1


def test_a_golden_title_is_scored_at_every_wave_and_is_never_an_example(
    mini_inputs, rows, models, simulated
):
    # Q07, Q08: its label is never learned, so no later title is matched against it.
    golden = TitleKey("SAL", "bdr")
    (bdr,) = [row for row in rows if row.key == golden]
    knowledge = Knowledge(golden_keys=[golden])
    matcher = Matcher(mini_inputs.catalog, models.embedder)
    later = new_row("SAL", "bdr ii")
    wave1 = matcher.match([bdr, later], knowledge)
    assert knowledge.learn(bdr, simulated.blind_label(bdr).role_id) is False
    for row in rows:
        if row.key != golden:
            knowledge.learn(row, simulated.blind_label(row).role_id)
    close = matcher.match([bdr, later], knowledge)
    assert golden in wave1 and golden in close
    for result in close.values():
        assert "bdr" not in [scored.matched_text for scored in result.shortlist]
        assert result.shortlist[0].score < 0.2


# --- ML-alone eligibility (1.6 step 4, Q04) --------------------------------------------


def test_the_ml_alone_route_needs_the_switch_the_margin_the_examples_and_the_family(
    mini_inputs, rows, models, simulated
):
    matcher = Matcher(mini_inputs.catalog, models.embedder)
    knowledge = Knowledge()
    later = new_row("FIN", "financial analyst ii", "Builds budgets and forecasts.")
    on = Settings(ml_alone=True)

    fresh = matcher.match([later], knowledge)[later.key]
    assert fresh.first_choice == "FIN-01" and fresh.margin > on.ml_margin_cutoff
    assert fresh.first_choice_examples == 0
    # During the backfill examples are scarce: the title goes to the LLM (1.6 step 4).
    assert ml_alone_eligible(fresh, "FIN", mini_inputs.catalog, on) is False

    # People approve the three mini titles whose true role is Financial Analyst.
    for row in rows:
        decision = simulated.wave1_approval(row)
        if decision.role_id == "FIN-01":
            knowledge.learn(row, decision.role_id)
    assert knowledge.example_count("FIN-01") == on.ml_min_examples == 3

    scored = matcher.match([later], knowledge)[later.key]
    assert scored.first_choice == "FIN-01" and scored.first_choice_examples == 3
    assert scored.margin > on.ml_margin_cutoff
    assert ml_alone_eligible(scored, "FIN", mini_inputs.catalog, on) is True
    # Built and off by default (2.4).
    assert ml_alone_eligible(scored, "FIN", mini_inputs.catalog, Settings()) is False
    # A first choice outside the recorded family is the agent's business, not this route's.
    assert ml_alone_eligible(scored, "SAL", mini_inputs.catalog, on) is False
    assert ml_alone_eligible(scored, "", mini_inputs.catalog, on) is False
    strict = Settings(ml_alone=True, ml_margin_cutoff=0.99)
    assert ml_alone_eligible(scored, "FIN", mini_inputs.catalog, strict) is False
    more = Settings(ml_alone=True, ml_min_examples=4)
    assert ml_alone_eligible(scored, "FIN", mini_inputs.catalog, more) is False


# --- The same answer every time (1.5, Q18) ----------------------------------------------


def test_two_matchers_give_the_same_answer(mini_inputs, rows, models, results):
    again = Matcher(mini_inputs.catalog, models.embedder).match(rows, Knowledge())
    assert again == results
    assert Matcher(mini_inputs.catalog, models.embedder).match(rows[::-1], Knowledge()) == results


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_another_process_gives_the_same_shortlists(repo_root, mini_inputs, rows, models, hash_seed):
    # Nothing rests on hash(), on the order of a set or on an earlier call.
    knowledge = Knowledge()
    load_seed(knowledge, mini_inputs.seed_lookup, rows, mini_inputs.abbreviations)
    expected = digest(Matcher(mini_inputs.catalog, models.embedder).match(rows, knowledge))
    result = subprocess.run(
        [sys.executable, "-c", DIGEST_SCRIPT],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
