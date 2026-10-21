"""The shared code around choose on the mini data set, with the modules of stage 1.

The title rows come from the real readers and the real prepare step, the roles
from titlestd.catalog, the vectors from the offline embedder and the answers
from the stand-in LLM, both built by the switch. What a model is shown is
rendered here, and what it answers is mapped and quote-checked here
(DESIGN.md 1.6 step 4, 2.5; docs/build-decisions.md Q15, Q18, S16).

The nearest-neighbour match (titlestd.matcher) is left out of these tests:
shortlist() ranks the roles in miniature by the rule of 1.6 step 3. It is
test-only code. No expected answer comes from the truth file: each is written
out by hand from Q17.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from titlestd import catalog, datafiles, fallback, prepare, reference, switch
from titlestd.contracts import (
    EMAIL_MASK,
    NAME_MASK,
    SHORTLIST_SIZE,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    InputData,
    Models,
    NumberedRole,
    TitleKey,
    TitleRow,
)
from titlestd.settings import Settings

SEED = Settings().seed

# What the stand-in LLM answers for each model-eligible mini row: the kind of
# answer, the role and the quote, worked out by hand from Q17.
EXPECTED = {
    TitleKey("FIN", "accounts payable clerk"): ("role", "FIN-05", "accounts payable clerk"),
    TitleKey("FIN", "financial analyst"): ("role", "FIN-01", "financial analyst"),
    TitleKey("FIN", "internal auditor"): ("role", "FIN-04", "internal auditor"),
    TitleKey("FIN", "payroll specialist"): ("role", "FIN-03", "payroll specialist"),
    TitleKey("FIN", "senior financial analyst"): ("role", "FIN-01", "financial analyst"),
    TitleKey("FIN", "special projects"): ("no_match", None, "special projects"),
    TitleKey("FIN", "treasury analyst"): ("role", "FIN-06", "treasury analyst"),
    TitleKey("SAL", "account manager ii"): ("role", "SAL-01", "account manager"),
    TitleKey("SAL", "bdr"): ("insufficient_data", None, "bdr"),
    TitleKey("SAL", "sales engineer"): ("role", "SAL-04", "sales engineer"),
    TitleKey("SAL", "sales operations analyst"): ("role", "SAL-03", "sales operations analyst"),
    TitleKey("SAL", "sales representative"): ("role", "SAL-02", "sales representative"),
    TitleKey("SAL", "senior account manager"): ("role", "SAL-01", "account manager"),
}
SENIOR_ACCOUNT_MANAGER = TitleKey("SAL", "senior account manager")


def shortlist(models: Models, row: TitleRow, visible: tuple[CatalogRole, ...]) -> list[CatalogRole]:
    """The ten roles nearest to the row's title and description, ties by role_id."""
    query = f"{row.record.cleaned_title} {row.record.description}".strip()
    texts = [f"{role.role_name}. {role.definition}" for role in visible]
    target, *vectors = models.embedder.embed([query, *texts])
    scores = [
        math.fsum(weight * vector.get(index, 0.0) for index, weight in target.items())
        for vector in vectors
    ]
    ranked = sorted(range(len(visible)), key=lambda at: (-scores[at], visible[at].role_id))
    return [visible[at] for at in ranked[:SHORTLIST_SIZE]]


@pytest.fixture(scope="module")
def rows(mini_inputs: InputData, mini_rows) -> dict[TitleKey, TitleRow]:
    """Every title row of the mini data set, as the real prepare step builds it."""
    prepared = prepare.prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )
    assert prepared == mini_rows
    return {row.key: row for row in prepared}


@pytest.fixture(scope="module")
def models(mini_inputs: InputData) -> Models:
    return switch.build_models(Settings(), mini_inputs.catalog.ladder, mini_inputs.abbreviations)


@pytest.fixture(scope="module")
def shown(mini_inputs: InputData, rows, models) -> dict[TitleKey, tuple[NumberedRole, ...]]:
    """The numbered list of every row a model may see."""
    visible = catalog.model_visible_roles(mini_inputs.catalog)
    return {
        key: fallback.number_roles(shortlist(models, row, visible), key, SEED)
        for key, row in rows.items()
        if row.model_eligible
    }


@pytest.fixture(scope="module")
def answers(rows, models, shown) -> dict[TitleKey, ChooseAnswer]:
    """What the stand-in LLM answers when it is shown each list."""
    return {key: models.llm.chooser.choose(rows[key].record, shown[key]) for key in shown}


# --- The record block -------------------------------------------------------------------------


def test_the_record_block_of_a_prepared_row(rows):
    assert fallback.render_record(rows[SENIOR_ACCOUNT_MANAGER].record) == (
        "Title: senior account manager\n"
        "Family: SAL\n"
        "Department: Enterprise Sales\n"
        "Description: Owns the largest customer accounts and negotiates renewals. "
        "Reports to [NAME]."
    )
    # A row whose employees have no description: the title alone (2.4).
    assert fallback.render_record(rows[TitleKey("SAL", "bdr")].record) == (
        "Title: bdr\nFamily: SAL\nDepartment: Inside Sales\nDescription:"
    )


def test_every_block_holds_the_four_fields_of_its_record_and_nothing_else(rows):
    for row in rows.values():
        record = row.record
        lines = fallback.render_record(record).split("\n")
        assert [line.partition(":")[0] for line in lines] == [
            "Title",
            "Family",
            "Department",
            "Description",
        ]
        assert [line.partition(":")[2].removeprefix(" ") for line in lines] == [
            record.cleaned_title,
            record.family_code,
            record.department,
            record.description,
        ]


def test_no_name_and_no_email_address_of_the_employees_file_is_in_a_block(mini_inputs, rows):
    # 1.4: names are masked before any text reaches a model. The masking is
    # the prepare step's; the block must show its masks and add nothing back.
    blocks = {key: fallback.render_record(row.record) for key, row in rows.items()}
    on_file = " ".join(
        f"{employee.job_title} {employee.job_description}" for employee in mini_inputs.employees
    ).lower()
    names = [name for name in (*reference.GIVEN_NAMES, *reference.SURNAMES) if name in on_file]
    assert sorted(names) == ["ashgrove", "maren", "oswin", "quillfeather", "tavi", "thistlewood"]
    assert reference.EMAIL_DOMAIN in on_file
    for block in blocks.values():
        assert not [name for name in names if name in block.lower()]
        assert "@" not in block and reference.EMAIL_DOMAIN not in block
    assert f"Reports to {NAME_MASK}." in blocks[SENIOR_ACCOUNT_MANAGER]
    assert f"Questions to {EMAIL_MASK}." in blocks[TitleKey("SAL", "sales operations analyst")]


# --- The numbered list: never a generic family role (2.7) -------------------------------------


def test_every_eligible_row_is_shown_ten_numbered_roles_and_never_a_generic_one(
    mini_inputs, rows, shown
):
    eligible = [key for key, row in rows.items() if row.model_eligible]
    assert sorted(shown) == sorted(EXPECTED) == eligible and len(eligible) == 13
    generic = {role.role_id for role in mini_inputs.catalog.roles if role.is_generic}
    assert generic == {"SAL-GEN", "FIN-GEN"}
    for numbered in shown.values():
        assert [each.number for each in numbered] == list(range(1, SHORTLIST_SIZE + 1))
        assert len({each.role.role_id for each in numbered}) == SHORTLIST_SIZE
        assert generic.isdisjoint(each.role.role_id for each in numbered)
        text = fallback.render_roles(numbered)
        assert "Generic role" not in text and len(text.split("\n")) == SHORTLIST_SIZE


def test_the_catalog_as_read_cannot_be_numbered_or_shown(mini_inputs, shown):
    # The catalog file holds the generic roles, so a caller that passed every
    # role of the catalog, or of a family, is stopped before a model reads one.
    every_role = mini_inputs.catalog.roles
    with pytest.raises(ValueError, match="generic family role FIN-GEN"):
        fallback.number_roles(every_role, SENIOR_ACCOUNT_MANAGER, SEED)
    sales = [role for role in every_role if role.family_code == "SAL"]
    with pytest.raises(ValueError, match="generic family role SAL-GEN"):
        fallback.number_roles(sales, SENIOR_ACCOUNT_MANAGER, SEED)
    for generic in (role for role in every_role if role.is_generic):
        slipped_in = (*shown[SENIOR_ACCOUNT_MANAGER][:9], NumberedRole(10, generic))
        with pytest.raises(ValueError, match=generic.role_id):
            fallback.render_roles(slipped_in)


def test_the_list_a_model_is_shown_for_one_title(shown):
    # Pinned in full: this text is part of a prompt, so of a recording's key (Q18).
    assert fallback.render_roles(shown[SENIOR_ACCOUNT_MANAGER]) == "\n".join(
        [
            "1. Accountant (family FIN): Keeps the general ledger and prepares monthly accounts.",
            "2. Accounts Payable Clerk (family FIN): Processes supplier invoices and payments.",
            "3. Customer Success Manager (family SAL): Helps existing customers adopt products "
            "and stay satisfied after the sale.",
            "4. Internal Auditor (family FIN): Tests financial controls and reports findings to "
            "the audit committee.",
            "5. Treasury Analyst (family FIN): Manages cash positions, bank accounts and "
            "short-term funding.",
            "6. Sales Engineer (family SAL): Explains the technical side of products during a "
            "sale and runs demonstrations.",
            "7. Financial Analyst (family FIN): Builds budgets, forecasts and variance reports.",
            "8. Sales Operations Analyst (family SAL): Maintains sales forecasts, territories "
            "and pipeline reports.",
            "9. Sales Representative (family SAL): Sells products to new customers and closes "
            "deals.",
            "10. Account Manager (family SAL): Manages a portfolio of customer accounts and "
            "renews their contracts.",
        ]
    )


def test_no_role_id_of_the_catalog_is_shown_in_any_list(mini_inputs, shown):
    role_ids = [role.role_id for role in mini_inputs.catalog.roles]
    for numbered in shown.values():
        text = fallback.render_roles(numbered)
        assert not [role_id for role_id in role_ids if role_id in text]


def test_the_roles_are_shuffled_from_title_to_title(shown, answers):
    orders = {tuple(each.role.role_id for each in numbered) for numbered in shown.values()}
    assert len(orders) == len(shown)
    # So the stand-in's answers are spread over the numbers, not all "1".
    assert len({answer.number for answer in answers.values() if answer.number}) >= 4


# Run in another process: number and render the lists again from the role ids.
RENDER_AGAIN = """
import json, sys
from pathlib import Path
from titlestd import datafiles, fallback
from titlestd.contracts import TitleKey

given = json.load(sys.stdin)
catalog = datafiles.read_inputs(Path(given["data_dir"])).catalog
by_id = {role.role_id: role for role in catalog.roles}
lists = {}
for family, title, role_ids in given["shortlists"]:
    numbered = fallback.number_roles(
        {by_id[role_id] for role_id in role_ids}, TitleKey(family, title), given["seed"]
    )
    lists[f"{family}/{title}"] = fallback.render_roles(numbered)
print(json.dumps(lists))
"""


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_another_process_shows_every_title_the_same_list(repo_root, mini_dir, shown, hash_seed):
    # Q18: the order depends only on the seed and the title. The other process
    # is handed the ten roles of each title as a set, in whatever order its
    # hash seed puts them.
    given = {
        "data_dir": str(mini_dir),
        "seed": SEED,
        "shortlists": [
            [key.family_code, key.cleaned_title, sorted(each.role.role_id for each in numbered)]
            for key, numbered in shown.items()
        ],
    }
    result = subprocess.run(
        [sys.executable, "-c", RENDER_AGAIN],
        input=json.dumps(given),
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        f"{key.family_code}/{key.cleaned_title}": fallback.render_roles(numbered)
        for key, numbered in shown.items()
    }


# --- The stand-in's answers through the mapping and the quote check ---------------------------


@pytest.mark.parametrize("key", sorted(EXPECTED), ids=" / ".join)
def test_the_stand_in_s_number_names_the_role_worked_out_by_hand(shown, answers, key):
    kind, role_id, quote = EXPECTED[key]
    answer = answers[key]
    assert (answer.answer, answer.quote) == (kind, quote)
    role = fallback.role_for_number(answer.number, shown[key])
    assert (role.role_id if role else None) == role_id
    if kind != "role":
        assert answer.number is None


def test_every_stand_in_quote_passes_the_quote_check_on_the_block_as_rendered(rows, answers):
    # Q15, Q17: the stand-in copies its quote from the record, and the block
    # shows the record verbatim, so the stand-in never fails the check itself.
    for key, answer in answers.items():
        assert fallback.quote_in_record(answer.quote, fallback.render_record(rows[key].record)), key


def test_the_legacy_title_as_recorded_is_not_a_quote_from_the_record(rows, answers):
    # The model is shown the cleaned title. "ACCT MGR II" as the employees file
    # has it was never shown, so it is no evidence: the abbreviations do not
    # stand in the block, their long forms do.
    row = rows[TitleKey("SAL", "account manager ii")]
    assert row.raw_titles == ("ACCT MGR II",)
    block = fallback.render_record(row.record)
    assert not fallback.quote_in_record("ACCT MGR II", block)
    assert not fallback.quote_in_record("acct mgr", block)
    assert fallback.quote_in_record("ACCOUNT MANAGER II", block)
    assert answers[row.key].quote == "account manager"


def test_a_piece_of_a_word_of_the_record_is_not_a_quote_from_it(rows):
    # "account" is in this record twice, inside "accounts" and inside
    # "Accounting", and never as a word of its own.
    block = fallback.render_record(rows[TitleKey("FIN", "accounts payable clerk")].record)
    assert block.lower().count("account") == 2
    assert not fallback.quote_in_record("account", block)
    assert not fallback.quote_in_record("account payable clerk", block)
    assert fallback.quote_in_record("accounts", block)
    assert fallback.quote_in_record("Accounting", block)


def test_a_quote_is_evidence_for_its_own_record_only(rows, answers):
    blocks = {key: fallback.render_record(rows[key].record) for key in answers}
    for key, answer in answers.items():
        elsewhere = [
            other
            for other, block in blocks.items()
            if other != key and fallback.quote_in_record(answer.quote, block)
        ]
        # Two titles share the words "financial analyst" and two "account manager".
        allowed = {
            TitleKey("FIN", "financial analyst"): [TitleKey("FIN", "senior financial analyst")],
            TitleKey("FIN", "senior financial analyst"): [TitleKey("FIN", "financial analyst")],
            TitleKey("SAL", "account manager ii"): [SENIOR_ACCOUNT_MANAGER],
            SENIOR_ACCOUNT_MANAGER: [TitleKey("SAL", "account manager ii")],
        }
        assert elsewhere == allowed.get(key, []), key


def test_a_number_the_list_does_not_hold_names_no_role_of_the_catalog(shown):
    for numbered in shown.values():
        for number in (0, SHORTLIST_SIZE + 1, 12, -1, None):
            assert fallback.role_for_number(number, numbered) is None
        named = [fallback.role_for_number(number, numbered) for number in range(1, 11)]
        assert named == [each.role for each in numbered]


def test_the_kinds_of_answer_the_design_names_all_occur(answers):
    kinds = {answer.answer for answer in answers.values()}
    assert kinds == {ChooseKind.ROLE, ChooseKind.NO_MATCH, ChooseKind.INSUFFICIENT_DATA}


def test_the_readers_give_back_the_same_lists_from_a_copy_of_the_files(
    tmp_path, mini_inputs, shown
):
    # The lists are made from what the catalog file says, not from objects in memory.
    datafiles.write_inputs(tmp_path, mini_inputs)
    copied = datafiles.read_inputs(tmp_path)
    by_id = {role.role_id: role for role in copied.catalog.roles}
    for key, numbered in shown.items():
        again = fallback.number_roles([by_id[each.role.role_id] for each in numbered], key, SEED)
        assert again == numbered
