"""The offline embedder on the mini data set: meaningful and repeatable (1.5, Q12).

The nearest-neighbour match itself is built later (titlestd.matcher). Here the
texts are put together by hand, the way 1.6 step 3 describes them: a role's
name and one-line definition, a title row's cleaned title and description.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from titlestd.contracts import CatalogRole, InputData, Record, TitleRow, Vector
from titlestd.offline_embedder import build_embedder
from titlestd.settings import Settings

# The role each model-eligible mini row is nearest to, written out by hand.
# "bdr" (an unexpanded acronym, no description) and "special projects" (wording
# shared with no role) are left out: nothing in the catalog is near them.
NEAREST = {
    ("FIN", "accounts payable clerk"): "FIN-05",
    ("FIN", "financial analyst"): "FIN-01",
    ("FIN", "internal auditor"): "FIN-04",
    ("FIN", "payroll specialist"): "FIN-03",
    ("FIN", "senior financial analyst"): "FIN-01",
    ("FIN", "treasury analyst"): "FIN-06",
    ("SAL", "account manager ii"): "SAL-01",
    ("SAL", "sales engineer"): "SAL-04",
    ("SAL", "sales operations analyst"): "SAL-03",
    ("SAL", "sales representative"): "SAL-02",
    ("SAL", "senior account manager"): "SAL-01",
}
FAR_FROM_EVERYTHING = [("FIN", "special projects"), ("SAL", "bdr")]

# Prints one digest of the vectors of the texts given on standard input.
DIGEST_SCRIPT = """
import hashlib, json, sys

from titlestd.offline_embedder import build_embedder
from titlestd.settings import Settings

vectors = build_embedder(Settings()).embed(json.load(sys.stdin))
plain = json.dumps([sorted(vector.items()) for vector in vectors])
print(hashlib.sha256(plain.encode()).hexdigest())
"""


def dot(a: Vector, b: Vector) -> float:
    return math.fsum(weight * b.get(index, 0.0) for index, weight in a.items())


def role_text(role: CatalogRole) -> str:
    return f"{role.role_name}. {role.definition}"


def record_text(record: Record) -> str:
    return f"{record.cleaned_title} {record.description}".strip()


def digest(vectors: list[Vector]) -> str:
    plain = json.dumps([sorted(vector.items()) for vector in vectors])
    return hashlib.sha256(plain.encode()).hexdigest()


@pytest.fixture(scope="module")
def roles(mini_inputs: InputData) -> list[CatalogRole]:
    """The roles a model may be shown, by role_id."""
    visible = [role for role in mini_inputs.catalog.roles if not role.is_generic]
    return sorted(visible, key=lambda role: role.role_id)


@pytest.fixture(scope="module")
def eligible(mini_rows: tuple[TitleRow, ...]) -> list[TitleRow]:
    return [row for row in mini_rows if row.model_eligible]


@pytest.fixture(scope="module")
def scores(roles, eligible) -> dict[tuple[str, str], list[tuple[float, str]]]:
    """Per row: (cosine, role_id) for every role, best first, ties by role_id."""
    embedder = build_embedder(Settings())
    role_vectors = embedder.embed([role_text(role) for role in roles])
    row_vectors = embedder.embed([record_text(row.record) for row in eligible])
    return {
        tuple(row.key): sorted(
            (
                (dot(vector, role_vector), role.role_id)
                for role, role_vector in zip(roles, role_vectors, strict=True)
            ),
            key=lambda scored: (-scored[0], scored[1]),
        )
        for row, vector in zip(eligible, row_vectors, strict=True)
    }


def test_the_table_covers_every_model_eligible_row(eligible):
    assert sorted([*NEAREST, *FAR_FROM_EVERYTHING]) == [tuple(row.key) for row in eligible]


def test_each_title_is_nearest_to_its_role_by_a_clear_margin(scores):
    for key, role_id in NEAREST.items():
        (first, first_id), (second, _) = scores[key][:2]
        assert first_id == role_id, key
        assert first - second > 0.2, key  # the margin of 1.6 step 3


def test_the_two_spellings_of_the_worked_example_meet_at_one_role(scores):
    # 1.3: "ACCT MGR II" and "Senior Account Manager" are the same job.
    assert scores[("SAL", "account manager ii")][0][1] == "SAL-01"
    assert scores[("SAL", "senior account manager")][0][1] == "SAL-01"


def test_a_title_that_resembles_no_role_has_a_low_score_and_no_margin(scores):
    for key in FAR_FROM_EVERYTHING:
        (first, _), (second, _) = scores[key][:2]
        assert first < 0.2, key
        assert first - second < 0.05, key


def test_every_cosine_is_between_zero_and_one(scores):
    for ranked in scores.values():
        assert all(-1e-12 <= cosine <= 1 + 1e-12 for cosine, _ in ranked)


def test_the_title_alone_is_enough_when_there_is_no_description(eligible, scores):
    # 2.4, missing data: "No description: use the title alone."
    (row,) = [row for row in eligible if row.key.cleaned_title == "payroll specialist"]
    assert row.record.description == ""
    assert scores[tuple(row.key)][0][1] == "FIN-03"


@pytest.mark.parametrize("hash_seed", ["0", "1", "12345"])
def test_another_process_gives_the_same_vectors(repo_root: Path, roles, eligible, hash_seed):
    # "The same text gives the same vector in every process": nothing rests on
    # hash(), on the order of a set or on anything kept from an earlier call.
    texts = [role_text(role) for role in roles]
    texts += [record_text(row.record) for row in eligible]
    result = subprocess.run(
        [sys.executable, "-c", DIGEST_SCRIPT],
        input=json.dumps(texts),
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": hash_seed},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == digest(build_embedder(Settings()).embed(texts))
