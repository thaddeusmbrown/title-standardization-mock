"""embedder = local on the real model: the one smoke test (Q30, Q32 d).

It is marked local_model, and "make check" leaves that marker out, so it is
never part of the merge gate. Asked for by its marker where the extra is
installed,

    uv run --isolated --extra local pytest -m local_model

it downloads the model on first use into the user's Hugging Face cache,
outside the repository, and embeds the title rows of the mini data set. In
any other run it is skipped unless the library is installed and the model is
already in that cache, so no other test run reaches the network.

The nearest-neighbour match is built beside this branch (titlestd.matcher), so
the texts are put together by hand, the way 1.6 step 3 describes them: a
role's name and one-line definition, a row's cleaned title and description.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from titlestd import catalog, switch
from titlestd.contracts import CatalogRole, InputData, Record, Vector
from titlestd.local_embedder import MODEL_FILES, MODEL_NAME, MODEL_REVISION, LocalEmbedder
from titlestd.prepare import prepare
from titlestd.settings import Settings

MARKER = "local_model"
DIMENSIONS = 256  # of minishlab/potion-base-8M

# The role each title of the mini data set is nearest to, written out by hand.
# "bdr" (an unexpanded acronym, no description) and "special projects" (wording
# shared with no role) are left out: nothing in the catalog is near them.
NEAREST = {
    ("FIN", "accounts payable clerk"): "FIN-05",
    ("FIN", "financial analyst"): "FIN-01",
    ("FIN", "internal auditor"): "FIN-04",
    ("FIN", "payroll specialist"): "FIN-03",  # no description: the title alone (2.4)
    ("FIN", "senior financial analyst"): "FIN-01",
    ("FIN", "treasury analyst"): "FIN-06",
    ("SAL", "account manager ii"): "SAL-01",  # the worked example of 1.3,
    ("SAL", "sales engineer"): "SAL-04",
    ("SAL", "sales operations analyst"): "SAL-03",
    ("SAL", "sales representative"): "SAL-02",
    ("SAL", "senior account manager"): "SAL-01",  # in both its spellings
}

# Prints one digest of the vectors of the texts given on standard input.
DIGEST_SCRIPT = """
import hashlib, json, sys

from titlestd.local_embedder import build_embedder
from titlestd.settings import Settings

vectors = build_embedder(Settings(embedder="local")).embed(json.load(sys.stdin))
plain = json.dumps([sorted(vector.items()) for vector in vectors])
print(hashlib.sha256(plain.encode()).hexdigest())
"""


def asked_for_by_marker(expression: str) -> bool:
    """True when a run's -m expression names the marker to select it."""
    return MARKER in expression.replace(f"not {MARKER}", "")


def model_is_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(
        isinstance(try_to_load_from_cache(MODEL_NAME, name, revision=MODEL_REVISION), str)
        for name in MODEL_FILES
    )


def dot(a: Vector, b: Vector) -> float:
    return math.fsum(weight * b.get(index, 0.0) for index, weight in a.items())


def role_text(role: CatalogRole) -> str:
    return f"{role.role_name}. {role.definition}"


def record_text(record: Record) -> str:
    return f"{record.cleaned_title} {record.description}".strip()


def digest(vectors: list[Vector]) -> str:
    plain = json.dumps([sorted(vector.items()) for vector in vectors])
    return hashlib.sha256(plain.encode()).hexdigest()


@pytest.mark.parametrize(
    ("expression", "asked"),
    [
        ("local_model", True),
        ("local_model or live", True),
        ("integration and local_model", True),
        ("", False),  # plain pytest
        ("not live and not local_model", False),  # the merge gate (S14)
        ("not local_model", False),
        ("integration", False),
    ],
)
def test_only_a_run_that_names_the_marker_may_fetch_the_model(expression: str, asked: bool):
    assert asked_for_by_marker(expression) is asked


@pytest.mark.local_model
def test_the_local_model_embeds_the_titles_of_the_mini_data_set(
    request: pytest.FixtureRequest, repo_root: Path, mini_inputs: InputData, mini_settings: Settings
):
    pytest.importorskip("model2vec", reason='the optional extra "local" is not installed')
    asked = asked_for_by_marker(request.config.getoption("markexpr") or "")
    if not asked and not model_is_cached():
        pytest.skip(f"{MODEL_NAME} is not in the Hugging Face cache; pytest -m {MARKER} fetches it")

    # Through the switch, as the pipeline will build it.
    settings = dataclasses.replace(mini_settings, embedder="local")
    models = switch.build_models(settings, mini_inputs.catalog.ladder, mini_inputs.abbreviations)
    embedder = models.embedder
    assert isinstance(embedder, LocalEmbedder)

    roles = catalog.model_visible_roles(mini_inputs.catalog)
    rows = prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )
    eligible = {tuple(row.key): row for row in rows if row.model_eligible}
    role_vectors = embedder.embed([role_text(role) for role in roles])
    row_vectors = embedder.embed([record_text(row.record) for row in eligible.values()])

    # One dense vector of length one per text, as plain floats.
    assert len(role_vectors) == len(roles) == 12
    assert len(row_vectors) == len(eligible) == 13
    for vector in [*role_vectors, *row_vectors]:
        assert list(vector) == list(range(DIMENSIONS))
        assert all(type(weight) is float for weight in vector.values())
        assert math.fsum(weight * weight for weight in vector.values()) == pytest.approx(
            1.0, abs=1e-12
        )

    # Each title that resembles a role is nearest to it, by a clear margin.
    by_key = dict(zip(eligible, row_vectors, strict=True))
    for key, role_id in NEAREST.items():
        ranked = sorted(
            (
                (dot(by_key[key], role_vector), role.role_id)
                for role, role_vector in zip(roles, role_vectors, strict=True)
            ),
            key=lambda scored: (-scored[0], scored[1]),
        )
        (first, first_id), (second, _) = ranked[:2]
        assert first_id == role_id, key
        assert first - second > 0.1, key  # the margin of 1.6 step 3

    # The same text gives the same vector: asked again, asked alone, asked
    # beside texts with no token, from a model loaded a second time, and in
    # another process.
    texts = [record_text(row.record) for row in eligible.values()]
    assert embedder.embed(texts) == row_vectors
    assert [embedder.embed([text])[0] for text in texts] == row_vectors
    assert embedder.embed(["", *texts, " \n"]) == [{}, *row_vectors, {}]
    assert switch.build_models(settings, [], []).embedder.embed(texts) == row_vectors
    result = subprocess.run(
        [sys.executable, "-c", DIGEST_SCRIPT],
        input=json.dumps(texts),
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), "PYTHONHASHSEED": "12345"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == digest(row_vectors)
