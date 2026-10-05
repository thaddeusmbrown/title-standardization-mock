"""Seeding: a draw depends only on the seed, the stream and the item key (Q06)."""

from __future__ import annotations

import ast
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

import titlestd
from titlestd import seeding

SEED = 20260101
PACKAGE_DIR = Path(titlestd.__file__).parent


def test_the_digest_is_pinned():
    # Pins the hashed bytes: the compact UTF-8 JSON array [seed, stream, *key].
    assert (
        seeding.digest(SEED, "shuffle", "SAL", "account manager ii").hex()
        == "c8f86c93006e8d871a35566a479dd2b263131bc8b8cc74a4803de0f85bd0544c"
    )
    assert (
        seeding.digest(0, "s", "é").hex()
        == "999399c63cb9df9166239eff132cc1b8362d1e9f3e08bcad55e77bc1b39c0932"
    )


def test_the_generator_is_pinned_on_every_supported_python():
    items = list(range(1, 11))
    seeding.rng(SEED, seeding.STREAM_SHUFFLE, "SAL", "account manager ii").shuffle(items)
    assert items == [4, 7, 1, 8, 10, 5, 2, 3, 9, 6]
    generator = seeding.rng(SEED, seeding.STREAM_GOLDEN)
    assert generator.sample(range(100), 5) == [18, 23, 20, 75, 63]
    assert generator.random() == 0.06989215885375599
    assert seeding.unit(SEED, seeding.STREAM_UNDECIDED, "FIN", "special projects") == (
        0.9463280335719298
    )


def test_the_same_inputs_give_the_same_draws():
    first = seeding.rng(SEED, "shuffle", "SAL", "sales representative")
    second = seeding.rng(SEED, "shuffle", "SAL", "sales representative")
    assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]


def test_seed_stream_and_key_each_change_the_draw():
    base = seeding.digest(SEED, "shuffle", "SAL", "sales representative")
    assert seeding.digest(SEED + 1, "shuffle", "SAL", "sales representative") != base
    assert seeding.digest(SEED, "audit", "SAL", "sales representative") != base
    assert seeding.digest(SEED, "shuffle", "FIN", "sales representative") != base
    assert seeding.digest(SEED, "shuffle", "SAL", "sales engineer") != base


def test_key_parts_do_not_run_together():
    assert seeding.digest(SEED, "s", "a", "bc") != seeding.digest(SEED, "s", "ab", "c")
    assert seeding.digest(SEED, "s", "ab") != seeding.digest(SEED, "s", "a", "b")
    assert seeding.digest(SEED, "s") != seeding.digest(SEED, "s", "")


def _interleaved_draws(keys: list[tuple[str, str]]) -> dict[tuple[str, str], list[float]]:
    """Build every generator first, then draw from them in turn, three rounds."""
    generators = {key: seeding.rng(SEED, "sme_error", *key) for key in keys}
    drawn: dict[tuple[str, str], list[float]] = {key: [] for key in keys}
    for _ in range(3):
        for key in keys:
            drawn[key].append(generators[key].random())
    return drawn


def test_a_draw_does_not_depend_on_the_order_of_other_draws():
    # Generators that shared state would give each key different numbers when
    # the keys are built and drawn in another order.
    keys = [("SAL", "a"), ("SAL", "b"), ("FIN", "c")]
    forward = _interleaved_draws(keys)
    backward = _interleaved_draws(list(reversed(keys)))
    assert forward == backward
    assert len({tuple(draws) for draws in forward.values()}) == len(keys)


def test_seeding_leaves_the_shared_generator_alone():
    state = random.getstate()
    seeding.rng(SEED, "shuffle", "SAL", "a").shuffle(list(range(10)))
    seeding.rng(SEED, "golden").sample(range(100), 5)
    seeding.unit(SEED, "undecided", "SAL", "a")
    assert random.getstate() == state


def test_unit_is_a_fraction_and_roughly_uniform():
    draws = [seeding.unit(SEED, "undecided", "FAM", f"title {n}") for n in range(4000)]
    assert all(0.0 <= draw < 1.0 for draw in draws)
    assert 0.47 < sum(draws) / len(draws) < 0.53
    share_under_a_tenth = sum(draw < 0.10 for draw in draws) / len(draws)
    assert 0.08 < share_under_a_tenth < 0.12


def test_stream_names_are_distinct():
    assert len(set(seeding.STREAMS)) == len(seeding.STREAMS) == 5


@pytest.mark.parametrize(
    "arguments",
    [
        ("7", "shuffle"),
        (True, "shuffle"),
        (1.0, "shuffle"),
        (SEED, ""),
        (SEED, 3),
        (SEED, "shuffle", 5),
        (SEED, "shuffle", ("SAL", "title")),
    ],
)
def test_wrong_argument_types_are_refused(arguments):
    with pytest.raises(TypeError):
        seeding.digest(*arguments)


def test_the_digest_is_the_same_in_another_process_with_another_hash_seed():
    script = (
        "from titlestd import seeding;"
        "print(seeding.digest(20260101, 'shuffle', 'SAL', 'account manager ii').hex())"
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PACKAGE_DIR.parent,
            env={
                **os.environ,
                "PYTHONPATH": str(PACKAGE_DIR.parent),
                "PYTHONHASHSEED": hash_seed,
            },
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert outputs == {"c8f86c93006e8d871a35566a479dd2b263131bc8b8cc74a4803de0f85bd0544c"}


# --- Rules for the whole package (Q06) -----------------------------------------


def _modules() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


def test_only_the_seeding_module_imports_random():
    offenders = []
    for path in _modules():
        if path.name == "seeding.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(name.split(".")[0] in ("random", "secrets") for name in names):
                offenders.append(path.name)
    assert offenders == [], f"use titlestd.seeding instead of random in: {offenders}"


def test_nothing_calls_the_builtin_hash():
    offenders = []
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"hash() changes between processes; use hashlib: {offenders}"
