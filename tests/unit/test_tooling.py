"""The tooling the branches share: packaging, the check command, ignores, CI."""

from __future__ import annotations

import importlib.util
import os
import tomllib
from pathlib import Path

import pytest

from titlestd import __main__ as entry_point

# Set by the Makefile to the extras a test run was given; unset outside make.
GATE_EXTRAS = "TITLESTD_GATE_EXTRAS"


@pytest.fixture(scope="module")
def pyproject(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


def test_no_runtime_dependencies(pyproject):
    assert pyproject["project"]["name"] == "titlestd"
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["requires-python"] == ">=3.12"


def test_the_entry_point_asks_for_the_python_the_project_requires(pyproject):
    # "python3 -m titlestd" says so itself on an older Python (titlestd/__main__.py).
    needed = ".".join(str(part) for part in entry_point.MINIMUM_PYTHON)
    assert pyproject["project"]["requires-python"] == f">={needed}"


def test_the_two_optional_extras(pyproject, extra_modules):
    assert pyproject["project"]["optional-dependencies"] == {
        "live": ["anthropic>=1.11,<2"],
        "local": ["model2vec>=0.9,<0.10"],
    }
    assert set(extra_modules) == {"live", "local"}


def test_the_dev_group_is_pytest_and_ruff(pyproject):
    names = sorted(spec.split(">")[0] for spec in pyproject["dependency-groups"]["dev"])
    assert names == ["pytest", "ruff"]


def test_the_package_is_not_installed_to_be_used(pyproject, repo_root: Path):
    assert pyproject["tool"]["uv"]["package"] is False
    assert "build-system" not in pyproject
    assert (repo_root / "titlestd" / "__main__.py").is_file()


def test_pytest_collects_only_from_tests_and_knows_the_markers(pyproject):
    options = pyproject["tool"]["pytest"]["ini_options"]
    assert options["testpaths"] == ["tests"]
    markers = [marker.split(":")[0] for marker in options["markers"]]
    assert markers == ["integration", "e2e", "replay", "live", "local_model"]
    assert "--strict-markers" in options["addopts"]
    assert {".worktrees", ".claude", ".venv", "out"} <= set(options["norecursedirs"])


def test_ruff_targets_312_and_skips_the_same_folders(pyproject):
    ruff = pyproject["tool"]["ruff"]
    assert ruff["target-version"] == "py312"
    assert {".worktrees", ".claude", ".venv", "out"} <= set(ruff["extend-exclude"])


def test_the_pinned_python_is_312(repo_root: Path):
    assert (repo_root / ".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_the_lock_file_is_committed_beside_the_project(repo_root: Path):
    lock = (repo_root / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "titlestd"' in lock
    assert 'name = "torch"' not in lock  # no PyTorch anywhere (Q30)


def test_gitignore_covers_secrets_outputs_and_tool_folders(repo_root: Path):
    ignored = (repo_root / ".gitignore").read_text(encoding="utf-8").split()
    for entry in (
        ".env",
        ".env.*",
        "out/",
        "*.sqlite",
        "__pycache__/",
        ".venv/",
        ".worktrees/",
        ".claude/settings.local.json",
        ".claude/worktrees/",
        ".pytest_cache/",
        ".ruff_cache/",
    ):
        assert entry in ignored, entry


def test_make_check_is_lint_then_format_check_then_the_suite_twice(repo_root: Path):
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "\ncheck: lint format-check test test-bare\n" in makefile
    assert "ruff check ." in makefile
    assert "ruff format --check ." in makefile
    assert 'pytest -m "not live and not local_model"' in makefile
    for target in ("check", "check-all", "lint", "format", "test", "test-bare", "run"):
        assert f"\n{target}:" in makefile


def test_the_gate_syncs_exactly_and_the_bare_run_installs_no_extra(repo_root: Path):
    lines = (repo_root / "Makefile").read_text(encoding="utf-8").splitlines()
    (uv_run,) = [line for line in lines if line.startswith("UV_RUN :=")]
    (uv_bare,) = [line for line in lines if line.startswith("UV_BARE :=")]
    assert "--exact" in uv_run and "--extra" in uv_run
    assert "--isolated" in uv_bare and "--extra" not in uv_bare and "UV_FLAGS" not in uv_bare
    assert "\nEXTRAS ?= live\n" in "\n".join(lines)


def test_make_check_all_adds_the_other_supported_pythons(repo_root: Path):
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "\ncheck-all: check\n" in makefile
    assert "\nOTHER_PYTHONS ?= 3.13 3.14\n" in makefile


@pytest.mark.skipif(GATE_EXTRAS not in os.environ, reason="not started by the Makefile")
def test_the_environment_holds_exactly_the_extras_the_makefile_names(extra_modules):
    # "make test" must really have the SDK, or its request-shape tests skip
    # unseen; "make test-bare" must really lack it, or it proves nothing (Q32).
    named = os.environ[GATE_EXTRAS].split()
    assert set(named) <= set(extra_modules), named
    expected = sorted(module for extra in named for module in extra_modules[extra])
    importable = sorted(
        module
        for modules in extra_modules.values()
        for module in modules
        if importlib.util.find_spec(module) is not None
    )
    assert importable == expected


def test_ci_runs_the_same_check_on_three_pythons(repo_root: Path):
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "make check" in workflow
    for version in ('"3.12"', '"3.13"', '"3.14"'):
        assert version in workflow
