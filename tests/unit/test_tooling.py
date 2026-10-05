"""The tooling the branches share: packaging, the check command, ignores, CI."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pyproject(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


def test_no_runtime_dependencies(pyproject):
    assert pyproject["project"]["name"] == "titlestd"
    assert pyproject["project"]["dependencies"] == []
    assert pyproject["project"]["requires-python"] == ">=3.12"


def test_the_two_optional_extras(pyproject):
    assert pyproject["project"]["optional-dependencies"] == {
        "live": ["anthropic>=1.11,<2"],
        "local": ["model2vec>=0.9,<0.10"],
    }


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
        ".claude/",
        ".pytest_cache/",
        ".ruff_cache/",
    ):
        assert entry in ignored, entry


def test_make_check_is_lint_then_format_check_then_tests(repo_root: Path):
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "\ncheck: lint format-check test\n" in makefile
    assert "ruff check ." in makefile
    assert "ruff format --check ." in makefile
    assert 'pytest -m "not live and not local_model"' in makefile
    for target in ("check", "lint", "format", "test", "run"):
        assert f"\n{target}:" in makefile


def test_ci_runs_the_same_check_on_three_pythons(repo_root: Path):
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "make check" in workflow
    for version in ('"3.12"', '"3.13"', '"3.14"'):
        assert version in workflow
