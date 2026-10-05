"""The program on a bare interpreter: nothing installed, nothing optional imported.

docs/design.md 2.5: "Import the SDK and the embedding library lazily, so the
default run installs nothing." These tests start a fresh interpreter from the
repository root and stay valid as the components are built.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Run after the import guard of tests/conftest.py: imports every module of the
# package, then reports how many there are and what the guard saw.
IMPORT_EVERYTHING = """
import importlib, pkgutil

import titlestd

names = sorted(module.name for module in pkgutil.iter_modules(titlestd.__path__))
for name in names:
    if name != "__main__":
        importlib.import_module(f"titlestd.{name}")
print(len(names), sorted(set(attempted_imports)))
"""

# Imports every module of the package and reports what was loaded from outside
# the standard library.
LOADED_BY_IMPORTING_EVERYTHING = """
import importlib, pkgutil, sys

before = set(sys.modules)
import titlestd

for module in pkgutil.iter_modules(titlestd.__path__):
    if module.name != "__main__":
        importlib.import_module(f"titlestd.{module.name}")
new = {name.partition(".")[0] for name in set(sys.modules) - before}
print(sorted(new - set(sys.stdlib_module_names) - {"titlestd"}))
"""


def test_help_works_with_nothing_installed(bare_cli):
    result = bare_cli("--help")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    # Only the names: argparse lays the lines out by the width of the terminal.
    for command in ("run", "generate-data", "record"):
        assert re.search(rf"^ +{re.escape(command)}\b", result.stdout, re.MULTILINE), command


def test_a_wrong_setting_is_reported_with_nothing_installed(bare_cli):
    result = bare_cli("run", "--llm", "telepathy")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "setting llm must be one of offline, replay, anthropic" in result.stderr


def test_live_mode_without_a_model_stops_before_anything_is_imported(bare_cli):
    result = bare_cli("record")
    assert result.returncode == 2
    assert "llm_model" in result.stderr and "claude-haiku-4-5" in result.stderr


def test_no_module_imports_anything_outside_the_standard_library(bare_python, import_guard):
    result = bare_python(import_guard + IMPORT_EVERYTHING)
    assert result.returncode == 0, result.stderr
    count, attempted = result.stdout.strip().split(" ", 1)
    assert int(count) >= 30
    assert attempted == "[]", f"imported when the package is imported: {attempted}"


def test_the_import_guard_sees_an_import_that_swallows_its_own_failure(bare_python, import_guard):
    # The usual idiom for an optional library. The refusal alone would vanish
    # in the except clause; the guard's record of the attempt does not.
    script = (
        import_guard
        + """
try:
    import an_optional_library
except ImportError:
    an_optional_library = None
import csv, json, sqlite3, tomllib  # the standard library is let through
print(attempted_imports)
"""
    )
    result = bare_python(script)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['an_optional_library']"


def test_importing_every_module_loads_only_the_standard_library(repo_root: Path):
    # With site-packages this time: where an optional library is installed, a
    # module-level import of it succeeds, and the library shows up here.
    result = subprocess.run(
        [sys.executable, "-c", LOADED_BY_IMPORTING_EVERYTHING],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
