"""The program on a bare interpreter: nothing installed, nothing optional imported.

docs/design.md 2.5: "Import the SDK and the embedding library lazily, so the
default run installs nothing." These tests start a fresh interpreter from the
repository root, without site-packages, and stay valid as the components are
built.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Imports every module of the package while refusing the optional libraries,
# then reports which of them ended up loaded anyway.
IMPORT_EVERYTHING = """
import importlib, pkgutil, sys

BLOCKED = {blocked!r}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} must not be imported by the default run")
        return None


sys.meta_path.insert(0, Blocker())
import titlestd

names = sorted(module.name for module in pkgutil.iter_modules(titlestd.__path__))
for name in names:
    if name != "__main__":
        importlib.import_module(f"titlestd.{{name}}")
loaded = sorted(BLOCKED & {{name.split(".")[0] for name in sys.modules}})
print(len(names), loaded)
"""


def test_help_works_with_nothing_installed(bare_cli):
    result = bare_cli("--help")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    for command in ("run", "generate-data", "record"):
        assert f"\n    {command} " in result.stdout


def test_a_wrong_setting_is_reported_with_nothing_installed(bare_cli):
    result = bare_cli("run", "--llm", "telepathy")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "setting llm must be one of offline, replay, anthropic" in result.stderr


def test_live_mode_without_a_model_stops_before_anything_is_imported(bare_cli):
    result = bare_cli("record")
    assert result.returncode == 2
    assert "llm_model" in result.stderr and "claude-haiku-4-5" in result.stderr


def test_no_module_imports_an_optional_library(repo_root: Path, optional_modules):
    script = IMPORT_EVERYTHING.format(blocked=set(optional_modules))
    result = subprocess.run(
        [sys.executable, "-S", "-E", "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    count, loaded = result.stdout.strip().split(" ", 1)
    assert int(count) >= 30
    assert loaded == "[]"


def test_importing_the_package_loads_only_the_standard_library(repo_root: Path):
    script = (
        "import sys; before = set(sys.modules);"
        "import titlestd, titlestd.cli, titlestd.settings, titlestd.datafiles;"
        "new = {name.split('.')[0] for name in set(sys.modules) - before};"
        "print(sorted(new - set(sys.stdlib_module_names) - {'titlestd'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
