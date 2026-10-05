"""Stand-in models never read the truth file: the structural half (2.4, Q22).

The truth file's name, reader and writer exist only in titlestd.people, and
only the pipeline and the data generator may import that module. These tests
parse the package source, so they hold whatever the modules do when they run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import titlestd
from titlestd import datafiles, people

PACKAGE_DIR = Path(titlestd.__file__).parent

# The only modules that may import titlestd.people.
MAY_IMPORT_PEOPLE = {"pipeline.py", "mockdata.py"}
# The modules behind the model switch. Listed by name so the rule reads plainly;
# each is also covered by the stricter rule above.
MODEL_MODULES = {
    "switch.py",
    "offline_embedder.py",
    "standin.py",
    "local_embedder.py",
    "live.py",
    "replay.py",
    "recording.py",
}
TRUTH_NAMES = {"read_truth", "write_truth", "validate_truth", "truth_fingerprint", "TruthRow"}
TRUTH_NAMES |= {"TRUTH_FILE", "TRUTH_COLUMNS"}


def _sources() -> dict[str, ast.Module]:
    return {
        path.name: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(PACKAGE_DIR.rglob("*.py"))
    }


def _imports_people(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "titlestd.people" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "titlestd.people":
                return True
            if node.module == "titlestd" and any(alias.name == "people" for alias in node.names):
                return True
            if node.level and (
                node.module == "people" or any(alias.name == "people" for alias in node.names)
            ):
                return True
    return False


def test_the_package_is_flat_so_file_names_identify_modules():
    assert all(path.parent == PACKAGE_DIR for path in PACKAGE_DIR.rglob("*.py"))


def test_only_the_pipeline_and_the_generator_import_the_people_module():
    importers = {name for name, tree in _sources().items() if _imports_people(tree)}
    assert importers <= MAY_IMPORT_PEOPLE, sorted(importers - MAY_IMPORT_PEOPLE)


def test_no_model_module_imports_the_people_module():
    sources = _sources()
    assert set(sources) >= MODEL_MODULES
    assert not [name for name in sorted(MODEL_MODULES) if _imports_people(sources[name])]


def test_the_truth_file_name_appears_only_in_the_people_module():
    offenders = []
    for name, tree in _sources().items():
        if name == "people.py":
            continue
        for node in ast.walk(tree):
            is_text = isinstance(node, ast.Constant) and isinstance(node.value, str)
            if is_text and people.TRUTH_FILE in node.value:
                offenders.append(f"{name}:{node.lineno}")
    assert offenders == []


def test_the_truth_reader_is_used_only_by_the_generator_and_the_pipeline():
    offenders = []
    for name, tree in _sources().items():
        if name == "people.py" or name in MAY_IMPORT_PEOPLE:
            continue
        for node in ast.walk(tree):
            used = None
            if isinstance(node, ast.Name):
                used = node.id
            elif isinstance(node, ast.Attribute):
                used = node.attr
            elif isinstance(node, ast.alias):
                used = node.name
            if used in TRUTH_NAMES:
                offenders.append(f"{name}:{getattr(node, 'lineno', '?')} {used}")
    assert offenders == []


def test_the_input_files_do_not_include_the_truth_file():
    assert people.TRUTH_FILE not in datafiles.INPUT_FILES
    assert len({*datafiles.INPUT_FILES, people.TRUTH_FILE}) == 7  # the seven files of 2.3


def test_the_people_module_is_the_only_one_defining_truth_names():
    for name in TRUTH_NAMES:
        assert hasattr(people, name), name
        assert not hasattr(datafiles, name), name
