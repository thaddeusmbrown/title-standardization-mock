"""The two vendored browser libraries: exactly the files the owner approved.

Owner: chore/browser-libs.
Governed by: DESIGN.md 2.8 (as amended by the owner on 2026-10-21: two
browser libraries, copied unchanged into static/vendor/ at a version and
checksum the owner approved); docs/build-decisions.md Q28 and rule 8.

static/vendor/ holds the two libraries, their licences, SHA256SUMS and the
README that says where each came from, and nothing else. SHA256SUMS names
exactly the four library and licence files, and each file's bytes match it,
so an edit, an upgrade or an added file fails here until the owner approves
it and the sums change with it.

Decided here, where the record is silent:

- The sums are read from SHA256SUMS in the folder, not written into this
  test, so the files and their sums change in one place, in one commit.
- The README must name both versions and both source URLs, so the note of
  where each file came from cannot drift from the sums without failing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

LIBRARIES = ("htmx.min.js", "LICENSE-htmx.txt", "mermaid.min.js", "LICENSE-mermaid.txt")
SOURCES = (
    "https://cdn.jsdelivr.net/npm/htmx.org@2.0.11/dist/htmx.min.js",
    "https://cdn.jsdelivr.net/npm/mermaid@11.17.2/dist/mermaid.min.js",
)


@pytest.fixture
def vendor(repo_root: Path) -> Path:
    return repo_root / "static" / "vendor"


def sums(vendor: Path) -> dict[str, str]:
    """SHA256SUMS as a mapping of file name to digest, one "<digest>  <name>" per line."""
    listed: dict[str, str] = {}
    for line in (vendor / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        assert separator and name not in listed, f"malformed or repeated line: {line!r}"
        listed[name] = digest
    return listed


def test_the_folder_holds_only_the_approved_files(vendor: Path):
    assert sorted(path.name for path in vendor.iterdir()) == sorted(
        (*LIBRARIES, "README.md", "SHA256SUMS")
    )


def test_every_file_is_an_ordinary_file(vendor: Path):
    for path in vendor.iterdir():
        assert path.is_file() and not path.is_symlink(), path.name


def test_the_sums_name_exactly_the_four_files(vendor: Path):
    assert sorted(sums(vendor)) == sorted(LIBRARIES)


@pytest.mark.parametrize("name", LIBRARIES)
def test_each_file_matches_its_sum(vendor: Path, name: str):
    digest = hashlib.sha256((vendor / name).read_bytes()).hexdigest()
    assert digest == sums(vendor)[name]


def test_the_readme_names_both_versions_and_sources(vendor: Path):
    readme = (vendor / "README.md").read_text(encoding="utf-8")
    for text in ("2.0.11", "11.17.2", *SOURCES):
        assert text in readme, text
