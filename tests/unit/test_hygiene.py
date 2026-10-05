"""Repository hygiene: what a tracked file may be and may hold.

Owner: docs/readme-hygiene (stage 5).
Governed by: docs/design.md 2.3 (invented emails use example.com), 2.8
(hygiene); docs/build-decisions.md Q02 (a test scans the tracked files for a
key), Q28 (what the hygiene tests check, and that none searches for a vendor's
name).

Every tracked file is an ordinary file, none is a .env file, none holds a
key-shaped string, and every email address is at example.com. The README is
one of those files; what it says is checked in test_hygiene_readme.py, here
and under tests/integration.

Decided here, where the record is silent:

- "Tracked" is what "git ls-files" lists in the checkout the tests run in,
  read from the working tree. Where there is no checkout nothing is tracked
  and those tests are skipped; in a checkout a git that fails is a failure.
- "Key-shaped" is the prefix of the hosted API's keys followed by at least
  one more character a key may hold. A file may name the bare prefix, as the
  decisions record does, and nothing longer. This file puts the prefix
  together from two pieces, like the other tests that need it.
- An email address must be at titlestd.reference.EMAIL_DOMAIN exactly, in
  every tracked text file: no file is exempt and a subdomain does not pass.
  A file that is not UTF-8 text (the two diagrams) is searched for a key but
  not for addresses, because the bytes of an image can look like one.
- "Nothing from outside this folder goes into the repository" (2.8) is
  tested in the one way a program can: no tracked entry is a symbolic link,
  which may point outside, or another repository.
- No test looks for the name of a real company or person (Q28): a program
  cannot tell an invented name from a real one.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from titlestd import reference

# Never written out whole, so that this file passes its own check (Q02).
KEY_PREFIX = "sk-" + "ant-"
KEY_SHAPED = re.compile((KEY_PREFIX + "[A-Za-z0-9_-]+").encode("ascii"))
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")
ORDINARY_FILE_MODES = {"100644", "100755"}  # not a symbolic link (120000) or a submodule (160000)


def key_shaped(content: bytes) -> list[str]:
    """Every key-shaped string in the content of a file."""
    return [found.decode("ascii") for found in KEY_SHAPED.findall(content)]


def foreign_emails(text: str) -> list[str]:
    """Every email address in a text that is not at the reserved domain (2.3)."""
    return [
        found[0] for found in EMAIL.finditer(text) if found[1].lower() != reference.EMAIL_DOMAIN
    ]


def is_env_file(path: str) -> bool:
    """Whether a tracked path is one that .gitignore keeps out: .env or .env.*"""
    return any(part == ".env" or part.startswith(".env.") for part in PurePosixPath(path).parts)


@pytest.fixture(scope="module")
def tracked(repo_root: Path) -> dict[str, str]:
    """The mode of every file git tracks in this checkout, by its path."""
    if not (repo_root / ".git").exists():
        pytest.skip("not a git checkout: no file is tracked here")
    # A hook hands its own GIT_* variables down; the listing must be this checkout's.
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        timeout=60,
        check=True,
    ).stdout.decode("utf-8")
    files = {}
    for entry in filter(None, listing.split("\0")):
        details, path = entry.split("\t", 1)
        files[path] = details.split()[0]
    return files


def _content(repo_root: Path, path: str) -> bytes:
    file = repo_root / path
    return file.read_bytes() if file.is_file() else b""


def test_the_listing_holds_the_files_of_the_repository(tracked):
    # Otherwise every test below would pass on an empty listing.
    assert {"docs/design.md", "titlestd/__init__.py", "data/employees.csv"} <= set(tracked)


def test_every_tracked_file_is_an_ordinary_file(tracked):
    odd = {path: mode for path, mode in tracked.items() if mode not in ORDINARY_FILE_MODES}
    assert odd == {}


def test_no_env_file_is_tracked(tracked):
    assert [path for path in tracked if is_env_file(path)] == []


def test_no_tracked_file_holds_a_key_shaped_string(tracked, repo_root: Path):
    found = {path: len(hits) for path in tracked if (hits := key_shaped(_content(repo_root, path)))}
    assert found == {}  # the count only: a key must not be printed by the test that finds it


def test_every_email_in_a_tracked_file_is_at_the_reserved_domain(tracked, repo_root: Path):
    found = {}
    for path in tracked:
        try:
            text = _content(repo_root, path).decode("utf-8")
        except UnicodeDecodeError:
            continue  # not a text file
        if foreign := foreign_emails(text):
            found[path] = foreign
    assert found == {}


def test_the_mock_data_holds_emails_for_that_check_to_read(repo_root: Path, mini_dir: Path):
    # The descriptions are given a few addresses on purpose (2.3), so the
    # check above reads real ones and is not passed by a pattern that finds none.
    assert reference.EMAIL_DOMAIN == "example.com"
    for folder in (repo_root / "data", mini_dir):
        employees = (folder / "employees.csv").read_text(encoding="utf-8")
        addresses = [found[0] for found in EMAIL.finditer(employees)]
        assert addresses, folder
        assert all(address.endswith("@example.com") for address in addresses)


@pytest.mark.parametrize(
    ("text", "found"),
    [
        (KEY_PREFIX + "api03-AbC_123-xyz", [KEY_PREFIX + "api03-AbC_123-xyz"]),
        (f'KEY = "{KEY_PREFIX}9"', [KEY_PREFIX + "9"]),
        (f"one {KEY_PREFIX}a, two {KEY_PREFIX}b", [KEY_PREFIX + "a", KEY_PREFIX + "b"]),
        (f"a test scans for `{KEY_PREFIX}`.", []),  # the bare prefix, as the record writes it
        ('KEY_PREFIX = "sk-" + "ant-"', []),
        ("task-antenna, ask-anthony", []),
        ("", []),
    ],
)
def test_a_key_shaped_string_is_the_prefix_and_more(text, found):
    assert key_shaped(text.encode("utf-8")) == found


def test_a_key_is_found_in_a_file_that_is_not_text():
    assert key_shaped(b"\x89PNG\xff\x00" + KEY_PREFIX.encode("ascii") + b"Zz09\xfe") == [
        KEY_PREFIX + "Zz09"
    ]


def _address(local: str, domain: str) -> str:
    """An address put together here, so that none at another domain stands in this file."""
    return f"{local}@{domain}"


@pytest.mark.parametrize(
    "address",
    [
        _address("maren", "example.org"),
        _address("maren", "mail.example.com"),  # a subdomain is not the domain
        _address("maren", "example.com.au"),
        _address("maren+hr", "employer.invalid"),
        _address("MAREN", "EXAMPLE.NET"),
    ],
)
def test_an_address_at_another_domain_is_found(address):
    assert foreign_emails(f"Questions to {address}. Or to oswin@example.com.") == [address]


@pytest.mark.parametrize(
    "text",
    [
        "Questions to maren.ashgrove@example.com.",
        "Write to Maren Ashgrove <MAREN@Example.COM>, please.",
        "uses: actions/checkout@v4",  # no address: what follows the @ is no domain
        "masked as [EMAIL] and [NAME]",
        "",
    ],
)
def test_an_address_at_the_reserved_domain_passes_and_so_does_what_is_no_address(text):
    assert foreign_emails(text) == []


@pytest.mark.parametrize(
    ("path", "kept_out"),
    [
        (".env", True),
        (".env.local", True),
        ("config/.env", True),
        (".env.d/key", True),
        ("docs/environment.md", False),
        (".envrc", False),
        ("tests/unit/test_env.py", False),
    ],
)
def test_an_env_file_is_named_as_gitignore_names_it(path, kept_out):
    assert is_env_file(path) is kept_out
