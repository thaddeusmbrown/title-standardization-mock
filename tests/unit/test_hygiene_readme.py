"""The README, held to the program it describes.

Owner: docs/readme-hygiene (stage 5).
Governed by: DESIGN.md 2.4 (the two statements it asks of the README),
2.6 (what a run prints and writes); docs/build-decisions.md owner decision 10
(a short README) and "What the README must say", Q02 and Q39 (the record
command and the key), Q29 and Q31 (replay), Q30 and S09 (the command that
runs the program with an extra), Q38 (the output files), S14 (the check
command).

A README cannot be tested for being true, only for agreeing with the program:
the output it shows is the frozen table of the default run, every command it
gives is one the command line or the Makefile takes, what it names exists,
and the two statements of 2.4 are in it. No test asks any other statement of
it (owner decision 10).

Decided here, where the record is silent:

- A command is a line of a fenced block marked "sh" that is not a comment,
  or inline code that starts with "python3 -m titlestd", "uv run" or "make".
  A fenced block of any other kind than "sh" and "text" is refused, so a
  command in a block is always one that is checked here.
- A command that starts the program is put through the real parser and the
  real settings, with what the command forces (S13). Two words are set aside
  first: "--help", which the parser answers by leaving, and uv's own
  "--env-file" with its file, which is how the key reaches the environment
  (Q39) and no part of the command of S09.
- How a mode is started is part of the mode: the default run and replay on a
  bare "python3", which is all they need; a mode that needs a library with
  contracts.run_with_extra, the one command text of S09.
- A statement of 2.4 is in the README when one paragraph or list item holds
  its key words. The README says it in its own words, so this is the most a
  test can ask.

Decided for the README itself, which has no docstring to hold it:

- It shows everything the default run prints, not the table alone, so the
  block is the frozen file byte for byte and one comparison covers it.
- On the second statement of 2.4 it gives what was measured in place of
  "provisional is the usual result": the models-agree path is Confirmed on
  the default seed and, in runs made for this branch with --seed 1 to 10, on
  seven of those seeds and Provisional on three. No test repeats those runs.
  Whoever changes what the default run prints re-pastes the block from the
  frozen file and measures these numbers again.
- The file the record command takes the key from is an example path in the
  home folder, outside any repository (Q39).
- The replay command, and the sentence that the recording is under
  recordings/, rest on the committed recording, which
  tests/integration/test_recordings_replay.py replays. Replay and record were
  not run for this branch.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest

from titlestd import cli, live, local_embedder, pipeline, store
from titlestd.contracts import run_with_extra
from titlestd.settings import (
    EXAMPLE_LLM_MODEL,
    Settings,
    SettingsError,
    settings_from_arguments,
)

# What the default run prints, written by the run itself (Q23).
FROZEN_OUTPUT = Path("tests") / "fixtures" / "expected" / "default_run.txt"
BARE = "python3 -m titlestd"
STARTS = (BARE, "uv run ", "make ")
FENCED = re.compile(r"^```(\w*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)


@pytest.fixture(scope="module")
def readme(repo_root: Path) -> str:
    return (repo_root / "README.md").read_text(encoding="utf-8")


def blocks(readme: str, language: str) -> list[str]:
    """The text of every fenced block of one language."""
    return [text for marked, text in FENCED.findall(readme) if marked == language]


def prose(readme: str) -> str:
    """The README without its fenced blocks."""
    return FENCED.sub("", readme)


def commands(readme: str) -> list[str]:
    """Every command the README gives, in the blocks first."""
    lines = [line.strip() for block in blocks(readme, "sh") for line in block.splitlines()]
    inline = re.findall(r"`([^`\n]+)`", prose(readme))
    return [line for line in lines if line and not line.startswith("#")] + [
        code for code in inline if code.startswith(STARTS)
    ]


def program_command(command: str) -> tuple[str, str, Settings]:
    """A command that starts the program: how it is started, the program's own
    command and the settings it runs with. A command line the program does not
    take leaves through the parser or raises a SettingsError."""
    words = shlex.split(command)
    if "--env-file" in words:
        at = words.index("--env-file")
        del words[at : at + 2]
    at = words.index("titlestd") + 1
    arguments = cli.build_parser().parse_args([word for word in words[at:] if word != "--help"])
    forced = cli.COMMANDS[arguments.command][3]
    return " ".join(words[:at]), arguments.command, settings_from_arguments(arguments, **forced)


def test_the_readme_shows_what_the_default_run_prints(readme, repo_root: Path):
    frozen = (repo_root / FROZEN_OUTPUT).read_text(encoding="utf-8")
    assert blocks(readme, "text") == [frozen]


def test_every_command_of_the_readme_is_one_the_program_or_the_makefile_takes(
    readme, repo_root: Path
):
    # A block of another kind would hold commands that this test does not read.
    assert {marked for marked, _ in FENCED.findall(readme)} == {"sh", "text"}
    assert "```" not in prose(readme)
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    given = commands(readme)
    assert {"python3 -m titlestd run", "make check"} <= set(given)
    for command in given:
        if "titlestd" in shlex.split(command):
            program_command(command)
        else:
            target = re.fullmatch(r"make ([a-z][a-z-]*)", command)
            assert target and f"\n{target[1]}:" in makefile, command


def test_the_readme_gives_the_default_run_and_the_three_optional_modes(readme):
    modes = set()
    for command in commands(readme):
        if "titlestd" in shlex.split(command):
            started, name, settings = program_command(command)
            modes.add((started, name, settings.embedder, settings.llm, settings.llm_model))
    assert modes == {
        (BARE, "run", "offline", "offline", None),
        (BARE, "run", "offline", "replay", None),
        (run_with_extra("local"), "run", "local", "offline", None),
        (run_with_extra("live"), "record", "offline", "anthropic", EXAMPLE_LLM_MODEL),
    }


def test_the_files_the_readme_says_a_run_writes_are_the_files_it_writes(readme):
    named = set(re.findall(r"`(?:out/)?([a-z_]+\.(?:csv|sqlite))`", prose(readme)))
    assert named == {*pipeline.OUTPUT_FILES, store.STATE_FILE}


def test_every_module_and_document_the_readme_names_exists(readme, repo_root: Path):
    named = set(re.findall(r"`((?:titlestd|docs)/[^`\s]+|DESIGN\.md)`", readme))
    assert {"titlestd/pipeline.py", "titlestd/switch.py", "DESIGN.md"} <= named
    assert [path for path in sorted(named) if not (repo_root / path).is_file()] == []


def test_the_readme_names_the_python_the_extras_the_model_and_the_key_of_the_program(
    readme, repo_root: Path
):
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert f"Python {project['requires-python'].removeprefix('>=')} or later" in readme
    assert set(re.findall(r"extra `(\w+)`", readme)) == set(project["optional-dependencies"])
    assert f"`{local_embedder.MODEL_NAME}`" in readme
    assert f"`{EXAMPLE_LLM_MODEL}`" in readme
    assert set(re.findall(r"\b[A-Z][A-Z0-9_]*_KEY\b", readme)) == {live.API_KEY_VARIABLE}


@pytest.mark.parametrize(
    "key_words",
    [
        ("offline accuracy", "the flow", "not model quality"),
        ("60 titles per path", "the audit", "little power"),
    ],
)
def test_the_readme_makes_the_two_statements_of_section_2_4(readme, key_words):
    passages = [
        " ".join(passage.split()).lower() for passage in re.split(r"\n\s*\n|\n- ", prose(readme))
    ]
    assert any(all(words in passage for words in key_words) for passage in passages), key_words


def test_a_readme_command_the_program_does_not_take_is_refused():
    # The check above is only as good as the parser's refusal.
    for command in (f"{BARE} run --no-such-flag", f"{BARE} replay"):
        with pytest.raises(SystemExit):
            program_command(command)
    for command in (f"{BARE} run --llm recorded", f"{run_with_extra('live')} record"):
        with pytest.raises(SettingsError):  # the second names no model (Q31)
            program_command(command)
