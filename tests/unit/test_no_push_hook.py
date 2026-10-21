"""The coding tool's push guard, .claude/hooks/no-push.py: the spellings it refuses and the
commands that only mention the words.

The hook is tooling, not part of titlestd, and .claude/ is kept out of collection and of ruff
(S14), so it is loaded here by its path. Pushing and gh are the owner's step (CLAUDE.md, Git).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REFUSED = [
    "git push origin main",
    "git -C /tmp/repo push",
    "git -c push.default=current push",
    "/usr/bin/git push",
    "/usr/libexec/git-core/git-push origin main",
    "git-push origin main",
    "git send-pack origin",
    "git http-push https://example.com/repo.git main",
    "git subtree push --prefix=docs origin pages",
    "git lfs push origin main",
    "env GIT_TRACE=1 git push",
    "timeout 60 git push",
    "uv run git push",
    "git status && git push",
    "git \\\n  push origin main",
    "bash -c 'git push'",
    "bash -c -- 'git push'",
    "sudo -u owner bash -lc 'git push'",
    "su -c 'gh pr list' owner",
    "script -qc 'git push' /dev/null",
    "env -S 'git push'",
    "echo $(git push)",
    "echo `git push`",
    "eval 'git push'",
    "echo it's; git push",
    "gh pr create",
    "/usr/bin/gh api user",
    "FOO=1 gh repo view",
    "sudo gh auth login",
    "flock /tmp/lock gh pr create",
    "ssh host gh pr list",
    "systemd-run --user gh pr list",
    "xargs gh pr close",
]

ALLOWED = [
    "git status",
    "git log --oneline -3",
    "git stash push -m wip",
    "git commit -m 'Say why git push is refused'",
    "make check",
    "command -v gh",
    "which gh",
    "uv run pytest -k gh",
    "git ls-files | xargs grep -ln gh",
    "timeout 60 grep -rnw gh docs",
    "uv run --isolated python -m titlestd run --out-dir /tmp/gh",
    "nice -n 10 ls ~/.config/gh",
    "sudo dnf info gh",
    "timeout 10 grep -c 'git push' CLAUDE_CODE.md",
    "git ls-files | xargs grep -c 'git push'",
    "nice -n 5 grep -c 'gh pr' README.md",
]


@pytest.fixture(scope="module")
def hook_path(repo_root: Path) -> Path:
    return repo_root / ".claude" / "hooks" / "no-push.py"


@pytest.fixture(scope="module")
def hook(hook_path):
    spec = importlib.util.spec_from_file_location("no_push", hook_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", REFUSED)
def test_a_push_or_a_gh_call_is_refused(hook, command):
    assert hook.refused(command) is not None


@pytest.mark.parametrize("command", ALLOWED)
def test_a_command_that_only_mentions_the_words_is_let_through(hook, command):
    assert hook.refused(command) is None


def run_hook(hook_path: Path, stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(hook_path)],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_hook_refuses_with_exit_status_2_and_says_why(hook_path):
    result = run_hook(hook_path, json.dumps({"tool_input": {"command": "git -C . push"}}))
    assert result.returncode == 2
    assert result.stderr.startswith("Refused: this command would run `git -C . push`.")


def test_a_fault_of_its_own_lets_the_call_through(hook_path):
    result = run_hook(hook_path, "not json")
    assert (result.returncode, result.stderr) == (0, "")


def test_a_long_line_the_splitter_cannot_read_is_read_quickly(hook):
    # An unbalanced quote sends it to the coarse pattern, which once grew with the square of the
    # line: about 3 seconds on this one, and over a minute at four times its length, past the
    # hook's 10-second timeout.
    command = "'" + "git x " * 5_000
    started = time.monotonic()
    assert hook.refused(command) is None
    assert time.monotonic() - started < 1
