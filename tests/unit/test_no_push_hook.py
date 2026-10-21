"""The coding tool's push guard, .claude/hooks/no-push.py: the spellings it refuses, the
commands that only mention the words, and the PR flow's commands, the only pushes and gh calls
it lets through, each with near misses it refuses. Also the deny rules of .claude/settings.json:
none of them stops a command of the flow, and the plain spellings outside it meet one.

The hook is tooling, not part of titlestd, and .claude/ is kept out of collection and of ruff
(S14), so it is loaded here by its path. A build reaches main as a pull request, and only the
user-level /pr-open, /pr-watch and /pr-setup push or call gh (owner decision 11; CLAUDE.md, Git).
"""

from __future__ import annotations

import importlib.util
import json
import re
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

PUSH = "git push claude build/pr-flow:refs/heads/build/pr-flow"
BODY = "/tmp/scratchpad/pr-body.md"
PULL_REQUEST = (
    f"gh pr create --draft --head build/pr-flow --base main --title 'Add the PR flow' "
    f"--body-file {BODY}"
)
FIELDS = "url,state,mergeStateStatus,headRefOid,isCrossRepository,headRepositoryOwner,baseRefName"
VIEW = f"gh pr view 12 --json {FIELDS}"
WATCH = "gh pr checks 12 --watch --fail-fast"
CHECKS = "gh pr checks 12 --json name,bucket,link"
SHA = "0123456789abcdef0123456789abcdef01234567"
RUNS = f"gh run list --commit {SHA} --json databaseId,event,headSha,status,conclusion,workflowName"
RUN_REPOSITORY = "gh api repos/owner/mock/actions/runs/12345678901 --jq .head_repository.full_name"
LOG = "gh run view 12345678901 --log-failed"
INSTALLATION = "gh api --paginate installation/repositories --jq '.repositories[].full_name'"
VISIBILITY = "gh repo view owner/mock --json visibility --jq .visibility"
SET_DEFAULT = "gh repo set-default owner/mock"
CHECK_RUNS = (
    "gh api repos/owner/mock/commits/main/check-runs --jq "
    "'.check_runs[] | select(.app.slug == \"github-actions\") | .name'"
)

# The PR flow's commands, word for word as the user-level skills spell them: /pr-open's push and
# draft pull request, /pr-watch's reads, and /pr-setup's, which the owner runs once.
FLOW = [
    PUSH,
    "git push claude build/web/app.v2:refs/heads/build/web/app.v2",
    PULL_REQUEST,
    'gh pr create --title "Don\'t push main" --body-file /tmp/pr-body.md --base main '
    "--head build/pr-flow --draft",
    "gh pr create --draft --head build/pr-flow --base main --title 'Fix #12 (and #13)' "
    f"--body-file {BODY}",
    VIEW,
    WATCH,
    CHECKS,
    RUNS,
    RUN_REPOSITORY,
    LOG,
    INSTALLATION,
    VISIBILITY,
    SET_DEFAULT,
    CHECK_RUNS,
]

# Near misses of the flow's commands, all refused: another remote, main, a force, a tag, another
# refspec or option, a wrapper or path before the program, a second command, a redirection, a
# comment, a variable, a substitution, a continued line.
NEAR_MISSES = [
    # the push
    "git push origin build/pr-flow:refs/heads/build/pr-flow",
    "git push https://example.com/mock.git build/pr-flow:refs/heads/build/pr-flow",
    "git push . build/pr-flow:main",
    "git push claude main:refs/heads/main",
    "git push claude build/pr-flow:refs/heads/main",
    "git push claude main",
    "git push claude",
    "git push claude build/pr-flow",
    "git push claude build/pr-flow:build/pr-flow",
    "git push claude build/one:refs/heads/build/two",
    "git push claude HEAD:refs/heads/build/pr-flow",
    "git push claude refs/heads/build/pr-flow:refs/heads/build/pr-flow",
    "git push claude feat/pr-flow:refs/heads/feat/pr-flow",
    "git push claude build/../main:refs/heads/build/../main",
    "git push claude build/-x:refs/heads/build/-x",
    "git push claude build/x.lock:refs/heads/build/x.lock",
    "git push claude build/:refs/heads/build/",
    "git push --force claude build/pr-flow:refs/heads/build/pr-flow",
    f"{PUSH} --force",
    f"{PUSH} -f",
    "git push --force-with-lease claude build/pr-flow:refs/heads/build/pr-flow",
    "git push claude +build/pr-flow:refs/heads/build/pr-flow",
    "git push claude :refs/heads/build/pr-flow",
    "git push --delete claude build/pr-flow",
    "git push claude refs/tags/v1:refs/tags/v1",
    "git push claude v1",
    "git push --tags claude",
    f"{PUSH} --tags",
    "git push --all claude",
    "git push --mirror claude",
    f"{PUSH} build/other:refs/heads/build/other",
    "git -C . push claude build/pr-flow:refs/heads/build/pr-flow",
    "git -c remote.claude.pushurl=https://example.com/mock.git push claude "
    "build/pr-flow:refs/heads/build/pr-flow",
    "/usr/bin/git push claude build/pr-flow:refs/heads/build/pr-flow",
    "git-push claude build/pr-flow:refs/heads/build/pr-flow",
    f"env GIT_TRACE=1 {PUSH}",
    f"GIT_SSH_COMMAND=ssh {PUSH}",
    f"timeout 60 {PUSH}",
    f"uv run {PUSH}",
    f"sudo {PUSH}",
    f"bash -c '{PUSH}'",
    f"echo $({PUSH})",
    f"eval '{PUSH}'",
    f"git status && {PUSH}",
    f"{PUSH} && git push origin main",
    f"{PUSH}; gh pr merge 12",
    f"{PUSH} 2>&1",
    f"{PUSH} &",
    f"{PUSH} # the build",
    f"{PUSH}\ngit status",
    "git push claude \\\n  build/pr-flow:refs/heads/build/pr-flow",
    "git push claude $BRANCH:refs/heads/build/pr-flow",
    # a carriage return, which the splitter takes for a space and a shell keeps in the word
    f"{PUSH}\r",
    # the draft pull request
    f"gh pr create --head build/pr-flow --base main --title 'Add the PR flow' --body-file {BODY}",
    f"{PULL_REQUEST} --draft",
    f"{PULL_REQUEST} --web",
    f"{PULL_REQUEST} -R owner/mock",
    f"{PULL_REQUEST} --repo owner/mock",
    f"{PULL_REQUEST} --fill",
    f"{PULL_REQUEST} --label ready",
    f"{PULL_REQUEST} --reviewer owner",
    "gh pr create --draft --head main --base main --title t --body-file /tmp/pr-body.md",
    "gh pr create --draft --head feat/x --base main --title t --body-file /tmp/pr-body.md",
    "gh pr create --draft --head owner:build/x --base main --title t --body-file /tmp/pr-body.md",
    "gh pr create --draft --head build/x --base build/y --title t --body-file /tmp/pr-body.md",
    "gh pr create --draft --head build/x --base main --title t --body 'Text'",
    "gh pr create --draft --head build/x --base main --title t --body-file pr-body.md",
    "gh pr create --draft --head build/x --base main --title t --body-file /tmp/notes.md",
    "gh pr create --draft --head build/x --base main --title t --body-file /tmp/../pr-body.md",
    "gh pr create --draft --head build/x --base main --title t --body-file /srv/pr-body.md",
    "gh pr create --draft --head build/x --base main --title --body-file /tmp/pr-body.md",
    "gh pr create --draft=true --head build/x --base main --title t --body-file /tmp/pr-body.md",
    "gh pr create -d -H build/x -B main -t t -F /tmp/pr-body.md",
    "gh pr create --draft --head build/x --base main --title '$(id)' --body-file /tmp/pr-body.md",
    "gh pr create --draft --head build/x --base main --title 'a*' --body-file /tmp/pr-body.md",
    # a title that a shell reads as the end of the command, or as a comment
    "gh pr create --draft --head build/x --base main --body-file /tmp/pr-body.md --title ;",
    "gh pr create --draft --head build/x --base main --body-file /tmp/pr-body.md --title #",
    f"/usr/bin/gh {PULL_REQUEST.removeprefix('gh ')}",
    f"GH_REPO=owner/other {PULL_REQUEST}",
    f"command {PULL_REQUEST}",
    f"gh -R owner/other {PULL_REQUEST.removeprefix('gh ')}",
    # /pr-watch's reads
    "gh pr view 12 --json url,state,body",
    "gh pr view 12 --comments",
    f"gh pr view build/pr-flow --json {FIELDS}",
    f"{VIEW} --web",
    f"{VIEW} -R owner/other",
    "gh pr checks 12 --watch",
    "gh pr checks build/pr-flow --watch --fail-fast",
    "gh pr checks 012 --json name,bucket,link",
    f"{WATCH} > /tmp/checks.txt",
    f"{CHECKS},description",
    "gh run list --commit 0123456 --json databaseId,event,headSha,status,conclusion,workflowName",
    "gh run list --json databaseId,event,headSha,status,conclusion,workflowName",
    "gh api repos/owner/mock/actions/runs/1/rerun -X POST",
    "gh api -X DELETE repos/owner/mock/actions/runs/1",
    f"{RUN_REPOSITORY} --method PATCH",
    "gh api repos/owner/mock/issues/1/comments --jq .head_repository.full_name",
    "gh api graphql -f query='mutation { resolveReviewThread }'",
    "gh run view 12345678901 --log",
    "gh run rerun 12345678901",
    "gh run cancel 12345678901",
    "gh pr merge 12",
    "gh pr ready 12",
    "gh pr comment 12 --body 'Done'",
    "gh pr review 12 --approve",
    "gh pr close 12",
    "gh pr edit 12 --title 'Other'",
    "gh pr checkout 12",
    # /pr-setup's reads
    "gh api --paginate installation/repositories",
    "gh repo view owner/mock --json visibility,description --jq .visibility",
    "gh repo set-default owner/mock/../other",
    "gh repo set-default --unset",
    "gh api repos/owner/mock/commits/build/x/check-runs --jq "
    "'.check_runs[] | select(.app.slug == \"github-actions\") | .name'",
    "gh repo edit owner/mock --visibility private",
    "gh repo delete owner/mock --yes",
    "gh auth status",
    "gh release create v1",
]

# Plain spellings outside the flow that a deny rule of .claude/settings.json meets, so they are
# refused even where the hook is not run.
DENIED = [
    "git push",
    "git push origin main",
    "git push origin build/pr-flow:refs/heads/build/pr-flow",
    "git push https://example.com/mock.git build/pr-flow:refs/heads/build/pr-flow",
    "git push . build/pr-flow:main",
    "git push claude",
    "git push claude main",
    "git push claude main:refs/heads/main",
    "git push claude build/pr-flow:refs/heads/main",
    "git push claude HEAD:refs/heads/build/pr-flow",
    "git push --force claude build/pr-flow:refs/heads/build/pr-flow",
    f"{PUSH} --force",
    "git push claude +build/pr-flow:refs/heads/build/pr-flow",
    "git push claude :refs/heads/build/pr-flow",
    "git push claude refs/tags/v1:refs/tags/v1",
    "git push --tags claude",
    "gh",
    "gh --version",
    "gh auth token",
    "gh issue create",
    "gh release create v1",
    "gh workflow run ci.yml",
    "gh secret set NAME",
    "gh codespace list",
    "gh -R owner/mock pr list",
    "gh pr list",
    "gh pr merge 12",
    "gh pr ready 12",
    "gh pr comment 12 --body 'Done'",
    "gh pr review 12 --approve",
    "gh pr close 12",
    "gh pr edit 12 --title 'Other'",
    "gh pr checkout 12",
    f"{PULL_REQUEST} -R owner/mock",
    f"{PULL_REQUEST} --web",
    "gh pr view 12 --comments",
    "gh pr view 12 --json body",
    "gh pr view 12 --json latestReviews",
    "gh run rerun 12345678901",
    "gh run cancel 12345678901",
    "gh repo delete owner/mock --yes",
    "gh repo edit owner/mock --visibility private",
    "gh api graphql -f query='mutation { resolveReviewThread }'",
    "gh api -X DELETE repos/owner/mock/actions/runs/1",
    "gh api repos/owner/mock/issues/1/comments -f body=Done",
]


def matches(rule: str, command: str) -> bool:
    """Whether a Bash permission rule matches a command, as Claude Code's permission docs describe
    the wildcard: * stands for any run of characters, spaces included, and a rule with none must
    equal the command. A rule ending in " *" is also taken to match the command without it, which
    errs toward a match for the test that no rule meets a command of the flow."""
    pattern = rule.removeprefix("Bash(").removesuffix(")")
    regex = ".*".join(map(re.escape, pattern.split("*")))
    if pattern.endswith(" *"):
        regex = f"(?:{regex}|{re.escape(pattern[:-2])})"
    return re.fullmatch(regex, command, re.DOTALL) is not None


@pytest.fixture(scope="module")
def deny_rules(repo_root: Path) -> list[str]:
    settings = json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))
    return settings["permissions"]["deny"]


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


@pytest.mark.parametrize("command", FLOW)
def test_a_command_of_the_pr_flow_is_let_through(hook, command):
    assert hook.refused(command) is None


@pytest.mark.parametrize("command", list(dict.fromkeys(NEAR_MISSES + DENIED)))
def test_a_near_miss_of_the_pr_flow_is_refused(hook, command):
    assert hook.refused(command) is not None


@pytest.mark.parametrize("command", FLOW)
def test_no_deny_rule_stops_a_command_of_the_pr_flow(deny_rules, command):
    assert [rule for rule in deny_rules if matches(rule, command)] == []


@pytest.mark.parametrize("command", DENIED)
def test_a_plain_spelling_outside_the_pr_flow_meets_a_deny_rule(deny_rules, command):
    assert any(matches(rule, command) for rule in deny_rules)


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


def test_the_hook_lets_the_pr_flows_push_through_with_exit_status_0(hook_path):
    result = run_hook(hook_path, json.dumps({"tool_input": {"command": PUSH}}))
    assert (result.returncode, result.stderr) == (0, "")


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
