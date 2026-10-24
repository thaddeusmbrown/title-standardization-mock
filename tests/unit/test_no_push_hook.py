"""The coding tool's push guard, .claude/hooks/no-push.py: the spellings it refuses, the
commands that only mention the words, and the PR flow's commands, the only pushes and gh calls
it lets through, each with near misses it refuses. Also the deny rules of .claude/settings.json:
none of them stops a command of the flow, each is the only rule to meet one of the plain
spellings in DENIED, so none is dead or covered by another, and each of those spellings meets
one. And the wiring that runs the hook.

The hook is tooling, not part of titlestd, and .claude/ is kept out of collection and of ruff
(S14), so it is loaded here by its path. A build reaches main as a pull request, and only the
user-level /pr-open, /pr-watch and /pr-setup push or call gh (owner decision 11; CLAUDE.md, Git).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import types
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
    # gh after another word, with its own options before its command, or one of its aliases
    "uv run gh -R owner/mock pr merge 12",
    "systemd-run --user gh --repo owner/mock pr merge 12",
    "timeout 5 gh -R owner/mock pr merge 12",
    "uv run gh --repo=owner/mock pr merge 12",
    "uv run gh co 12",
    "nice gh cs list",
    "uv run gh ext list",
    "timeout 5 gh help pr",
    "env gh version",
    # more options before gh's command than the search for it reads
    "uv run gh " + "-R owner/mock " * 16 + "pr merge 12",
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
    # options after a gh that is only a word, as gh's own would be: no command of gh follows
    "grep -c gh -r docs",
    "uv run --isolated python -m titlestd run --out-dir /tmp/gh",
    "nice -n 10 ls ~/.config/gh",
    "sudo dnf info gh",
    "timeout 10 grep -c 'git push' CLAUDE_CODE.md",
    "git ls-files | xargs grep -c 'git push'",
    "nice -n 5 grep -c 'gh pr' README.md",
]

UID = os.getuid()
# A session scratchpad, where /pr-open's preflight writes the body: the coding tool's per-user
# temporary folder, the project's path in dashes, and the session's id.
SCRATCHPAD = f"/tmp/claude-{UID}/-home-owner-mock/0123abcd-0000-4000-8000-000000000000/scratchpad"
BODY = f"{SCRATCHPAD}/pr-body.md"
REPOSITORY = "owner-name/title-standardization-mock"
PUSH = "git push claude build/pr-flow:refs/heads/build/pr-flow"
TITLE = "'Open the PR flow'"
FIELDS = "url,state,mergeStateStatus,headRefOid,isCrossRepository,headRepositoryOwner,baseRefName"
VIEW = f"gh pr view 12 --json {FIELDS}"
WATCH = "gh pr checks 12 --watch --fail-fast"
CHECKS = "gh pr checks 12 --json name,bucket,link"
SHA = "0123456789abcdef0123456789abcdef01234567"
SHA256 = "0123456789abcdef" * 4
RUN_FIELDS = "databaseId,event,headSha,status,conclusion,workflowName"
RUNS = f"gh run list --commit {SHA} --json {RUN_FIELDS}"
LOG = "gh run view 12345678901 --log-failed"
INSTALLATION = "gh api --paginate installation/repositories --jq '.repositories[].full_name'"
VISIBILITY = f"gh repo view {REPOSITORY} --json visibility --jq .visibility"
SET_DEFAULT = f"gh repo set-default {REPOSITORY}"
CHECK_RUNS_JQ = "'.check_runs[] | select(.app.slug == \"github-actions\") | .name'"


def pull_request(title: str) -> str:
    """/pr-open's draft pull request with the given title, as the command line spells it."""
    return (
        f"gh pr create --draft --head build/pr-flow --base main --title {title} --body-file {BODY}"
    )


def run_repository(repository: str) -> str:
    return f"gh api repos/{repository}/actions/runs/12345678901 --jq .head_repository.full_name"


def check_runs(repository: str) -> str:
    return f"gh api repos/{repository}/commits/main/check-runs --jq {CHECK_RUNS_JQ}"


PULL_REQUEST = pull_request(TITLE)
RUN_REPOSITORY = run_repository(REPOSITORY)
CHECK_RUNS = check_runs(REPOSITORY)

# Titles quoted as bash quotes them, which /pr-open's preflight accepts and bash hands to gh as
# the hook reads them: in single quotes any text but a control or format character, $, backticks,
# backslashes, globs, braces and tildes included; in double quotes globs, braces and tildes;
# outside quotes a character that a backslash makes text; a redirection to a file that only
# starts with nul, which the Bash tool leaves alone; an emoji on its own; and the words of gh's
# options, which no deny rule may meet inside a title (DROPPED_RULES).
TITLES = [
    "'Fix #12 (and #13)'",
    "'Read ~/.config at start'",
    "'Treat * as text'",
    "'a*'",
    "'Why is the cap 256?'",
    "'Match the [x] marker'",
    "'Expand {owner} in the report'",
    "'Price in $'",
    "'$(id)'",
    "'Quote `make check` in the README'",
    "'Normalize C:\\Users paths'",
    '"Match *.md, [x] and {a,b} under ~?"',
    "'Don'\"'\"'t say \"push\"'",
    "'Don'\\''t push main'",
    "Fix\\*",
    "'Write 2>null and >nul.txt'",
    "'Keep a\u00a0no-break space'",
    "'Ship it \U0001f680'",
    "'Deny --web, -w and -R in the push guard'",
    "'Drop --repo and --hostname from the notes'",
    "'Refuse --fill, --body and -b flags'",
    "'Rebase --base build/x handling'",
    f"'{'a' * 256}'",
]

# Rules that .claude/settings.json once held on the options of gh and of gh pr create, dropped
# because a title is free text: each meets one of TITLES, so
# test_no_deny_rule_stops_a_command_of_the_pr_flow fails if one comes back.
DROPPED_RULES = [
    "Bash(gh * -R *)",
    "Bash(gh *--repo*)",
    "Bash(gh *--hostname*)",
    "Bash(gh *--web*)",
    "Bash(gh pr create *--fill*)",
    "Bash(gh pr create *--body *)",
    "Bash(gh pr create * -b *)",
    "Bash(gh pr create *--base build/*)",
]

# Repository names holding a word that a deny rule names: an option of gh api, or an endpoint that
# /pr-watch never reads.
AWKWARD_REPOSITORIES = [
    "owner/data-feed",
    "owner/Hub-X",
    "owner/x-Frontend",
    "owner/web-Hooks",
    "owner/my-graphql-api",
    "owner/pulls",
    "owner/issues-tracker",
    "owner/comments",
]

# The PR flow's commands, word for word as the user-level skills spell them, with invented values
# in their placeholders, main for <default> and gh pr create's options in any order: /pr-open's
# push and draft pull request, /pr-watch's reads, and /pr-setup's, which the owner runs once.
FLOW = [
    PUSH,
    "git push claude build/web/app.v2:refs/heads/build/web/app.v2",
    PULL_REQUEST,
    *map(pull_request, TITLES),
    f'gh pr create --title "Don\'t push main" --body-file {BODY} --base main '
    "--head build/pr-flow --draft",
    "gh pr create --draft --head build/pr-flow --base main --title t "
    f"--body-file /tmp/claude-{UID}/pr-body",
    VIEW,
    WATCH,
    CHECKS,
    RUNS,
    f"gh run list --commit {SHA256} --json {RUN_FIELDS}",
    RUN_REPOSITORY,
    LOG,
    INSTALLATION,
    VISIBILITY,
    SET_DEFAULT,
    CHECK_RUNS,
    *map(run_repository, AWKWARD_REPOSITORIES),
    *map(check_runs, AWKWARD_REPOSITORIES),
]


def draft(title: str = "t", body: str = BODY) -> str:
    """A draft pull request of build/x with the given title and body file, as written."""
    return f"gh pr create --draft --head build/x --base main --title {title} --body-file {body}"


# Characters that plain_line refuses wherever they stand, as each can make a title read otherwise
# than it reaches gh: format characters (Cf: bidi embeddings, overrides and isolates, zero-width
# characters and marks, the byte-order mark, a soft hyphen, a word joiner and a tag character),
# the line and paragraph separators (Zl, Zp) and a lone surrogate (Cs), which json.load gives for
# a \ud800 escape.
UNSEEN = (
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    "\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\U000e0041"
    "\u2028\u2029\ud800"
)

# Near misses of the flow's commands, all refused: another remote, main, a force, a tag, another
# refspec or option, a wrapper or path before the program, a second command, a redirection, a
# comment, a line break or another control character, something a shell expands, a value out of
# its shape.
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
    "git push claude build/x/:refs/heads/build/x/",
    # a build name that git reads as a pattern or a revision, quoted so that only its shape decides
    "git push claude 'build/*:refs/heads/build/*'",
    "git push claude build/*:refs/heads/build/*",
    "git push claude 'build/x~1:refs/heads/build/x~1'",
    "git push claude build/x~1:refs/heads/build/x~1",
    "git push claude 'build/x@{1}:refs/heads/build/x@{1}'",
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
    # one command split by a line break, which a shell runs as two
    "git push claude\nbuild/pr-flow:refs/heads/build/pr-flow",
    "git push claude \\\n  build/pr-flow:refs/heads/build/pr-flow",
    "git push claude $BRANCH:refs/heads/build/pr-flow",
    # a carriage return, which the splitter takes for a space and a shell keeps in the word
    f"{PUSH}\r",
    # the draft pull request, without each of its options in turn, and with others
    f"gh pr create --head build/pr-flow --base main --title {TITLE} --body-file {BODY}",
    f"gh pr create --draft --base main --title {TITLE} --body-file {BODY}",
    f"gh pr create --draft --head build/pr-flow --title {TITLE} --body-file {BODY}",
    f"gh pr create --draft --head build/pr-flow --base main --body-file {BODY}",
    f"gh pr create --draft --head build/pr-flow --base main --title {TITLE}",
    f"gh pr new --draft --head build/pr-flow --base main --title {TITLE} --body-file {BODY}",
    f"gh pr create --draft --base main --title t --body-file {BODY} --head",
    f"gh pr create --draft --head build/x --base main --body-file {BODY} --title",
    f"{PULL_REQUEST} --draft",
    f"{PULL_REQUEST} --web",
    f"{PULL_REQUEST} -R owner/mock",
    f"{PULL_REQUEST} --repo owner/mock",
    f"{PULL_REQUEST} --fill",
    f"{PULL_REQUEST} --label ready",
    f"{PULL_REQUEST} --reviewer owner",
    f"gh pr create --draft --head main --base main --title t --body-file {BODY}",
    f"gh pr create --draft --head feat/x --base main --title t --body-file {BODY}",
    f"gh pr create --draft --head owner:build/x --base main --title t --body-file {BODY}",
    f"gh pr create --draft --head build/x --base build/y --title t --body-file {BODY}",
    "gh pr create --draft --head build/x --base main --title t --body 'Text'",
    f"gh pr create --draft --head build/x --base main --title --body-file {BODY}",
    f"gh pr create --draft=true --head build/x --base main --title t --body-file {BODY}",
    f"gh pr create -d -H build/x -B main -t t -F {BODY}",
    # a title that a shell expands: unquoted, or in double quotes
    draft("a?"),
    draft("a*"),
    draft("[ab]"),
    draft("a[b"),
    draft("a]b"),
    draft("{t,--repo=other/mock}"),
    draft("a{b"),
    draft("a}b"),
    draft("~"),
    draft("$HOME"),
    draft("`id`"),
    draft('"$(id)"'),
    draft('"Quote `id`"'),
    draft('"C:\\Users"'),
    # a backslash that ends the line, with nothing for it to make text
    f"gh pr create --draft --head build/x --base main --body-file {BODY} --title t\\",
    # a title out of bounds: too long, empty, or holding a control character (C0, DEL, C1)
    draft(f"'{'a' * 257}'"),
    draft("''"),
    draft("'a\tb'"),
    draft("'a\x01b'"),
    draft("'a\x7fb'"),
    draft("'a\x85b'"),
    draft("'a\x9fb'"),
    # a title holding a format character, a separator or a lone surrogate, an emoji joined with a
    # zero-width joiner among them
    *(draft(f"'a{char}b'") for char in UNSEEN),
    draft("'Pair \U0001f469\u200d\U0001f4bb in review'"),
    # a title that Claude Code's Bash tool would change to a redirection to /dev/null, inside
    # quotes too, and whether or not the line holds the <, $ or backtick that its rewrite skips
    draft("'Send 2>nul output'"),
    draft('"Send >>NUL output"'),
    draft("'Drop &>Nul| tail'"),
    draft("'Close it > nul)'"),
    draft("'Stop >nul;now'"),
    draft("a\\>nul"),
    draft("'Pay $5 >nul now'"),
    draft("'Read <in >nul now'"),
    draft("'Quote `x` >nul now'"),
    # a title that a shell reads as the end of the command, or as a comment, or one made only of
    # operators, which the hook takes for the end of the command even quoted
    f"gh pr create --draft --head build/x --base main --body-file {BODY} --title ;",
    f"gh pr create --draft --head build/x --base main --body-file {BODY} --title #",
    draft("';'"),
    draft("'&&'"),
    # a body file outside the coding tool's per-user temporary folder, or out of its shape
    draft(body="pr-body.md"),
    draft(body="/tmp/pr-body.md"),
    draft(body="/srv/pr-body.md"),
    draft(body="/tmpfs/pr-body.md"),
    draft(body=f"/tmp/claude-{UID + 1}/x/pr-body.md"),
    draft(body=f"/tmp/claude-{UID}0/pr-body.md"),
    draft(body=f"/tmp/claude-{UID}/notes.md"),
    draft(body=f"/tmp/claude-{UID}/../pr-body.md"),
    draft(body=f"/tmp/claude-{UID}/./pr-body.md"),
    draft(body=f"/tmp/claude-{UID}/pr-body/notes.md"),
    draft(body=f"/tmp/claude-{UID}/pr-body/../../etc/hostname"),
    f"/usr/bin/gh {PULL_REQUEST.removeprefix('gh ')}",
    f"GH_REPO=owner/other {PULL_REQUEST}",
    f"command {PULL_REQUEST}",
    f"gh -R owner/other {PULL_REQUEST.removeprefix('gh ')}",
    # /pr-watch's reads
    "gh pr view 12 --json url,state,body",
    "gh pr view 12 --comments",
    f"gh pr view build/pr-flow --json {FIELDS}",
    f"gh pr view 12345678901234567890 --json {FIELDS}",
    f"{VIEW} --web",
    f"{VIEW} -R owner/other",
    f"/usr/bin/gh pr view 12 --json {FIELDS}",
    "gh pr checks 12 --watch",
    "gh pr checks build/pr-flow --watch --fail-fast",
    "gh pr checks 012 --json name,bucket,link",
    "gh pr checks 12\n--watch --fail-fast",
    f"{WATCH} > /tmp/checks.txt",
    f"{CHECKS},description",
    f"gh run list --commit 0123456 --json {RUN_FIELDS}",
    f"gh run list --commit {'g' * 40} --json {RUN_FIELDS}",
    f"gh run list --commit {SHA.upper()} --json {RUN_FIELDS}",
    f"gh run list --json {RUN_FIELDS}",
    "gh api repos/owner/mock/actions/runs/1/rerun -X POST",
    "gh api -X DELETE repos/owner/mock/actions/runs/1",
    f"{RUN_REPOSITORY} --method PATCH",
    "gh api repos/owner/../actions/runs/1 --jq .head_repository.full_name",
    "gh api repos/owner/mock/pulls/1/actions/runs/1 --jq .head_repository.full_name",
    "gh api repos/owner/mock/actions/runs/1/logs --jq .head_repository.full_name",
    "gh api repos/owner/mock/issues/1/comments --jq .head_repository.full_name",
    "gh api graphql -f query='mutation { resolveReviewThread }'",
    "gh run view 12345678901 --log",
    f"{LOG} | tail -n 80",
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
    "gh repo view owner/.. --json visibility --jq .visibility",
    "gh repo set-default owner/mock/../other",
    "gh repo set-default owner/..",
    "gh repo set-default -owner/mock",
    "gh repo set-default --unset",
    "gh repo sync owner/mock",
    "gh repo fork owner/mock",
    f"gh api repos/owner/mock/commits/build/x/check-runs --jq {CHECK_RUNS_JQ}",
    f"gh api repos/owner/mock/pulls/1/commits/main/check-runs --jq {CHECK_RUNS_JQ}",
    "gh repo edit owner/mock --visibility private",
    "gh repo delete owner/mock --yes",
    "gh auth status",
    "gh release create v1",
]

# Plain spellings outside the flow that a deny rule of .claude/settings.json meets, so they are
# refused even where the hook is not run. Each rule is the only one to meet at least one of them.
DENIED = [
    "git push",
    "git push origin",
    "git push origin main",
    "git push origin build/pr-flow:refs/heads/build/pr-flow",
    "git push https://example.com/mock.git build/pr-flow:refs/heads/build/pr-flow",
    "git push . build/pr-flow:main",
    "git push ../mock build/pr-flow:refs/heads/build/pr-flow",
    "git push /srv/git/mock.git build/pr-flow:refs/heads/build/pr-flow",
    # a tilde that bash expands, which plain_line refuses (test_each_entry_is_one_plain_command)
    "git push ~/mock.git build/pr-flow:refs/heads/build/pr-flow",
    "git push claude",
    "git push claude main",
    "git push claude main build/pr-flow",
    "git push claude main:refs/heads/main",
    "git push claude main:build/x",
    "git push claude build/pr-flow:main",
    "git push claude build/pr-flow:main build/pr-flow",
    "git push claude build/pr-flow:refs/heads/main",
    "git push claude HEAD:refs/heads/build/pr-flow",
    "git push claude @:refs/heads/build/pr-flow",
    "git push --force claude build/pr-flow:refs/heads/build/pr-flow",
    f"{PUSH} --force",
    "git push claude +build/pr-flow:refs/heads/build/pr-flow",
    "git push claude :refs/heads/build/pr-flow",
    "git push claude :build/x",
    "git push claude refs/heads/build/pr-flow:refs/heads/build/pr-flow",
    "git push claude refs/tags/v1:refs/tags/v1",
    "git push claude v1:refs/tags/v1",
    "git push claude tag v1",
    "git push --tags claude",
    "gh",
    "gh --version",
    "gh version",
    "gh help",
    "gh agent-task list",
    "gh alias list",
    "gh attestation verify mock.tar.gz --owner owner",
    "gh auth token",
    "gh browse",
    "gh cache list",
    "gh cs list",
    "gh ext list",
    "gh gist create notes.md",
    "gh gpg-key list",
    "gh issue create",
    "gh label list",
    "gh org list",
    "gh preview prompter",
    "gh project list",
    "gh release create v1",
    "gh ruleset list",
    "gh search prs --author owner",
    "gh ssh-key list",
    "gh status",
    "gh variable list",
    "gh workflow run ci.yml",
    "gh secret set NAME",
    "gh codespace list",
    "gh -R owner/mock pr list",
    "gh pr",
    "gh pr --help",
    "gh pr list",
    "gh pr ls",
    "gh pr diff 12",
    "gh pr lock 12",
    "gh pr merge 12",
    "gh pr ready 12",
    "gh pr reopen 12",
    "gh pr revert 12",
    "gh pr status",
    "gh pr unlock 12",
    "gh pr update-branch 12",
    "gh pr comment 12 --body 'Done'",
    "gh pr review 12 --approve",
    "gh pr close 12",
    "gh pr edit 12 --title 'Other'",
    "gh pr checkout 12",
    f"gh pr new --draft --head build/pr-flow --base main --title {TITLE} --body-file {BODY}",
    "gh pr view 12 --comments",
    "gh pr view 12 -c",
    "gh pr view 12 --json body",
    "gh pr view 12 --json reviews",
    "gh pr view 12 --json latestReviews",
    "gh pr view 12 --web",
    "gh pr view -w 12",
    "gh pr view 12 -w",
    "gh pr view -R owner/other 12",
    "gh pr view 12 -R owner/other",
    "gh pr view 12 --repo owner/other",
    "gh pr checks 12 --web",
    "gh pr checks -w 12",
    "gh pr checks 12 -w",
    "gh pr checks -R owner/other 12",
    "gh pr checks 12 -R owner/other",
    "gh pr checks 12 --repo owner/other",
    "gh run ls",
    "gh run list -R owner/other",
    "gh run list --limit 5 -R owner/other",
    "gh run list --repo owner/other",
    "gh run view 12345678901 --web",
    "gh run view -w 12345678901",
    "gh run view 12345678901 -w",
    "gh run view -R owner/other 12345678901",
    "gh run view 12345678901 -R owner/other",
    "gh run view 12345678901 --repo owner/other",
    "gh run view 12345678901 --log",
    "gh run view --log 12345678901",
    "gh run view 12345678901 --log --job 1",
    "gh run rerun 12345678901",
    "gh run cancel 12345678901",
    "gh run delete 12345678901",
    "gh run download 12345678901",
    "gh run watch 12345678901",
    "gh repo ls",
    "gh repo list owner",
    "gh repo archive owner/mock",
    "gh repo unarchive owner/mock",
    "gh repo autolink list",
    "gh repo clone owner/mock",
    "gh repo create owner/other --public",
    "gh repo new other --private",
    "gh repo deploy-key list",
    "gh repo fork owner/mock",
    "gh repo gitignore list",
    "gh repo license list",
    "gh repo rename other",
    "gh repo sync",
    "gh repo delete owner/mock --yes",
    "gh repo edit owner/mock --visibility private",
    "gh repo view owner/mock --web",
    "gh repo view --web owner/mock",
    "gh repo view -w owner/mock",
    "gh repo view owner/mock -w",
    "gh api graphql -f query='mutation { resolveReviewThread }'",
    "gh api --paginate graphql -F owner=owner",
    # -i and -f run together, which no rule on -f meets: only the graphql rules do
    "gh api graphql -if query='query { viewer { login } }'",
    "gh api --paginate graphql -if query='query { viewer { login } }'",
    "gh api -X DELETE repos/owner/mock/actions/runs/1",
    "gh api repos/owner/mock/actions/runs/1 -X DELETE",
    "gh api --method PATCH repos/owner/mock/actions/runs/1",
    "gh api repos/owner/mock/actions/runs/1 --method PATCH",
    "gh api -f state=success repos/owner/mock/statuses/1",
    "gh api repos/owner/mock/statuses/1 -f state=success",
    "gh api --raw-field state=success repos/owner/mock/statuses/1",
    "gh api repos/owner/mock/statuses/1 --raw-field state=success",
    "gh api -F state=success repos/owner/mock/statuses/1",
    "gh api repos/owner/mock/statuses/1 -F state=success",
    "gh api --field state=success repos/owner/mock/statuses/1",
    "gh api repos/owner/mock/statuses/1 --field state=success",
    "gh api --input body.json repos/owner/mock/statuses/1",
    "gh api repos/owner/mock/statuses/1 --input body.json",
    "gh api -H 'Accept: text/plain' user",
    "gh api user -H 'Accept: text/plain'",
    "gh api --header 'Accept: text/plain' user",
    "gh api user --header 'Accept: text/plain'",
    "gh api --hostname github.example.com user",
    "gh api user --hostname github.example.com",
    "gh api repos/owner/mock/issues/1/comments -f body=Done",
    "gh api repos/owner/mock/issues/7",
    "gh api repos/owner/mock/issues/7/comments",
    "gh api repos/owner/mock/pulls/7/comments",
    "gh api repos/owner/mock/pulls/7/reviews",
    "gh api repos/owner/mock/comments/1",
]

SPACES = re.compile(r"[ \t]+")


def matches(rule: str, command: str) -> bool:
    """Whether a Bash rule of .claude/settings.json meets one plain command, as Claude Code
    2.1.310 matches it: read from its binary and from code.claude.com/docs/en/permissions on
    2026-10-23 (re-check with claude --version and that page).

    The command's ends are trimmed. A rule with no * must equal it. A rule ending in ":*" is the
    legacy prefix form: the text before ":*" is literal, a * in it included, and meets the
    command equal to it or followed by a space. Otherwise * stands for any run of characters, and
    a rule whose one * is a trailing " *" also meets the command without it. In those two forms
    runs of spaces or tabs count as one, in the rule and in the command. This models one plain
    command only: the real matcher also splits a compound command and, for a deny rule, strips
    wrappers, assignments and a program's path, which no entry here uses
    (test_each_entry_is_one_plain_command).
    """
    pattern = rule.removeprefix("Bash(").removesuffix(")")
    command = command.strip()
    if "*" not in pattern:
        return command == pattern
    pattern, command = SPACES.sub(" ", pattern), SPACES.sub(" ", command)
    if pattern.endswith(":*"):
        prefix = pattern.removesuffix(":*")
        return command == prefix or command.startswith(f"{prefix} ")
    if pattern.count("*") == 1 and pattern.endswith(" *"):
        regex = re.escape(pattern.removesuffix(" *")) + "(?: .*)?"
    else:
        regex = ".*".join(map(re.escape, pattern.split("*")))
    return re.fullmatch(regex, command, re.DOTALL) is not None


def one_word_changed(command: str) -> list[str]:
    """The command with each word in turn made x, but the title, which is free text.

    The program becomes a path to it and git push's push becomes send-pack, so that each line is
    still a push or a gh call for the hook to refuse.
    """
    words = shlex.split(command)
    changed = []
    for at, word in enumerate(words):
        if at and words[at - 1] == "--title":
            continue
        if at == 0:
            new = f"/usr/bin/{word}"
        elif words[:2] == ["git", "push"] and at == 1:
            new = "send-pack"
        else:
            new = "x"
        changed.append(shlex.join([*words[:at], new, *words[at + 1 :]]))
    return changed


ONE_WORD_CHANGED = list(
    dict.fromkeys(line for command in FLOW for line in one_word_changed(command))
)


@pytest.fixture(scope="module")
def settings(repo_root: Path) -> dict:
    return json.loads((repo_root / ".claude" / "settings.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def deny_rules(settings) -> list[str]:
    return settings["permissions"]["deny"]


@pytest.fixture(scope="module")
def hook_path(repo_root: Path) -> Path:
    return repo_root / ".claude" / "hooks" / "no-push.py"


@pytest.fixture(scope="module")
def hook(hook_path):
    # Compiled from the source each time, with no bytecode written or read: a cached file is
    # checked only by the source's size and time to the second, so a quick edit of the same size
    # could run the old code.
    module = types.ModuleType("no_push")
    module.__file__ = str(hook_path)
    exec(compile(hook_path.read_text(encoding="utf-8"), str(hook_path), "exec"), module.__dict__)
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


@pytest.mark.parametrize("command", NEAR_MISSES)
def test_a_near_miss_of_the_pr_flow_is_refused(hook, command):
    assert hook.in_flow(command) is False
    assert hook.refused(command) is not None


@pytest.mark.parametrize("command", ONE_WORD_CHANGED)
def test_a_command_of_the_pr_flow_with_one_word_changed_is_refused(hook, command):
    assert hook.in_flow(command) is False
    assert hook.refused(command) is not None


@pytest.mark.parametrize(
    "title", ["", "a" * 257, "a\tb", "a\nb", "a\x01b", "a\x7fb", "a\x85b", "a\x9fb"]
)
def test_a_title_out_of_bounds_is_outside_the_flow(hook, title):
    words = [
        *("gh", "pr", "create", "--draft", "--head", "build/x", "--base", "main"),
        *("--title", title, "--body-file", BODY),
    ]
    assert hook.draft_pull_request(words) is False


@pytest.mark.parametrize(
    "line",
    [
        "gh pr view 'a",
        'gh pr view "a',
        "gh pr view 'a'\"b",
        *(f"gh pr view 'a{char}b'" for char in "\t\n\r\x01\x1f\x7f\x80\x85\x9f"),
    ],
)
def test_a_quote_left_open_or_a_control_character_is_not_a_plain_line(hook, line):
    # The splitter refuses an open quote, and the shapes a control character, on their own;
    # plain_line refuses both before them.
    assert hook.plain_line(line) is False


def test_single_quotes_hold_text_and_double_quotes_a_glob_brace_or_tilde(hook):
    assert hook.plain_line("x '$`\\*?[]{}~\"' \"*?[]{}~'\" '\u00a0'") is True


@pytest.mark.parametrize("char", UNSEEN)
def test_a_format_character_a_separator_or_a_lone_surrogate_is_not_a_plain_line(hook, char):
    assert hook.plain_line(f"gh pr view 'a{char}b'") is False


def test_outside_quotes_a_backslash_makes_the_next_character_text(hook):
    assert hook.plain_line("x Fix\\* \\$HOME \\~ a\\{b\\} 'Don'\\''t' \\\\ \\'") is True
    assert hook.plain_line("x t\\") is False


def test_a_fault_in_the_allowance_refuses_the_call(hook, monkeypatch):
    def fault(command):
        raise RuntimeError(command)

    monkeypatch.setattr(hook, "in_flow", fault)
    assert hook.refused(PUSH) is not None


@pytest.mark.parametrize("command", DENIED)
def test_the_hook_refuses_a_spelling_that_a_deny_rule_meets(hook, command):
    assert hook.refused(command) is not None


@pytest.mark.parametrize("command", FLOW + DENIED)
def test_each_entry_is_one_plain_command(hook, command):
    words = command.split(" ")
    assert words[:2] == ["git", "push"] or words[0] == "gh"
    # The tilde that the git push ~* rule needs bare is read escaped, as text: bash expands it in
    # place, and the line is still one command of the same words.
    assert hook.lone_command(command.replace("git push ~", "git push \\~")) is not None


@pytest.mark.parametrize("command", FLOW)
def test_no_deny_rule_stops_a_command_of_the_pr_flow(deny_rules, command):
    assert [rule for rule in deny_rules if matches(rule, command)] == []


@pytest.mark.parametrize("command", DENIED)
def test_a_plain_spelling_outside_the_pr_flow_meets_a_deny_rule(deny_rules, command):
    assert any(matches(rule, command) for rule in deny_rules)


def test_each_deny_rule_is_the_only_one_to_meet_a_plain_spelling(deny_rules):
    # Without a rule its own spelling meets none, and
    # test_a_plain_spelling_outside_the_pr_flow_meets_a_deny_rule fails. A rule that the others
    # already cover can have no spelling of its own, as git push origin had none beside
    # git push origin *, whose trailing " *" also meets the bare command.
    met_alone = set()
    for command in DENIED:
        met = [rule for rule in deny_rules if matches(rule, command)]
        if len(met) == 1:
            met_alone.update(met)
    assert [rule for rule in deny_rules if rule not in met_alone] == []


@pytest.mark.parametrize("rule", DROPPED_RULES)
def test_a_dropped_rule_would_stop_a_title_of_the_pr_flow(rule):
    assert [command for command in FLOW if matches(rule, command)] != []


def test_each_deny_rule_meets_a_command_that_the_hook_refuses(hook, deny_rules):
    instances = {
        rule: rule.removeprefix("Bash(").removesuffix(")").replace("*", "x") for rule in deny_rules
    }
    assert [rule for rule, command in instances.items() if not matches(rule, command)] == []
    assert [rule for rule, command in instances.items() if hook.refused(command) is None] == []


def test_no_deny_rule_mixes_a_wildcard_with_the_prefix_form(deny_rules):
    # Claude Code reads such a rule as a literal prefix, with its * not expanded.
    patterns = [rule.removeprefix("Bash(").removesuffix(")") for rule in deny_rules]
    assert [pattern for pattern in patterns if pattern.endswith(":*") and "*" in pattern[:-2]] == []


def test_the_hook_runs_before_each_bash_and_monitor_call(settings):
    matchers = [
        entry["matcher"]
        for entry in settings["hooks"]["PreToolUse"]
        for handler in entry["hooks"]
        if handler["type"] == "command"
        and '"$CLAUDE_PROJECT_DIR/.claude/hooks/no-push.py"' in handler["command"]
        and 'exec python3 -I "$f"' in handler["command"]
    ]
    assert matchers == ["Bash|Monitor"]


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


def test_a_read_refused_for_a_pipe_is_run_again_alone_and_read_whole(hook_path):
    # /pr-watch finds a failure in the tail of the log, which invites the pipe that is refused.
    result = run_hook(hook_path, json.dumps({"tool_input": {"command": f"{LOG} | tail -n 80"}}))
    assert result.returncode == 2
    assert result.stderr.startswith(f"Refused: this command would run `{LOG}`.")
    assert (
        "is run again alone and spelled as its skill spells it, and its whole output read. "
        "Never respell a refused command any other way"
    ) in result.stderr


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


@pytest.mark.parametrize(
    "command",
    [
        # a run of > that the nul pattern backtracks over: 5 seconds at this length unbounded
        "git push origin main " + ">" * 20_000,
        # many gh words, each searched for its command to the end: 6 seconds unbounded
        "echo " + "gh -R " * 12_000 + "; git push origin main",
    ],
    ids=["a run of >", "many gh words"],
)
def test_a_long_flagged_line_is_refused_quickly(hook, command):
    started = time.monotonic()
    assert hook.refused(command) is not None
    assert time.monotonic() - started < 2
