#!/usr/bin/env python3
"""Refuse a shell command of the coding tool that pushes or calls gh, but the PR flow's own.

A PreToolUse hook on the Bash and Monitor tools, wired in .claude/settings.json.
A build reaches main as a pull request (owner decision 11; CLAUDE.md, Git): the
user-level /pr-open pushes build/<topic> to the claude remote and opens a draft
pull request into main, /pr-watch reads its checks, and /pr-setup, which the
owner runs once, reads the repository and sets gh's default repository. Their
commands, in FLOW below, are the only pushes and gh calls let through, and each
only as the whole command line, word for word: no other remote, branch, refspec
or option, no wrapper and no path before the program, no second command,
redirection, comment, line break, control or format character, and nothing a
shell would expand outside quotes. A title may be quoted as bash quotes it. In
single quotes it holds any text but a control or format character. In double
quotes it holds no $, backtick or backslash. Outside quotes a backslash makes
the next character text. A title made only of shell operators, or holding text
that Claude Code's Bash tool would rewrite (>nul), is refused. Every other push
and gh call is refused.

The deny rules beside this hook refuse the plain spellings of many pushes and
gh calls that the flow never makes, not all of them, and miss other spellings
such as git -C <dir> push. This file splits the command into words much as a
shell does, so it also finds the other spellings: an option before the
subcommand (git -C <dir> push), a path before the program (/usr/bin/git push),
git's dashed programs (git-push), the pushes of git subtree and git lfs, gh
after any other word when one of gh's commands follows it and its own options
(flock <file> gh -R <repo> pr create), a command handed to another shell
(bash -c "git push", su -c, script -c, env -S) and a command substitution.
tests/unit/test_no_push_hook.py holds the spellings, the flow's commands and a
near miss of each.

It reads text, so it guards against a slip and is not a wall. A push made from
inside a script, a make recipe or another program is not seen here. Neither is
one hidden behind a variable or a git alias, a program named by a command
substitution ("$(command -v git)" push), one given to a shell through a pipe or
a here-string, a gh that xargs runs with its words from standard input, or one that stands between two here-documents whose texts each
hold an apostrophe. The flow's own scripts call git and gh inside, unseen here:
the preflight fetches from the claude remote and reads the repository's
visibility, and two scripts list pull requests and review threads. Nor does it
read git's configuration, which decides where the claude remote points and can
also add refs to a push: push.followTags sends annotated tags along with the
build. gh's own resolution, GH_REPO or the default that set-default records,
decides which repository a gh call reaches. A shell function or alias named git
or gh, from the profile the Bash tool sources, would run in place of the
program. The title, the body file, the pull request's number and the run's id
are taken on the skill's word: nothing here ties them to what /pr-open's
preflight checked or to the App's own pull request. Once the owner has imported
the rulesets /pr-setup writes, they are the boundary for what reaches the
repository.

It errs the other way too: text that only mentions a push or a gh command can
be refused where a shell would not run it, as in a here-document, a comment, a
command substitution inside single quotes, an argument of echo or man, or a
search for gh beside one of its command words (grep -c gh -r pr). The message
says what to do then.

Exit status 2 refuses the call, and the message on standard error goes back to
the model. A fault of this file's own lets the call through, but for a fault in
the PR flow's allowance, which refuses a call already found to push or call gh.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
import unicodedata
from collections.abc import Iterator

# The git subcommands that send commits to a remote.
PUSHING = frozenset({"push", "send-pack", "http-push"})
# The same as programs of their own: git's dashed forms.
PUSHING_PROGRAMS = frozenset(f"git-{subcommand}" for subcommand in PUSHING)
# git subcommands whose own "push" subcommand sends to a remote.
PUSHING_EXTENSIONS = frozenset({"lfs", "subtree"})
# The commands of gh: a gh word followed by one of them, past gh's own options, is a gh call.
# co is gh's alias for pr checkout, cs for codespace and ext for extension.
GH_COMMANDS = frozenset(
    {
        "agent-task",
        "alias",
        "api",
        "attestation",
        "auth",
        "browse",
        "cache",
        "co",
        "codespace",
        "completion",
        "config",
        "copilot",
        "cs",
        "ext",
        "extension",
        "gist",
        "gpg-key",
        "help",
        "issue",
        "label",
        "org",
        "pr",
        "preview",
        "project",
        "release",
        "repo",
        "ruleset",
        "run",
        "search",
        "secret",
        "ssh-key",
        "status",
        "variable",
        "version",
        "workflow",
    }
)
# Options of gh itself whose value is the next word.
GH_OPTIONS_WITH_A_VALUE = frozenset({"-R", "--repo"})

# Options of git itself whose value is the next word.
GIT_OPTIONS_WITH_A_VALUE = frozenset(
    {
        "-C",
        "-c",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--super-prefix",
        "--attr-source",
        "--config-env",
        "--shallow-file",
    }
)

KEYWORDS = frozenset({"{", "!", "if", "then", "elif", "else", "while", "until", "do"})
SHELLS = frozenset({"bash", "dash", "fish", "ksh", "sh", "zsh"})
# Programs whose -c option takes a command line for a shell.
SHELL_RUNNERS = SHELLS | {"runuser", "script", "su"}

ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
OPERATOR = re.compile(r"[();<>|&]+")
# An operator that names a file for the command it stands in: its next word is not a command.
REDIRECTION = re.compile(r"(?!<<<)[<>]+[&|]?|&>+")
BACKTICKS = re.compile(r"`([^`]*)`")
SHELL_COMMAND_FLAG = re.compile(r"-[A-Za-z]*c[A-Za-z]*")
# A backslash that is not itself escaped, at the end of a line: the shell joins the two lines.
CONTINUATION = re.compile(r"(?<!\\)((?:\\\\)*)\\\n")
# A line that is only a comment. Its apostrophes would open quotes for the splitter.
COMMENT_LINE = re.compile(r"^[ \t]*#.*$", re.MULTILINE)
# A comment at the end of a line as well. Inside quotes such a # is text, so this is a reading
# of its own: it may hide what the others show.
COMMENT = re.compile(r"(?:^|(?<=[ \t]))#.*$", re.MULTILINE)

# For a command the splitter cannot read (an unbalanced quote, as in the text
# of a here-document): any "git ... push" or "gh <word>" on one line. The span
# after git is bounded, so a long line is read in linear time.
_PUSHING = "|".join(sorted(map(re.escape, PUSHING)))
COARSE = re.compile(
    rf"(?:^|[\s;&|(`'\"/])git\b[^\n;&|]{{0,200}}?\s(?<!\bstash\s)(?:{_PUSHING})"
    r"(?=$|[\s;&|)`'\"])"
    rf"|(?:^|[\s;&|(`'\"/])git-(?:{_PUSHING})(?=$|[\s;&|)`'\"])"
    r"|(?:^|[\s;&|(`/])gh\s+[-a-z]",
    re.MULTILINE,
)

# An owner and a repository as GitHub names them, and as the flow's own scripts check them.
OWNER_REPO = r"[A-Za-z0-9](?:[A-Za-z0-9]|-[A-Za-z0-9]){0,38}/(?!\.\.?(?:/|$))[A-Za-z0-9._-]{1,100}"
REPOSITORY = re.compile(OWNER_REPO)
NUMBER = re.compile(r"[1-9][0-9]{0,18}")
COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
RUN = re.compile(rf"repos/{OWNER_REPO}/actions/runs/[1-9][0-9]{{0,18}}")
CHECK_RUNS = re.compile(rf"repos/{OWNER_REPO}/commits/main/check-runs")
# The PR flow's commands but its push and its pull request (owner decision 11; FLOW below), word
# for word as the user-level /pr-watch and /pr-setup spell them. Each word of a shape is a literal
# or a pattern it must match whole.
SHAPES = (
    # /pr-watch: the pull request by its number, its checks and the runs behind them.
    (
        "gh",
        "pr",
        "view",
        NUMBER,
        "--json",
        "url,state,mergeStateStatus,headRefOid,isCrossRepository,headRepositoryOwner,baseRefName",
    ),
    ("gh", "pr", "checks", NUMBER, "--watch", "--fail-fast"),
    ("gh", "pr", "checks", NUMBER, "--json", "name,bucket,link"),
    (
        "gh",
        "run",
        "list",
        "--commit",
        COMMIT,
        "--json",
        "databaseId,event,headSha,status,conclusion,workflowName",
    ),
    ("gh", "api", RUN, "--jq", ".head_repository.full_name"),
    ("gh", "run", "view", NUMBER, "--log-failed"),
    # /pr-setup, which the owner runs once to add the claude remote.
    ("gh", "api", "--paginate", "installation/repositories", "--jq", ".repositories[].full_name"),
    ("gh", "repo", "view", REPOSITORY, "--json", "visibility", "--jq", ".visibility"),
    ("gh", "repo", "set-default", REPOSITORY),
    (
        "gh",
        "api",
        CHECK_RUNS,
        "--jq",
        '.check_runs[] | select(.app.slug == "github-actions") | .name',
    ),
)
# The Unicode categories that plain_line refuses anywhere on a line, in quotes or not: a control
# character (Cc: C0, DEL and C1, a tab, a line break and a carriage return among them), a format
# character (Cf: the bidi controls such as U+202E and the isolates U+2066 to U+2069, the
# zero-width characters U+200B to U+200F, U+FEFF), a lone surrogate (Cs), and the line and
# paragraph separators U+2028 and U+2029 (Zl, Zp). Each can make a title read one way at the
# owner's approval and reach gh as another. An emoji joined with the zero-width joiner U+200D is
# refused with them.
CONTROL_OR_FORMAT = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
# Text that Claude Code's Bash tool rewrites before bash reads the line, inside quotes too: a
# redirection to nul, the Windows null device, becomes one to /dev/null. The rule is this pattern,
# as read from the 2.1.310 binary on 2026-10-23, which skips a line holding <, $ or a backtick;
# plain_line refuses a line it meets whatever else the line holds. Re-check with
# printf '[%s]\n' 'Send 2>nul output', which printed [Send 2>/dev/null output] that day.
NUL_REDIRECTION = re.compile(r"(\d?&?>+[ \t]*)[Nn][Uu][Ll](?=\s|$|[|&;)\n])")
# What bash expands inside double quotes: a variable or substitution, a backtick and a backslash.
EXPANDED_IN_DOUBLE_QUOTES = frozenset("$`\\")
# What bash expands outside quotes: a variable or substitution, a backtick, a glob, a brace or a
# tilde. A backslash there is no expansion: it makes the next character text.
EXPANDED_OUTSIDE_QUOTES = frozenset("$`*?[]{}~")
# A pull request's title: 1 to GitHub's 256 characters, with no control character (C0, DEL, C1).
# plain_line refuses more on the whole line: a format character, what a shell would expand and
# what the Bash tool would rewrite, so a title in single quotes may hold anything else.
TITLE = re.compile(r"[^\x00-\x1f\x7f-\x9f]{1,256}")
# The body file /pr-open's preflight writes to the session scratchpad: a pr-body file under the
# coding tool's per-user temporary folder, /tmp/claude-<uid>, with no . or .. folder. That folder
# is the user's own, mode 0700, while /tmp itself is open to every account to write in.
BODY_FILE = re.compile(
    rf"/tmp/claude-{os.getuid()}(?:/(?!\.\.?/)[A-Za-z0-9._-]+)*/pr-body[A-Za-z0-9._-]*"
)

MESSAGE = (
    "Refused: this command would run `{found}`. In this repository only the PR flow pushes or "
    "calls gh (CLAUDE.md, Git; owner decision 11). A command of the user-level /pr-open, "
    "/pr-watch or /pr-setup passes only alone on the command line and spelled as its skill "
    "spells it, such as `git push claude build/<topic>:refs/heads/build/<topic>`. A read of the "
    "flow refused for what stood beside it, such as a log piped to tail, is run again alone and "
    "spelled as its skill spells it, and its whole output read. Never respell a refused command "
    "any other way to get past this. If it was spelled so and is still refused, stop and say "
    "what is committed but not pushed, or, if /pr-open's push already went out, that the build "
    "is pushed and which command was refused. "
    "If the command only mentions these words, keep them off the command line: write the text "
    "to a file outside the repository with the Write tool, not with a here-document, and pass "
    "the file (git commit -F <file>), or use the search tool to look for them."
)


# The name of the program a word starts, without its folder.
program = os.path.basename


def words_of(command: str, commenters: str = "") -> list[str]:
    """The words of a command line as a shell would split them, operators as words of their own."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = commenters
    return list(lexer)


def split(command: str) -> list[list[str]]:
    """The simple commands of a command line, each as its words."""
    commands: list[list[str]] = [[]]
    target = False
    for word in words_of(command.replace("\n", " ; ")):
        if OPERATOR.fullmatch(word):
            target = bool(REDIRECTION.fullmatch(word))
            if not target:
                commands.append([])
            elif commands[-1] and commands[-1][-1].isdigit():
                commands[-1].pop()  # the descriptor number of 2>file
        elif target:
            target = False
        else:
            commands[-1].append(word)
    return [words for words in commands if words]


def substitutions(command: str) -> Iterator[str]:
    """The text inside each ` ` and $( ), the inner ones first, then the command without them.

    One word stands where a substitution was, so `git -C $(pwd) push` reads as four words.
    """
    for inner in BACKTICKS.finditer(command):
        yield from substitutions(inner[1])
    texts: list[list[str]] = [[]]
    dollar: list[bool] = []
    for char in BACKTICKS.sub("_", command):
        if char == "(":
            dollar.append(texts[-1][-1:] == ["$"])
            if dollar[-1]:
                texts[-1].pop()
                texts.append([])
                continue
        elif char == ")" and dollar and dollar.pop():
            yield "".join(texts.pop())
            texts[-1].append("_")
            continue
        texts[-1].append(char)
    yield "$(".join("".join(text) for text in texts)


def subcommand(words: list[str], start: int, options_with_a_value: frozenset[str]) -> int | None:
    """Where the subcommand of the git or gh call whose program word is at start stands.

    That is the first word past the program's own options and their values, or None when no word
    is left.
    """
    at = start + 1
    while at < len(words):
        word = words[at]
        if word in options_with_a_value:
            at += 2
        elif word.startswith("-"):
            at += 1
        else:
            return at
    return None


def git_pushes(words: list[str], start: int) -> bool:
    """Whether the git call whose program word is at start sends to a remote."""
    at = subcommand(words, start, GIT_OPTIONS_WITH_A_VALUE)
    if at is None:
        return False
    if words[at] in PUSHING:
        return True
    if words[at] in PUSHING_EXTENSIONS:
        following = [word for word in words[at + 1 :] if not word.startswith("-")]
        return following[:1] == ["push"]
    return False


def shell_command(words: list[str], at: int, first: int) -> str | None:
    """The command line that the option at at hands to a shell, or None.

    The option counts only for a program that runs a shell: the nearest word
    before it that is not an option. "bash -c -- text" runs text as well.
    """
    word = words[at]
    runner = at - 1
    while runner > first and words[runner].startswith("-"):
        runner -= 1
    name = program(words[runner])
    if name == "env" and word in ("-S", "--split-string"):
        return words[at + 1] if at + 1 < len(words) else None
    if name not in SHELL_RUNNERS or not SHELL_COMMAND_FLAG.fullmatch(word):
        return None
    following = words[at + 1 : at + 3]
    if following[:1] == ["--"]:
        following = following[1:]
    return following[0] if following else None


def build_branch(word: str) -> bool:
    """Whether a word is build/<name> as the PR flow names a build.

    <name> holds letters, digits and . _ - / only, has no "..", does not start with - and does not
    end in / or .lock: the test of the user-level /pr-open's own scripts.
    """
    name = word.removeprefix("build/")
    return (
        word.startswith("build/")
        and re.fullmatch(r"[A-Za-z0-9._/-]+", name) is not None
        and not name.startswith("-")
        and ".." not in name
        and not name.endswith(("/", ".lock"))
    )


def build_push(words: list[str]) -> bool:
    """git push claude build/<name>:refs/heads/build/<name>, /pr-open's push.

    The destination is spelled out and is the source's own name, so no configuration of the
    remote picks where the branch lands, though push.followTags can still send tags with it.
    """
    if len(words) != 4 or words[:3] != ["git", "push", "claude"]:
        return False
    source, _, destination = words[3].partition(":")
    return build_branch(source) and destination == f"refs/heads/{source}"


PULL_REQUEST_OPTIONS = {
    "--draft": None,
    "--head": build_branch,
    "--base": lambda value: value == "main",
    "--title": TITLE.fullmatch,
    "--body-file": BODY_FILE.fullmatch,
}


def draft_pull_request(words: list[str]) -> bool:
    """/pr-open's pull request: gh pr create with exactly its five options, each once.

    --draft, --head build/<name>, --base main, --title <title> and --body-file <file>, in any
    order: the skill lists them in one, and no other option, -R and --web among them, passes.
    """
    if words[:3] != ["gh", "pr", "create"]:
        return False
    given: dict[str, str] = {}
    rest = iter(words[3:])
    for option in rest:
        if option not in PULL_REQUEST_OPTIONS or option in given:
            return False
        test = PULL_REQUEST_OPTIONS[option]
        value = "" if test is None else next(rest, None)
        if value is None or (test is not None and not test(value)):
            return False
        given[option] = value
    return given.keys() == PULL_REQUEST_OPTIONS.keys()


def one_of_the_shapes(words: list[str]) -> bool:
    """Whether the words are one of SHAPES, word for word."""
    return any(
        len(words) == len(shape)
        and all(
            word == part if isinstance(part, str) else part.fullmatch(word) is not None
            for word, part in zip(words, shape, strict=True)
        )
        for shape in SHAPES
    )


FLOW = (build_push, draft_pull_request, one_of_the_shapes)


def plain_line(command: str) -> bool:
    """Whether a shell takes every character of a command line as written, quotes aside.

    False for a control or format character anywhere, in quotes or not (CONTROL_OR_FORMAT); for
    text that the Bash tool would rewrite, wherever it stands (NUL_REDIRECTION); for $, ` or \\
    inside double quotes, where bash expands them; for $, `, a glob, brace or tilde character
    outside quotes; for a \\ that ends the line; and for a quote left open. Outside quotes a \\
    makes the character after it text, as in bash and the splitter. Inside single quotes
    everything else is text, as in bash, and so is a ' inside double quotes or a " inside single
    quotes.
    """
    if NUL_REDIRECTION.search(command) or any(
        unicodedata.category(char) in CONTROL_OR_FORMAT for char in command
    ):
        return False
    quote = ""
    escaped = False
    for char in command:
        if escaped:
            escaped = False
        elif quote == "'":
            if char == "'":
                quote = ""
        elif quote == '"':
            if char == '"':
                quote = ""
            elif char in EXPANDED_IN_DOUBLE_QUOTES:
                return False
        elif char == "\\":
            escaped = True
        elif char in "'\"":
            quote = char
        elif char in EXPANDED_OUTSIDE_QUOTES:
            return False
    return not quote and not escaped


def lone_command(command: str) -> list[str] | None:
    """The words of a command line that is one simple command a shell runs as written, or None.

    None when the line holds an operator (a second command, a pipe, a redirection, a background
    job) or a word made only of operator characters, quoted or not; a comment; or a control or
    format character, something a shell would expand or text the Bash tool would rewrite
    (plain_line).
    """
    if not plain_line(command):
        return None
    try:
        words = words_of(command)
        uncommented = words_of(command, commenters="#")
    except ValueError:
        return None
    if words != uncommented or any(OPERATOR.fullmatch(word) for word in words):
        return None
    return words


def in_flow(command: str) -> bool:
    """Whether the whole command line is one of the PR flow's commands."""
    words = lone_command(command)
    return words is not None and any(shape(words) for shape in FLOW)


def refused(command: str) -> str | None:
    """What makes a command a push or a gh call outside the PR flow, or None when nothing does.

    A fault in reading the flow's commands counts as outside the flow, so the call is refused
    rather than let through by main()'s handling of a fault.
    """
    found = found_in(command)
    if found is None:
        return None
    try:
        allowed = in_flow(command)
    except Exception:
        allowed = False
    return None if allowed else found


def found_in(command: str, depth: int = 0) -> str | None:
    """What makes a command a push or a gh call, or None when nothing does."""
    if depth > 4:
        return None
    # Four readings: as written, with continued lines joined, that without comment lines, and
    # that without the comments at the end of a line. Only the first may fall back on the coarse
    # pattern.
    joined = CONTINUATION.sub(r"\1", command)
    bare = CONTINUATION.sub(r"\1", COMMENT_LINE.sub("", command))
    barer = CONTINUATION.sub(r"\1", COMMENT.sub("", command))
    for reading in dict.fromkeys((command, joined, bare, barer)):
        for text in dict.fromkeys((*substitutions(reading), reading)):
            if found := refused_words(text, depth, coarse=reading is command):
                return found
    return None


def refused_words(command: str, depth: int, coarse: bool) -> str | None:
    """The same, for a text whose command substitutions have been taken out."""
    try:
        commands = split(command)
    except ValueError:
        found = COARSE.search(command) if coarse else None
        return found[0].strip() if found else None
    for words in commands:
        for at, word in enumerate(words):
            name = program(word)
            if name in PUSHING_PROGRAMS or (name == "git" and git_pushes(words, at)):
                return " ".join(words[at:])
        first = 0
        while first < len(words) and (ASSIGNMENT.match(words[first]) or words[first] in KEYWORDS):
            first += 1
        if first == len(words):
            continue
        head = program(words[first])
        if head == "gh":
            return " ".join(words[first:])
        for at in range(first + 1, len(words)):
            # gh after another word (sudo, flock <file>, ssh <host>, xargs, uv run) is a call only
            # when one of its commands follows it and its own options, so a search for the word
            # gh is not refused.
            if program(words[at]) == "gh":
                called = subcommand(words, at, GH_OPTIONS_WITH_A_VALUE)
                if called is not None and words[called] in GH_COMMANDS:
                    return " ".join(words[at:])
            text = shell_command(words, at, first)
            if text is not None and (found := found_in(text, depth + 1)):
                return found
        if head == "eval" and (found := found_in(" ".join(words[first + 1 :]), depth + 1)):
            return found
    return None


def main() -> int:
    try:
        command = json.load(sys.stdin)["tool_input"]["command"]
        found = refused(command) if isinstance(command, str) else None
    except Exception:
        return 0
    if found is None:
        return 0
    print(MESSAGE.format(found=found), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
