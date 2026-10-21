#!/usr/bin/env python3
"""Refuse a shell command of the coding tool that pushes or calls gh.

A PreToolUse hook on the Bash and Monitor tools, wired in .claude/settings.json.
Pushing and gh are the owner's step in this repository (CLAUDE.md, Git). The two
deny rules beside this hook match "git push ..." and "gh ...", and miss other
spellings such as git -C <dir> push. This file splits the command into words
much as a shell does, so it also refuses the other spellings: an option before
the subcommand (git -C <dir> push), a path before the program
(/usr/bin/git push), git's dashed programs (git-push), the pushes of git
subtree and git lfs, gh after any other word (flock <file> gh pr create), a
command handed to another shell (bash -c "git push", su -c, script -c,
env -S) and a command substitution. tests/unit/test_no_push_hook.py holds the
spellings.

It reads text, so it guards against a slip and is not a wall. A push made from
inside a script, a make recipe or another program is not seen here. Neither is
one hidden behind a variable or a git alias, a program named by a command
substitution ("$(command -v git)" push), one given to a shell through a pipe or
a here-string, or one that stands between two here-documents whose texts each
hold an apostrophe.

It errs the other way too: text that only mentions a push or a gh command can
be refused where a shell would not run it, as in a here-document, a comment, a
command substitution inside single quotes, or an argument of echo or man. The
message says what to do then.

Exit status 2 refuses the call, and the message on standard error goes back to
the model. A fault of this file's own lets the call through.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from collections.abc import Iterator

# The git subcommands that send commits to a remote.
PUSHING = frozenset({"push", "send-pack", "http-push"})
# The same as programs of their own: git's dashed forms.
PUSHING_PROGRAMS = frozenset(f"git-{subcommand}" for subcommand in PUSHING)
# git subcommands whose own "push" subcommand sends to a remote.
PUSHING_EXTENSIONS = frozenset({"lfs", "subtree"})
# The commands of gh: a gh word followed by one of them is a gh call.
GH_COMMANDS = frozenset(
    {
        "agent-task",
        "alias",
        "api",
        "attestation",
        "auth",
        "browse",
        "cache",
        "codespace",
        "completion",
        "config",
        "copilot",
        "extension",
        "gist",
        "gpg-key",
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
        "workflow",
    }
)

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

MESSAGE = (
    "Refused: this command would run `{found}`. Pushing and gh are the owner's step in this "
    "repository (CLAUDE.md, Git): stop at the local commit and say what is ready to push. "
    "If the command only mentions these words, keep them off the command line: write the text "
    "to a file outside the repository with the Write tool, not with a here-document, and pass "
    "the file (git commit -F <file>), or use the search tool to look for them."
)


# The name of the program a word starts, without its folder.
program = os.path.basename


def split(command: str) -> list[list[str]]:
    """The simple commands of a command line, each as its words."""
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    commands: list[list[str]] = [[]]
    target = False
    for word in lexer:
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


def git_subcommand(words: list[str], start: int) -> int | None:
    """Where the subcommand of the git call whose program word is at start stands."""
    at = start + 1
    while at < len(words):
        word = words[at]
        if word in GIT_OPTIONS_WITH_A_VALUE:
            at += 2
        elif word.startswith("-"):
            at += 1
        else:
            return at
    return None


def git_pushes(words: list[str], start: int) -> bool:
    """Whether the git call whose program word is at start sends to a remote."""
    at = git_subcommand(words, start)
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


def refused(command: str, depth: int = 0) -> str | None:
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
        for at in range(first + 1, len(words) - 1):
            # gh after another word (sudo, flock <file>, ssh <host>, xargs) is a call only when
            # one of its commands follows, so a search for the word gh is not refused.
            if program(words[at]) == "gh" and words[at + 1] in GH_COMMANDS:
                return " ".join(words[at:])
            text = shell_command(words, at, first)
            if text is not None and (found := refused(text, depth + 1)):
                return found
        if head == "eval" and (found := refused(" ".join(words[first + 1 :]), depth + 1)):
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
