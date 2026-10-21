#!/usr/bin/env bash
# Format a Python file of this repository after the coding tool writes or edits it.
#
# A PostToolUse hook on the Write and Edit tools, wired in .claude/settings.json.
# It runs the locked ruff of the checkout that holds the file, so a file in a
# worktree under .worktrees/ gets that worktree's ruff and config: import
# sorting, then the format the gate checks. A file outside the project, one
# ruff's config excludes, or a checkout with no .venv yet is left alone. It
# never fails a tool call: a fault of its own is silent.
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty' 2>/dev/null) || exit 0
case "$f" in
"$CLAUDE_PROJECT_DIR"/*.py) ;;
*) exit 0 ;;
esac
ruff="$(git -C "${f%/*}" rev-parse --show-toplevel 2>/dev/null)/.venv/bin/ruff"
[ -x "$ruff" ] || exit 0
{
    "$ruff" check --select I --fix --force-exclude --quiet "$f"
    "$ruff" format --force-exclude --quiet "$f"
} >/dev/null 2>&1
exit 0
