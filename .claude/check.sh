#!/usr/bin/env bash
# What "green" means in this repository, for a check gate that runs when a turn
# of the coding tool ends: the same "make check" that CI runs. A non-zero exit
# is a red check.
#
# It starts from this file's own place, so it works from any directory and in
# any worktree.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# A git variable inherited from a hook (GIT_DIR, GIT_INDEX_FILE) would point the
# checks below at another repository or index. The hygiene tests drop them too.
unset $(git rev-parse --local-env-vars 2>/dev/null || true)

# One run per checkout at a time: the make targets re-sync .venv exactly and
# the suite rewrites out/, so two runs turn each other red for no reason in the
# code. If a make or pytest run is already going here, say so and fail. Two
# gates that start at the same moment can both pass this. Where pgrep or /proc
# is missing the test finds nothing and the gate runs.
here=$(pwd -P)
for pid in $(pgrep -x 'make|pytest' || true); do
    if [ "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" = "$here" ]; then
        echo "Another $(cat "/proc/$pid/comm" 2>/dev/null || echo make) run (pid $pid)" \
            "is still going in this checkout. Wait until it exits (the Monitor tool," \
            "until ! kill -0 $pid), then end the turn again."
        exit 1
    fi
done

# The hygiene tests list files with "git ls-files", so a file git does not
# track is scanned by nothing and "make check" would pass it unread. Where git
# is missing or fails the list is empty and the gate runs.
untracked=$(git ls-files --others --exclude-standard 2>/dev/null || true)
if [ -n "$untracked" ]; then
    count=$(printf '%s\n' "$untracked" | wc -l)
    echo "The hygiene scan reads only files git tracks. Stage each new file of" \
        "this change by its path. Ask the owner about any other: it can go in" \
        "$(git rev-parse --git-path info/exclude). Untracked now ($count):"
    printf '%s\n' "$untracked" | head -n 20
    exit 1
fi

exec make check
