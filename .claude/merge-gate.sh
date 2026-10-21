#!/usr/bin/env bash
# The merge gate the user-level land skill runs on a branch and again on its
# merge: `make check-all`, the gate CI runs (`make check`) and then the suite on
# Python 3.13 and 3.14. That is more than rule 7 and Q34 ask, which is
# `make check` on the merge result. A non-zero exit is a red gate.
#
# It starts from this file's own place, so it works in any worktree, each with
# its own .venv and out/.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# A git variable inherited from a hook would point the suite's git calls at
# another repository or index. The hygiene tests drop them too.
# shellcheck disable=SC2046  # one variable name per word
unset $(git rev-parse --local-env-vars 2>/dev/null || true)
exec make check-all
