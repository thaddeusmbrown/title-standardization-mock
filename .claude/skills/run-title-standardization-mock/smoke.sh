#!/usr/bin/env bash
# Drive the titlestd command line through every mode that costs nothing, each
# run into its own folder outside the repository, and check exit statuses and
# the lines that show each mode worked. It never runs "record" or
# "--llm anthropic": both call a paid API.
#
#   .claude/skills/run-title-standardization-mock/smoke.sh [folder]
#
# With a folder, the runs stay there to be read. Without one, a temporary
# folder is used and removed at the end. SMOKE_LOCAL=1 adds the local-embedder
# mode, which needs uv and model2vec's model in the Hugging Face cache (or the
# network once). PYTHON overrides the interpreter.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)

if [ $# -gt 0 ]; then
    # Resolved from where it was called, before the cd below.
    dir=$(realpath -m "$1")
    case "$dir/" in
    "$repo"/*)
        echo "smoke.sh: the folder must be outside the repository, not $dir" >&2
        exit 2
        ;;
    esac
    mkdir -p "$dir"
else
    dir=$(mktemp -d)
    trap 'rm -r "$dir"' EXIT
fi
cd "$repo"

# The frozen output is compared on Python 3.12 only (Q38). "uv python find"
# names an interpreter without syncing .venv, and -S -E keeps the run on the
# standard library, as a clean clone has it.
py=${PYTHON:-$(uv python find 3.12 2>/dev/null || command -v python3)}
version=$("$py" -S -E -c 'import sys; print("%d.%d" % sys.version_info[:2])')
run() { "$py" -S -E -m titlestd "$@"; }

failures=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; failures=$((failures + 1)); }
# Run a command with its output in last.out and last.err and its exit status in $status.
status_of() { set +e; "$@" >"$dir/last.out" 2>"$dir/last.err"; status=$?; set -e; }
# The last command exited with $1, and last.$2 (out or err) matches $3: pass or fail as $4.
expect() {
    if [ "$status" = "$1" ] && grep -q -- "$3" "$dir/last.$2"; then
        pass "$4"
    else fail "$4 (exit $status): $(tail -1 "$dir/last.err")"; fi
}
agree() { grep '^models agree' "$dir/last.out" || true; }

# Every file under out/ with its size and modification time, to the nanosecond.
out_state() { { find out -type f -printf '%P %s %T@\n' 2>/dev/null || true; } | sort | sha256sum; }
out_before=$(out_state)
echo "interpreter: $py (Python $version); runs in: $dir"

status_of run --help
expect 0 out 'generate-data' "help lists the commands"

status_of run run --out-dir "$dir/default"
cp "$dir/last.out" "$dir/default.txt"
expect 0 out '^Job-title standardization mock:' "default run: $(agree)"
missing=""
for file in load.csv reverse.csv exceptions.csv review_queue.csv state.sqlite; do
    [ -s "$dir/default/$file" ] || missing="$missing $file"
done
if [ -z "$missing" ]; then
    pass "default run wrote its five files"
else fail "default run wrote no$missing"; fi
if [ "$version" = 3.12 ]; then
    # The last line names the output folder, so it differs from the frozen file's.
    if diff <(head -n -1 "$dir/default.txt") <(head -n -1 tests/fixtures/expected/default_run.txt) >/dev/null; then
        pass "default run equals the frozen output, apart from the folder line"
    else fail "default run differs from tests/fixtures/expected/default_run.txt"; fi
else
    echo "SKIP  frozen output: compared on Python 3.12 only, this is $version"
fi

status_of run run --llm replay --out-dir "$dir/replay"
expect 0 out '^LLM and agent answers: recorded' "replay: $(agree)"

status_of run run --llm replay --seed 2 --out-dir "$dir/replay-miss"
expect 1 err 'replay miss' "replay with another seed stops with exit 1 and a replay miss"

status_of run run --llm bogus --out-dir "$dir/bad-setting"
expect 2 err 'must be one of' "a bad setting exits 2: $(tail -1 "$dir/last.err")"

status_of run run --seed 2 --out-dir "$dir/seed-2"
expect 0 out '^models agree' "seed 2: $(agree)"

status_of run generate-data --data-dir "$dir/data"
if [ "$status" = 0 ] && diff -r "$dir/data" data >/dev/null; then
    pass "generate-data writes data/ again byte for byte"
else fail "generate-data (exit $status) or its files differ from data/"; fi

if [ "${SMOKE_LOCAL:-0}" = 1 ]; then
    # --isolated leaves .venv as the gate synced it.
    status_of uv run --locked --isolated --extra local python -m titlestd run --embedder local --out-dir "$dir/local"
    expect 0 out 'Embedder: local' "local embedder: $(agree)"
fi

if [ "$(out_state)" = "$out_before" ]; then
    pass "out/ is as it was"
else fail "something wrote to out/ (or a test run in this checkout did)"; fi

rm "$dir/last.out" "$dir/last.err"
if [ "$failures" -gt 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
