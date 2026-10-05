# One check command for every branch: "make check" is the merge gate.
# It also works from another directory: make -C <this directory> check
#
#   check      lint, format check, the suite with the "live" extra installed,
#              then the suite again with no optional extra at all
#   check-all  check, then the suite on the other supported interpreters
#
# The gate runs in the uv environment pinned by .python-version and uv.lock.
# Every uv call syncs that environment exactly, so it holds the dev group and
# the extras named in EXTRAS and nothing an earlier command left behind.
# EXTRAS defaults to "live", so the request-shape tests for the hosted model
# run in the gate. "make test-bare" runs the suite in a throwaway environment
# that holds no optional extra and leaves .venv alone.
# TITLESTD_GATE_EXTRAS tells one tooling test which extras the run was given,
# so a run whose environment does not match fails instead of passing by luck.
# To run the gate on another interpreter without touching .venv:
#   UV_PYTHON=3.13 make check UV_FLAGS=--isolated

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
EXTRAS ?= live
UV_FLAGS ?=
OTHER_PYTHONS ?= 3.13 3.14
UV := uv run --directory $(ROOT) --locked
UV_RUN := $(UV) --exact $(UV_FLAGS) $(addprefix --extra ,$(EXTRAS))
UV_BARE := $(UV) --isolated
PYTEST := pytest -m "not live and not local_model"
PYTHON ?= python3
ARGS ?=

.PHONY: check check-all lint format-check format test test-bare run

check: lint format-check test test-bare

check-all: check
	@set -e; for version in $(OTHER_PYTHONS); do \
		echo "--- the suite on Python $$version"; \
		UV_PYTHON=$$version TITLESTD_GATE_EXTRAS="$(EXTRAS)" \
			$(UV) --isolated $(addprefix --extra ,$(EXTRAS)) $(PYTEST); \
	done

lint:
	$(UV_RUN) ruff check .

format-check:
	$(UV_RUN) ruff format --check .

format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

test:
	TITLESTD_GATE_EXTRAS="$(EXTRAS)" $(UV_RUN) $(PYTEST)

# The suite on the standard library, pytest and ruff only (Q32 a).
test-bare:
	TITLESTD_GATE_EXTRAS= $(UV_BARE) $(PYTEST)

# The default run needs nothing installed: a bare interpreter from the repository root.
run:
	cd $(ROOT) && $(PYTHON) -m titlestd run $(ARGS)
