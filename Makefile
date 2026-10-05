# One check command for every branch: "make check" is the merge gate.
# It also works from another directory: make -C <this directory> check
#
# The gate runs in the uv environment pinned by .python-version and uv.lock.
# To run it on another interpreter without touching .venv:
#   UV_PYTHON=3.13 make check UV_FLAGS=--isolated
# EXTRAS lists the optional extras installed for the tests. The default is
# "live", so the request-shape tests for the hosted model run in the gate;
# "make check EXTRAS=" runs on the standard library plus pytest only.

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
EXTRAS ?= live
UV_FLAGS ?=
UV_RUN := uv run --directory $(ROOT) --locked $(UV_FLAGS) $(addprefix --extra ,$(EXTRAS))
PYTHON ?= python3
ARGS ?=

.PHONY: check lint format-check format test run

check: lint format-check test

lint:
	$(UV_RUN) ruff check .

format-check:
	$(UV_RUN) ruff format --check .

format:
	$(UV_RUN) ruff check --fix .
	$(UV_RUN) ruff format .

test:
	$(UV_RUN) pytest -m "not live and not local_model"

# The default run needs nothing installed: a bare interpreter from the repository root.
run:
	cd $(ROOT) && $(PYTHON) -m titlestd run $(ARGS)
