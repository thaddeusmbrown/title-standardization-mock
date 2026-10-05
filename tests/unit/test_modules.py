"""The package layout: every module is in place, documented and imports cleanly."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import titlestd
from titlestd import live, local_embedder, pipeline

# Module -> the branch that built it ("scaffold" for the shared branch's modules).
# docs/build-decisions.md, "How the build is cut", is the record of this table.
OWNERS = {
    "__main__": "scaffold",
    "cli": "scaffold",
    "contracts": "scaffold",
    "datafiles": "scaffold",
    "reference": "scaffold",
    "seeding": "scaffold",
    "settings": "scaffold",
    "catalog": "feat/catalog-naming",
    "prepare": "feat/prepare",
    "switch": "feat/model-switch-offline",
    "offline_embedder": "feat/model-switch-offline",
    "standin": "feat/model-switch-offline",
    "people": "feat/simulated-people",
    "mockdata": "feat/mock-data",
    "lookup": "feat/lookup-ml-match",
    "matcher": "feat/lookup-ml-match",
    "fallback": "feat/llm-fallback-rules",
    "rules": "feat/llm-fallback-rules",
    "golden": "feat/golden-audit-status",
    "audit": "feat/golden-audit-status",
    "close": "feat/close-release-files",
    "release": "feat/close-release-files",
    "local_embedder": "feat/local-embedder",
    "agent": "feat/agent",
    "pipeline": "feat/pipeline-report-cli",
    "report": "feat/pipeline-report-cli",
    "store": "feat/pipeline-report-cli",
    "live": "feat/live-replay",
    "replay": "feat/live-replay",
    "recording": "feat/live-replay",
}
COMPONENTS = sorted(name for name, owner in OWNERS.items() if owner != "scaffold")


def _present() -> set[str]:
    return {module.name for module in pkgutil.iter_modules(titlestd.__path__)}


def test_every_planned_module_is_present():
    assert set(OWNERS) <= _present()


@pytest.mark.parametrize("name", sorted(OWNERS))
def test_module_imports_cleanly_and_is_documented(name):
    module = importlib.import_module(f"titlestd.{name}")
    assert module.__doc__ and module.__doc__.strip()


@pytest.mark.parametrize("name", COMPONENTS)
def test_component_names_its_owner_and_what_governs_it(name):
    doc = importlib.import_module(f"titlestd.{name}").__doc__
    assert f"Owner: {OWNERS[name]} (stage " in doc
    assert "Governed by: docs/design.md" in doc
    assert "docs/build-decisions.md Q" in doc


def test_every_module_in_the_package_imports_cleanly():
    # Also covers modules added after the scaffold.
    for name in sorted(_present()):
        importlib.import_module(f"titlestd.{name}")


def test_the_package_has_a_version():
    assert titlestd.__version__ == "0.1.0"


REQUIRED = inspect.Parameter.empty


def _keeps(function, declared: list[tuple[str, object]]) -> bool:
    """True when the function still starts with the declared parameters and their
    defaults (REQUIRED for none), and anything added after them is optional."""
    parameters = [
        (name, parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    ]
    added = parameters[len(declared) :]
    return parameters[: len(declared)] == declared and all(
        default is not REQUIRED for _, default in added
    )


def test_the_seams_for_tests_are_declared():
    # S09: a test hands in its own models or its own client here, so these
    # functions keep these parameters.
    assert _keeps(pipeline.run_pipeline, [("settings", REQUIRED), ("models", None)])
    assert _keeps(pipeline.data_fingerprint, [("data_dir", REQUIRED)])
    assert _keeps(
        live.build_llm,
        [("settings", REQUIRED), ("data_fingerprint", REQUIRED), ("client", None)],
    )


def test_a_seam_check_refuses_a_changed_signature():
    def renamed(settings, doubles=None): ...
    def needs_more(settings, models=None, *, people): ...
    def extended(settings, models=None, *, people=None): ...

    declared = [("settings", REQUIRED), ("models", None)]
    assert not _keeps(renamed, declared)
    assert not _keeps(needs_more, declared)
    assert _keeps(extended, declared)


def test_each_optional_provider_names_its_extra(extra_modules):
    assert live.EXTRA == "live" and local_embedder.EXTRA == "local"
    assert {live.EXTRA, local_embedder.EXTRA} == set(extra_modules)
