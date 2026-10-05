"""The model switch: each setting value builds its own provider, and no other (2.5, S09).

The five providers are replaced by recorders here, so these tests say what the
switch hands each provider and nothing about the providers themselves, which
have their own tests.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from titlestd import live, local_embedder, offline_embedder, replay, settings, standin, switch
from titlestd.contracts import (
    Abbreviation,
    LadderLevel,
    Llm,
    Models,
    TitlestdError,
    run_with_extra,
)
from titlestd.settings import Settings, SettingsError

LADDER = (LadderLevel("Senior", "Senior", "Sr"),)
ABBREVIATIONS = (Abbreviation("mgr", "manager"),)
FINGERPRINT = "0f" * 32

PROVIDERS = {
    ("embedder", "offline"): (offline_embedder, "build_embedder"),
    ("embedder", "local"): (local_embedder, "build_embedder"),
    ("llm", "offline"): (standin, "build_llm"),
    ("llm", "replay"): (replay, "build_llm"),
    ("llm", "anthropic"): (live, "build_llm"),
}


def make_settings(embedder: str = "offline", llm: str = "offline") -> Settings:
    model = "claude-haiku-4-5" if llm == "anthropic" else None
    return Settings(embedder=embedder, llm=llm, llm_model=model)


@pytest.fixture
def calls(monkeypatch) -> list[tuple]:
    """Replace all five providers by recorders; each call is listed in order as
    (setting, value, positional arguments, keyword arguments)."""
    made: list[tuple] = []

    def recorder(setting: str, value: str):
        def provider(*args, **kwargs):
            made.append((setting, value, args, kwargs))
            return f"the {value} {setting}"

        return provider

    for (setting, value), (module, name) in PROVIDERS.items():
        monkeypatch.setattr(module, name, recorder(setting, value))
    return made


# --- The default --------------------------------------------------------------


def test_the_default_settings_build_the_offline_embedder_and_the_stand_in():
    models = switch.build_models(Settings(), LADDER, ABBREVIATIONS)
    assert isinstance(models, Models)
    assert isinstance(models.embedder, offline_embedder.OfflineEmbedder)
    assert models.embedder.dim == Settings().embed_dim
    assert isinstance(models.llm, Llm)
    assert isinstance(models.llm.chooser, standin.StandIn)
    assert models.llm.reviewer is models.llm.chooser
    assert models.llm.finish() is None


def test_the_stand_in_is_given_the_ladder_and_the_abbreviations():
    models = switch.build_models(Settings(), LADDER, ABBREVIATIONS)
    # "senior" is a level word and "mgr" an abbreviation only if both arrived.
    assert models.llm.chooser.content_words("Senior Sales Mgr") == {"sales", "manager"}


def test_the_embedder_is_given_the_settings():
    models = switch.build_models(Settings(embed_dim=64), LADDER, ABBREVIATIONS)
    assert models.embedder.dim == 64


def test_the_fingerprint_is_optional_for_a_run_that_records_nothing():
    parameters = inspect.signature(switch.build_models).parameters
    assert list(parameters) == ["settings", "ladder", "abbreviations", "data_fingerprint"]
    assert parameters["data_fingerprint"].default == ""


# --- The provider table of S09 ---------------------------------------------------


def test_every_value_of_the_two_settings_has_a_provider():
    assert {value for setting, value in PROVIDERS if setting == "embedder"} == set(
        settings.EMBEDDERS
    )
    assert {value for setting, value in PROVIDERS if setting == "llm"} == set(settings.LLMS)


@pytest.mark.parametrize("llm", settings.LLMS)
@pytest.mark.parametrize("embedder", settings.EMBEDDERS)
def test_each_combination_builds_exactly_its_two_providers(calls, embedder, llm):
    models = switch.build_models(make_settings(embedder, llm), LADDER, ABBREVIATIONS, FINGERPRINT)
    assert models == Models(embedder=f"the {embedder} embedder", llm=f"the {llm} llm")
    assert [(setting, value) for setting, value, _, _ in calls] == [
        ("embedder", embedder),
        ("llm", llm),
    ]


@pytest.mark.parametrize("embedder", settings.EMBEDDERS)
def test_an_embedder_provider_is_given_the_settings_alone(calls, embedder):
    chosen = make_settings(embedder)
    switch.build_models(chosen, LADDER, ABBREVIATIONS, FINGERPRINT)
    assert calls[0] == ("embedder", embedder, (chosen,), {})


def test_the_stand_in_provider_is_given_settings_ladder_and_abbreviations(calls):
    chosen = make_settings()
    switch.build_models(chosen, LADDER, ABBREVIATIONS, FINGERPRINT)
    # Not the fingerprint, and never the catalog or anything of the truth.
    assert calls[1] == ("llm", "offline", (chosen, LADDER, ABBREVIATIONS), {})


@pytest.mark.parametrize("llm", ["replay", "anthropic"])
def test_a_recording_provider_is_given_settings_and_the_data_fingerprint(calls, llm):
    chosen = make_settings(llm=llm)
    switch.build_models(chosen, LADDER, ABBREVIATIONS, FINGERPRINT)
    # No client: the switch takes no double and always builds the real thing (S09).
    assert calls[1] == ("llm", llm, (chosen, FINGERPRINT), {})


# --- No silent fall-back (Q30, Q31, Q39) ---------------------------------------


@pytest.mark.parametrize(
    ("setting", "value", "extra"),
    [("embedder", "local", "local"), ("llm", "anthropic", "live"), ("llm", "replay", None)],
)
def test_a_provider_that_cannot_start_stops_the_run_and_nothing_stands_in(
    calls, monkeypatch, setting, value, extra
):
    message = f"cannot start; run: {run_with_extra(extra)} run" if extra else "no recording"

    def refuses(*args, **kwargs):
        raise TitlestdError(message)

    module, name = PROVIDERS[(setting, value)]
    monkeypatch.setattr(module, name, refuses)
    with pytest.raises(TitlestdError) as raised:
        switch.build_models(make_settings(**{setting: value}), LADDER, ABBREVIATIONS, FINGERPRINT)
    assert str(raised.value) == message  # let through unchanged (S09)
    assert (setting, "offline") not in [(made[0], made[1]) for made in calls]


def test_the_embedder_is_built_first_so_its_failure_starts_no_llm_provider(calls, monkeypatch):
    def refuses(*args, **kwargs):
        raise TitlestdError("the local extra is not installed")

    monkeypatch.setattr(local_embedder, "build_embedder", refuses)
    with pytest.raises(TitlestdError, match="local extra"):
        switch.build_models(make_settings("local", "anthropic"), LADDER, ABBREVIATIONS, FINGERPRINT)
    assert calls == []


def test_an_error_of_another_kind_passes_through_too(calls, monkeypatch):
    # The switch catches nothing: an error that is no TitlestdError is not hidden either.
    def breaks(*args, **kwargs):
        raise OSError("the recording cannot be read")

    monkeypatch.setattr(replay, "build_llm", breaks)
    with pytest.raises(OSError, match="cannot be read"):
        switch.build_models(make_settings(llm="replay"), LADDER, ABBREVIATIONS, FINGERPRINT)


@pytest.mark.parametrize(
    ("embedder", "llm", "named"),
    [
        ("telepathy", "offline", "embedder"),
        ("offline", "telepathy", "llm"),
        ("", "offline", "embedder"),
    ],
)
def test_a_value_with_no_provider_is_an_error_not_the_offline_one(calls, embedder, llm, named):
    # Settings refuses such a value itself; the switch must not rely on that
    # and quietly build the default.
    unchecked = SimpleNamespace(embedder=embedder, llm=llm)
    with pytest.raises(SettingsError, match=f"setting {named} has no provider"):
        switch.build_models(unchecked, LADDER, ABBREVIATIONS)
    assert all(made[0] != named for made in calls)


# --- Lazy imports (2.5, Q32) -----------------------------------------------------


def test_the_switch_imports_a_provider_module_only_inside_a_function():
    # At module level: the contracts and the settings, nothing else of the package.
    tree = ast.parse(Path(switch.__file__).read_text(encoding="utf-8"))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            top_level.add(node.module)
    assert top_level == {"__future__", "collections.abc", "titlestd.contracts", "titlestd.settings"}


def test_the_switch_names_the_five_provider_modules_and_no_other():
    tree = ast.parse(Path(switch.__file__).read_text(encoding="utf-8"))
    lazy = set()
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                if isinstance(node, ast.ImportFrom):
                    assert node.module == "titlestd"
                    lazy |= {alias.name for alias in node.names}
                assert not isinstance(node, ast.Import)
    assert lazy == {"offline_embedder", "local_embedder", "standin", "replay", "live"}
