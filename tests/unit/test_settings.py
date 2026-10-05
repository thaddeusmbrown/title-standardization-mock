"""Settings: decided defaults, loading, precedence and validation."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest

from titlestd.settings import (
    EMBEDDERS,
    HELP,
    LLMS,
    Settings,
    SettingsError,
    add_setting_arguments,
    load_settings,
    setting_names,
    settings_from_arguments,
)

# Every setting and its decided default (design 2.5 and the Q entries named in
# titlestd/settings.py). A change here is a change to docs/build-decisions.md.
DEFAULTS = {
    "embedder": "offline",
    "llm": "offline",
    "llm_model": None,
    "seed": 20260101,
    "data_dir": "data",
    "out_dir": "out",
    "recording_dir": "recordings",
    "request_cap": 3000,
    "wave1_size": 40,
    "golden_size": 40,
    "ml_alone": False,
    "ml_margin_cutoff": 0.10,
    "ml_min_examples": 3,
    "embed_dim": 32768,
    "agent_search_k": 5,
    "standin_min_title_words": 2,
    "standin_no_match_below": 2,
    "sme_error_rate": 0.02,
    "undecided_rate": 0.10,
    "generic_cap_share": 0.02,
    "release_stopped_paths": False,
}


def test_defaults_are_the_decided_ones():
    assert Settings().as_dict() == DEFAULTS
    assert list(Settings().as_dict()) == list(DEFAULTS)


def test_setting_values_of_the_design():
    assert EMBEDDERS == ("offline", "local")
    assert LLMS == ("offline", "replay", "anthropic")


def test_every_setting_has_a_help_line():
    assert set(HELP) == set(setting_names())


def test_settings_are_immutable():
    settings = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.seed = 1  # type: ignore[misc]


def test_field_types_after_construction():
    settings = Settings(data_dir="somewhere", sme_error_rate=0, ml_margin_cutoff=1)
    assert settings.data_dir == Path("somewhere")
    assert isinstance(settings.out_dir, Path)
    assert settings.sme_error_rate == 0.0 and isinstance(settings.sme_error_rate, float)
    assert isinstance(settings.ml_margin_cutoff, float)


def test_anthropic_needs_a_model_and_the_message_names_the_example():
    with pytest.raises(SettingsError) as error:
        Settings(llm="anthropic")
    assert "llm_model" in str(error.value)
    assert "claude-haiku-4-5" in str(error.value)


def test_model_string_is_kept_exactly_as_written():
    settings = Settings(llm="anthropic", llm_model="claude-haiku-4-5")
    assert settings.llm_model == "claude-haiku-4-5"


def test_offline_and_replay_do_not_need_a_model():
    assert Settings(llm="offline", llm_model="anything").llm_model == "anything"
    assert Settings(llm="replay").llm_model is None


def test_replace_is_checked_too():
    with pytest.raises(SettingsError):
        dataclasses.replace(Settings(), llm="anthropic")
    assert dataclasses.replace(Settings(), seed=7).seed == 7


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("embedder", "hosted"),
        ("llm", "openai"),
        ("llm", None),
        ("llm_model", ""),
        ("llm_model", " claude-haiku-4-5"),
        ("llm_model", 5),
        ("seed", "abc"),
        ("seed", 1.5),
        ("seed", True),
        ("data_dir", ""),
        ("data_dir", 3),
        ("request_cap", 0),
        ("embed_dim", 0),
        ("agent_search_k", 0),
        ("wave1_size", -1),
        ("golden_size", -1),
        ("ml_min_examples", -1),
        ("ml_margin_cutoff", -0.1),
        ("ml_margin_cutoff", float("nan")),
        ("ml_margin_cutoff", float("inf")),
        ("standin_min_title_words", -1),
        ("standin_no_match_below", -1),
        ("ml_alone", "yes"),
        ("ml_alone", 1),
        ("release_stopped_paths", None),
        ("sme_error_rate", 1.5),
        ("sme_error_rate", "0.1"),
        ("undecided_rate", -0.1),
        ("generic_cap_share", 2),
    ],
)
def test_bad_values_are_refused_by_name(name, value):
    with pytest.raises(SettingsError) as error:
        Settings(**{name: value})
    assert name in str(error.value)


@pytest.mark.parametrize("name", ["sme_error_rate", "undecided_rate", "generic_cap_share"])
@pytest.mark.parametrize("value", [0, 0.0, 1, 1.0])
def test_rates_may_be_zero_or_one(name, value):
    assert getattr(Settings(**{name: value}), name) == float(value)


def test_load_without_anything_gives_the_defaults():
    assert load_settings() == Settings()


def test_load_from_a_toml_file(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text(
        'llm = "replay"\nseed = 7\nundecided_rate = 0.5\nml_alone = true\ndata_dir = "elsewhere"\n',
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.llm == "replay"
    assert settings.seed == 7
    assert settings.undecided_rate == 0.5
    assert settings.ml_alone is True
    assert settings.data_dir == Path("elsewhere")
    assert settings.golden_size == 40  # untouched settings keep their defaults


def test_overrides_beat_the_file_and_none_means_not_given(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("seed = 7\nwave1_size = 5\n", encoding="utf-8")
    settings = load_settings(config, {"seed": 9, "wave1_size": None})
    assert settings.seed == 9
    assert settings.wave1_size == 5


def test_unknown_setting_in_the_file_is_named(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("sead = 7\n", encoding="utf-8")
    with pytest.raises(SettingsError) as error:
        load_settings(config)
    assert "sead" in str(error.value)
    assert "seed" in str(error.value)  # the valid names are listed


def test_unknown_override_is_refused():
    with pytest.raises(SettingsError, match="nonsense"):
        load_settings(None, {"nonsense": 1})


def test_wrong_type_in_the_file_is_refused(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text('seed = "seven"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="seed"):
        load_settings(config)


def test_a_table_in_the_file_is_refused(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("[models]\nllm = 'replay'\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="models"):
        load_settings(config)


def test_missing_and_malformed_files_are_reported(tmp_path):
    with pytest.raises(SettingsError, match="not found"):
        load_settings(tmp_path / "absent.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("seed = = 7\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="not valid TOML"):
        load_settings(broken)
    with pytest.raises(SettingsError, match="cannot be read"):
        load_settings(tmp_path)  # a folder, not a file


def test_the_file_is_checked_like_any_other_source(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text('llm = "anthropic"\n', encoding="utf-8")
    with pytest.raises(SettingsError, match="llm_model"):
        load_settings(config)
    assert load_settings(config, {"llm_model": "claude-haiku-4-5"}).llm == "anthropic"


# --- Command-line flags --------------------------------------------------------

FLAG_SAMPLES = {
    "embedder": (["--embedder", "local"], "local"),
    "llm": (["--llm", "replay"], "replay"),
    "llm_model": (["--llm-model", "claude-haiku-4-5"], "claude-haiku-4-5"),
    "seed": (["--seed", "11"], 11),
    "data_dir": (["--data-dir", "d"], Path("d")),
    "out_dir": (["--out-dir", "o"], Path("o")),
    "recording_dir": (["--recording-dir", "r"], Path("r")),
    "request_cap": (["--request-cap", "12"], 12),
    "wave1_size": (["--wave1-size", "4"], 4),
    "golden_size": (["--golden-size", "3"], 3),
    "ml_alone": (["--ml-alone"], True),
    "ml_margin_cutoff": (["--ml-margin-cutoff", "0.2"], 0.2),
    "ml_min_examples": (["--ml-min-examples", "5"], 5),
    "embed_dim": (["--embed-dim", "64"], 64),
    "agent_search_k": (["--agent-search-k", "3"], 3),
    "standin_min_title_words": (["--standin-min-title-words", "1"], 1),
    "standin_no_match_below": (["--standin-no-match-below", "3"], 3),
    "sme_error_rate": (["--sme-error-rate", "0.5"], 0.5),
    "undecided_rate": (["--undecided-rate", "0"], 0.0),
    "generic_cap_share": (["--generic-cap-share", "0.05"], 0.05),
    "release_stopped_paths": (["--release-stopped-paths"], True),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_setting_arguments(parser)
    return parser


def test_there_is_a_flag_sample_for_every_setting():
    assert set(FLAG_SAMPLES) == set(setting_names())


@pytest.mark.parametrize("name", sorted(FLAG_SAMPLES))
def test_each_setting_has_a_flag(name):
    flags, expected = FLAG_SAMPLES[name]
    settings = settings_from_arguments(_parser().parse_args(flags))
    assert getattr(settings, name) == expected
    others = {n: v for n, v in settings.as_dict().items() if n != name}
    assert others == {n: v for n, v in DEFAULTS.items() if n != name}


def test_no_flags_gives_the_defaults():
    assert settings_from_arguments(_parser().parse_args([])) == Settings()


def test_a_boolean_flag_can_be_switched_off_over_the_file(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("ml_alone = true\n", encoding="utf-8")
    arguments = _parser().parse_args(["--config", str(config), "--no-ml-alone"])
    assert settings_from_arguments(arguments).ml_alone is False
    arguments = _parser().parse_args(["--config", str(config)])
    assert settings_from_arguments(arguments).ml_alone is True


def test_forced_values_win_over_file_and_flags(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text('llm = "replay"\n', encoding="utf-8")
    arguments = _parser().parse_args(
        ["--config", str(config), "--llm", "offline", "--llm-model", "claude-haiku-4-5"]
    )
    assert settings_from_arguments(arguments, llm="anthropic").llm == "anthropic"
