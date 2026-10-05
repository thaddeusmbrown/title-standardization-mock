"""The command line: commands, settings, exit statuses.

These tests replace the command functions, so they hold before and after the
components behind the commands are built.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from titlestd import cli, mockdata, pipeline
from titlestd.contracts import TitlestdError
from titlestd.settings import Settings


@pytest.fixture
def calls(monkeypatch) -> dict[str, list[Settings]]:
    """Replace the command functions with recorders that succeed."""
    seen: dict[str, list[Settings]] = {"pipeline": [], "mockdata": []}

    def run_command(settings: Settings) -> int:
        seen["pipeline"].append(settings)
        return 0

    def generate_command(settings: Settings) -> int:
        seen["mockdata"].append(settings)
        return 0

    monkeypatch.setattr(pipeline, "run_command", run_command)
    monkeypatch.setattr(mockdata, "generate_command", generate_command)
    return seen


def test_help_lists_the_three_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for command in ("run", "generate-data", "record"):
        assert f"\n    {command} " in output


def test_the_commands_are_exactly_the_decided_ones():
    assert list(cli.COMMANDS) == ["run", "generate-data", "record"]


@pytest.mark.parametrize("arguments", [[], ["frobnicate"], ["run", "--no-such-flag"]])
def test_a_wrong_command_line_exits_with_2(arguments, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(arguments)
    assert exit_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_a_flag_value_of_the_wrong_type_exits_with_2(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--seed", "seven"])
    assert exit_info.value.code == 2
    assert "--seed" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run", "generate-data", "record"])
def test_each_command_has_help_with_the_settings(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([command, "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for flag in ("--config", "--embedder", "--llm", "--llm-model", "--seed", "--data-dir"):
        assert flag in output


def test_run_passes_the_settings_to_the_pipeline(calls):
    assert cli.main(["run", "--seed", "5", "--data-dir", "elsewhere"]) == 0
    assert calls["mockdata"] == []
    (settings,) = calls["pipeline"]
    assert settings == Settings(seed=5, data_dir=Path("elsewhere"))


def test_run_reads_a_settings_file_and_flags_win(calls, tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text("seed = 7\ngolden_size = 12\n", encoding="utf-8")
    assert cli.main(["run", "--config", str(config), "--seed", "9"]) == 0
    (settings,) = calls["pipeline"]
    assert (settings.seed, settings.golden_size) == (9, 12)


def test_generate_data_goes_to_the_generator(calls):
    assert cli.main(["generate-data", "--data-dir", "fresh"]) == 0
    assert calls["pipeline"] == []
    assert calls["mockdata"][0].data_dir == Path("fresh")


def test_record_is_a_run_with_the_live_model(calls):
    assert cli.main(["record", "--llm-model", "claude-haiku-4-5"]) == 0
    (settings,) = calls["pipeline"]
    assert settings.llm == "anthropic"
    assert settings.llm_model == "claude-haiku-4-5"


def test_record_is_live_whatever_the_llm_flag_says(calls):
    assert cli.main(["record", "--llm", "offline", "--llm-model", "claude-haiku-4-5"]) == 0
    assert calls["pipeline"][0].llm == "anthropic"


def test_record_without_a_model_exits_with_2_and_says_why(calls, capsys):
    assert cli.main(["record"]) == 2
    assert calls["pipeline"] == []
    error = capsys.readouterr().err
    assert "llm_model" in error and "claude-haiku-4-5" in error


def test_a_bad_setting_exits_with_2_and_names_it(calls, capsys):
    assert cli.main(["run", "--llm", "telepathy"]) == 2
    assert calls["pipeline"] == []
    error = capsys.readouterr().err
    assert error.startswith("titlestd: ") and "llm" in error and "telepathy" in error


def test_a_bad_settings_file_exits_with_2(calls, tmp_path, capsys):
    assert cli.main(["run", "--config", str(tmp_path / "absent.toml")]) == 2
    assert "not found" in capsys.readouterr().err


def test_the_command_status_is_returned(monkeypatch):
    monkeypatch.setattr(pipeline, "run_command", lambda settings: 3)
    assert cli.main(["run"]) == 3
    monkeypatch.setattr(pipeline, "run_command", lambda settings: None)
    assert cli.main(["run"]) == 0


def test_a_command_that_is_not_built_exits_with_1_and_says_so(monkeypatch, capsys):
    def not_built(settings: Settings) -> int:
        raise NotImplementedError("titlestd.pipeline is built by feat/pipeline-report-cli")

    monkeypatch.setattr(pipeline, "run_command", not_built)
    assert cli.main(["run"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "the 'run' command is not built yet" in captured.err
    assert "feat/pipeline-report-cli" in captured.err


def test_a_part_that_is_not_built_is_reported_the_same_way(monkeypatch, capsys):
    def half_built(settings: Settings) -> int:
        raise NotImplementedError

    monkeypatch.setattr(mockdata, "generate_command", half_built)
    assert cli.main(["generate-data"]) == 1
    assert "the 'generate-data' command is not built yet" in capsys.readouterr().err


def test_a_run_error_exits_with_1_and_prints_its_message(monkeypatch, capsys):
    def stopped(settings: Settings) -> int:
        raise TitlestdError("replay miss at step choose for title 'bdr'")

    monkeypatch.setattr(pipeline, "run_command", stopped)
    assert cli.main(["run"]) == 1
    assert capsys.readouterr().err == "titlestd: replay miss at step choose for title 'bdr'\n"


def test_an_unexpected_error_is_not_swallowed(monkeypatch):
    def broken(settings: Settings) -> int:
        raise ZeroDivisionError

    monkeypatch.setattr(pipeline, "run_command", broken)
    with pytest.raises(ZeroDivisionError):
        cli.main(["run"])
