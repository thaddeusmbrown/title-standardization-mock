"""Settings: one immutable object with every setting, its default and its checks.

Governed by DESIGN.md 2.5 and docs/build-decisions.md Q38 (mechanism),
Q31 (llm_model has no default) and the entries named beside each field.

A value comes from, in rising order of precedence: the default here, an
optional TOML file given with --config, and a command-line flag. There are no
environment-variable settings: the environment carries only ANTHROPIC_API_KEY,
and only live mode reads it.

Every construction is checked, including dataclasses.replace(), so a Settings
object in hand is always valid.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_type_hints

from titlestd.contracts import TitlestdError

EMBEDDERS = ("offline", "local")
LLMS = ("offline", "replay", "anthropic")
EXAMPLE_LLM_MODEL = "claude-haiku-4-5"


class SettingsError(TitlestdError):
    """A setting is unknown, has the wrong type or an impossible value."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Every setting of a run. Field names are the setting names."""

    # Models (2.5)
    embedder: str = "offline"
    llm: str = "offline"
    llm_model: str | None = None
    # Reproducibility and places (Q06, Q29, Q38)
    seed: int = 20260101
    data_dir: Path = Path("data")
    out_dir: Path = Path("out")
    recording_dir: Path = Path("recordings")
    request_cap: int = 3000
    # Waves and the golden set (Q24, Q08)
    wave1_size: int = 40
    golden_size: int = 40
    # The ML match (2.4, Q04, Q12)
    ml_alone: bool = False
    ml_margin_cutoff: float = 0.10
    ml_min_examples: int = 3
    embed_dim: int = 32768
    # The agent and the stand-ins (Q10, Q17)
    agent_search_k: int = 5
    standin_min_title_words: int = 2
    standin_no_match_below: int = 2
    # Simulated people (Q27, Q01)
    sme_error_rate: float = 0.02
    undecided_rate: float = 0.10
    # The close and release (Q25, 2.4)
    generic_cap_share: float = 0.02
    release_stopped_paths: bool = False

    def __post_init__(self) -> None:
        for name, kind in _field_kinds().items():
            object.__setattr__(self, name, _coerce(name, getattr(self, name), kind))
        _validate(self)

    def as_dict(self) -> dict[str, Any]:
        """The settings in field order, with paths as strings (for manifests and logs)."""
        values = dataclasses.asdict(self)
        return {name: str(v) if isinstance(v, Path) else v for name, v in values.items()}


# One line per setting, shown by --help. A test checks none is missing.
HELP = {
    "embedder": "offline (built-in) or local (a small local embedding model)",
    "llm": "offline (stand-in), replay (recorded answers) or anthropic (live)",
    "llm_model": f"hosted model for llm = anthropic, for example {EXAMPLE_LLM_MODEL} (Q31)",
    "seed": "seed of every random draw in a run (Q06)",
    "data_dir": "folder holding the seven data files",
    "out_dir": "folder for the output files and the state file",
    "recording_dir": "folder holding recorded answers (Q29)",
    "request_cap": "most model requests one live run may make (Q02)",
    "wave1_size": "how many of the most widely held titles form the Wave 1 list (Q24)",
    "golden_size": "size of the golden set (Q08)",
    "ml_alone": "let a clear ML match skip the LLM; a person still decides (Q04)",
    "ml_margin_cutoff": "margin the ML-alone route needs; untuned (Q04)",
    "ml_min_examples": "approved examples the ML-alone route needs (Q04)",
    "embed_dim": "number of buckets of the offline embedder (Q12)",
    "agent_search_k": "results per agent search (Q10)",
    "standin_min_title_words": (
        "stand-in: with no description and fewer title words, insufficient data (Q17)"
    ),
    "standin_no_match_below": "stand-in: a best score below this is no match (Q17)",
    "sme_error_rate": "share of simulated human decisions that are wrong (Q27)",
    "undecided_rate": "share of review-queue titles a simulated reviewer cannot decide (Q01)",
    "generic_cap_share": "cap on generic family roles, as a share of employees (Q25)",
    "release_stopped_paths": (
        "release a stopped path as provisional instead of leaving it on legacy titles (2.4)"
    ),
}


@functools.cache
def _field_kinds() -> dict[str, Any]:
    return get_type_hints(Settings)


def _coerce(name: str, value: Any, kind: Any) -> Any:
    """Return value as the field's type, or raise SettingsError naming the setting."""
    if kind == (str | None):
        if value is None or isinstance(value, str):
            return value
        expected = "text"
    elif kind is bool:
        if isinstance(value, bool):
            return value
        expected = "true or false"
    elif kind is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        expected = "a whole number"
    elif kind is float:
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
        expected = "a number"
    elif kind is Path:
        if isinstance(value, str | Path) and str(value):
            return Path(value)
        expected = "a path"
    else:
        if isinstance(value, str):
            return value
        expected = "text"
    raise SettingsError(f"setting {name} must be {expected}, got {value!r}")


def _validate(settings: Settings) -> None:
    if settings.embedder not in EMBEDDERS:
        raise SettingsError(
            f"setting embedder must be one of {', '.join(EMBEDDERS)}; got {settings.embedder!r}"
        )
    if settings.llm not in LLMS:
        raise SettingsError(f"setting llm must be one of {', '.join(LLMS)}; got {settings.llm!r}")
    if settings.llm_model is not None and (
        not settings.llm_model or settings.llm_model != settings.llm_model.strip()
    ):
        raise SettingsError(
            f"setting llm_model must be a model name such as {EXAMPLE_LLM_MODEL}, "
            f"got {settings.llm_model!r}"
        )
    if settings.llm == "anthropic" and settings.llm_model is None:
        raise SettingsError(
            "llm = anthropic needs llm_model, which has no default; "
            f"name the model, for example --llm-model {EXAMPLE_LLM_MODEL}"
        )
    for name in ("request_cap", "embed_dim", "agent_search_k"):
        if getattr(settings, name) < 1:
            raise SettingsError(f"setting {name} must be at least 1, got {getattr(settings, name)}")
    for name in (
        "wave1_size",
        "golden_size",
        "ml_min_examples",
        "ml_margin_cutoff",
        "standin_min_title_words",
        "standin_no_match_below",
    ):
        if getattr(settings, name) < 0:
            raise SettingsError(
                f"setting {name} must not be negative, got {getattr(settings, name)}"
            )
    for name in ("sme_error_rate", "undecided_rate", "generic_cap_share"):
        if not 0.0 <= getattr(settings, name) <= 1.0:
            raise SettingsError(
                f"setting {name} must be between 0 and 1, got {getattr(settings, name)}"
            )


def setting_names() -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(Settings))


def load_settings(
    config: Path | None = None, overrides: Mapping[str, Any] | None = None
) -> Settings:
    """Build the settings from an optional TOML file and overrides.

    The TOML file is a flat table of setting names. An override of None means
    "not given". Unknown names are an error, in the file and in overrides.
    """
    values: dict[str, Any] = {}
    if config is not None:
        values.update(_read_config(Path(config)))
    for name, value in (overrides or {}).items():
        if value is not None:
            values[name] = value
    unknown = sorted(set(values) - set(setting_names()))
    if unknown:
        where = f" in {config}" if config is not None else ""
        raise SettingsError(
            f"unknown setting{'s' if len(unknown) > 1 else ''}{where}: {', '.join(unknown)}; "
            f"the settings are: {', '.join(setting_names())}"
        )
    return Settings(**values)


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise SettingsError(f"settings file not found: {path}") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise SettingsError(f"settings file {path} is not valid TOML: {error}") from None
    except OSError as error:
        raise SettingsError(f"settings file {path} cannot be read: {error}") from None


def add_setting_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --config and one flag per setting, named after it with dashes."""
    parser.add_argument(
        "--config", type=Path, default=None, metavar="FILE", help="TOML file of settings"
    )
    kinds = _field_kinds()
    for spec in dataclasses.fields(Settings):
        flag = "--" + spec.name.replace("_", "-")
        text = f"{HELP[spec.name]} (default: {_show(spec.default)})"
        kind = kinds[spec.name]
        if kind is bool:
            parser.add_argument(
                flag, dest=spec.name, default=None, action=argparse.BooleanOptionalAction, help=text
            )
        elif kind in (int, float, Path):
            parser.add_argument(
                flag,
                dest=spec.name,
                default=None,
                type=kind,
                metavar=kind.__name__.upper(),
                help=text,
            )
        else:
            parser.add_argument(flag, dest=spec.name, default=None, metavar="TEXT", help=text)


def settings_from_arguments(arguments: argparse.Namespace, **forced: Any) -> Settings:
    """Settings from parsed arguments. forced values win over file and flags."""
    overrides = {name: getattr(arguments, name, None) for name in setting_names()}
    overrides.update(forced)
    return load_settings(getattr(arguments, "config", None), overrides)


def _show(default: Any) -> str:
    if default is None:
        return "none"
    if isinstance(default, bool):
        return "on" if default else "off"
    return str(default)
