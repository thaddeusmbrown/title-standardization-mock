"""The command line: python3 -m titlestd <command> [settings].

Commands (Q38):
  run            set-up, Wave 1, Wave 2 and the close on the mock data; prints
                 one table and writes the output files (2.6)
  generate-data  write the seven mock data files into the data folder (2.3)
  record         the same run with llm = anthropic, which records every answer
                 (2.5, Q02); it needs --llm-model and ANTHROPIC_API_KEY

Each command is one function in the module that owns it, imported only when
the command runs. This file does not change when a component is built: a
command whose module is still a stub exits with status 1 and says so.

Exit status: 0 success, 1 the run stopped with an error, 2 the command line or
a setting is wrong.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from titlestd.contracts import TitlestdError
from titlestd.settings import SettingsError, add_setting_arguments, settings_from_arguments

# command -> (help, module, function taking Settings and returning an exit status,
#             settings the command forces)
COMMANDS = {
    "run": (
        "run set-up, Wave 1, Wave 2 and the close, print the table, write the files",
        "titlestd.pipeline",
        "run_command",
        {},
    ),
    "generate-data": (
        "write the seven mock data files into the data folder",
        "titlestd.mockdata",
        "generate_command",
        {},
    ),
    "record": (
        "run against the live model and record every answer (llm = anthropic)",
        "titlestd.pipeline",
        "run_command",
        {"llm": "anthropic"},
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m titlestd",
        description="A small offline mock of a job-title standardization workflow.",
        epilog="Every command takes the same settings; see: python3 -m titlestd run --help",
    )
    commands = parser.add_subparsers(dest="command", metavar="command", required=True)
    for name, (help_text, _, _, _) in COMMANDS.items():
        command = commands.add_parser(name, help=help_text, description=help_text)
        add_setting_arguments(command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    _, module_name, function_name, forced = COMMANDS[arguments.command]
    try:
        settings = settings_from_arguments(arguments, **forced)
    except SettingsError as error:
        print(f"titlestd: {error}", file=sys.stderr)
        return 2
    try:
        function = getattr(importlib.import_module(module_name), function_name)
        return int(function(settings) or 0)
    except NotImplementedError as error:
        detail = f" ({error})" if str(error) else ""
        print(
            f"titlestd: the '{arguments.command}' command is not built yet{detail}",
            file=sys.stderr,
        )
        return 1
    except TitlestdError as error:
        print(f"titlestd: {error}", file=sys.stderr)
        return 1
