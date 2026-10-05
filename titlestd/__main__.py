"""Entry point of "python3 -m titlestd".

The interpreter is checked here, before any other module of the package is
imported: on a Python older than the package needs, that import ends in a
traceback which does not say why. So this file itself uses nothing that only
a recent Python has.
"""

import sys

# The oldest Python the package runs on: requires-python of pyproject.toml (Q38).
MINIMUM_PYTHON = (3, 12)


def run() -> int:
    """Run the command line and return its exit status; 1 on a Python that is too old."""
    if sys.version_info < MINIMUM_PYTHON:
        needed = ".".join(str(part) for part in MINIMUM_PYTHON)
        found = ".".join(str(part) for part in sys.version_info[:2])
        sys.stderr.write(f"titlestd: Python {needed} or later is needed; this is Python {found}.\n")
        return 1
    from titlestd.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
