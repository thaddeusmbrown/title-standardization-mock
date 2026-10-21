"""titlestd: a small offline mock of a job-title standardization workflow.

Run it from the repository root with nothing installed:

    python3 -m titlestd run

DESIGN.md describes the design and scopes the mock. docs/build-decisions.md
records every point the design leaves open and how the build is cut into
modules. titlestd/contracts.py holds the types the modules exchange.

Importing this package imports only the standard library. The hosted-model SDK
and the local embedding library are optional and are imported only inside the
modules that use them, when their setting is chosen (2.5).
"""

__version__ = "0.1.0"
