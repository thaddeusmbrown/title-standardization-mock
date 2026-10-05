"""Shared test scaffolding.

This file belongs to the scaffold branch. A later branch that needs more
fixtures defines them in its own test files, not here, so branches built in
parallel never edit the same file.

tests/fixtures/mini is a tiny hand-written data set in the seven-file format of
docs/design.md 2.3: two families with six roles each plus their generic roles,
22 employees on 15 title rows. It reproduces the worked example of 1.3 and
holds one of each kind of mess: abbreviations, mixed case, a location suffix,
two requisition suffixes, level words in several forms, a person's name in a
title, a name and an email in descriptions, empty descriptions, missing grades
and a blank family code. Everything in it is invented. Tests read it and never
write into it.

MINI_ROWS is what the prepare step must produce from it, written out by hand
(Q11, Q14, Q26), so branches can test against real title rows before the
prepare step exists.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from titlestd import datafiles
from titlestd.contracts import NAME_MASK, Catalog, InputData, Record, TitleRow
from titlestd.settings import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
MINI_DIR = REPO_ROOT / "tests" / "fixtures" / "mini"

# The optional extras and the third-party modules each one brings (2.5, Q38).
EXTRA_MODULES = {"live": ("anthropic", "httpx2"), "local": ("model2vec", "numpy")}
# Third-party modules the default run must never import (2.5, Q32).
OPTIONAL_MODULES = tuple(module for modules in EXTRA_MODULES.values() for module in modules)


def _row(
    family: str,
    title: str,
    employee_ids: tuple[str, ...],
    raw_titles: tuple[str, ...],
    department: str,
    description: str = "",
    *,
    has_person_name: bool = False,
    family_usable: bool = True,
) -> TitleRow:
    return TitleRow(
        record=Record(
            cleaned_title=title, family_code=family, department=department, description=description
        ),
        employee_ids=employee_ids,
        raw_titles=raw_titles,
        has_person_name=has_person_name,
        family_usable=family_usable,
    )


# The 15 title rows of the mini data set, sorted by key (family code, cleaned title).
MINI_ROWS: tuple[TitleRow, ...] = (
    _row(
        "",
        "staff accountant",
        ("E021",),
        ("Staff Accountant",),
        "Accounting",
        "Keeps the general ledger and prepares monthly accounts.",
        family_usable=False,
    ),
    _row(
        "FIN",
        "accounts payable clerk",
        ("E017",),
        ("AP Clerk",),
        "Accounting",
        "Processes supplier invoices and payments.",
    ),
    _row(
        "FIN",
        "financial analyst",
        ("E014", "E015"),
        ("Fin Analyst",),
        "Corporate Finance",
        "Builds budgets and forecasts and explains variances.",
    ),
    _row(
        "FIN",
        "internal auditor",
        ("E018",),
        ("Internal Auditor",),
        "Internal Audit",
        "Tests financial controls and reports findings to the audit committee.",
    ),
    _row("FIN", "payroll specialist", ("E019",), ("Payroll Spec",), "Accounting"),
    _row(
        "FIN",
        "senior financial analyst",
        ("E016",),
        ("Senior Financial Analyst",),
        "Corporate Finance",
        "Leads the annual budget and the quarterly forecast.",
    ),
    _row(
        "FIN",
        "special projects",
        ("E022",),
        ("Special Projects",),
        "Corporate Finance",
        "Works on assignments set by the leadership team.",
    ),
    _row(
        "FIN",
        "treasury analyst",
        ("E020",),
        ("Treasury Analyst",),
        "Treasury",
        "Manages cash positions and bank accounts.",
    ),
    _row(
        "SAL",
        "account manager ii",
        ("E001", "E002", "E003"),
        ("ACCT MGR II",),
        "Enterprise Sales",
        # Held by two of the three employees: the most common description.
        "Manages a portfolio of enterprise accounts and renews their contracts.",
    ),
    _row("SAL", "bdr", ("E012",), ("BDR",), "Inside Sales"),
    _row(
        "SAL",
        f"customer success manager {NAME_MASK}",
        ("E011",),
        ("Cust Success Mgr (Tavi Quillfeather)",),
        "Customer Success",
        "Helps existing customers adopt products after the sale.",
        has_person_name=True,
    ),
    _row(
        "SAL",
        "sales engineer",
        ("E013",),
        ("Sales Engineer #20311",),
        "Solutions",
        "Explains the technical side of products and runs demonstrations during a sale.",
    ),
    _row(
        "SAL",
        "sales operations analyst",
        ("E010",),
        ("Sales Ops Analyst",),
        "Sales Operations",
        "Maintains the sales forecast, territories and pipeline reports. Questions to [EMAIL].",
    ),
    _row(
        "SAL",
        "sales representative",
        ("E007", "E008", "E009"),
        ("Sales Rep", "sales representative REQ-10442"),
        "Inside Sales",
        # Two descriptions are held once each: the tie goes to the lowest employee_id.
        "Sells products to new customers by phone and closes deals.",
    ),
    _row(
        "SAL",
        "senior account manager",
        ("E004", "E005", "E006"),
        ("Senior Account Manager", "Sr. Acct Mgr - NYC"),
        "Enterprise Sales",
        f"Owns the largest customer accounts and negotiates renewals. Reports to {NAME_MASK}.",
    ),
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Everything under tests/integration carries the integration marker."""
    integration = REPO_ROOT / "tests" / "integration"
    for item in items:
        if integration in Path(item.path).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def optional_modules() -> tuple[str, ...]:
    """The third-party modules the default run must never import."""
    return OPTIONAL_MODULES


@pytest.fixture(scope="session")
def extra_modules() -> dict[str, tuple[str, ...]]:
    """Each optional extra of pyproject.toml and the modules it makes importable."""
    return EXTRA_MODULES


@pytest.fixture(scope="session")
def mini_dir() -> Path:
    """The folder of the mini data set. Read-only: copy it before changing a file."""
    return MINI_DIR


@pytest.fixture(scope="session")
def mini_inputs() -> InputData:
    """The six input files of the mini data set, read and checked."""
    return datafiles.read_inputs(MINI_DIR)


@pytest.fixture(scope="session")
def mini_catalog(mini_inputs: InputData) -> Catalog:
    return mini_inputs.catalog


@pytest.fixture(scope="session")
def mini_rows() -> tuple[TitleRow, ...]:
    """The title rows of the mini data set, as the prepare step must build them."""
    return MINI_ROWS


@pytest.fixture
def mini_settings(tmp_path: Path) -> Settings:
    """Settings sized for the mini data set, writing only under tmp_path.

    The Wave 1 list is the four most widely held rows (headcounts 3, 3, 3, 2;
    the fifth has 1, so membership never rests on the tie-break), and the
    golden set has three titles.
    """
    return Settings(
        data_dir=MINI_DIR,
        out_dir=tmp_path / "out",
        recording_dir=tmp_path / "recordings",
        wave1_size=4,
        golden_size=3,
    )


def run_bare(*arguments: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """Run "python -m titlestd" on a bare interpreter: no site-packages (-S) and
    no PYTHON* environment variables (-E), from the repository root."""
    return subprocess.run(
        [sys.executable, "-S", "-E", "-m", "titlestd", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.fixture(scope="session")
def bare_cli():
    """Callable running the command line on a bare interpreter; see run_bare."""
    return run_bare
