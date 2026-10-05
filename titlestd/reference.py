"""Invented reference lists for cleaning, masking and data generation (Q14).

The mock has no name-masking service, so a person's name is recognised by
looking it up in these lists. They are module constants, not data files: the
data files are exactly the seven of docs/design.md 2.3. The prepare step and
the data generator both read them from here, so what the generator writes is
what the cleaner recognises.

All entries are lower-case single tokens, made up for this mock. No list
shares a word with another, or with a role name, abbreviation, level word or
department in the data (tests assert this for each data set).
"""

from __future__ import annotations

# Given names and surnames. An adjacent given name and surname are masked as
# one [NAME]; a lone listed token is masked too.
GIVEN_NAMES = (
    "bramwell",
    "calista",
    "dariel",
    "isolde",
    "kasimir",
    "liora",
    "maren",
    "odalys",
    "oswin",
    "sunniva",
    "tavi",
    "yarrow",
)

SURNAMES = (
    "ashgrove",
    "brackenridge",
    "fennimore",
    "harrowgate",
    "larkspur",
    "merriweather",
    "oakhollow",
    "pendlewick",
    "quillfeather",
    "stonebrook",
    "thistlewood",
    "wrenfield",
)

# Location tokens stripped from the end of a title, as in "Sr. Acct Mgr - NYC".
LOCATIONS = (
    "atl",
    "bos",
    "chi",
    "den",
    "ldn",
    "nyc",
    "remote",
    "sea",
    "sgp",
    "syd",
)

# Requisition numbers, matched without regard to case. At least four digits,
# so a level numeral such as "II" or "2" is never removed.
REQUISITION_PATTERNS = (
    r"\breq-\d{4,6}\b",
    r"#\d{4,6}\b",
)

# The reserved domain for every invented email address (2.3).
EMAIL_DOMAIN = "example.com"
