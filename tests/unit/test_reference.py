"""The invented reference lists for cleaning, masking and data generation (Q14)."""

from __future__ import annotations

import re

import pytest

from titlestd import reference

LISTS = {
    "GIVEN_NAMES": reference.GIVEN_NAMES,
    "SURNAMES": reference.SURNAMES,
    "LOCATIONS": reference.LOCATIONS,
}


@pytest.mark.parametrize("name", sorted(LISTS))
def test_entries_are_single_lower_case_tokens_in_order(name):
    entries = LISTS[name]
    assert all(re.fullmatch(r"[a-z]+", entry) for entry in entries)
    assert list(entries) == sorted(set(entries))


def test_the_lists_are_about_the_decided_size():
    assert len(reference.LOCATIONS) == 10
    assert len(reference.GIVEN_NAMES) >= 10 and len(reference.SURNAMES) >= 10


def test_no_word_is_in_two_lists():
    everything = [entry for entries in LISTS.values() for entry in entries]
    assert len(everything) == len(set(everything))


def _has_requisition(text: str) -> bool:
    return any(
        re.search(pattern, text, re.IGNORECASE) for pattern in reference.REQUISITION_PATTERNS
    )


@pytest.mark.parametrize(
    "title",
    [
        "sales representative REQ-10442",
        "Sales Engineer #20311",
        "analyst req-1234",
        "Analyst (#123456)",
        "REQ-999999 Clerk",
    ],
)
def test_requisition_numbers_are_recognised(title):
    assert _has_requisition(title)


@pytest.mark.parametrize(
    "title",
    [
        "ACCT MGR II",
        "Engineer 2",
        "Analyst #12",
        "Analyst #123",
        "Clerk REQ-123",
        "Frequent Flyer Desk",
        "prereq-1234 checker",
    ],
)
def test_level_numerals_and_short_numbers_are_left_alone(title):
    assert not _has_requisition(title)


def test_there_are_two_requisition_patterns():
    assert len(reference.REQUISITION_PATTERNS) == 2


def test_the_email_domain_is_the_reserved_one():
    assert reference.EMAIL_DOMAIN == "example.com"
