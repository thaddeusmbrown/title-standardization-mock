"""The prepare step: cleaning a title, masking, the decision key (Q14, S05, S06)."""

from __future__ import annotations

import pytest

from titlestd import reference
from titlestd.contracts import EMAIL_MASK, NAME_MASK, Abbreviation, TitleKey
from titlestd.prepare import CleanedTitle, clean_title, mask_text, title_key

ABBREVIATIONS = (
    Abbreviation("sr", "senior"),
    Abbreviation("acct", "account"),
    Abbreviation("mgr", "manager"),
    Abbreviation("rep", "representative"),
    Abbreviation("spec", "specialist"),
    Abbreviation("cust", "customer"),
    Abbreviation("ap", "accounts payable"),
)
NAMES = (*reference.GIVEN_NAMES, *reference.SURNAMES)


def cleaned(job_title: str, abbreviations=ABBREVIATIONS) -> str:
    return clean_title(job_title, abbreviations).text


# --- Cleaning a title, step by step (S06) ----------------------------------------


def test_the_worked_example_of_the_design():
    assert clean_title("Sr. Acct Mgr - NYC", ABBREVIATIONS) == CleanedTitle(
        text="senior account manager", has_person_name=False
    )


def test_a_title_is_lower_cased():
    assert cleaned("SENIOR Account MANAGER") == "senior account manager"


@pytest.mark.parametrize(
    ("job_title", "expected"),
    [
        ("sales representative REQ-10442", "sales representative"),
        ("Sales Engineer #20311", "sales engineer"),
        ("analyst req-1234", "analyst"),
        ("Analyst (#123456)", "analyst"),
        ("REQ-999999 Clerk", "clerk"),
        ("Clerk Req-10442 Payroll", "clerk payroll"),
    ],
)
def test_requisition_numbers_are_removed(job_title, expected):
    # Before the punctuation goes: stripped first, "REQ-10442" would be left
    # as the two tokens "req" and "10442".
    assert cleaned(job_title) == expected


@pytest.mark.parametrize(
    ("job_title", "expected"),
    [
        ("ACCT MGR II", "account manager ii"),
        ("Engineer 2", "engineer 2"),
        ("Analyst #12", "analyst 12"),
        ("Analyst #123", "analyst 123"),
        ("Clerk REQ-123", "clerk req 123"),
        ("Analyst #1234567", "analyst 1234567"),
    ],
)
def test_level_numerals_and_other_numbers_stay(job_title, expected):
    assert cleaned(job_title) == expected


@pytest.mark.parametrize(
    ("job_title", "expected"),
    [
        ("Sr. Acct Mgr", "senior account manager"),
        ("Acct/Mgr", "account manager"),
        ("Sr.Acct.Mgr", "senior account manager"),
        ("R&D_Lead,", "r d lead"),
        ("(Clerk) - [Payroll]!", "clerk payroll"),
        ('The "Front" Desk', "the front desk"),
        ("Clerk's Desk", "clerk s desk"),
    ],
)
def test_punctuation_is_stripped_and_separates_tokens(job_title, expected):
    assert cleaned(job_title) == expected


def test_spaces_are_collapsed():
    assert cleaned("  Sales \t Rep\n-  ") == "sales representative"


@pytest.mark.parametrize("location", reference.LOCATIONS)
def test_every_trailing_location_is_removed(location):
    assert cleaned(f"Sales Rep - {location.upper()}") == "sales representative"
    assert cleaned(f"Sales Rep {location}") == "sales representative"


def test_several_trailing_locations_are_all_removed():
    assert cleaned("Sales Rep - NYC / Remote") == "sales representative"


def test_a_location_is_removed_from_the_end_only():
    assert cleaned("NYC Sales Rep") == "nyc sales representative"
    assert cleaned("Sales NYC Rep - LDN") == "sales nyc representative"


def test_a_location_is_a_whole_token():
    assert cleaned("Analyst Seas") == "analyst seas"
    assert cleaned("Clerk Remotely") == "clerk remotely"


def test_a_location_in_front_of_a_requisition_number_is_still_trailing():
    assert cleaned("Sales Rep - NYC REQ-10442") == "sales representative"
    assert cleaned("Sales Rep #20311 - NYC") == "sales representative"


def test_abbreviations_are_expanded_as_whole_tokens():
    assert cleaned("AP Clerk") == "accounts payable clerk"
    assert cleaned("Payroll Spec") == "payroll specialist"
    # "spec" inside "special" and "ap" inside "apprentice" are not tokens.
    assert cleaned("Special Projects") == "special projects"
    assert cleaned("Apprentice Rep.") == "apprentice representative"


def test_a_title_with_no_abbreviation_is_only_cleaned():
    assert cleaned("BDR") == "bdr"
    assert cleaned("Sr. Acct Mgr", abbreviations=()) == "sr acct mgr"


def test_an_expansion_is_not_expanded_again():
    chained = (Abbreviation("vp", "vice pres"), Abbreviation("pres", "president"))
    assert cleaned("VP Sales", chained) == "vice pres sales"
    assert cleaned("Pres", chained) == "president"


def test_abbreviations_match_whatever_the_case_of_the_file():
    shouting = (Abbreviation("SR", "Senior"), Abbreviation("Mgr", "MANAGER"))
    assert cleaned("Sr Mgr", shouting) == "senior manager"


def test_locations_go_before_abbreviations_are_expanded():
    # The order of S06. Expanded first, "ldn" would no longer be a location
    # token and the title would keep "london".
    assert cleaned("Analyst LDN", (Abbreviation("ldn", "london"),)) == "analyst"
    # And a location that an expansion puts at the end is not removed.
    assert cleaned("Analyst HQ", (Abbreviation("hq", "office nyc"),)) == "analyst office nyc"


@pytest.mark.parametrize("job_title", ["#12345", "NYC", "REQ-1234 - Remote", "(-)", "   "])
def test_a_title_of_removable_parts_only_cleans_to_nothing(job_title):
    assert clean_title(job_title, ABBREVIATIONS) == CleanedTitle(text="", has_person_name=False)


def test_letters_outside_ascii_are_letters():
    assert cleaned("Café Mgr") == "café manager"


# --- Names in a title (Q14) --------------------------------------------------------


def test_a_given_name_and_surname_become_one_mask():
    assert clean_title("Cust Success Mgr (Tavi Quillfeather)", ABBREVIATIONS) == CleanedTitle(
        text=f"customer success manager {NAME_MASK}", has_person_name=True
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_listed_name_is_masked_when_it_stands_alone(name):
    for written in (name, name.upper(), name.capitalize()):
        assert clean_title(f"Assistant to {written}", ABBREVIATIONS) == CleanedTitle(
            text=f"assistant to {NAME_MASK}", has_person_name=True
        )


def test_a_name_anywhere_in_the_title_is_masked():
    assert cleaned("Tavi Quillfeather's Assistant") == f"{NAME_MASK} s assistant"
    assert cleaned("Office of Oswin Thistlewood, Sales") == f"office of {NAME_MASK} sales"


def test_only_a_given_name_followed_by_a_surname_is_one_mask():
    assert cleaned("Quillfeather, Tavi") == f"{NAME_MASK} {NAME_MASK}"
    assert cleaned("Tavi Maren Quillfeather") == f"{NAME_MASK} {NAME_MASK}"
    assert cleaned("Tavi Quillfeather / Oswin Thistlewood") == f"{NAME_MASK} {NAME_MASK}"


def test_a_word_that_only_contains_a_name_is_not_one():
    for job_title in ("Assistant to Tavita", "Marengo Analyst", "Preoswin Clerk", "Tavi2 Clerk"):
        result = clean_title(job_title, ABBREVIATIONS)
        assert NAME_MASK not in result.text and not result.has_person_name


def test_a_title_without_a_name_is_not_flagged():
    assert not clean_title("Sr. Acct Mgr - NYC", ABBREVIATIONS).has_person_name
    # The mask's own word is not a name: brackets are punctuation here.
    assert clean_title(f"Clerk {NAME_MASK}", ABBREVIATIONS) == CleanedTitle("clerk name", False)


def test_names_are_masked_after_the_title_is_cleaned():
    # The location goes although a requisition number follows it, and the name
    # is found although it was written in capitals and brackets.
    assert cleaned("Sr Acct Mgr (TAVI QUILLFEATHER) - NYC #20311") == (
        f"senior account manager {NAME_MASK}"
    )


def test_two_people_in_the_same_job_clean_to_the_same_title():
    assert cleaned("Assistant to Tavi Quillfeather") == cleaned("Assistant to Oswin Thistlewood")


# --- Masking a description or a department (Q14) -----------------------------------


def test_an_email_is_masked_and_the_full_stop_after_it_stays():
    text = "Maintains the pipeline reports. Questions to maren.ashgrove@example.com."
    assert mask_text(text) == f"Maintains the pipeline reports. Questions to {EMAIL_MASK}."


def test_emails_are_masked_before_names():
    # Names first would leave "[NAME].[NAME]@example.com" behind.
    assert mask_text("maren.ashgrove@example.com") == EMAIL_MASK
    assert mask_text("Write to Maren Ashgrove <maren@example.com>.") == (
        f"Write to {NAME_MASK} <{EMAIL_MASK}>."
    )


@pytest.mark.parametrize(
    ("local_part", "domain"),
    [
        ("payroll", reference.EMAIL_DOMAIN),
        ("first.last+tag", f"mail.{reference.EMAIL_DOMAIN}"),
        ("A_B-C", reference.EMAIL_DOMAIN.upper()),
        ("desk", "intranet"),  # a domain with no dot is masked too
    ],
)
def test_emails_of_several_shapes_are_masked(local_part, domain):
    text = f"Send it to {local_part}@{domain}, please."
    assert mask_text(text) == f"Send it to {EMAIL_MASK}, please."


def test_a_name_in_a_description_is_masked_as_one():
    text = "Owns the largest customer accounts. Reports to Oswin Thistlewood."
    assert mask_text(text) == f"Owns the largest customer accounts. Reports to {NAME_MASK}."


@pytest.mark.parametrize("name", NAMES)
def test_every_listed_name_is_masked_in_text(name):
    for written in (name, name.upper(), name.capitalize()):
        assert mask_text(f"Covers for {written}; see below.") == (
            f"Covers for {NAME_MASK}; see below."
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Oswin's team", f"{NAME_MASK}'s team"),
        ("OSWIN  \n THISTLEWOOD signs off", f"{NAME_MASK} signs off"),
        ("Thistlewood, Oswin", f"{NAME_MASK}, {NAME_MASK}"),
        ("Oswin-Thistlewood", f"{NAME_MASK}-{NAME_MASK}"),
        ("oswin_thistlewood", f"{NAME_MASK}_{NAME_MASK}"),
        ("Tavi and Liora Wrenfield", f"{NAME_MASK} and {NAME_MASK}"),
    ],
)
def test_names_in_other_spellings(text, expected):
    assert mask_text(text) == expected


def test_a_longer_word_is_not_a_name():
    text = "Marengo sauce, Tavita and the Oswinian method."
    assert mask_text(text) == text


def test_nothing_else_changes():
    for text in (
        "",
        "  Keeps   the\tgeneral ledger;  PREPARES monthly accounts (Sr. Acct Mgr - NYC, #20311). ",
        "Line one.\nLine two.",
    ):
        assert mask_text(text) == text


def test_a_department_is_masked_the_same_way():
    assert mask_text("Office of Calista Fennimore") == f"Office of {NAME_MASK}"
    assert mask_text("Enterprise Sales") == "Enterprise Sales"


# --- The decision key (S05, Q09, Q26) ----------------------------------------------


def test_the_key_is_the_family_code_and_the_cleaned_title():
    key = title_key("Sr. Acct Mgr - NYC", "SAL", ABBREVIATIONS)
    assert key == TitleKey(family_code="SAL", cleaned_title="senior account manager")
    assert isinstance(key, TitleKey)


def test_the_key_of_a_title_with_a_name_is_the_masked_title():
    assert title_key("Cust Success Mgr (Tavi Quillfeather)", "SAL", ABBREVIATIONS) == TitleKey(
        "SAL", f"customer success manager {NAME_MASK}"
    )


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        ("FIN ", "FIN"),  # padded: the same family (S05)
        (" \tFIN\n", "FIN"),
        ("", ""),  # blank stays blank (Q26)
        ("   ", ""),
        ("XYZ", "XYZ"),  # unknown: kept as recorded
        ("fin", "fin"),
        ("F IN", "F IN"),
    ],
)
def test_the_family_code_is_trimmed_and_otherwise_kept_as_recorded(recorded, expected):
    assert title_key("Treasury Analyst", recorded, ABBREVIATIONS) == TitleKey(
        expected, "treasury analyst"
    )


def test_spellings_of_one_title_share_a_key_and_families_do_not():
    one = title_key("Sr. Acct Mgr - NYC", "SAL", ABBREVIATIONS)
    assert title_key("Senior Account Manager", "SAL ", ABBREVIATIONS) == one
    assert title_key("senior account manager REQ-10442", "SAL", ABBREVIATIONS) == one
    assert title_key("Senior Account Manager", "FIN", ABBREVIATIONS) != one
    assert title_key("ACCT MGR II", "SAL", ABBREVIATIONS) != one
