"""The look-up table and the examples (design 1.2, 1.6 steps 2 and 10).

One test of section 2.7 starts here: golden-set titles never enter the look-up
table or the examples (T6). Knowledge.learn is the one place that adds either,
and it refuses a golden-set key whatever the decision (Q07).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from titlestd import lookup
from titlestd.contracts import (
    NAME_MASK,
    Example,
    Record,
    TitleKey,
    TitleRow,
)
from titlestd.lookup import Knowledge


def _row(
    family: str,
    title: str,
    description: str = "",
    *,
    has_person_name: bool = False,
    family_usable: bool = True,
    raw_titles: tuple[str, ...] = (),
) -> TitleRow:
    return TitleRow(
        record=Record(title, family, "Some Department", description),
        employee_ids=("E1",),
        raw_titles=raw_titles or (title,),
        has_person_name=has_person_name,
        family_usable=family_usable,
    )


MANAGER = _row("SAL", "account manager", "Looks after customer accounts.")
SENIOR = _row("SAL", "senior account manager")
ANALYST = _row("FIN", "financial analyst", "Builds budgets.")
NAMED = _row("SAL", f"customer success manager {NAME_MASK}", "Helps.", has_person_name=True)
NO_FAMILY = _row("", "staff accountant", "Keeps the ledger.", family_usable=False)
UNKNOWN_FAMILY = _row("XYZ", "staff accountant", "Keeps the ledger.", family_usable=False)


# --- The look-up (1.6 step 2) ---------------------------------------------------


def test_nothing_is_known_at_the_start():
    knowledge = Knowledge()
    assert knowledge.lookup(MANAGER.key) is None
    assert knowledge.entries() == ()
    assert knowledge.examples() == ()
    assert knowledge.example_count("SAL-01") == 0


def test_a_persons_decision_is_a_look_up_entry_and_an_example():
    knowledge = Knowledge()
    assert knowledge.learn(MANAGER, "SAL-01") is True
    assert knowledge.lookup(TitleKey("SAL", "account manager")) == "SAL-01"
    assert knowledge.entries() == ((TitleKey("SAL", "account manager"), "SAL-01"),)
    # The example keeps the title's record, the text it was queried with (Q12).
    assert knowledge.examples() == (Example(MANAGER.record, "SAL-01"),)
    assert knowledge.examples()[0].key == MANAGER.key


@pytest.mark.parametrize(
    "key",
    [
        TitleKey("FIN", "account manager"),  # the same title in another family
        TitleKey("SAL", "account managers"),
        TitleKey("SAL", "Account Manager"),
        TitleKey("SAL ", "account manager"),  # whoever builds a key trims the code (S05)
        TitleKey("", "account manager"),
        TitleKey("SAL", ""),
    ],
)
def test_the_look_up_is_exact_on_cleaned_title_plus_family(key):
    knowledge = Knowledge()
    knowledge.learn(MANAGER, "SAL-01")
    assert knowledge.lookup(key) is None


def test_a_plain_pair_looks_up_like_a_key():
    knowledge = Knowledge()
    knowledge.learn(MANAGER, "SAL-01")
    assert knowledge.lookup(("SAL", "account manager")) == "SAL-01"


def test_entries_and_examples_are_sorted_by_key_whatever_the_order_learned():
    rows = [SENIOR, ANALYST, NO_FAMILY, MANAGER]
    expected = [NO_FAMILY.key, ANALYST.key, MANAGER.key, SENIOR.key]
    for order in (rows, rows[::-1], rows[2:] + rows[:2]):
        knowledge = Knowledge()
        for row in order:
            knowledge.learn(row, "R-" + row.record.cleaned_title)
        assert [key for key, _ in knowledge.entries()] == expected
        assert [example.key for example in knowledge.examples()] == expected


def test_example_count_is_per_role():
    knowledge = Knowledge()
    knowledge.learn(MANAGER, "SAL-01")
    knowledge.learn(SENIOR, "SAL-01")
    knowledge.learn(ANALYST, "FIN-01")
    assert knowledge.example_count("SAL-01") == 2
    assert knowledge.example_count("FIN-01") == 1
    assert knowledge.example_count("FIN-02") == 0


def test_a_later_decision_replaces_the_earlier_one_in_both():
    knowledge = Knowledge()
    knowledge.learn(MANAGER, "SAL-01")
    assert knowledge.learn(MANAGER, "SAL-06") is True
    assert knowledge.entries() == ((MANAGER.key, "SAL-06"),)
    assert knowledge.examples() == (Example(MANAGER.record, "SAL-06"),)
    assert knowledge.example_count("SAL-01") == 0
    assert knowledge.example_count("SAL-06") == 1


def test_learning_the_same_decision_twice_changes_nothing():
    knowledge = Knowledge()
    knowledge.learn(MANAGER, "SAL-01")
    knowledge.learn(MANAGER, "SAL-01")
    assert knowledge.entries() == ((MANAGER.key, "SAL-01"),)
    assert knowledge.example_count("SAL-01") == 1


def test_what_was_handed_out_does_not_change_when_more_is_learned():
    knowledge = Knowledge()
    knowledge.learn(MANAGER, "SAL-01")
    entries, examples = knowledge.entries(), knowledge.examples()
    knowledge.learn(ANALYST, "FIN-01")
    assert entries == ((MANAGER.key, "SAL-01"),)
    assert examples == (Example(MANAGER.record, "SAL-01"),)
    assert len(knowledge.entries()) == len(knowledge.examples()) == 2


def test_two_knowledges_share_nothing():
    first, second = Knowledge(), Knowledge()
    first.learn(MANAGER, "SAL-01")
    assert second.entries() == () and second.examples() == ()


@pytest.mark.parametrize("role_id", [None, ""])
def test_a_decision_that_names_no_role_is_nothing_to_learn(role_id):
    # "Could not decide" and "no suitable role" carry no role (Q01, Q03). An
    # entry without one would read as a miss, so it is refused out loud.
    for knowledge in (Knowledge(), Knowledge(golden_keys=[MANAGER.key])):
        with pytest.raises(ValueError, match="no role to learn for 'account manager'"):
            knowledge.learn(MANAGER, role_id)
        assert knowledge.entries() == () and knowledge.examples() == ()


# --- The golden-set guard (2.7, Q07) --------------------------------------------


def test_a_golden_title_enters_neither_the_look_up_nor_the_examples():
    knowledge = Knowledge(golden_keys=[MANAGER.key])
    assert knowledge.learn(MANAGER, "SAL-01") is False
    assert knowledge.lookup(MANAGER.key) is None
    assert knowledge.entries() == ()
    assert knowledge.examples() == ()
    assert knowledge.example_count("SAL-01") == 0


def test_the_guard_holds_whatever_the_decision_and_however_often():
    # A golden label, an audit label, a small path labeled in full, a queue
    # decision: they all arrive as learn(row, role), and none is recorded.
    knowledge = Knowledge(golden_keys=[MANAGER.key, ANALYST.key])
    for role_id in ("SAL-01", "SAL-02", "SAL-01", "FIN-01"):
        assert knowledge.learn(MANAGER, role_id) is False
        assert knowledge.learn(ANALYST, role_id) is False
    assert knowledge.entries() == () and knowledge.examples() == ()


def test_the_guard_is_on_the_key_whatever_else_the_row_holds():
    knowledge = Knowledge(golden_keys=[MANAGER.key])
    same_key = TitleRow(
        record=Record("account manager", "SAL", "Another Department", "Another description."),
        employee_ids=("E7", "E8"),
        raw_titles=("ACCT MGR", "Account Manager - NYC"),
    )
    assert knowledge.learn(same_key, "SAL-01") is False
    assert knowledge.entries() == () and knowledge.examples() == ()


def test_titles_beside_the_golden_set_are_learned_as_before():
    knowledge = Knowledge(golden_keys=[MANAGER.key])
    assert knowledge.learn(SENIOR, "SAL-01") is True
    assert knowledge.learn(MANAGER, "SAL-01") is False
    assert knowledge.learn(ANALYST, "FIN-01") is True
    assert [key for key, _ in knowledge.entries()] == [ANALYST.key, SENIOR.key]
    assert [example.key for example in knowledge.examples()] == [ANALYST.key, SENIOR.key]


@pytest.mark.parametrize(
    "make_golden_keys",
    [
        lambda: (key for key in [MANAGER.key]),  # read once, when the knowledge is built
        lambda: {MANAGER.key},
        lambda: [("SAL", "account manager")],  # a plain pair names the same title
        lambda: [MANAGER.key, MANAGER.key],
    ],
    ids=["generator", "set", "plain pair", "repeated"],
)
def test_the_golden_keys_may_come_in_any_form(make_golden_keys):
    knowledge = Knowledge(make_golden_keys())
    assert knowledge.learn(MANAGER, "SAL-01") is False
    assert knowledge.learn(MANAGER, "SAL-01") is False
    assert knowledge.entries() == ()


def test_the_golden_set_is_fixed_when_the_knowledge_is_built():
    golden = [MANAGER.key]
    knowledge = Knowledge(golden)
    golden.clear()
    assert knowledge.learn(MANAGER, "SAL-01") is False


def test_learn_is_the_only_function_that_adds_to_the_table_or_the_examples():
    # Q07 rests on one guarded function. Parse the module: outside __init__,
    # only learn may assign to, or call a changing method on, the two tables.
    tables = {"_roles", "_examples"}
    reading = {"get", "items", "keys", "values"}
    writers = set()
    tree = ast.parse(Path(lookup.__file__).read_text(encoding="utf-8"))
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            stored = isinstance(getattr(node, "ctx", None), ast.Store | ast.Del)
            target = node.value if isinstance(node, ast.Subscript) else node
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                stored = node.func.attr not in reading
                target = node.func.value
            if stored and isinstance(target, ast.Attribute) and target.attr in tables:
                writers.add(function.name)
    assert writers == {"__init__", "learn"}
    public = {name for name in vars(Knowledge) if not name.startswith("_")}
    assert public == {"lookup", "learn", "entries", "examples", "example_count"}


# --- A person's name, and no usable family code (Q14, Q26) ----------------------


def test_a_title_with_a_persons_name_joins_the_look_up_and_never_the_examples():
    knowledge = Knowledge()
    assert knowledge.learn(NAMED, "SAL-06") is True
    # Under the masked key, so the next title of that shape is a look-up hit.
    assert knowledge.lookup(TitleKey("SAL", f"customer success manager {NAME_MASK}")) == "SAL-06"
    assert knowledge.entries() == ((NAMED.key, "SAL-06"),)
    # Joining the examples would mean embedding it.
    assert knowledge.examples() == ()
    assert knowledge.example_count("SAL-06") == 0


def test_a_name_bearing_title_does_not_count_toward_the_three_examples():
    knowledge = Knowledge()
    knowledge.learn(NAMED, "SAL-06")
    knowledge.learn(_row("SAL", "customer success manager"), "SAL-06")
    assert len(knowledge.entries()) == 2
    assert knowledge.example_count("SAL-06") == 1


@pytest.mark.parametrize("row", [NO_FAMILY, UNKNOWN_FAMILY], ids=["blank", "unknown"])
def test_a_decision_on_a_title_with_no_usable_family_joins_under_its_key(row):
    knowledge = Knowledge()
    assert knowledge.learn(row, "FIN-02") is True
    # The key holds the family value as received: blank stays blank (Q26).
    assert knowledge.lookup(TitleKey(row.record.family_code, "staff accountant")) == "FIN-02"
    assert knowledge.lookup(TitleKey("FIN", "staff accountant")) is None
    # And it is an example of the chosen role, whose family the key does not name.
    assert knowledge.examples() == (Example(row.record, "FIN-02"),)
    assert knowledge.example_count("FIN-02") == 1


def test_the_same_title_under_two_unusable_codes_is_two_entries():
    knowledge = Knowledge()
    knowledge.learn(NO_FAMILY, "FIN-02")
    knowledge.learn(UNKNOWN_FAMILY, "FIN-02")
    assert [key for key, _ in knowledge.entries()] == [NO_FAMILY.key, UNKNOWN_FAMILY.key]


def test_a_name_in_a_title_with_no_usable_family_still_keeps_it_out_of_the_examples():
    both = _row("", f"assistant to {NAME_MASK}", has_person_name=True, family_usable=False)
    knowledge = Knowledge()
    assert knowledge.learn(both, "FIN-02") is True
    assert knowledge.lookup(both.key) == "FIN-02"
    assert knowledge.examples() == ()
