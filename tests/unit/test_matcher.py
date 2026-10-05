"""The ML match (design 1.6 step 3, 2.4 "Role score"; Q04, Q12, Q18).

One test of section 2.7 starts here: no model is ever shown a generic family
role, so none is ever embedded or on a shortlist (T5).

The embedder here gives each word a bucket of its own, so a cosine is plain
word overlap and every expected score can be worked out by hand: two texts of
m and n different words that share k have cosine k / sqrt(m * n).
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence

import pytest

from titlestd.contracts import (
    NAME_MASK,
    SHORTLIST_SIZE,
    Catalog,
    CatalogRole,
    MatchResult,
    Record,
    ScoredRole,
    TitleKey,
    TitleRow,
    Vector,
)
from titlestd.lookup import Knowledge
from titlestd.matcher import Matcher, cosine, embed_text, ml_alone_eligible, role_text
from titlestd.settings import Settings


class WordEmbedder:
    """One bucket per word, L2-normalised. It records every text it is given."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._buckets: dict[str, int] = {}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        counts = Counter(re.findall(r"[a-z0-9]+", text.lower()))
        norm = math.sqrt(sum(count * count for count in counts.values()))
        return {
            self._buckets.setdefault(word, len(self._buckets)): count / norm
            for word, count in counts.items()
        }

    @property
    def texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _role(
    role_id: str, family: str, name: str, definition: str = "", *, generic: bool = False
) -> CatalogRole:
    return CatalogRole(role_id, family, name, name[:4], definition, False, generic)


def _catalog(*roles: CatalogRole) -> Catalog:
    return Catalog(roles=roles, ladder=(), grade_levels=())


def _row(
    family: str,
    title: str,
    description: str = "",
    *,
    has_person_name: bool = False,
    family_usable: bool = True,
) -> TitleRow:
    return TitleRow(
        record=Record(title, family, "Some Department", description),
        employee_ids=("E1",),
        raw_titles=(title,),
        has_person_name=has_person_name,
        family_usable=family_usable,
    )


def _ids(result: MatchResult) -> list[str]:
    return [scored.role_id for scored in result.shortlist]


def _only(results: dict[TitleKey, MatchResult]) -> MatchResult:
    (result,) = results.values()
    return result


# Three trades and the generic role of their family, out of role_id order.
BAKER = _role("T-02", "FOOD", "Baker", "bakes bread")
BUTCHER = _role("T-03", "FOOD", "Butcher", "cuts meat")
BREWER = _role("T-01", "FOOD", "Brewer", "brews beer")
FOOD = _role("T-00", "FOOD", "Food", "Generic role for the Food family.", generic=True)
TRADES = _catalog(BUTCHER, FOOD, BAKER, BREWER)


# --- The texts (1.4, 2.4, Q12) ----------------------------------------------------


def test_a_role_is_embedded_as_its_name_and_one_line_definition():
    assert role_text(BAKER) == "Baker. bakes bread"
    assert role_text(_role("X-1", "X", "Night Baker")) == "Night Baker"
    assert role_text(_role("X-1", "X", "Night Baker", "   ")) == "Night Baker"


def test_a_record_is_embedded_as_its_title_plus_its_description():
    record = Record("senior baker", "FOOD", "Kitchen", "Bakes bread. Opens the shop.")
    assert embed_text(record) == "senior baker Bakes bread. Opens the shop."


def test_no_description_means_the_title_alone():
    # 2.4, missing data.
    assert embed_text(Record("senior baker", "FOOD", "Kitchen", "")) == "senior baker"


def test_the_department_and_family_are_not_embedded():
    here = Record("baker", "FOOD", "Kitchen", "Bakes bread.")
    there = Record("baker", "SHOP", "Front of House", "Bakes bread.")
    assert embed_text(here) == embed_text(there) == "baker Bakes bread."


def test_an_empty_title_leaves_the_description_and_an_empty_record_nothing():
    assert embed_text(Record("", "FOOD", "Kitchen", "Bakes bread.")) == "Bakes bread."
    assert embed_text(Record("", "FOOD", "Kitchen", "")) == ""


# --- Cosine: a sparse dot product (Q12) ---------------------------------------------


def test_cosine_is_the_dot_product_over_the_shared_buckets():
    a = {1: 0.6, 5: 0.8}
    b = {5: 0.6, 9: 0.8}
    assert cosine(a, b) == pytest.approx(0.48)
    assert cosine(a, a) == pytest.approx(1.0)
    assert cosine(a, {2: 1.0}) == 0.0
    assert cosine(a, {}) == 0.0 and cosine({}, {}) == 0.0


def test_cosine_sums_exactly_so_the_order_of_a_vectors_items_cannot_change_it():
    # Summed exactly, ten products of 0.1 round to 1.0. A plain running total
    # (total += product) gives 0.9999999999999999.
    tenths, ones = dict.fromkeys(range(10), 0.1), dict.fromkeys(range(10), 1.0)
    assert cosine(tenths, ones) == cosine(ones, tenths) == 1.0
    # Added in this order a plain running total loses the 1.0 altogether.
    lossy, flat = {0: 1e16, 1: 1.0, 2: -1e16}, {0: 1.0, 1: 1.0, 2: 1.0}
    backwards = dict(reversed(list(lossy.items())))
    assert cosine(lossy, flat) == cosine(flat, lossy) == cosine(backwards, flat) == 1.0
    # The built-in sum is compensated since Python 3.12 and gets both of those
    # right, so they do not tell it from math.fsum. These five do, in every
    # order. Their exact sum is 99990000000011000 less 1e-16, a hair under the
    # midpoint of the neighbouring floats ...10992 and ...11008, so it rounds
    # down. sum() loses the 1e-16 and rounds the tie up.
    for values in itertools.permutations((1000.0, -1e13, 1e4, -1e-16, 1e17)):
        halfway = dict(enumerate(values))
        assert cosine(halfway, dict.fromkeys(halfway, 1.0)) == 99990000000010992.0


def test_a_dense_vector_is_the_mapping_of_every_index():
    dense = {0: 0.0, 1: 0.6, 2: 0.0, 3: 0.8}
    assert cosine(dense, {3: 1.0}) == pytest.approx(0.8)
    assert cosine(dense, dense) == pytest.approx(1.0)


# --- The shortlist, the first choice and the margin (1.6 step 3) --------------------


def test_the_shortlist_is_best_first_with_score_and_matched_text():
    embedder = WordEmbedder()
    row = _row("FOOD", "baker", "bakes bread and cuts meat")  # six different words
    result = _only(Matcher(TRADES, embedder).match([row], Knowledge()))
    assert _ids(result) == ["T-02", "T-03", "T-01"]
    assert [scored.score for scored in result.shortlist] == pytest.approx(
        [3 / math.sqrt(18), 2 / math.sqrt(18), 0.0]
    )
    assert [scored.matched_text for scored in result.shortlist] == [
        "Baker. bakes bread",
        "Butcher. cuts meat",
        "Brewer. brews beer",
    ]
    assert result.first_choice == "T-02"
    assert result.margin == pytest.approx(1 / math.sqrt(18))
    assert result.top_three == ("T-02", "T-03", "T-01")
    assert result.first_choice_examples == 0


def test_the_result_is_keyed_by_title_key_one_per_row():
    rows = [_row("FOOD", "brewer"), _row("FOOD", "butcher"), _row("SHOP", "brewer")]
    results = Matcher(TRADES, WordEmbedder()).match(rows, Knowledge())
    assert list(results) == [row.key for row in rows]
    assert [result.first_choice for result in results.values()] == ["T-01", "T-03", "T-01"]


def test_equal_scores_are_ordered_by_role_id_whatever_the_order_of_the_catalog():
    roles = [_role(f"C-{n:02d}", "FAM", "Clerk", "files papers") for n in (3, 1, 4, 2)]
    for order in (roles, roles[::-1]):
        matcher = Matcher(_catalog(*order), WordEmbedder())
        result = _only(matcher.match([_row("FAM", "clerk")], Knowledge()))
        assert _ids(result) == ["C-01", "C-02", "C-03", "C-04"]
        assert result.margin == 0.0


def test_the_shortlist_holds_ten_roles_and_the_cut_follows_the_same_order():
    names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    names += ["ten", "eleven", "twelve"]
    roles = [_role(f"N-{n:02d}", "FAM", name) for n, name in enumerate(names)]
    catalog = _catalog(*reversed(roles))
    # "eleven" twice and "four" once in a text of two different words.
    row = _row("FAM", "eleven four", "eleven")
    result = _only(Matcher(catalog, WordEmbedder()).match([row], Knowledge()))
    assert len(result.shortlist) == SHORTLIST_SIZE == 10
    # The two that match, then the lowest role ids among the eleven that tie at zero.
    assert _ids(result) == ["N-11", "N-04", *(f"N-{n:02d}" for n in (0, 1, 2, 3, 5, 6, 7, 8))]
    assert result.shortlist[0].score == pytest.approx(2 / math.sqrt(5))
    assert result.shortlist[1].score == pytest.approx(1 / math.sqrt(5))


def test_a_catalog_with_fewer_roles_gives_a_shorter_shortlist():
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "brewer")], Knowledge()))
    assert _ids(result) == ["T-01", "T-02", "T-03"]
    one = _only(Matcher(_catalog(BAKER), WordEmbedder()).match([_row("FOOD", "x")], Knowledge()))
    assert _ids(one) == ["T-02"] and one.margin == 0.0
    none = _only(Matcher(_catalog(FOOD), WordEmbedder()).match([_row("FOOD", "x")], Knowledge()))
    assert none.shortlist == () and none.first_choice is None
    assert none.first_choice_examples == 0


def test_a_title_that_resembles_nothing_still_gets_a_shortlist_with_no_margin():
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "zzz")], Knowledge()))
    assert _ids(result) == ["T-01", "T-02", "T-03"]
    assert [scored.score for scored in result.shortlist] == [0.0, 0.0, 0.0]
    assert result.margin == 0.0


# --- Role score: the best cosine across catalog text and approved titles (2.4) ------


def test_an_approved_title_lifts_its_roles_score():
    embedder = WordEmbedder()
    matcher = Matcher(TRADES, embedder)
    knowledge = Knowledge()
    query = _row("FOOD", "boulanger chef")
    before = _only(matcher.match([query], knowledge))
    assert _ids(before) == ["T-01", "T-02", "T-03"] and before.margin == 0.0

    # A person decides that a "boulanger" is a baker; the title becomes an example.
    knowledge.learn(_row("FOOD", "boulanger", "night shift"), "T-02")
    after = _only(matcher.match([query], knowledge))
    assert after.first_choice == "T-02"
    assert after.shortlist[0].score == pytest.approx(1 / math.sqrt(6))
    assert after.margin == pytest.approx(1 / math.sqrt(6))
    assert after.first_choice_examples == 1
    # The audit trail names the approved title that was matched (1.5).
    assert after.shortlist[0].matched_text == "boulanger"
    # The earlier answer is what it was: a wave is scored, then learned (Q18).
    assert _ids(before) == ["T-01", "T-02", "T-03"] and before.first_choice_examples == 0


def test_the_score_is_the_best_of_them_not_their_sum_or_their_mean():
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "baker apprentice", "kneads dough daily"), "T-02")  # 1 of 5
    knowledge.learn(_row("FOOD", "night shift worker"), "T-02")  # shares nothing
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "baker")], knowledge))
    best = result.shortlist[0]
    # The catalog text "Baker. bakes bread" (1 of 3) beats the example (1 of 5).
    assert best.role_id == "T-02"
    assert best.score == pytest.approx(1 / math.sqrt(3))
    assert best.matched_text == "Baker. bakes bread"
    assert result.first_choice_examples == 2


def test_the_margin_is_the_gap_between_two_roles_not_two_texts_of_one_role():
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "pastry cook"), "T-02")
    knowledge.learn(_row("FOOD", "pastry cook assistant"), "T-02")
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "pastry cook")], knowledge))
    assert _ids(result) == ["T-02", "T-01", "T-03"]
    assert result.shortlist[0].score == pytest.approx(1.0)
    assert result.margin == pytest.approx(1.0)


def test_an_example_counts_for_its_own_role_only():
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "boulanger"), "T-02")
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "boulanger")], knowledge))
    scores = {scored.role_id: scored.score for scored in result.shortlist}
    assert scores == {"T-02": pytest.approx(1.0), "T-01": 0.0, "T-03": 0.0}


def test_every_approved_title_is_compared_not_only_the_first_in_key_order():
    # 1.6 step 3: "every human-approved title". Of the baker's three approved
    # titles the one that matches is the middle one in key order; the
    # butcher's is the third of the four overall.
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "aaa helper"), "T-02")
    knowledge.learn(_row("FOOD", "boulanger"), "T-02")
    knowledge.learn(_row("FOOD", "charcutier"), "T-03")
    knowledge.learn(_row("FOOD", "zzz helper"), "T-02")
    rows = [_row("SHOP", "boulanger"), _row("SHOP", "charcutier")]
    results = Matcher(TRADES, WordEmbedder()).match(rows, knowledge)
    for result, role_id, title in zip(
        results.values(), ["T-02", "T-03"], ["boulanger", "charcutier"], strict=True
    ):
        assert result.first_choice == role_id
        assert result.shortlist[0].score == pytest.approx(1.0)
        assert result.shortlist[0].matched_text == title
        assert result.margin == pytest.approx(1.0)


def test_an_example_from_another_family_is_an_example_all_the_same():
    # The match is not filtered by family: the family rule is checked later (1.7).
    knowledge = Knowledge()
    knowledge.learn(_row("SHOP", "boulanger"), "T-02")
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "boulanger")], knowledge))
    assert result.first_choice == "T-02" and result.first_choice_examples == 1


def test_on_equal_scores_the_catalog_text_is_named_before_an_approved_title():
    catalog = _catalog(_role("B-1", "FOOD", "Baker"), BREWER)
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "baker"), "B-1")
    result = _only(Matcher(catalog, WordEmbedder()).match([_row("SHOP", "baker")], knowledge))
    assert result.shortlist[0].score == pytest.approx(1.0)
    assert result.shortlist[0].matched_text == "Baker"


def test_on_equal_scores_approved_titles_are_named_in_key_order():
    rows = [
        _row("FOOD", "boulanger two"),
        _row("FOOD", "boulanger one"),
        _row("AAA", "boulanger z"),
    ]
    for order in (rows, rows[::-1]):
        knowledge = Knowledge()
        for row in order:
            knowledge.learn(row, "T-02")
        result = _only(
            Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "boulanger")], knowledge)
        )
        assert result.shortlist[0].matched_text == "boulanger z"  # family AAA sorts first
        assert result.first_choice_examples == 3


def test_first_choice_examples_counts_the_first_choice_as_the_examples_stood():
    matcher = Matcher(TRADES, WordEmbedder())
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "master brewer"), "T-01")
    for title in ("baker one", "baker two", "baker three"):
        knowledge.learn(_row("FOOD", title), "T-02")
    results = matcher.match([_row("FOOD", "baker"), _row("FOOD", "brewer")], knowledge)
    assert [result.first_choice_examples for result in results.values()] == [3, 1]
    knowledge.learn(_row("FOOD", "baker four"), "T-02")
    assert [result.first_choice_examples for result in results.values()] == [3, 1]
    assert _only(matcher.match([_row("FOOD", "baker")], knowledge)).first_choice_examples == 4


# --- No generic family role is embedded or shortlisted (2.7, 1.2) -------------------


def test_a_generic_family_role_is_never_embedded_and_never_on_a_shortlist():
    embedder = WordEmbedder()
    # A record that reads exactly like the generic role: it would score 1.0.
    lookalike = _row("FOOD", "food", "Generic role for the Food family.")
    assert embed_text(lookalike.record).lower() == role_text(FOOD).lower().replace(".", "", 1)
    rows = [lookalike, _row("FOOD", "baker"), _row("FOOD", "food")]
    results = Matcher(TRADES, embedder).match(rows, Knowledge())
    for result in results.values():
        # Three roles, though a shortlist of ten had room for the fourth.
        assert sorted(_ids(result)) == ["T-01", "T-02", "T-03"]
    # embed was given the three visible roles and the three rows, and nothing else.
    assert sorted(embedder.texts) == sorted(
        [role_text(BAKER), role_text(BREWER), role_text(BUTCHER)]
        + [embed_text(row.record) for row in rows]
    )
    assert role_text(FOOD) not in embedder.texts


def test_a_catalog_of_generic_roles_only_embeds_no_role_at_all():
    embedder = WordEmbedder()
    row = _row("FOOD", "food")
    assert _only(Matcher(_catalog(FOOD), embedder).match([row], Knowledge())).shortlist == ()
    assert embedder.texts == ["food"]


@pytest.mark.parametrize("role_id", ["T-00", "T-99"], ids=["generic", "not in the catalog"])
def test_an_example_of_a_role_no_model_may_be_shown_reaches_no_shortlist(role_id):
    # No person's decision names such a role. If one did, the match would
    # still list only the roles a model may be shown.
    embedder = WordEmbedder()
    knowledge = Knowledge()
    stray = _row("FOOD", "food hall host", "greets guests")
    knowledge.learn(stray, role_id)
    result = _only(Matcher(TRADES, embedder).match([_row("FOOD", "food hall host")], knowledge))
    assert sorted(_ids(result)) == ["T-01", "T-02", "T-03"]
    assert [scored.score for scored in result.shortlist] == [0.0, 0.0, 0.0]
    assert embed_text(stray.record) not in embedder.texts


# --- Rows and titles no model may see (Q14, Q26, Q07) --------------------------------


NAMED = _row("FOOD", f"baker {NAME_MASK}", "bakes bread", has_person_name=True)
NO_FAMILY = _row("", "butcher", "cuts meat", family_usable=False)


def test_a_row_no_model_may_see_is_left_out_and_never_passed_to_embed():
    embedder = WordEmbedder()
    eligible = _row("FOOD", "brewer")
    results = Matcher(TRADES, embedder).match([NAMED, eligible, NO_FAMILY], Knowledge())
    assert list(results) == [eligible.key]
    assert embed_text(NAMED.record) not in embedder.texts
    assert embed_text(NO_FAMILY.record) not in embedder.texts
    assert not any(NAME_MASK in text for text in embedder.texts)


def test_with_no_eligible_row_the_embedder_is_not_called():
    embedder = WordEmbedder()
    matcher = Matcher(TRADES, embedder)
    assert matcher.match([NAMED, NO_FAMILY], Knowledge()) == {}
    assert matcher.match([], Knowledge()) == {}
    assert embedder.calls == []


def test_a_decided_title_with_a_persons_name_is_never_an_example_to_match():
    embedder = WordEmbedder()
    knowledge = Knowledge()
    knowledge.learn(NAMED, "T-03")  # a person says this one is a butcher
    assert knowledge.lookup(NAMED.key) == "T-03"
    result = _only(Matcher(TRADES, embedder).match([_row("FOOD", "baker")], knowledge))
    assert result.first_choice == "T-02"
    assert {scored.role_id: scored.score for scored in result.shortlist}["T-03"] == 0.0
    assert not any(NAME_MASK in text for text in embedder.texts)


def test_a_decided_title_with_no_usable_family_is_an_example_of_the_chosen_role():
    # Q26: it joins the chosen role's examples, so from then on it is matched against.
    knowledge = Knowledge()
    knowledge.learn(_row("", "charcutier", family_usable=False), "T-03")
    result = _only(Matcher(TRADES, WordEmbedder()).match([_row("FOOD", "charcutier")], knowledge))
    assert result.first_choice == "T-03"
    assert result.shortlist[0].matched_text == "charcutier"
    assert result.first_choice_examples == 1


def test_a_golden_title_is_scored_but_never_matched_against():
    # Q07, Q08: the golden set is re-scored at every wave and never becomes an example.
    embedder = WordEmbedder()
    golden = _row("FOOD", "boulanger")
    knowledge = Knowledge(golden_keys=[golden.key])
    matcher = Matcher(TRADES, embedder)
    before = matcher.match([golden, _row("FOOD", "boulanger chef")], knowledge)
    assert knowledge.learn(golden, "T-02") is False  # its label, from wherever
    after = matcher.match([golden, _row("FOOD", "boulanger chef")], knowledge)
    assert after == before
    assert all(result.margin == 0.0 for result in after.values())
    assert golden.key in after


# --- One embed call for the texts that are new ---------------------------------------


def test_each_text_is_embedded_once_however_often_it_is_matched():
    embedder = WordEmbedder()
    matcher = Matcher(TRADES, embedder)
    knowledge = Knowledge()
    rows = [_row("FOOD", "baker"), _row("FOOD", "brewer")]
    first = matcher.match(rows, knowledge)
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) == 5  # three roles, two rows
    assert matcher.match(rows, knowledge) == first
    assert len(embedder.calls) == 1
    example = _row("FOOD", "boulanger", "night shift")
    knowledge.learn(example, "T-02")
    matcher.match(rows, knowledge)
    assert embedder.calls[1:] == [["boulanger night shift"]]


def test_rows_with_the_same_text_share_one_embedding():
    embedder = WordEmbedder()
    rows = [_row("FOOD", "baker"), _row("SHOP", "baker"), _row("FOOD", "Baker. bakes bread")]
    results = Matcher(TRADES, embedder).match(rows, Knowledge())
    assert len(results) == 3
    assert len(embedder.texts) == len(set(embedder.texts)) == 4  # three roles and "baker"


@pytest.mark.parametrize("vectors", [1, 9], ids=["too few", "too many"])
def test_an_embedder_that_does_not_answer_text_for_text_is_refused(vectors):
    class Miscounting:
        def embed(self, texts):
            assert len(texts) == 4  # three roles and the row
            return [{0: 1.0}] * vectors

    with pytest.raises(ValueError):
        Matcher(TRADES, Miscounting()).match([_row("FOOD", "baker")], Knowledge())


# --- ML-alone eligibility (1.6 step 4, Q04) -------------------------------------------


def _match(*scores: float, first: str = "SAL-01", examples: int = 3) -> MatchResult:
    role_ids = [first, "SAL-02", "FIN-01"]
    pairs = zip(role_ids, scores, strict=False)
    shortlist = tuple(ScoredRole(role_id, score) for role_id, score in pairs)
    return MatchResult(shortlist, first_choice_examples=examples)


ON = Settings(ml_alone=True)  # cut-off 0.10, three examples


def test_the_route_is_off_by_default(mini_catalog):
    clear = _match(0.9, 0.2)
    assert Settings().ml_alone is False
    assert ml_alone_eligible(clear, "SAL", mini_catalog, Settings()) is False
    assert ml_alone_eligible(clear, "SAL", mini_catalog, ON) is True


@pytest.mark.parametrize(
    ("scores", "cutoff", "expected"),
    [
        ((0.9, 0.2), 0.10, True),
        ((0.5, 0.45), 0.10, False),
        ((0.5, 0.25), 0.25, True),  # at the cut-off: "at least"
        ((0.5, 0.25), 0.26, False),
        ((0.5, 0.5), 0.0, True),
        ((0.5,), 0.10, False),  # one role on the shortlist: no margin
        ((0.5,), 0.0, True),
    ],
)
def test_the_margin_must_clear_the_cut_off(mini_catalog, scores, cutoff, expected):
    settings = Settings(ml_alone=True, ml_margin_cutoff=cutoff)
    assert ml_alone_eligible(_match(*scores), "SAL", mini_catalog, settings) is expected


@pytest.mark.parametrize(
    ("examples", "needed", "expected"),
    [(3, 3, True), (2, 3, False), (0, 3, False), (7, 3, True), (2, 2, True), (0, 0, True)],
)
def test_the_first_choice_needs_enough_approved_examples(mini_catalog, examples, needed, expected):
    settings = Settings(ml_alone=True, ml_min_examples=needed)
    match = _match(0.9, 0.2, examples=examples)
    assert ml_alone_eligible(match, "SAL", mini_catalog, settings) is expected


@pytest.mark.parametrize(
    ("family_code", "expected"),
    [("SAL", True), (" SAL ", True), ("FIN", False), ("sal", False), ("", False), ("XYZ", False)],
)
def test_the_first_choice_must_be_in_the_recorded_family(mini_catalog, family_code, expected):
    assert ml_alone_eligible(_match(0.9, 0.2), family_code, mini_catalog, ON) is expected


@pytest.mark.parametrize("first", ["SAL-GEN", "SAL-99"], ids=["generic", "not in the catalog"])
def test_a_first_choice_no_model_may_be_shown_is_never_eligible(mini_catalog, first):
    assert ml_alone_eligible(_match(0.9, 0.2, first=first), "SAL", mini_catalog, ON) is False


def test_an_empty_shortlist_is_never_eligible(mini_catalog):
    settings = Settings(ml_alone=True, ml_margin_cutoff=0.0, ml_min_examples=0)
    assert ml_alone_eligible(MatchResult(()), "SAL", mini_catalog, settings) is False


def test_eligibility_follows_from_what_the_matcher_returned():
    matcher = Matcher(TRADES, WordEmbedder())
    knowledge = Knowledge()
    row = _row("FOOD", "baker")
    for title in ("baker one", "baker two"):
        knowledge.learn(_row("FOOD", title), "T-02")
    settings = Settings(ml_alone=True)
    two = _only(matcher.match([row], knowledge))
    assert two.margin > settings.ml_margin_cutoff and two.first_choice_examples == 2
    assert ml_alone_eligible(two, "FOOD", TRADES, settings) is False
    knowledge.learn(_row("FOOD", "baker three"), "T-02")
    three = _only(matcher.match([row], knowledge))
    assert ml_alone_eligible(three, "FOOD", TRADES, settings) is True
    assert ml_alone_eligible(three, "SHOP", TRADES, settings) is False
