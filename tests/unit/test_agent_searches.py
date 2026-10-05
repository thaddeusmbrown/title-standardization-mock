"""The agent's two searches and the numbering of what they show (Q10).

One test of section 2.7 continues here: no model is ever shown a generic
family role, so none is ever in a search result (T5).

The embedder here gives each word a bucket of its own, so a cosine is plain
word overlap and every expected order can be worked out by hand: two texts of
m and n different words that share k have cosine k / sqrt(m * n).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import pytest

from titlestd import agent
from titlestd.contracts import (
    AGENT_MAX_SEARCHES,
    NAME_MASK,
    AgentCase,
    AgentTranscript,
    Catalog,
    CatalogRole,
    ChooseAnswer,
    ChooseKind,
    MatchResult,
    NumberedRole,
    Reason,
    Record,
    ScoredRole,
    SearchHit,
    SearchRequest,
    SearchStep,
    SearchTool,
    TitleRow,
    Vector,
)
from titlestd.lookup import Knowledge
from titlestd.matcher import embed_text, role_text
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


def _role(role_id: str, name: str, definition: str, *, generic: bool = False) -> CatalogRole:
    return CatalogRole(role_id, role_id[:4], name, name[:4], definition, False, generic)


# Two families and their generic roles, out of role_id order. As embedded:
#   "Brewer. brews beer"            brewer brews beer
#   "Baker. bakes bread and rolls"  baker bakes bread and rolls
#   "Butcher. cuts meat"            butcher cuts meat
#   "Bread Taster. tastes bread"    bread (twice) taster tastes
#   "Cashier. sells bread and beer" cashier sells bread and beer
#   "Shop Manager. runs the shop"   shop (twice) manager runs the
BUTCHER = _role("FOOD-03", "Butcher", "cuts meat")
BAKER = _role("FOOD-02", "Baker", "bakes bread and rolls")
TASTER = _role("FOOD-04", "Bread Taster", "tastes bread")
BREWER = _role("FOOD-01", "Brewer", "brews beer")
FOOD = _role("FOOD-00", "Food", "Generic role for the Food family.", generic=True)
MANAGER = _role("SHOP-02", "Shop Manager", "runs the shop")
CASHIER = _role("SHOP-01", "Cashier", "sells bread and beer")
SHOP = _role("SHOP-00", "Shop", "Generic role for the Shop family.", generic=True)
CATALOG = Catalog(
    roles=(BUTCHER, FOOD, MANAGER, BAKER, SHOP, TASTER, CASHIER, BREWER), ladder=(), grade_levels=()
)
GENERIC = {"FOOD-00", "SHOP-00"}


def _row(family: str, title: str, description: str = "", **flags: bool) -> TitleRow:
    return TitleRow(
        Record(title, family, "Some Department", description), ("E1",), (title,), **flags
    )


def _transcript(
    family: str = "FOOD", shortlist: Sequence[CatalogRole] = (), *steps: SearchStep
) -> AgentTranscript:
    """A case as build_case makes it for a title of the family, and the searches run so far."""
    row = _row(family, "night baker", "bakes bread at night")
    shown = [NumberedRole(number, role) for number, role in enumerate(shortlist, start=1)]
    llm_answer = ChooseAnswer(ChooseKind.NO_MATCH, None, "night baker", "test")
    case = agent.build_case(row, shown, llm_answer, None, MatchResult(()), (Reason.NO_MATCH,))
    return AgentTranscript(case, tuple(steps))


def _tools(knowledge: Knowledge | None = None, **settings: int) -> agent.SearchTools:
    return agent.SearchTools(
        CATALOG, knowledge or Knowledge(), WordEmbedder(), Settings(**settings)
    )


def _catalog(query: str) -> SearchRequest:
    return SearchRequest(SearchTool.CATALOG, query)


def _approved(query: str) -> SearchRequest:
    return SearchRequest(SearchTool.APPROVED_TITLES, query)


def _ids(step: SearchStep) -> list[str]:
    return [hit.role.role.role_id for hit in step.hits]


def _numbers(step: SearchStep) -> list[int | None]:
    return [hit.role.number for hit in step.hits]


def _titles(step: SearchStep) -> list[tuple[str, str, str]]:
    return [(hit.approved_title, hit.title_family_code, hit.role.role.role_id) for hit in step.hits]


def _approved_titles() -> Knowledge:
    """Three titles a person has decided. As embedded:
    "night baker bakes bread at night"  night (twice) baker bakes bread at
    "bread seller"                      bread seller
    "head brewer brews beer"            head brewer brews beer
    """
    knowledge = Knowledge()
    knowledge.learn(_row("FOOD", "night baker", "bakes bread at night"), "FOOD-02")
    knowledge.learn(_row("SHOP", "bread seller"), "SHOP-01")
    knowledge.learn(_row("FOOD", "head brewer", "brews beer"), "FOOD-01")
    return knowledge


# --- The catalog search (Q10) -----------------------------------------------------


def test_the_catalog_search_ranks_the_roles_by_similarity_to_the_query():
    step = _tools().search(_catalog("bread"), _transcript())
    # Bread Taster 2/sqrt(6), then Baker and Cashier at 1/sqrt(5): the tie goes
    # by role_id. The rest share nothing with the query and follow by role_id.
    assert _ids(step) == ["FOOD-04", "FOOD-02", "SHOP-01", "FOOD-01", "FOOD-03"]
    assert step.request == _catalog("bread")
    assert all((hit.approved_title, hit.title_family_code) == ("", "") for hit in step.hits)


def test_the_query_is_free_text_embedded_as_written():
    embedder = WordEmbedder()
    tools = agent.SearchTools(CATALOG, Knowledge(), embedder, Settings())
    step = tools.search(_catalog("Someone who BREWS, or sells, beer!"), _transcript())
    assert _ids(step)[:2] == ["FOOD-01", "SHOP-01"]  # 2 of 3 words, then 2 of 5
    assert embedder.texts[0] == "Someone who BREWS, or sells, beer!"


def test_a_search_returns_the_top_agent_search_k():
    assert Settings().agent_search_k == 5
    for top in (1, 2, 5, 6, 40):
        step = _tools(agent_search_k=top).search(_catalog("bread"), _transcript())
        assert (
            _ids(step) == ["FOOD-04", "FOOD-02", "SHOP-01", "FOOD-01", "FOOD-03", "SHOP-02"][:top]
        )


def test_a_query_that_resembles_nothing_still_gets_the_top_few_by_role_id():
    # "top 5" has no score threshold.
    for query in ("zebra", "", "   "):
        step = _tools().search(_catalog(query), _transcript())
        assert _ids(step) == ["FOOD-01", "FOOD-02", "FOOD-03", "FOOD-04", "SHOP-01"]


def test_the_catalog_search_is_not_filtered_by_family():
    # "the whole catalog": a shop role is the best hit for a title recorded in food.
    step = _tools().search(_catalog("runs the shop"), _transcript("FOOD"))
    assert _ids(step)[0] == "SHOP-02"
    assert step.hits[0].role == NumberedRole(None, MANAGER)


# --- No model is ever shown a generic family role (2.7) -----------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Food",
        "Generic role for the Food family.",
        role_text(FOOD),
        role_text(SHOP),
        "shop",
        "generic",
        "",
    ],
)
@pytest.mark.parametrize("top", [5, 8, 100])
def test_a_generic_family_role_is_never_in_a_catalog_search_result(query, top):
    embedder = WordEmbedder()
    tools = agent.SearchTools(CATALOG, Knowledge(), embedder, Settings(agent_search_k=top))
    for family in ("FOOD", "SHOP"):
        step = tools.search(_catalog(query), _transcript(family))
        assert not GENERIC & set(_ids(step))
        assert not any(hit.role.role.is_generic for hit in step.hits)
        assert len(step.hits) == min(top, 6)  # the six roles a model may be shown
    # Never embedded either (1.2), unless the query itself is that text.
    assert [text for text in embedder.texts if text != query] == [
        role_text(role)
        for role in sorted(CATALOG.roles, key=lambda r: r.role_id)
        if not role.is_generic
    ]


def test_an_approved_title_of_a_generic_role_is_never_in_a_search_result():
    # No person's decision names a generic role (1.2) and learn does not check
    # it. Were the examples to hold one all the same, no search returns it.
    knowledge = _approved_titles()
    knowledge.learn(_row("FOOD", "bread person", "bread bread bread"), "FOOD-00")
    knowledge.learn(_row("SHOP", "bread helper"), "SHOP-77")  # a role the catalog does not hold
    embedder = WordEmbedder()
    tools = agent.SearchTools(CATALOG, knowledge, embedder, Settings(agent_search_k=100))
    step = tools.search(_approved("bread person helper"), _transcript())
    assert _titles(step) == [
        ("bread seller", "SHOP", "SHOP-01"),
        ("night baker", "FOOD", "FOOD-02"),
        ("head brewer", "FOOD", "FOOD-01"),
    ]
    assert "bread person bread bread bread" not in embedder.texts
    assert "bread helper" not in embedder.texts


def test_a_title_left_out_does_not_cost_a_place_among_the_top_few():
    # What no model may be shown is dropped before ranking, not after.
    knowledge = _approved_titles()
    knowledge.learn(_row("FOOD", "bread bread", "bread"), "FOOD-00")
    step = _tools(knowledge, agent_search_k=2).search(_approved("bread"), _transcript())
    assert [title for title, _, _ in _titles(step)] == ["bread seller", "night baker"]


def test_the_shared_numbering_refuses_a_generic_role_wherever_it_comes_from():
    for generic in (FOOD, SHOP):
        with pytest.raises(ValueError, match=rf"generic family role {generic.role_id}"):
            _transcript("FOOD", [BAKER, generic])


# --- The approved-title search (Q10, Q12) -------------------------------------------


def test_the_approved_title_search_ranks_the_titles_people_have_decided():
    step = _tools(_approved_titles()).search(_approved("bread"), _transcript())
    # "bread seller" 1/sqrt(2), "night baker ..." 1/sqrt(8), "head brewer ..." 0.
    assert _titles(step) == [
        ("bread seller", "SHOP", "SHOP-01"),
        ("night baker", "FOOD", "FOOD-02"),
        ("head brewer", "FOOD", "FOOD-01"),
    ]
    assert [hit.role.role for hit in step.hits] == [CASHIER, BAKER, BREWER]


def test_an_approved_title_is_compared_as_the_ml_match_compares_it():
    # Its example's text: the cleaned title plus the description (Q12). "night"
    # stands only in the description of the baker's title.
    knowledge = _approved_titles()
    embedder = WordEmbedder()
    tools = agent.SearchTools(CATALOG, knowledge, embedder, Settings())
    step = tools.search(_approved("works at night"), _transcript())
    assert _titles(step)[0] == ("night baker", "FOOD", "FOOD-02")
    assert embedder.texts[1:] == [embed_text(example.record) for example in knowledge.examples()]
    assert "night baker bakes bread at night" in embedder.texts


def test_with_nothing_approved_the_search_finds_nothing():
    step = _tools().search(_approved("bread"), _transcript())
    assert step == SearchStep(_approved("bread"), ())


def test_approved_title_ties_go_by_title_then_family_code():
    knowledge = Knowledge()
    for family, title, role_id in (
        ("SHOP", "bread seller", "SHOP-01"),
        ("FOOD", "bread seller", "FOOD-02"),
        ("SHOP", "bread baker", "SHOP-02"),
        ("FOOD", "bread taster", "FOOD-04"),
    ):
        knowledge.learn(_row(family, title), role_id)
    # Four titles of two words that share one with the query: all at 1/sqrt(2).
    step = _tools(knowledge).search(_approved("bread"), _transcript())
    assert _titles(step) == [
        ("bread baker", "SHOP", "SHOP-02"),
        ("bread seller", "FOOD", "FOOD-02"),
        ("bread seller", "SHOP", "SHOP-01"),
        ("bread taster", "FOOD", "FOOD-04"),
    ]


def test_two_titles_approved_for_one_role_are_two_results_with_one_number():
    knowledge = _approved_titles()
    knowledge.learn(_row("FOOD", "bread baker"), "FOOD-02")
    step = _tools(knowledge).search(_approved("baker"), _transcript("FOOD", [BREWER]))
    # "bread baker" 1/sqrt(2), "night baker ..." 1/sqrt(8), the other two 0.
    assert _titles(step)[:2] == [
        ("bread baker", "FOOD", "FOOD-02"),
        ("night baker", "FOOD", "FOOD-02"),
    ]
    assert _numbers(step)[:2] == [2, 2]


def test_a_search_reads_the_approved_titles_as_they_stand_when_it_runs():
    # One SearchTools serves a wave and the next: what is learned in between is found.
    knowledge = Knowledge()
    tools = _tools(knowledge)
    assert tools.search(_approved("bread"), _transcript()).hits == ()
    knowledge.learn(_row("SHOP", "bread seller"), "SHOP-01")
    assert _titles(tools.search(_approved("bread"), _transcript())) == [
        ("bread seller", "SHOP", "SHOP-01")
    ]
    # A later decision on the same title replaces the earlier one (titlestd.lookup).
    knowledge.learn(_row("SHOP", "bread seller"), "SHOP-02")
    assert _titles(tools.search(_approved("bread"), _transcript())) == [
        ("bread seller", "SHOP", "SHOP-02")
    ]


def test_a_title_with_a_person_s_name_is_never_searched_or_embedded():
    # Q14: a person's decision on it joins the look-up, never the examples.
    knowledge = _approved_titles()
    named = _row("FOOD", f"bread baker {NAME_MASK}", "bakes bread", has_person_name=True)
    assert knowledge.learn(named, "FOOD-02")
    assert knowledge.lookup(named.key) == "FOOD-02"
    embedder = WordEmbedder()
    tools = agent.SearchTools(CATALOG, knowledge, embedder, Settings(agent_search_k=100))
    step = tools.search(_approved("bread baker"), _transcript())
    assert len(step.hits) == 3
    assert not any(NAME_MASK in title for title, _, _ in _titles(step))
    assert not any(NAME_MASK in text for text in embedder.texts)


def test_a_title_with_no_usable_family_code_is_found_under_the_code_it_has():
    # Q26: decided by a person, it is an example of the chosen role.
    knowledge = Knowledge()
    knowledge.learn(_row("", "bread seller", family_usable=False), "SHOP-01")
    knowledge.learn(_row("ZZZ", "bread baker", family_usable=False), "FOOD-02")
    step = _tools(knowledge).search(_approved("bread"), _transcript("FOOD"))
    assert _titles(step) == [("bread baker", "ZZZ", "FOOD-02"), ("bread seller", "", "SHOP-01")]
    assert _numbers(step) == [1, None]  # the number follows the role's family, not the title's


# --- The numbering (Q10) --------------------------------------------------------------


def test_the_case_numbers_the_in_family_roles_first_in_the_order_the_llm_saw_them():
    case = _transcript("FOOD", [CASHIER, BUTCHER, MANAGER, BAKER]).case
    assert case.roles == (
        NumberedRole(None, CASHIER),
        NumberedRole(1, BUTCHER),
        NumberedRole(None, MANAGER),
        NumberedRole(2, BAKER),
    )


def test_a_new_in_family_role_gets_the_next_number_on_first_appearance():
    transcript = _transcript("FOOD", [CASHIER, BUTCHER, BAKER])
    step = _tools().search(_catalog("bread"), transcript)
    assert _ids(step) == ["FOOD-04", "FOOD-02", "SHOP-01", "FOOD-01", "FOOD-03"]
    # New: the taster 3 and the brewer 4. Shown before: the baker 2, the butcher 1.
    assert _numbers(step) == [3, 2, None, 4, 1]


def test_a_number_never_changes_and_is_never_reused():
    tools = _tools(_approved_titles())
    transcript = _transcript("FOOD", [CASHIER, BUTCHER])
    numbers: dict[str, int] = {"FOOD-03": 1}
    for request in (
        _catalog("beer"),  # the brewer, then the baker and the taster
        _approved("bread"),
        _catalog("bread"),
        _catalog("cuts meat"),
    ):
        step = tools.search(request, transcript)
        transcript = AgentTranscript(transcript.case, (*transcript.steps, step))
        for hit in step.hits:
            if hit.role.number is not None:
                assert numbers.setdefault(hit.role.role.role_id, hit.role.number) == hit.role.number
    assert numbers == {"FOOD-03": 1, "FOOD-01": 2, "FOOD-02": 3, "FOOD-04": 4}
    assert [shown.number for shown in transcript.numbered_roles] == [1, 2, 3, 4]
    assert [shown.role for shown in transcript.numbered_roles] == [BUTCHER, BREWER, BAKER, TASTER]


def test_an_out_of_family_role_is_shown_without_a_number_however_often_it_is_found():
    tools = _tools(_approved_titles(), agent_search_k=100)
    transcript = _transcript("FOOD", [CASHIER, BAKER])
    for request in (_catalog("bread"), _approved("bread"), _catalog("shop"), _approved("beer")):
        step = tools.search(request, transcript)
        transcript = AgentTranscript(transcript.case, (*transcript.steps, step))
        for hit in step.hits:
            assert (hit.role.number is None) == (hit.role.role.family_code == "SHOP")
    shown = {hit.role.role.role_id for step in transcript.steps for hit in step.hits}
    assert {"SHOP-01", "SHOP-02"} <= shown
    assert all(role.role.family_code == "FOOD" for role in transcript.numbered_roles)


def test_the_same_role_is_in_the_family_of_one_case_and_outside_that_of_another():
    tools = _tools()
    in_food = tools.search(_catalog("bread"), _transcript("FOOD"))
    in_shop = tools.search(_catalog("bread"), _transcript("SHOP"))
    assert _ids(in_food) == _ids(in_shop) == ["FOOD-04", "FOOD-02", "SHOP-01", "FOOD-01", "FOOD-03"]
    assert _numbers(in_food) == [1, 2, None, 3, 4]
    assert _numbers(in_shop) == [None, None, 1, None, None]


def test_the_next_number_follows_the_highest_one_in_the_transcript():
    # Also when the transcript was not numbered here: no number is given twice.
    case = _transcript("FOOD").case
    earlier = SearchStep(_catalog("x"), (SearchHit(NumberedRole(7, BUTCHER)),))
    step = _tools().search(_catalog("bread"), AgentTranscript(case, (earlier,)))
    assert dict(zip(_ids(step), _numbers(step), strict=True)) == {
        "FOOD-04": 8,
        "FOOD-02": 9,
        "SHOP-01": None,
        "FOOD-01": 10,
        "FOOD-03": 7,
    }


def test_the_recorded_family_is_read_trimmed_and_a_blank_one_is_no_family():
    # As titlestd.catalog reads a code (S05, Q26).
    def case_for(family: str) -> AgentCase:
        row = _row(family, "night baker")
        shown = [NumberedRole(1, CASHIER), NumberedRole(2, BAKER)]
        answer = ChooseAnswer(ChooseKind.NO_MATCH, None, "night baker", "test")
        return agent.build_case(row, shown, answer, None, MatchResult(()), ())

    assert [shown.number for shown in case_for(" FOOD ").roles] == [None, 1]
    assert [shown.number for shown in case_for("food").roles] == [None, None]
    for blank in ("", "   "):
        case = case_for(blank)
        assert [shown.number for shown in case.roles] == [None, None]
        step = _tools().search(_catalog("bread"), AgentTranscript(case))
        assert _numbers(step) == [None] * 5


# --- What a search keeps, and what it refuses ------------------------------------------


def test_each_text_is_embedded_once():
    embedder = WordEmbedder()
    knowledge = _approved_titles()
    tools = agent.SearchTools(CATALOG, knowledge, embedder, Settings())
    assert embedder.calls == []  # nothing is embedded until a search runs
    tools.search(_catalog("bread"), _transcript())
    assert len(embedder.calls) == 1 and len(embedder.texts) == 1 + 6
    tools.search(_catalog("beer"), _transcript())
    assert embedder.calls[1:] == [["beer"]]
    tools.search(_approved("beer"), _transcript())
    assert len(embedder.calls[2]) == 3  # the three approved titles; "beer" is known
    tools.search(_approved("beer"), _transcript())
    tools.search(_catalog("bread"), _transcript())
    assert len(embedder.calls) == 3
    assert len(set(embedder.texts)) == len(embedder.texts)


def test_search_refuses_a_search_past_the_cap_whoever_calls_it():
    tools = _tools()
    steps = [SearchStep(_catalog("x"), ())] * AGENT_MAX_SEARCHES
    assert tools.search(_catalog("bread"), _transcript("FOOD", (), *steps[:-1])).hits
    with pytest.raises(ValueError, match="4 searches"):
        tools.search(_catalog("bread"), _transcript("FOOD", (), *steps))
    with pytest.raises(ValueError, match="4 searches"):
        tools.search(_approved("bread"), _transcript("FOOD", (), *steps, *steps))


def test_search_refuses_a_tool_that_is_not_one_of_the_two():
    with pytest.raises(ValueError, match="search_the_web"):
        _tools().search(SearchRequest("search_the_web", "bread"), _transcript())
    # The tool's name as plain text is the tool.
    step = _tools().search(SearchRequest("search_catalog", "bread"), _transcript())
    assert _ids(step)[0] == "FOOD-04"


def test_a_search_changes_nothing_it_was_given():
    knowledge = _approved_titles()
    before = (knowledge.entries(), knowledge.examples())
    transcript = _transcript("FOOD", [CASHIER, BAKER])
    tools = _tools(knowledge)
    first = tools.search(_approved("bread"), transcript)
    assert transcript.steps == () and transcript.searches_left == AGENT_MAX_SEARCHES
    assert (knowledge.entries(), knowledge.examples()) == before  # read-only
    assert tools.search(_approved("bread"), transcript) == first


def test_the_ml_match_is_not_what_numbers_a_role():
    # The ML first choice is marked in the case, in the family or not; only the
    # family decides the number.
    row = _row("FOOD", "night baker")
    match = MatchResult((ScoredRole("SHOP-01", 0.9), ScoredRole("FOOD-02", 0.4)))
    answer = ChooseAnswer(ChooseKind.ROLE, 1, "night baker", "test")
    shown = [NumberedRole(1, CASHIER), NumberedRole(2, BAKER)]
    case = agent.build_case(row, shown, answer, "SHOP-01", match, (Reason.FAMILY_MISMATCH,))
    assert (case.llm_role_id, case.ml_first_choice) == ("SHOP-01", "SHOP-01")
    assert case.roles == (NumberedRole(None, CASHIER), NumberedRole(1, BAKER))
