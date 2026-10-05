"""The offline embedder: hashed character n-grams, as Q12 fixes them."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from titlestd import offline_embedder
from titlestd.contracts import Vector
from titlestd.offline_embedder import OfflineEmbedder, bucket, build_embedder, ngrams
from titlestd.settings import Settings

DIM = 32768


def dot(a: Vector, b: Vector) -> float:
    """Cosine similarity of two normalised vectors."""
    return math.fsum(weight * b.get(index, 0.0) for index, weight in a.items())


def embed(text: str, dim: int = DIM) -> Vector:
    (vector,) = OfflineEmbedder(dim).embed([text])
    return vector


# --- The features -------------------------------------------------------------


def test_features_are_the_3_4_and_5_grams_of_each_wrapped_token():
    assert list(ngrams("mgr")) == ["<mg", "mgr", "gr>", "<mgr", "mgr>", "<mgr>"]
    assert list(ngrams("sales")) == [
        "<sa",
        "sal",
        "ale",
        "les",
        "es>",
        "<sal",
        "sale",
        "ales",
        "les>",
        "<sale",
        "sales",
        "ales>",
    ]


def test_a_token_shorter_than_an_ngram_size_has_none_of_that_size():
    assert list(ngrams("a")) == ["<a>"]
    assert list(ngrams("ii")) == ["<ii", "ii>", "<ii>"]


def test_text_is_case_folded_and_split_on_everything_but_letters_and_digits():
    assert list(ngrams("Sr. Mgr")) == list(ngrams("sr")) + list(ngrams("mgr"))
    assert list(ngrams("ACCT-MGR/2")) == [*ngrams("acct"), *ngrams("mgr"), *ngrams("2")]
    # No n-gram spans two tokens: each token is wrapped on its own.
    assert not any(" " in feature for feature in ngrams("account manager"))


def test_a_repeated_token_repeats_its_features():
    assert list(ngrams("ab ab")) == ["<ab", "ab>", "<ab>"] * 2


# --- The buckets --------------------------------------------------------------


def test_the_bucket_of_a_feature_is_pinned():
    # An 8-byte blake2b digest, read big-endian, modulo the dimension.
    assert bucket("<sr", DIM) == 7628
    assert bucket("sr>", DIM) == 14128
    assert bucket("<sr>", DIM) == 13587
    assert bucket("<a>", DIM) == 8836
    assert bucket("<sr", 2048) == 1484
    assert bucket("ount>", 2048) == 1990


def test_a_bucket_is_always_inside_the_dimension():
    for dim in (1, 2, 7, 64):
        assert all(0 <= bucket(feature, dim) < dim for feature in ngrams("senior account manager"))


# --- The vector ---------------------------------------------------------------


def test_the_vector_of_a_short_text_is_pinned():
    assert embed("a") == {8836: 1.0}
    third = 1 / math.sqrt(3)
    assert embed("Sr") == {7628: third, 13587: third, 14128: third}


def test_weight_is_one_plus_the_log_of_the_count():
    # "ab" twice, "cd" once: before normalising, 1 + ln 2 against 1.
    vector = embed("ab ab cd")
    heavy = {vector[bucket(feature, DIM)] for feature in ngrams("ab")}
    light = {vector[bucket(feature, DIM)] for feature in ngrams("cd")}
    assert len(heavy) == len(light) == 1
    assert heavy.pop() / light.pop() == pytest.approx(1 + math.log(2), rel=1e-12)


def test_features_that_share_a_bucket_add_their_weights():
    # Two buckets only, so the six features of "ab ab cd" must share them.
    expected = {0: 0.0, 1: 0.0}
    for feature in ("<ab", "ab>", "<ab>"):
        expected[bucket(feature, 2)] += 1 + math.log(2)
    for feature in ("<cd", "cd>", "<cd>"):
        expected[bucket(feature, 2)] += 1.0
    norm = math.hypot(*expected.values())
    vector = embed("ab ab cd", dim=2)
    assert vector == pytest.approx({i: w / norm for i, w in expected.items() if w})
    assert embed("ab ab cd", dim=1) == {0: 1.0}


@pytest.mark.parametrize(
    "text",
    [
        "a",
        "senior account manager",
        "Manages a portfolio of enterprise accounts and renews their contracts.",
        "budget budget budget forecast",
    ],
)
def test_a_vector_has_length_one(text):
    vector = embed(text)
    assert math.fsum(weight * weight for weight in vector.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(weight > 0 for weight in vector.values())


def test_a_vector_lists_its_buckets_in_rising_order():
    indices = list(embed("Explains the technical side of products and runs demonstrations."))
    assert indices == sorted(indices)
    assert len(indices) > 50


def test_a_text_with_no_token_gives_the_empty_vector():
    assert embed("") == {}
    assert embed(" -- (?) ") == {}
    assert dot(embed(""), embed("account manager")) == 0.0


def test_case_and_punctuation_do_not_change_the_vector():
    assert embed("Sr. Acct Mgr - NYC") == embed("sr acct mgr nyc")


# --- embed(texts) -------------------------------------------------------------


def test_one_vector_per_text_in_order():
    texts = ["account manager", "", "treasury analyst", "account manager"]
    vectors = OfflineEmbedder(DIM).embed(texts)
    assert vectors == [embed(text) for text in texts]
    assert vectors[0] == vectors[3] and vectors[0] != vectors[2]


def test_no_texts_no_vectors():
    assert OfflineEmbedder(DIM).embed([]) == []


def test_a_single_string_is_refused():
    with pytest.raises(TypeError, match="sequence of texts"):
        OfflineEmbedder(DIM).embed("account manager")


def test_nothing_is_fitted_so_a_vector_never_changes():
    # No weights learned from what was embedded before (no IDF): the vector of
    # a text is the same before and after any other text, on any instance.
    embedder = OfflineEmbedder(DIM)
    before = embedder.embed(["financial analyst"])
    embedder.embed(["financial analyst"] * 50 + ["analyst", "financial controller"])
    assert embedder.embed(["financial analyst"]) == before
    assert OfflineEmbedder(DIM).embed(["financial analyst"]) == before


# --- Similarity ---------------------------------------------------------------


def test_cosine_is_one_for_the_same_text_and_falls_with_shared_ngrams():
    assert dot(embed("account manager"), embed("Account Manager")) == pytest.approx(1.0)
    close = dot(embed("senior account manager"), embed("account manager"))
    far = dot(embed("senior account manager"), embed("treasury analyst"))
    assert close > 0.8 > 0.2 > far


def test_it_matches_on_parts_of_words():
    # Two different words that share most of their letters are still close,
    # which a comparison of whole words cannot see (Q12).
    assert dot(embed("accountant"), embed("accounting")) > 0.5
    assert dot(embed("accountant"), embed("treasury")) < 0.05


def test_few_buckets_make_unrelated_texts_look_alike():
    # Why the default is 32,768 and not 2,048 (Q12): collisions add similarity.
    one = "Manages a portfolio of customer accounts and renews their contracts."
    other = "Tests financial controls and reports findings to the audit committee."
    wide = dot(embed(one), embed(other))
    narrow = dot(embed(one, dim=64), embed(other, dim=64))
    assert narrow > wide + 0.2


# --- The provider -------------------------------------------------------------


def test_the_provider_takes_its_dimension_from_the_settings():
    assert isinstance(build_embedder(Settings()), OfflineEmbedder)
    assert build_embedder(Settings()).dim == DIM
    small = build_embedder(Settings(embed_dim=16))
    assert small.dim == 16
    (vector,) = small.embed(["senior account manager"])
    assert vector and all(0 <= index < 16 for index in vector)


def test_two_providers_give_the_same_vectors():
    texts = ["senior account manager", "Builds budgets, forecasts and variance reports."]
    assert build_embedder(Settings()).embed(texts) == build_embedder(Settings()).embed(texts)


# --- What the module may use --------------------------------------------------


def test_it_imports_only_the_standard_library_the_contracts_and_the_settings():
    tree = ast.parse(Path(offline_embedder.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {
        "__future__",
        "collections",
        "collections.abc",
        "hashlib",
        "math",
        "re",
        "titlestd.contracts",
        "titlestd.settings",
    }
