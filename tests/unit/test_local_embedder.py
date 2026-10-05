"""embedder = local, with a stub in place of the library (2.5, Q30, Q32).

The merge gate never has the "local" extra, and these tests must not depend on
whether it is installed. Each one puts a stub of model2vec and of
huggingface_hub where the lazy imports look, in sys.modules, or blocks the
import there. The stub is put in place long after titlestd.local_embedder was
imported, so a test that sees the stub used has also seen that the library is
looked for when the embedder is built and not before.

The real model has one smoke test of its own, marked local_model, in
tests/integration/test_local_embedder_model.py.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from titlestd import local_embedder
from titlestd.contracts import TitlestdError, run_with_extra
from titlestd.local_embedder import (
    EXTRA,
    MODEL_FILES,
    MODEL_NAME,
    MODEL_REVISION,
    LocalEmbedder,
    build_embedder,
)
from titlestd.settings import Settings

LOCAL = Settings(embedder="local")


class Single:
    """A number that is not a Python float, as the model's single-precision
    numbers are not. It can be turned into one and does nothing else."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


def row(*weights: float) -> list[Single]:
    return [Single(weight) for weight in weights]


# What the stub model returns for each text, before any normalising.
ROWS = {
    "account manager": row(3.0, 4.0, 0.0),
    "treasury analyst": row(0.0, 0.0, 2.0),
    "sales engineer": row(-1.0, 2.0, -2.0),
    "already of length one": row(0.6, 0.0, 0.8),
    "": row(0.0, 0.0, 0.0),
    " \n": row(0.0, 0.0, 0.0),
}


class StubModel:
    """Stands in for a loaded model2vec.StaticModel and lists how it was asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, sentences, **options):
        self.calls.append((list(sentences), options))
        return [ROWS[sentence] for sentence in sentences]


def module(name: str, **members) -> ModuleType:
    made = ModuleType(name)
    for member, value in members.items():
        setattr(made, member, value)
    return made


@pytest.fixture
def library(monkeypatch, tmp_path: Path) -> SimpleNamespace:
    """Stubs of the two libraries of the extra, in place of the real ones.

    cached maps a file name to what the cache answers for it: nothing is in
    the cache unless a test puts it there. looked lists every file looked for
    in the cache, fetched every file asked of the hub and loaded every model
    folder opened, with the options of each call.
    """
    folder = tmp_path / "hub" / "snapshots" / MODEL_REVISION
    stub = SimpleNamespace(
        model=StubModel(), folder=folder, cached={}, looked=[], fetched=[], loaded=[]
    )

    def try_to_load_from_cache(repo_id, filename, **options):
        stub.looked.append((repo_id, filename, options))
        return stub.cached.get(filename)

    def hf_hub_download(repo_id, filename, **options):
        stub.fetched.append((repo_id, filename, options))
        return str(folder / filename)

    class StaticModel:
        @classmethod
        def from_pretrained(cls, path, **options):
            stub.loaded.append((path, options))
            return stub.model

    monkeypatch.setitem(sys.modules, "model2vec", module("model2vec", StaticModel=StaticModel))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        module(
            "huggingface_hub",
            try_to_load_from_cache=try_to_load_from_cache,
            hf_hub_download=hf_hub_download,
        ),
    )
    return stub


@pytest.fixture
def no_library(monkeypatch) -> None:
    """The extra is not installed: importing model2vec fails, whatever is on the machine."""
    monkeypatch.setitem(sys.modules, "model2vec", None)


# --- The library and the model of Q30 ------------------------------------------


def test_the_extra_the_model_and_its_revision_are_those_of_q30():
    assert EXTRA == "local"
    assert MODEL_NAME == "minishlab/potion-base-8M"
    assert MODEL_REVISION == "bf8b056651a2c21b8d2565580b8569da283cab23"
    assert MODEL_FILES == ("config.json", "model.safetensors", "tokenizer.json")


def test_the_three_files_are_asked_for_at_the_pinned_revision(library):
    build_embedder(LOCAL)
    assert [(repo_id, filename) for repo_id, filename, _ in library.fetched] == [
        (MODEL_NAME, "config.json"),
        (MODEL_NAME, "model.safetensors"),
        (MODEL_NAME, "tokenizer.json"),
    ]
    assert all(options["revision"] == MODEL_REVISION for _, _, options in library.fetched)
    # A full commit id, not a branch: a branch can move, so only the hub knows
    # where it points today. A commit is the same files wherever it is found.
    assert len(MODEL_REVISION) == 40 and set(MODEL_REVISION) <= set("0123456789abcdef")


def test_no_credential_is_sent_for_the_public_model(library):
    build_embedder(LOCAL)
    # token=False, not None: None would send a token stored on the machine (2.5).
    assert [options["token"] for _, _, options in library.fetched] == [False, False, False]
    assert all(set(options) == {"revision", "token"} for _, _, options in library.fetched)


# --- The cache before the hub (Q30) ------------------------------------------------


def put_in_cache(library: SimpleNamespace, *names: str) -> None:
    """The cache answers for these files with their path, as the hub library does."""
    for name in names:
        library.cached[name] = str(library.folder / name)


def test_each_file_is_looked_for_in_the_cache_at_the_pinned_revision(library):
    build_embedder(LOCAL)
    assert library.looked == [
        (MODEL_NAME, "config.json", {"revision": MODEL_REVISION}),
        (MODEL_NAME, "model.safetensors", {"revision": MODEL_REVISION}),
        (MODEL_NAME, "tokenizer.json", {"revision": MODEL_REVISION}),
    ]


def test_a_model_in_the_cache_is_not_asked_of_the_hub(library):
    put_in_cache(library, *MODEL_FILES)
    embedder = build_embedder(LOCAL)
    # Not even to hear that the files are there: the hub library's download
    # call can contact the hub before it looks in the cache, so a later run
    # would not be free of the network.
    assert library.fetched == []
    assert library.loaded == [(library.folder, {})]
    assert embedder.model is library.model


@pytest.mark.parametrize("missing", MODEL_FILES)
def test_only_a_file_missing_from_the_cache_is_asked_of_the_hub(library, missing):
    put_in_cache(library, *(name for name in MODEL_FILES if name != missing))
    build_embedder(LOCAL)
    assert library.fetched == [(MODEL_NAME, missing, {"revision": MODEL_REVISION, "token": False})]
    assert library.loaded == [(library.folder, {})]


def test_a_cache_answer_that_is_no_path_sends_the_file_to_the_hub(library):
    # The hub library answers with a marker, not a path, for a file it was
    # once told the revision does not hold. The hub is asked again.
    put_in_cache(library, *MODEL_FILES)
    library.cached["tokenizer.json"] = object()
    build_embedder(LOCAL)
    assert [filename for _, filename, _ in library.fetched] == ["tokenizer.json"]


def test_the_model_is_read_from_the_folder_of_the_pinned_revision(library):
    embedder = build_embedder(LOCAL)
    # The folder, not the repository name: by name the library would fetch the
    # newest revision, and all of the repository.
    assert library.loaded == [(library.folder, {})]
    assert isinstance(embedder, LocalEmbedder)
    assert embedder.model is library.model


def test_the_model_is_loaded_when_the_embedder_is_built_and_only_then(library):
    embedder = build_embedder(LOCAL)
    assert len(library.loaded) == 1 and len(library.looked) == 3 and len(library.fetched) == 3
    assert library.model.calls == []
    embedder.embed(["account manager"])
    embedder.embed(["treasury analyst", "account manager"])
    assert len(library.loaded) == 1 and len(library.looked) == 3 and len(library.fetched) == 3


# --- The lazy import (2.5) --------------------------------------------------------


def test_the_library_is_looked_for_each_time_an_embedder_is_built(library, monkeypatch):
    first = build_embedder(LOCAL)
    other = StubModel()
    replacement = SimpleNamespace(from_pretrained=lambda path: other)
    monkeypatch.setitem(sys.modules, "model2vec", module("model2vec", StaticModel=replacement))
    assert first.model is library.model
    assert build_embedder(LOCAL).model is other


def test_the_module_imports_the_two_libraries_only_inside_a_function():
    tree = ast.parse(Path(local_embedder.__file__).read_text(encoding="utf-8"))

    def imported(nodes) -> set[str]:
        names = set()
        for node in nodes:
            if isinstance(node, ast.Import):
                names |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module)
        return names

    def outside_functions(node):
        """Everything that runs when the module is imported: all but the
        bodies of functions. An import under try or if is among it (S15)."""
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                yield child
                yield from outside_functions(child)

    # On import: the standard library, the contracts and the settings.
    assert imported(outside_functions(tree)) == {
        "__future__",
        "collections.abc",
        "math",
        "pathlib",
        "typing",
        "titlestd.contracts",
        "titlestd.settings",
    }
    lazy = {
        node.name: imported(ast.walk(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and imported(ast.walk(node))
    }
    assert lazy == {"build_embedder": {"model2vec", "huggingface_hub"}}


# --- A missing extra (Q30, Q38, S09) ----------------------------------------------


def test_a_missing_extra_stops_with_the_command_that_installs_and_runs_it(no_library):
    with pytest.raises(TitlestdError) as raised:
        build_embedder(LOCAL)
    message = str(raised.value)
    assert message.startswith("embedder = local needs the model2vec library, which cannot be")
    assert 'the optional extra "local"' in message
    assert run_with_extra("local") == "uv run --extra local python -m titlestd"
    assert message.endswith(f"for example: {run_with_extra('local')} run --embedder local")


def test_the_message_repeats_what_the_failed_import_said(no_library):
    with pytest.raises(TitlestdError) as raised:
        build_embedder(LOCAL)
    cause = raised.value.__cause__
    assert isinstance(cause, ImportError) and "model2vec" in str(cause)
    assert f"({cause})" in str(raised.value)


def test_a_missing_extra_fetches_nothing_and_nothing_stands_in(library, no_library):
    # The hub library is there, the embedding library is not: no file is looked
    # for or asked for, and no embedder of another kind comes back.
    with pytest.raises(TitlestdError, match="model2vec"):
        build_embedder(LOCAL)
    assert library.looked == [] and library.fetched == [] and library.loaded == []


def test_a_hub_library_that_cannot_be_imported_is_a_missing_extra_too(library, monkeypatch):
    # The embedding library is there, the one the files are fetched with is
    # not. That is a library of the extra that cannot be imported (S09), not a
    # failed download: the message of a failed load would blame the network.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(TitlestdError) as raised:
        build_embedder(LOCAL)
    message = str(raised.value)
    cause = raised.value.__cause__
    assert isinstance(cause, ImportError) and "huggingface_hub" in str(cause)
    assert message == (
        f"embedder = local needs the huggingface_hub library, which cannot be imported ({cause}). "
        'It comes with the optional extra "local"; run the program with that extra, '
        f"for example: {run_with_extra('local')} run --embedder local"
    )
    assert library.looked == [] and library.fetched == [] and library.loaded == []


def test_a_hub_library_without_what_is_needed_of_it_is_a_missing_extra_too(library, monkeypatch):
    # The name can be imported and the function cannot: still an import that failed.
    monkeypatch.delattr(sys.modules["huggingface_hub"], "try_to_load_from_cache")
    with pytest.raises(TitlestdError) as raised:
        build_embedder(LOCAL)
    assert isinstance(raised.value.__cause__, ImportError)
    assert str(raised.value).startswith("embedder = local needs the huggingface_hub library,")
    assert str(raised.value).endswith(f"{run_with_extra('local')} run --embedder local")
    assert library.fetched == [] and library.loaded == []


def test_with_neither_library_the_message_names_the_one_the_extra_declares(no_library, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(TitlestdError) as raised:
        build_embedder(LOCAL)
    assert str(raised.value).startswith("embedder = local needs the model2vec library,")


# --- A failed load (Q30) ----------------------------------------------------------


def expect_failed_load(error: Exception) -> str:
    with pytest.raises(TitlestdError) as raised:
        build_embedder(LOCAL)
    assert raised.value.__cause__ is error
    message = str(raised.value)
    assert message.startswith(
        f"embedder = local could not load the model {MODEL_NAME} at revision {MODEL_REVISION} ("
    )
    assert message.endswith(
        "The first run downloads it into the Hugging Face cache and needs the network; "
        "later runs do not."
    )
    # The library is there, so nobody is told to install it.
    assert run_with_extra("local") not in message
    return message


def test_a_failed_download_stops_with_a_message_naming_the_model(library, monkeypatch):
    error = OSError("Network is unreachable")

    def unreachable(repo_id, filename, **options):
        raise error

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", unreachable)
    assert "(OSError: Network is unreachable)" in expect_failed_load(error)
    assert library.loaded == []  # and no model was opened in its place


def test_a_download_that_fails_half_way_stops_the_same_way(library, monkeypatch):
    error = TimeoutError("timed out after 10 seconds")
    fetch = sys.modules["huggingface_hub"].hf_hub_download

    def second_file_fails(repo_id, filename, **options):
        if filename == MODEL_FILES[1]:
            raise error
        return fetch(repo_id, filename, **options)

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", second_file_fails)
    assert "(TimeoutError: timed out after 10 seconds)" in expect_failed_load(error)
    assert library.loaded == []


def test_a_cache_that_cannot_be_looked_in_stops_the_same_way(library, monkeypatch):
    error = PermissionError("the cache folder cannot be read")

    def unreadable(repo_id, filename, **options):
        raise error

    monkeypatch.setattr(sys.modules["huggingface_hub"], "try_to_load_from_cache", unreadable)
    assert "(PermissionError: the cache folder cannot be read)" in expect_failed_load(error)
    assert library.fetched == [] and library.loaded == []


def test_a_model_folder_that_cannot_be_read_stops_the_same_way(library, monkeypatch):
    error = ValueError("Could not find expected model files")

    def broken(path):
        raise error

    monkeypatch.setattr(sys.modules["model2vec"].StaticModel, "from_pretrained", broken)
    assert "(ValueError: Could not find expected model files)" in expect_failed_load(error)


def test_an_interrupted_download_is_not_reported_as_a_failed_one(library, monkeypatch):
    def interrupted(repo_id, filename, **options):
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_embedder(LOCAL)


# --- embed(texts) -----------------------------------------------------------------


def test_one_vector_per_text_in_order(library):
    embedder = build_embedder(LOCAL)
    texts = ["account manager", "", "treasury analyst", "account manager"]
    vectors = embedder.embed(texts)
    assert vectors == [
        {0: 0.6, 1: 0.8, 2: 0.0},
        {},
        {0: 0.0, 1: 0.0, 2: 1.0},
        {0: 0.6, 1: 0.8, 2: 0.0},
    ]
    assert embedder.embed(texts) == vectors


def test_a_vector_maps_every_index_to_a_python_float(library):
    (vector,) = build_embedder(LOCAL).embed(["treasury analyst"])
    # Dense (S09): an index whose weight is zero is listed like any other.
    assert list(vector) == [0, 1, 2]
    assert all(type(weight) is float for weight in vector.values())


@pytest.mark.parametrize("text", ["account manager", "sales engineer", "already of length one"])
def test_a_vector_has_length_one_whatever_the_model_returns(library, text):
    (vector,) = build_embedder(LOCAL).embed([text])
    assert math.fsum(weight * weight for weight in vector.values()) == pytest.approx(1.0, abs=1e-15)
    # The direction is the model's: each weight divided by the row's length.
    raw = [float(weight) for weight in ROWS[text]]
    length = math.sqrt(math.fsum(weight * weight for weight in raw))
    assert vector == {index: weight / length for index, weight in enumerate(raw)}


def test_a_negative_weight_stays_negative(library):
    (vector,) = build_embedder(LOCAL).embed(["sales engineer"])
    assert vector == pytest.approx({0: -1 / 3, 1: 2 / 3, 2: -2 / 3})


@pytest.mark.parametrize("text", ["", " \n"])
def test_a_text_with_no_token_gives_the_empty_vector(library, text):
    # The model answers such a text with a row of zeros, which has no direction.
    assert build_embedder(LOCAL).embed([text]) == [{}]


def test_each_text_is_encoded_in_a_call_of_its_own_and_whole(library):
    build_embedder(LOCAL).embed(["account manager", "", "treasury analyst"])
    # One text per call, so no text can change the vector of another; and no
    # cut of the embedder's own: max_length=None lifts the library's.
    assert library.model.calls == [
        (["account manager"], {"max_length": None}),
        ([""], {"max_length": None}),
        (["treasury analyst"], {"max_length": None}),
    ]


def test_no_texts_no_vectors(library):
    assert build_embedder(LOCAL).embed([]) == []
    assert library.model.calls == []


def test_any_sequence_of_texts_will_do(library):
    embedder = build_embedder(LOCAL)
    assert embedder.embed(("account manager", "treasury analyst")) == embedder.embed(
        ["account manager", "treasury analyst"]
    )


def test_a_single_string_is_refused(library):
    embedder = build_embedder(LOCAL)
    with pytest.raises(TypeError, match="sequence of texts"):
        embedder.embed("account manager")
    assert library.model.calls == []


def test_no_setting_changes_a_vector(library):
    # embed_dim is the number of buckets of the offline embedder (Q12); the
    # size of a local vector is the model's.
    plain = build_embedder(LOCAL).embed(["account manager"])
    other = Settings(embedder="local", embed_dim=2, seed=7, agent_search_k=1)
    assert build_embedder(other).embed(["account manager"]) == plain
    assert len(plain[0]) == 3
