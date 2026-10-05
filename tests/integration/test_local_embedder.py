"""embedder = local behind the real switch, without the real library (2.5, Q30, S09).

The first tests start a bare interpreter (no site-packages) behind the import
guard of tests/conftest.py, where the "local" extra cannot be installed: the
setting stops the run with the command of S09, nothing stands in for the
embedder, and the module's own code needs nothing beyond the standard library.

The others run in this process on the mini data set, with the real readers,
the real prepare step, the real catalog functions and the real switch, and a
stub in place of the library. The real model has one smoke test of its own,
marked local_model, in tests/integration/test_local_embedder_model.py.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from titlestd import catalog, datafiles, offline_embedder, standin, switch
from titlestd.contracts import (
    EMAIL_MASK,
    CatalogRole,
    InputData,
    Record,
    TitleRow,
    TitlestdError,
    run_with_extra,
)
from titlestd.local_embedder import MODEL_REVISION, LocalEmbedder
from titlestd.prepare import prepare
from titlestd.settings import Settings

# After the import guard: import the module, then choose the setting through
# the switch, and report what was looked for, what was loaded and what was said.
MISSING_EXTRA = """
import json

from titlestd import local_embedder

on_import = list(attempted_imports)

from titlestd import switch
from titlestd.contracts import TitlestdError
from titlestd.settings import Settings

llm = sys.argv[1]
model = "claude-haiku-4-5" if llm == "anthropic" else None
try:
    built = switch.build_models(Settings(embedder="local", llm=llm, llm_model=model), [], [])
    message = None
except TitlestdError as error:
    built = None
    message = str(error)
print(json.dumps({
    "on_import": on_import,
    "attempted": attempted_imports,
    "built": built is not None,
    "message": message,
    "loaded": sorted(name for name in sys.modules if name.partition(".")[0] == "titlestd"),
}))
"""

# After the import guard: put a stub of the two libraries in place, as the
# extra would put the real ones, and embed through the switch.
WITH_A_STUB = """
import json, types


class Model:
    def encode(self, sentences, **options):
        return [[float(len(sentence)), 1.0] for sentence in sentences]


class StaticModel:
    @classmethod
    def from_pretrained(cls, path):
        return Model()


sys.modules["model2vec"] = types.ModuleType("model2vec")
sys.modules["model2vec"].StaticModel = StaticModel
sys.modules["huggingface_hub"] = types.ModuleType("huggingface_hub")
sys.modules["huggingface_hub"].hf_hub_download = lambda repo_id, filename, **options: filename

from titlestd import switch
from titlestd.settings import Settings

models = switch.build_models(Settings(embedder="local"), [], [])
vectors = models.embedder.embed(["abc", ""])
print(json.dumps({
    "embedder": type(models.embedder).__name__,
    "vectors": [sorted(vector.items()) for vector in vectors],
    "attempted": attempted_imports,
    "loaded": sorted(name for name in sys.modules if name.partition(".")[0] == "titlestd"),
}))
"""

DIMENSIONS = 16


class WordModel:
    """Stands in for the loaded model: a text becomes counts of its words in
    DIMENSIONS buckets. It lists every text it was asked for."""

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def encode(self, sentences, **options):
        self.asked.append(list(sentences))
        return [self.row(sentence) for sentence in sentences]

    @staticmethod
    def row(text: str) -> list[float]:
        counts = [0.0] * DIMENSIONS
        for word in text.lower().split():
            counts[hashlib.sha256(word.encode("utf-8")).digest()[0] % DIMENSIONS] += 1.0
        return counts


@pytest.fixture
def model(monkeypatch) -> WordModel:
    """A stub of the two libraries of the extra, where the lazy imports look."""
    loaded = WordModel()

    class StaticModel:
        @classmethod
        def from_pretrained(cls, path):
            assert Path(path).name == MODEL_REVISION
            return loaded

    def hf_hub_download(repo_id, filename, *, revision, token):
        return f"/nowhere/{repo_id}/snapshots/{revision}/{filename}"

    stubs = {
        "model2vec": {"StaticModel": StaticModel},
        "huggingface_hub": {"hf_hub_download": hf_hub_download},
    }
    for name, members in stubs.items():
        stub = ModuleType(name)
        vars(stub).update(members)
        monkeypatch.setitem(sys.modules, name, stub)
    return loaded


@pytest.fixture
def local_settings(mini_settings: Settings) -> Settings:
    return dataclasses.replace(mini_settings, embedder="local")


@pytest.fixture(scope="module")
def roles(mini_inputs: InputData) -> tuple[CatalogRole, ...]:
    return catalog.model_visible_roles(mini_inputs.catalog)


@pytest.fixture(scope="module")
def eligible(mini_inputs: InputData) -> list[TitleRow]:
    rows = prepare(
        mini_inputs.employees, mini_inputs.abbreviations, catalog.family_codes(mini_inputs.catalog)
    )
    return [row for row in rows if row.model_eligible]


def role_text(role: CatalogRole) -> str:
    return f"{role.role_name}. {role.definition}"


def record_text(record: Record) -> str:
    return f"{record.cleaned_title} {record.description}".strip()


def build(settings: Settings, data: InputData):
    """The models as the pipeline will ask for them."""
    return switch.build_models(
        settings,
        data.catalog.ladder,
        data.abbreviations,
        datafiles.input_fingerprint(settings.data_dir),
    )


# --- Without the extra, on a bare interpreter --------------------------------------


@pytest.mark.parametrize("llm", ["offline", "anthropic"])
def test_without_the_extra_the_setting_stops_the_run_and_nothing_stands_in(
    bare_python, import_guard, llm
):
    result = bare_python(import_guard + MISSING_EXTRA, llm)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    # Importing the module looked for nothing; choosing the setting looked for
    # the library, once, and for nothing else.
    assert report["on_import"] == []
    assert report["attempted"] == ["model2vec"]
    assert report["built"] is False
    assert report["message"] == (
        "embedder = local needs the model2vec library, which cannot be imported "
        "(model2vec is outside the standard library: not to be imported here). "
        'It comes with the optional extra "local"; run the program with that extra, '
        f"for example: {run_with_extra('local')} run --embedder local"
    )
    # No fall-back to the offline embedder (Q30), and no LLM provider was
    # started after the embedder failed.
    assert "titlestd.local_embedder" in report["loaded"]
    assert not {"titlestd.offline_embedder", "titlestd.standin", "titlestd.live"} & set(
        report["loaded"]
    )


def test_with_the_library_in_place_the_module_needs_only_the_standard_library(
    bare_python, import_guard
):
    result = bare_python(import_guard + WITH_A_STUB)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["embedder"] == "LocalEmbedder"
    assert report["vectors"] == [
        [[0, 3 / math.sqrt(10)], [1, 1 / math.sqrt(10)]],
        [[0, 0.0], [1, 1.0]],
    ]
    # Nothing else was looked for: what the model returns is turned into
    # vectors with the standard library alone.
    assert report["attempted"] == []
    assert "titlestd.offline_embedder" not in report["loaded"]


# --- On the mini data set, behind the real switch -----------------------------------


def test_the_switch_builds_the_local_embedder_beside_the_stand_in(
    model, local_settings, mini_inputs, monkeypatch
):
    def never(settings):
        raise AssertionError("the offline embedder was built for embedder = local")

    monkeypatch.setattr(offline_embedder, "build_embedder", never)
    models = build(local_settings, mini_inputs)
    assert isinstance(models.embedder, LocalEmbedder)
    assert models.embedder.model is model
    assert isinstance(models.llm.chooser, standin.StandIn)


def test_every_text_reaches_the_model_as_it_is_one_at_a_time(
    model, local_settings, mini_inputs, roles, eligible
):
    embedder = build(local_settings, mini_inputs).embedder
    texts = [role_text(role) for role in roles] + [record_text(row.record) for row in eligible]
    assert len(texts) == 12 + 13
    vectors = embedder.embed(texts)
    assert model.asked == [[text] for text in texts]
    # Among them the row with no description (the title alone, 2.4) and the
    # description whose email the prepare step masked.
    assert ["payroll specialist"] in model.asked
    assert sum(EMAIL_MASK in text for text in texts) == 1
    assert len(vectors) == len(texts)


def test_each_vector_is_the_models_row_at_length_one(
    model, local_settings, mini_inputs, roles, eligible
):
    embedder = build(local_settings, mini_inputs).embedder
    texts = [role_text(role) for role in roles] + [record_text(row.record) for row in eligible]
    for text, vector in zip(texts, embedder.embed(texts), strict=True):
        counts = WordModel.row(text)
        length = math.sqrt(sum(count * count for count in counts))
        assert list(vector) == list(range(DIMENSIONS))
        assert vector == pytest.approx({i: count / length for i, count in enumerate(counts)})
        assert math.fsum(weight * weight for weight in vector.values()) == pytest.approx(
            1.0, abs=1e-12
        )


def test_a_missing_extra_passes_through_the_switch_unchanged(
    local_settings, mini_inputs, monkeypatch
):
    monkeypatch.setitem(sys.modules, "model2vec", None)

    def never(*args, **kwargs):
        raise AssertionError("a provider was built after the embedder failed")

    monkeypatch.setattr(offline_embedder, "build_embedder", never)
    monkeypatch.setattr(standin, "build_llm", never)
    with pytest.raises(TitlestdError) as raised:
        build(local_settings, mini_inputs)
    assert str(raised.value).endswith(f"{run_with_extra('local')} run --embedder local")
