"""embedder = local: a small local embedding model behind an optional extra.

Owner: feat/local-embedder (stage 2).
Governed by: docs/design.md 2.5 (lazy import; the default run installs
nothing); docs/build-decisions.md Q30 (library, model and revision), Q32 (how
it is tested), Q38 with S09 (the message for a missing extra).

The library is model2vec, from the optional extra "local". It and the hub
library the model's files are fetched with are imported only inside
build_embedder. No PyTorch anywhere. The weights download on first use into
the user's Hugging Face cache, never into the repository. A missing extra or
a failed download stops with a clear message and never falls back to the
offline embedder. For a missing extra the message holds
contracts.run_with_extra(EXTRA), the one command text of S09.

Vectors are returned as contracts.Vector, L2-normalised, like every embedder.
The smoke test is marked local_model and is never part of the merge gate.

Decided here, where Q30 leaves the detail open:
- How the revision is pinned. Given a repository name, model2vec 0.9 takes
  no revision: it fetches whatever is newest, and all of the repository. So
  the three files it reads (MODEL_FILES) are fetched here at MODEL_REVISION
  with huggingface_hub, the library model2vec itself downloads with, which
  the extra installs too, and model2vec is handed the folder they are in.
- The cache is looked in first, and the hub is asked only for a file that
  is not there. So a run with the three files in the cache makes no request
  to the hub, and only the first run needs the network. Looking first is
  what makes this true: the hub library's download call prepares its
  request before it looks in the cache, and preparing it can contact the
  hub by itself (huggingface_hub 1.33 fetches a list for its user agent,
  about once a day), even for a file it then finds in the cache.
- A missing extra is either library that cannot be imported. The module
  imports two, both only inside build_embedder: model2vec, which the extra
  names, and then huggingface_hub, which the extra does not name and which
  comes with it (model2vec's tokenizer library needs it, and uv.lock pins
  it). S09 speaks of a provider whose library cannot be imported, so for
  either one the message names the library and holds the command of S09.
  model2vec is looked for first: without the extra, nothing else is.
- No credential. The model is public, so the files are asked for without a
  token (token=False): a Hugging Face token kept on the machine is never
  sent, and the program's only secret stays ANTHROPIC_API_KEY (2.5). While
  it downloads, the hub library prints its own notice about unauthenticated
  requests on standard error.
- build_embedder loads the model at once, not on the first embed. The switch
  builds the embedder before the LLM, so a missing extra or a failed
  download stops the run before any model is asked anything.
- A failed load is any error raised while the files are fetched or read: no
  network on the first run, a download cut short, a file that cannot be
  read. The message names the model and the revision and repeats the
  library's own words.
- A vector is the dense mapping of every index (S09), 0 to 255 for this
  model, as Python floats. It is normalised here in double precision,
  whatever the model's configuration says about normalising. A text in which
  the model finds no token (an empty one, or white space only) gives the
  empty vector, whose cosine with anything is 0, as in the offline embedder.
- Each text is encoded in a call of its own, so its vector cannot depend on
  the texts asked for with it. In one call the library works in single
  precision, unless a text without a token is among them: then it works in
  double precision for all of them, and the last digits of the others move.
- A text is embedded whole (max_length=None; the library would otherwise
  stop at 512 tokens). The cut to 200 words is the prepare step's (1.4), and
  the embedder adds none of its own.
- settings.embed_dim is the number of buckets of the offline embedder and
  does not apply here. No setting changes a vector of this embedder.
- embed takes a sequence of texts. A single string is refused, as in the
  offline embedder, because it would be embedded letter by letter.
- The smoke test downloads the model only when it is asked for by its
  marker (pytest -m local_model). In any other test run it is skipped unless
  the model is already in the cache (Q32 d), so no other run reaches the
  network.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from titlestd.contracts import Embedder, TitlestdError, Vector, run_with_extra
from titlestd.settings import Settings

EXTRA = "local"
MODEL_NAME = "minishlab/potion-base-8M"
MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
# What model2vec reads from a model folder. The repository holds more, an
# ONNX copy of the weights among it, and none of that is fetched.
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json")


class LocalEmbedder:
    """A loaded model2vec model behind embed(texts). It keeps nothing between calls."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, str):
            raise TypeError("embed takes a sequence of texts, not one string")
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        (row,) = self.model.encode([text], max_length=None)
        return _unit(row)


def _unit(row: Iterable[Any]) -> Vector:
    """One row of the model's output as an L2-normalised dense vector."""
    weights = [float(weight) for weight in row]
    norm = math.sqrt(math.fsum(weight * weight for weight in weights))
    if norm == 0.0:
        return {}
    return {index: weight / norm for index, weight in enumerate(weights)}


def _model_folder(in_cache: Callable[..., Any], download: Callable[..., str]) -> Path:
    """The folder of MODEL_REVISION in the Hugging Face cache, filled on first use.

    in_cache and download are the hub library's try_to_load_from_cache and
    hf_hub_download. A file that is in the cache is taken from it, and the
    download call is never reached for it: that call can contact the hub even
    for a cached file.
    """
    files = []
    for name in MODEL_FILES:
        # A path when the file is in the cache; otherwise None or a marker.
        found = in_cache(MODEL_NAME, name, revision=MODEL_REVISION)
        if not isinstance(found, str):
            found = download(MODEL_NAME, name, revision=MODEL_REVISION, token=False)
        files.append(found)
    return Path(files[0]).parent


def _missing_extra(library: str, error: ImportError) -> TitlestdError:
    """What stops the run when a library of the extra cannot be imported (S09)."""
    return TitlestdError(
        f"embedder = local needs the {library} library, which cannot be imported ({error}). "
        f'It comes with the optional extra "{EXTRA}"; run the program with that extra, '
        f"for example: {run_with_extra(EXTRA)} run --embedder local"
    )


def build_embedder(settings: Settings) -> Embedder:
    """The local embedder. Raises TitlestdError when the extra is not installed
    or the model cannot be loaded."""
    try:
        from model2vec import StaticModel
    except ImportError as error:
        raise _missing_extra("model2vec", error) from error
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
    except ImportError as error:
        raise _missing_extra("huggingface_hub", error) from error
    try:
        model = StaticModel.from_pretrained(_model_folder(try_to_load_from_cache, hf_hub_download))
    except Exception as error:
        raise TitlestdError(
            f"embedder = local could not load the model {MODEL_NAME} at revision "
            f"{MODEL_REVISION} ({type(error).__name__}: {error}). The first run downloads it "
            "into the Hugging Face cache and needs the network; later runs do not."
        ) from error
    return LocalEmbedder(model)
