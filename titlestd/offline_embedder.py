"""embedder = offline: hashed character n-grams, standard library only.

Owner: feat/model-switch-offline (stage 1).
Governed by: DESIGN.md 1.5 (off the shelf, not trained), 2.5;
docs/build-decisions.md Q12.

The method (Q12): case-fold; split into [a-z0-9]+ tokens; wrap each as
"<token>"; features are all 3-, 4- and 5-character n-grams of the wrapped
token; weight 1 + ln(count); bucket = an 8-byte hashlib.blake2b digest taken
modulo settings.embed_dim; L2-normalise. Vectors are sparse (contracts.Vector).
There are no fitted weights, so a vector never changes as approvals are added,
and the same text gives the same vector in every process and on every machine.

It must be a different calculation from the word-overlap stand-in (Q12).

Decided here, where Q12 leaves the detail open:
- count is how often one n-gram occurs in the whole text. Each n-gram is
  weighted on its own, and n-grams that share a bucket add their weights.
- The digest is read as a big-endian whole number before the modulo.
- A text with no token gives the empty vector, whose cosine with anything
  is 0. A wrapped token shorter than an n-gram size has no n-gram of that
  size: "<a>" is one 3-gram.
- A vector lists its buckets in rising order, so two equal texts give
  vectors that are equal item for item.
- embed takes a sequence of texts. A single string is refused, because it
  would otherwise be embedded letter by letter.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence

from titlestd.contracts import Embedder, Vector
from titlestd.settings import Settings

NGRAM_SIZES = (3, 4, 5)
DIGEST_BYTES = 8

_TOKEN = re.compile(r"[a-z0-9]+")


def ngrams(text: str) -> Iterator[str]:
    """Every feature of a text, in order of appearance, repeats included."""
    for token in _TOKEN.findall(text.casefold()):
        wrapped = f"<{token}>"
        for size in NGRAM_SIZES:
            for start in range(len(wrapped) - size + 1):
                yield wrapped[start : start + size]


def bucket(feature: str, dim: int) -> int:
    """The index a feature is hashed to, the same in every process."""
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") % dim


class OfflineEmbedder:
    """Hashed character n-grams in dim buckets. It keeps nothing between calls."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if isinstance(texts, str):
            raise TypeError("embed takes a sequence of texts, not one string")
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> Vector:
        weights: dict[int, float] = {}
        for feature, count in Counter(ngrams(text)).items():
            index = bucket(feature, self.dim)
            weights[index] = weights.get(index, 0.0) + 1.0 + math.log(count)
        norm = math.sqrt(math.fsum(weight * weight for weight in weights.values()))
        return {index: weights[index] / norm for index in sorted(weights)}


def build_embedder(settings: Settings) -> Embedder:
    """The offline embedder with settings.embed_dim buckets."""
    return OfflineEmbedder(settings.embed_dim)
