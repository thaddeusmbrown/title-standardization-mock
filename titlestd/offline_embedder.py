"""embedder = offline: hashed character n-grams, standard library only.

Owner: feat/model-switch-offline (stage 1).
Governed by: docs/design.md 1.5 (off the shelf, not trained), 2.5;
docs/build-decisions.md Q12.

The method (Q12): case-fold; split into [a-z0-9]+ tokens; wrap each as
"<token>"; features are all 3-, 4- and 5-character n-grams of the wrapped
token; weight 1 + ln(count); bucket = an 8-byte hashlib.blake2b digest taken
modulo settings.embed_dim; L2-normalise. Vectors are sparse (contracts.Vector).
There are no fitted weights, so a vector never changes as approvals are added,
and the same text gives the same vector in every process and on every machine.

It must be a different calculation from the word-overlap stand-in (Q12).
"""

from __future__ import annotations

from titlestd.contracts import Embedder
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.offline_embedder is built by feat/model-switch-offline"


def build_embedder(settings: Settings) -> Embedder:
    """The offline embedder with settings.embed_dim buckets."""
    raise NotImplementedError(_NOT_BUILT)
