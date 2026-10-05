"""embedder = local: a small local embedding model behind an optional extra.

Owner: feat/local-embedder (stage 2).
Governed by: docs/design.md 2.5 (lazy import; the default run installs
nothing); docs/build-decisions.md Q30 (library, model and revision), Q32 (how
it is tested).

The library is model2vec, from the optional extra "local", imported only
inside build_embedder. No PyTorch anywhere. The weights download on first use
into the user's Hugging Face cache, never into the repository. A missing extra
or a failed download stops with a clear message that holds the exact install
command, and never falls back to the offline embedder.

Vectors are returned as contracts.Vector, L2-normalised, like every embedder.
The smoke test is marked local_model and is never part of the merge gate.
"""

from __future__ import annotations

from titlestd.contracts import Embedder
from titlestd.settings import Settings

_NOT_BUILT = "titlestd.local_embedder is built by feat/local-embedder"

EXTRA = "local"
MODEL_NAME = "minishlab/potion-base-8M"
MODEL_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"


def build_embedder(settings: Settings) -> Embedder:
    """The local embedder. Raises TitlestdError when the extra is not installed
    or the model cannot be loaded."""
    raise NotImplementedError(_NOT_BUILT)
