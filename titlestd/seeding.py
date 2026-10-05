"""Seeded randomness: every random use builds its own generator (Q06).

A draw depends only on the run seed, a stream name and an item key, never on
the order in which titles are processed or on another draw. That is what lets
a recorded prompt repeat exactly (Q18).

Rules for the whole package, checked by tests/unit/test_seeding.py:
- this is the only module that imports random;
- nothing calls the built-in hash(), whose value changes between processes;
- nothing lets the iteration order of a set reach an output: sort first.

The stream names are fixed here so two branches cannot pick the same one.
"""

from __future__ import annotations

import hashlib
import json
import random

# One stream per random use. The item key of a per-title stream is the two
# fields of its TitleKey: family code, then cleaned title.
STREAM_SHUFFLE = "shuffle"  # order of the ten roles shown to the LLM, per title (Q18)
STREAM_GOLDEN = "golden"  # the golden-set draw, no item key (Q08)
STREAM_AUDIT = "audit"  # the audit sample, item key: the machine path's name (2.4)
STREAM_SME_ERROR = "sme_error"  # whether, and how, a simulated label is wrong, per title (Q27)
STREAM_UNDECIDED = "undecided"  # whether a simulated reviewer cannot decide, per title (Q01)

STREAMS = (STREAM_SHUFFLE, STREAM_GOLDEN, STREAM_AUDIT, STREAM_SME_ERROR, STREAM_UNDECIDED)


def digest(seed: int, stream: str, *key: str) -> bytes:
    """SHA-256 of the seed, the stream name and the item key.

    The hashed bytes are the compact UTF-8 JSON array [seed, stream, *key], so
    ("a", "bc") and ("ab", "c") never collide.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}")
    if not isinstance(stream, str) or not stream:
        raise TypeError(f"stream must be a non-empty str, got {stream!r}")
    for part in key:
        if not isinstance(part, str):
            raise TypeError(f"item key parts must be str, got {part!r}")
    payload = json.dumps([seed, stream, *key], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def rng(seed: int, stream: str, *key: str) -> random.Random:
    """A fresh generator for one random use, for example one title's shuffle."""
    return random.Random(int.from_bytes(digest(seed, stream, *key), "big"))


def unit(seed: int, stream: str, *key: str) -> float:
    """One number in [0, 1) for a rate test such as "wrong with probability p".

    Taken straight from the digest: draw < rate decides the event.
    """
    return (int.from_bytes(digest(seed, stream, *key)[:7], "big") >> 3) / float(1 << 53)
