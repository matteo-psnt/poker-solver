"""Ownership and stable-hashing helpers for shared-array storage."""

from __future__ import annotations

import xxhash

from src.engine.solver.infoset import InfoSetKey

# Every component of a key is drawn from a small domain -- 2 positions, 4 streets,
# a few thousand betting sequences, 169 preflop hands, a few hundred buckets, 3 SPRs
# -- while the number of keys is in the millions. Memoizing the encodings turns the
# per-key str()/encode() work into a dict lookup. The joined bytes, and therefore the
# digest, are unchanged; ownership of existing checkpoints is unaffected.
_TEXT_BYTES: dict[str, bytes] = {}
_INT_BYTES: dict[int, bytes] = {}


def stable_hash(key: InfoSetKey) -> int:
    """
    Compute stable hash of InfoSetKey using xxhash.

    Python's built-in hash() is randomized per process, which breaks
    ownership consistency across workers.
    """
    text = _TEXT_BYTES
    ints = _INT_BYTES

    position = key.player_position
    position_bytes = ints.get(position)
    if position_bytes is None:
        position_bytes = ints[position] = str(position).encode()

    street_name = key.street.name
    street_bytes = text.get(street_name)
    if street_bytes is None:
        street_bytes = text[street_name] = street_name.encode()

    sequence = key.betting_sequence
    sequence_bytes = text.get(sequence)
    if sequence_bytes is None:
        sequence_bytes = text[sequence] = sequence.encode()

    hand = key.preflop_hand or ""
    hand_bytes = text.get(hand)
    if hand_bytes is None:
        hand_bytes = text[hand] = hand.encode()

    bucket = key.postflop_bucket
    if bucket is None:
        bucket = -1
    bucket_bytes = ints.get(bucket)
    if bucket_bytes is None:
        bucket_bytes = ints[bucket] = str(bucket).encode()

    spr = key.spr_bucket
    spr_bytes = ints.get(spr)
    if spr_bytes is None:
        spr_bytes = ints[spr] = str(spr).encode()

    key_bytes = b"|".join(
        (position_bytes, street_bytes, sequence_bytes, hand_bytes, bucket_bytes, spr_bytes)
    )
    return xxhash.xxh64(key_bytes).intdigest()


def owner_for_key(key: InfoSetKey, num_workers: int) -> int:
    """Determine owner worker for an infoset key."""
    return stable_hash(key) % num_workers
