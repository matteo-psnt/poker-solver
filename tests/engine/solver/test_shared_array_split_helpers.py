"""Tests for extracted shared-array helper modules."""

from src.core.game.state import Street
from src.engine.solver.storage.shared_array.ownership import (
    owner_for_id,
    owner_for_key,
    stable_hash,
)
from src.engine.solver.storage.shared_array_layout import get_shm_name
from src.pipeline.abstraction.utils.infoset import InfoSetKey


def _sample_key() -> InfoSetKey:
    return InfoSetKey(
        player_position=0,
        street=Street.PREFLOP,
        betting_sequence="",
        preflop_hand="AKs",
        postflop_bucket=None,
        spr_bucket=2,
    )


def test_stable_hash_is_deterministic():
    key = _sample_key()
    assert stable_hash(key) == stable_hash(key)


def test_owner_for_key_with_two_workers():
    key = _sample_key()
    owner = owner_for_key(key, 2)
    assert owner in (0, 1)


def test_owner_for_id_uses_base_and_extra_regions():
    # Base region example
    assert (
        owner_for_id(
            1,
            unknown_id=0,
            base_slots_per_worker=10,
            num_workers=2,
            extra_regions=[],
        )
        == 0
    )
    assert (
        owner_for_id(
            11,
            unknown_id=0,
            base_slots_per_worker=10,
            num_workers=2,
            extra_regions=[],
        )
        == 1
    )

    # Extra region: start=21, total=3, base=1, remainder=1
    # Worker 0 gets first 2 ids (21,22), worker 1 gets next 1 id (23).
    assert (
        owner_for_id(
            22,
            unknown_id=0,
            base_slots_per_worker=10,
            num_workers=2,
            extra_regions=[(21, 3, 1, 1)],
        )
        == 0
    )
    assert (
        owner_for_id(
            23,
            unknown_id=0,
            base_slots_per_worker=10,
            num_workers=2,
            extra_regions=[(21, 3, 1, 1)],
        )
        == 1
    )


def test_get_shm_name_formats_session():
    assert get_shm_name("sas_reg", "abcd1234") == "sas_reg_abcd1234"
