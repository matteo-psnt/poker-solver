"""Tests for extracted shared-array helper modules."""

from src.core.game.state import Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.shared_array.ownership import owner_for_key, stable_hash
from src.engine.solver.storage.shared_array_layout import get_shm_name


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


def test_get_shm_name_formats_session():
    assert get_shm_name("sas_reg", "abcd1234") == "sas_reg_abcd1234"
