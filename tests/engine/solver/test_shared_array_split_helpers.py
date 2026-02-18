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


def test_get_shm_name_distinct_and_short():
    # Same-day run ids share a long common prefix; names must still differ
    # (the old 8-char truncation collapsed these to one namespace).
    a = get_shm_name("sas_reg", "run-20260718_135615-ac0c83")
    b = get_shm_name("sas_reg", "run-20260718_140703-d90aac")
    assert a != b
    assert a.startswith("sas_reg_")
    # macOS caps POSIX shm names at 31 chars including the leading slash.
    assert len(get_shm_name("sas_reach", "run-20260718_135615-ac0c83")) <= 30


def test_get_shm_name_deterministic():
    # Master and workers derive names independently from the same session id.
    assert get_shm_name("sas_reg", "some-session") == get_shm_name("sas_reg", "some-session")
