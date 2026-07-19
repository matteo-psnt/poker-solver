"""The columnar key table must round-trip exactly and shard consistently.

This format exists so a worker can load only the rows it owns: the pickled
key→id map and action-signature dicts were all-or-nothing, forcing every worker
to materialize the whole checkpoint (~28 GB per worker at 18.9M keys, ~444 GB
across 16) to keep its ~1/N shard.

The properties that matter are therefore (a) nothing is lost or reordered in the
round trip, since row index *is* the infoset id, and (b) the union of every
worker's shard is exactly the whole table, with no key in two shards -- a
partition that disagrees with ``stable_hash`` would silently drop updates.
"""

import numpy as np
import pytest

from src.core.game.actions import Action, ActionType
from src.core.game.state import Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage import key_table
from src.engine.solver.storage.shared_array.ownership import stable_hash


def _key(i: int) -> InfoSetKey:
    """Mix preflop and postflop rows so both encodings are exercised."""
    if i % 4 == 0:
        return InfoSetKey(
            player_position=i % 2,
            street=Street.PREFLOP,
            betting_sequence=f"b{i % 7}-c",
            preflop_hand=["AKs", "72o", "TT", "QJs"][i % 4],
            postflop_bucket=None,
            spr_bucket=i % 3,
        )
    return InfoSetKey(
        player_position=i % 2,
        street=Street(2 + i % 3),
        betting_sequence=f"b{i % 11}-c-r{i % 5}",
        preflop_hand=None,
        postflop_bucket=i % 97,
        spr_bucket=i % 3,
    )


def _actions(i: int) -> list[Action]:
    return [Action(ActionType.FOLD, 0), Action(ActionType.BET, 10 + i % 13)][: 1 + i % 2]


@pytest.fixture
def table(tmp_path):
    keys = [_key(i) for i in range(200)]
    actions = [_actions(i) for i in range(200)]
    key_table.write_key_table(
        tmp_path / "keys",
        keys=keys,
        key_hashes=[stable_hash(k) for k in keys],
        action_lists=actions,
    )
    return tmp_path / "keys", keys, actions


def test_round_trips_keys_and_actions_in_row_order(table):
    """Row index is the infoset id, so order is part of the contract."""
    table_dir, keys, actions = table
    rows = key_table.read_all_rows(table_dir)

    assert rows.keys == keys
    assert rows.action_lists == actions
    assert rows.row_ids.tolist() == list(range(len(keys)))


def test_row_count_matches_without_reading_columns(table):
    table_dir, keys, _ = table
    assert key_table.num_rows(table_dir) == len(keys)


@pytest.mark.parametrize("num_workers", [1, 2, 3, 4, 8, 16])
def test_shards_partition_the_table_exactly(table, num_workers):
    """Union of shards == whole table, and no key appears twice."""
    table_dir, keys, _ = table

    seen: list[InfoSetKey] = []
    for worker_id in range(num_workers):
        shard = key_table.read_owned_rows(table_dir, num_workers, worker_id)
        seen.extend(shard.keys)
        # Each shard's rows must be exactly what stable_hash assigns to it.
        for key in shard.keys:
            assert stable_hash(key) % num_workers == worker_id

    assert len(seen) == len(keys), "shards do not cover the table exactly once"
    assert set(seen) == set(keys)


def test_shard_row_ids_index_the_original_rows(table):
    """row_ids must stay usable as checkpoint array indices."""
    table_dir, keys, actions = table
    shard = key_table.read_owned_rows(table_dir, num_workers=4, worker_id=2)

    for key, row, acts in zip(shard.keys, shard.row_ids.tolist(), shard.action_lists):
        assert keys[row] == key
        assert actions[row] == acts


def test_single_worker_owns_everything(table):
    table_dir, keys, _ = table
    shard = key_table.read_owned_rows(table_dir, num_workers=1, worker_id=0)
    assert shard.keys == keys


def test_missing_action_list_is_rejected(tmp_path):
    """A row with no actions was a KeyError in the dict format; keep it an error."""
    keys = [_key(0), _key(1)]
    key_table.write_key_table(
        tmp_path / "keys",
        keys=keys,
        key_hashes=[stable_hash(k) for k in keys],
        action_lists=[_actions(0), None],
    )
    with pytest.raises(ValueError, match="missing action signatures"):
        key_table.read_all_rows(tmp_path / "keys")


def test_writer_rejects_misaligned_inputs(tmp_path):
    keys = [_key(0), _key(1)]
    with pytest.raises(ValueError, match="disagree"):
        key_table.write_key_table(
            tmp_path / "keys",
            keys=keys,
            key_hashes=[stable_hash(keys[0])],
            action_lists=[_actions(0), _actions(1)],
        )


def test_hash_column_matches_stable_hash(table):
    """Ownership is decided on this column alone; it must equal the live function."""
    table_dir, keys, _ = table
    stored = np.load(table_dir / "key_hash.npy")
    assert stored.tolist() == [stable_hash(k) for k in keys]
