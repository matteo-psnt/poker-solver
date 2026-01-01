import uuid

import numpy as np

from src.core.game.actions import bet, call, fold
from src.core.game.state import Street
from src.engine.solver.storage.in_memory import InMemoryStorage
from src.engine.solver.storage.shared_array import SharedArrayStorage
from src.pipeline.abstraction.utils.infoset import InfoSetKey


def _action_signature(actions):
    return [(action.type.name, action.amount) for action in actions]


def test_checkpoint_roundtrip_smoke(tmp_path):
    session_id = f"test_{uuid.uuid4().hex[:8]}"
    storage = SharedArrayStorage(
        num_workers=1,
        worker_id=0,
        session_id=session_id,
        initial_capacity=64,
        max_actions=5,
        is_coordinator=True,
        checkpoint_dir=tmp_path,
    )

    try:
        key_one = InfoSetKey(
            player_position=0,
            street=Street.PREFLOP,
            betting_sequence="",
            preflop_hand="AKs",
            postflop_bucket=None,
            spr_bucket=1,
        )
        actions_one = [fold(), call(), bet(50)]
        infoset_one = storage.get_or_create_infoset(key_one, actions_one)
        infoset_one.regrets[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        infoset_one.strategy_sum[:] = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        infoset_one.increment_reach_count(5)
        infoset_one.add_cumulative_utility(12.5)

        key_two = InfoSetKey(
            player_position=1,
            street=Street.PREFLOP,
            betting_sequence="r2.0",
            preflop_hand="72o",
            postflop_bucket=None,
            spr_bucket=2,
        )
        actions_two = [fold(), call()]
        infoset_two = storage.get_or_create_infoset(key_two, actions_two)
        infoset_two.regrets[:] = np.array([0.5, -1.25], dtype=np.float32)
        infoset_two.strategy_sum[:] = np.array([3.0, 7.0], dtype=np.float64)
        infoset_two.increment_reach_count(2)
        infoset_two.add_cumulative_utility(-4.0)

        storage.checkpoint(iteration=1)
    finally:
        storage.cleanup()

    loaded = InMemoryStorage(checkpoint_dir=tmp_path)
    assert loaded.num_infosets() == 2

    loaded_one = loaded.get_infoset(key_one)
    assert loaded_one is not None
    np.testing.assert_allclose(loaded_one.regrets, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(loaded_one.strategy_sum, [10.0, 20.0, 30.0])
    assert loaded_one.reach_count == 5
    assert loaded_one.cumulative_utility == 12.5
    assert _action_signature(loaded_one.legal_actions) == _action_signature(actions_one)

    loaded_two = loaded.get_infoset(key_two)
    assert loaded_two is not None
    np.testing.assert_allclose(loaded_two.regrets, [0.5, -1.25])
    np.testing.assert_allclose(loaded_two.strategy_sum, [3.0, 7.0])
    assert loaded_two.reach_count == 2
    assert loaded_two.cumulative_utility == -4.0
    assert _action_signature(loaded_two.legal_actions) == _action_signature(actions_two)
