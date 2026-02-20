"""Training must release the session's bootstrap storage before claiming its name.

``TrainingSession.__init__`` builds a single-worker coordinator ``SharedArrayStorage``
so ``resume`` can verify the checkpoint loads. ``train_partitioned`` then builds its
own coordinator storage under the *same* ``session_id``, and that constructor's
``cleanup_stale_shm`` unlinks the bootstrap segments regardless. Left implicit, the
bootstrap storage keeps mapping unlinked memory for the rest of the run -- a full
capacity-sized allocation nobody ever frees (GBs at production capacity), plus, on
resume, the checkpoint it loaded into them.

Fixtures are deliberately local rather than imported from ``test_resume``: these
assertions are about storage lifetime, and should not break when the resume tests'
config changes.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.core.game.state import Street
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.pipeline.training import components
from src.pipeline.training.trainer.session import TrainingSession
from src.shared.config import Config
from tests.test_helpers import DummyCardAbstraction


@pytest.fixture
def temp_run_dir():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def mock_card_abstraction(monkeypatch):
    """Keep these tests off persisted abstraction artifacts."""
    monkeypatch.setattr(
        components, "build_card_abstraction", lambda *_a, **_k: DummyCardAbstraction()
    )
    monkeypatch.setattr(
        components, "resolve_card_abstraction_hash", lambda config: "dummy-abstraction-hash"
    )


@pytest.fixture
def test_config(temp_run_dir):
    return Config.from_dict(
        {
            "system": {"config_name": "test_bootstrap_release", "seed": 42},
            "game": {"starting_stack": 200, "small_blind": 1, "big_blind": 2},
            "action_model": {
                "preflop_templates": {"sb_first_in": ["fold", "call", 2.0]},
                "postflop_templates": {"first_aggressive": [1.0]},
                "jam_spr_threshold": 2.0,
            },
            "resolver": {"max_raises_per_street": 2},
            "card_abstraction": {"config": "default"},
            "training": {
                "runs_dir": str(temp_run_dir.parent),
                "num_iterations": 50,
                "checkpoint_frequency": 100,
                "verbose": False,
            },
        }
    )


def _key(i: int) -> InfoSetKey:
    return InfoSetKey(
        player_position=i % 2,
        street=Street.FLOP,
        betting_sequence=f"b{i}",
        preflop_hand=None,
        postflop_bucket=i,
        spr_bucket=0,
    )


def _live_segments(session: TrainingSession) -> list[str]:
    """Shared-memory handles the session's bootstrap storage still holds."""
    return [
        spec.shm_attr
        for spec in ARRAY_SPECS
        if getattr(session.storage.state, spec.shm_attr) is not None
    ]


def test_bootstrap_storage_is_live_before_training(test_config, temp_run_dir):
    """Guards the test below: it must be observing a real release, not a no-op."""
    session = TrainingSession(test_config, run_id=temp_run_dir.name)
    assert _live_segments(session) == [spec.shm_attr for spec in ARRAY_SPECS]
    session.release_bootstrap_storage()


def test_training_releases_the_bootstrap_storage(test_config, temp_run_dir):
    session = TrainingSession(test_config, run_id=temp_run_dir.name)
    session.train(num_iterations=1, num_workers=1)
    assert _live_segments(session) == [], "bootstrap segments leaked past training start"


def test_release_drops_the_key_dicts_before_the_fork(test_config, temp_run_dir):
    """The COW storm, not the shared memory, is what OOMs a resume.

    With num_workers=1 this storage owns every key, so on resume owned_keys holds
    one entry per infoset (~289 bytes each: ~3 GB at 10.6M infosets, ~5 GB at 18M).
    Workers are forked and CPython refcounting touches inherited pages, so every
    child copies it. Freeing only the shared memory leaves that heap in place.
    """
    session = TrainingSession(test_config, run_id=temp_run_dir.name)

    # A fresh session's dicts are empty, so stand in for the resume case by
    # populating them; otherwise this test passes with or without the fix.
    state = session.storage.state
    state.owned_keys = {_key(i): i for i in range(64)}
    state.legal_actions_cache = {i: () for i in range(64)}
    state.remote_keys = {_key(i): i for i in range(64, 96)}
    state.requested_id_keys = {_key(i) for i in range(96, 100)}
    assert state.owned_keys, "guard: the dicts must be populated for this to mean anything"

    session.train(num_iterations=1, num_workers=1)

    assert state.owned_keys == {}
    assert state.remote_keys == {}
    assert state.legal_actions_cache == {}
    assert state.requested_id_keys == set()


def test_release_is_idempotent(test_config, temp_run_dir):
    """train() may be entered more than once on one session."""
    session = TrainingSession(test_config, run_id=temp_run_dir.name)
    session.release_bootstrap_storage()
    session.release_bootstrap_storage()
    assert _live_segments(session) == []
