"""Resume must be bit-identical to an uninterrupted run at the same batch grid.

The scattered resume tests check that state is *restored* (iteration counter,
infoset count). They do not check the stronger property a long, repeatedly
resumed run actually depends on: that splitting training at a batch boundary and
resuming lands on the *same learned policy* as training straight through.

This is the regression guard for the resume RNG/iteration hazard seen in the 25M
postmortem: the deal stream is reseeded per batch from the batch's ABSOLUTE
starting iteration (``_compute_seed`` in parallel_worker.py). If resume ever
restarted that counter at 0 (a per-leg counter), the resumed batch would replay
the opening deal stream and this test would diverge, even though every existing
restore-state assertion still passes.

The batch grid must line up for bit-identity to be the right bar: each batch
draws an independent SeedSequence stream, so ``iterations_per_worker`` is pinned
to the split point. Then the uninterrupted run batches at {0, split} and the
interrupted run runs batch{0} then, on resume, batch{split} -- the same two
streams. A mismatched grid would still be *correct* (disjoint streams, no
replay) but not byte-equal, which is why the split is aligned deliberately.

Seeded, single-worker, dummy abstraction so the run is deterministic and cheap.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.pipeline.training import components
from src.pipeline.training.trainer import TrainingSession
from src.pipeline.training.versioning import checkpoint_fingerprint
from src.shared.config import Config
from tests.test_helpers import DummyCardAbstraction


@pytest.fixture
def runs_parent():
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(autouse=True)
def mock_card_abstraction(monkeypatch):
    """Dummy abstraction so the round-trip doesn't depend on persisted artifacts."""
    monkeypatch.setattr(
        components, "build_card_abstraction", lambda *a, **k: DummyCardAbstraction()
    )
    monkeypatch.setattr(
        components, "resolve_card_abstraction_hash", lambda config: "dummy-abstraction-hash"
    )


def _config(runs_parent: Path, batch_iters: int) -> Config:
    return Config.from_dict(
        {
            "system": {"config_name": "resume_roundtrip", "seed": 42},
            "game": {"starting_stack": 200, "small_blind": 1, "big_blind": 2},
            "card_abstraction": {"config": "default"},
            "training": {
                "runs_dir": str(runs_parent),
                # Batch grid == the split point, so the uninterrupted and resumed
                # runs draw the same per-batch (absolute-iteration) deal streams.
                "iterations_per_worker": batch_iters,
                # Checkpoint only at the end of each train() call, not mid-run.
                "checkpoint_frequency": 10_000,
                "verbose": False,
            },
        }
    )


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_resume_is_bit_identical_to_uninterrupted(runs_parent):
    """train(N) -> resume -> train(M) must equal train(N+M) byte-for-byte."""
    n_first, n_second = 20, 20

    # Uninterrupted reference: N+M iterations in a single session, batched at N.
    uninterrupted = TrainingSession(_config(runs_parent, n_first), run_id="uninterrupted")
    uninterrupted.train(num_iterations=n_first + n_second, num_workers=1)
    fp_reference = checkpoint_fingerprint(runs_parent / "uninterrupted")

    # Interrupted: train N, drop the session, resume from checkpoint, train M.
    first = TrainingSession(_config(runs_parent, n_first), run_id="interrupted")
    first.train(num_iterations=n_first, num_workers=1)
    fp_midpoint = checkpoint_fingerprint(runs_parent / "interrupted")
    del first

    resumed = TrainingSession.resume(runs_parent / "interrupted")
    assert resumed.solver.iteration == n_first, (
        f"resume restored the wrong iteration counter: {resumed.solver.iteration} != {n_first}"
    )
    resumed.train(num_iterations=n_second, num_workers=1)
    fp_resumed = checkpoint_fingerprint(runs_parent / "interrupted")

    # Sanity: the midpoint checkpoint must differ from the final one, otherwise
    # the second training leg was a no-op and the equality below is vacuous.
    assert fp_midpoint != fp_reference, "midpoint == final: second leg trained nothing"

    assert fp_resumed == fp_reference, (
        "resume is NOT equivalent to an uninterrupted run: "
        f"interrupted fingerprint {fp_resumed[:16]} != reference {fp_reference[:16]}. "
        "A resumed long run would drift from the policy it would have learned unbroken."
    )


@pytest.mark.slow
@pytest.mark.timeout(180)
def test_repeated_resume_does_not_drift(runs_parent):
    """Many resume legs (the Modal guillotine case) must still equal one run.

    A single resume being correct does not prove that resuming dozens of times
    compounds without drift -- an off-by-one in the batch grid would accumulate.
    Train the same total in K legs with a fresh session per leg and require the
    final checkpoint to match the uninterrupted reference byte-for-byte.
    """
    leg = 15
    num_legs = 4
    total = leg * num_legs

    reference = TrainingSession(_config(runs_parent, leg), run_id="reference")
    reference.train(num_iterations=total, num_workers=1)
    fp_reference = checkpoint_fingerprint(runs_parent / "reference")

    chunked = TrainingSession(_config(runs_parent, leg), run_id="chunked")
    chunked.train(num_iterations=leg, num_workers=1)
    del chunked
    for expected_start in range(leg, total, leg):
        session = TrainingSession.resume(runs_parent / "chunked")
        assert session.solver.iteration == expected_start, (
            f"resume leg started at {session.solver.iteration}, expected {expected_start}"
        )
        session.train(num_iterations=leg, num_workers=1)
        del session

    fp_chunked = checkpoint_fingerprint(runs_parent / "chunked")
    assert fp_chunked == fp_reference, (
        f"{num_legs}-leg resumed run drifted from the uninterrupted run: "
        f"{fp_chunked[:16]} != {fp_reference[:16]}"
    )
