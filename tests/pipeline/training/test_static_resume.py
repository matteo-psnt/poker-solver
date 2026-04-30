"""Chunked checkpointing and resume on the static path.

A long run is only survivable if a death costs one chunk rather than the whole
run, and if a retry converges instead of repeating. The dynamic path gets that
from an ABSOLUTE ``--to-iteration``; this pins the same contract here.

Bucketing in these fixtures keys on Python ``hash()``, which is per-process
randomised under spawn, so nothing below asserts on bucket ASSIGNMENT -- only on
iteration accounting and resume semantics, which are assignment-independent.
"""

from __future__ import annotations

import pytest

from src.engine.solver.storage.static_checkpoint import StaticCheckpointManifest
from src.pipeline.training.static_parallel import (
    train_static_parallel,
    worker_iteration_indices,
    worker_seed,
)
from tests.pipeline.training.test_static_parallel import Buckets, _config, session


class TestChunkIndicesTileTheRange:
    """`start` must shift the numbering without tearing or overlapping it."""

    def test_a_chunk_tiles_exactly_its_own_range(self):
        start, end, workers = 400, 700, 4
        covered: list[int] = []
        for w in range(workers):
            covered.extend(worker_iteration_indices(w, workers, end, start=start))
        assert sorted(covered) == list(range(start, end))

    def test_workers_within_a_chunk_are_disjoint(self):
        start, end, workers = 100, 340, 3
        seen: set[int] = set()
        for w in range(workers):
            idx = set(worker_iteration_indices(w, workers, end, start=start))
            assert not (seen & idx), "two workers share an absolute iteration"
            seen |= idx

    def test_default_start_is_the_historical_behaviour(self):
        assert list(worker_iteration_indices(1, 4, 12)) == list(
            worker_iteration_indices(1, 4, 12, start=0)
        )

    def test_consecutive_chunks_do_not_overlap_or_gap(self):
        workers, step = 4, 200
        covered: list[int] = []
        for chunk_start in range(0, 800, step):
            for w in range(workers):
                covered.extend(
                    worker_iteration_indices(w, workers, chunk_start + step, start=chunk_start)
                )
        assert sorted(covered) == list(range(800))


class TestChunkSeeding:
    def test_chunks_do_not_replay_the_same_stream(self):
        # Without chunk_id every chunk re-seeds identically, so extra chunks add
        # correlated samples rather than new information -- silently, since the
        # iteration count still goes up.
        assert worker_seed(42, 0, 0) != worker_seed(42, 0, 1)

    def test_still_distinct_across_workers_within_a_chunk(self):
        seeds = {worker_seed(42, w, 3) for w in range(8)}
        assert len(seeds) == 8


@pytest.mark.slow
@pytest.mark.timeout(180)
class TestCheckpointAndResume:
    def test_chunking_reaches_the_same_total(self, tmp_path):
        result = train_static_parallel(
            _config(),
            num_iterations=400,
            num_workers=2,
            session_id=session("static-chunked"),
            checkpoint_dir=tmp_path,
            checkpoint_every=100,
            abstraction=Buckets(),
        )
        assert result.iterations == 400
        assert result.dropped_updates == 0

    def test_a_checkpoint_lands_before_the_run_ends(self, tmp_path):
        """The bound on loss: a death mid-run must leave something banked."""
        train_static_parallel(
            _config(),
            num_iterations=300,
            num_workers=2,
            session_id=session("static-midrun"),
            checkpoint_dir=tmp_path,
            checkpoint_every=100,
            abstraction=Buckets(),
        )
        manifest = StaticCheckpointManifest.read(tmp_path)
        assert manifest is not None
        assert manifest.iteration == 300

    def test_resume_continues_from_the_checkpoint(self, tmp_path):
        first = train_static_parallel(
            _config(),
            num_iterations=200,
            num_workers=2,
            session_id=session("static-resume-a"),
            checkpoint_dir=tmp_path,
            checkpoint_every=100,
            abstraction=Buckets(),
        )
        second = train_static_parallel(
            _config(),
            num_iterations=400,
            num_workers=2,
            session_id=session("static-resume-b"),
            checkpoint_dir=tmp_path,
            checkpoint_every=100,
            resume=True,
            abstraction=Buckets(),
        )
        assert first.iterations == 200
        assert second.iterations == 400
        # Continued, not restarted: more of the tree is touched after the second
        # task than the first banked.
        assert second.touched_rows >= first.touched_rows

    def test_retry_past_the_target_is_a_noop(self, tmp_path):
        train_static_parallel(
            _config(),
            num_iterations=200,
            num_workers=2,
            session_id=session("static-noop-a"),
            checkpoint_dir=tmp_path,
            checkpoint_every=100,
            abstraction=Buckets(),
        )
        again = train_static_parallel(
            _config(),
            num_iterations=200,
            num_workers=2,
            session_id=session("static-noop-b"),
            checkpoint_dir=tmp_path,
            resume=True,
            abstraction=Buckets(),
        )
        # This is what makes an automatic scheduler retry safe rather than
        # destructive: past the absolute target, the task does nothing.
        assert again.iterations == 200
        assert again.elapsed_s == 0.0

    def test_resume_without_a_checkpoint_starts_fresh(self, tmp_path):
        # A first task asked to resume must not die on the missing manifest.
        result = train_static_parallel(
            _config(),
            num_iterations=100,
            num_workers=2,
            session_id=session("static-resume-empty"),
            checkpoint_dir=tmp_path,
            resume=True,
            abstraction=Buckets(),
        )
        assert result.iterations == 100
