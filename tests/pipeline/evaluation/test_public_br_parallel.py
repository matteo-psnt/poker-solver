"""Parallel exact-BR must agree with serial exactly, not approximately.

Exact BR is the zero-variance instrument: its whole value is that two
checkpoints under one board plan are exactly paired, so a difference is pure
signal. A parallel path that shifted the number even slightly would destroy
that property while still looking plausible.
"""

from __future__ import annotations

import pytest

from src.pipeline.evaluation.public_tree_br import PublicBRConfig


class TestConfigContract:
    def test_defaults_to_serial(self):
        # Parallel needs a factory; defaulting to >1 would make every existing
        # caller fail rather than simply not speed up.
        assert PublicBRConfig().num_workers == 1

    def test_board_plan_still_defines_the_tier(self):
        """num_workers must NOT be part of what makes two evals comparable.

        It changes only how the same four walks are scheduled, so treating it as
        a tier knob would refuse to pair a serial eval with a parallel one that
        computed exactly the same number.
        """
        from src.pipeline.evaluation import ledger as eval_ledger

        knobs = eval_ledger.build_exact_br_knobs_from_params(
            num_flops=4, num_turns=2, num_rivers=2, board_seed=7
        )

        assert "num_workers" not in knobs
        # base_seed mirrors board_seed so the shared pairing guard applies.
        assert {"num_flops", "num_turns", "num_rivers", "base_seed"} <= set(knobs)


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestParallelMatchesSerial:
    def _blueprint(self, tmp_path):
        from src.pipeline.services.evaluation import build_blueprint_for
        from src.pipeline.services.runs import load_run_metadata
        from src.pipeline.services.static_training import train_static

        out = train_static(
            "quick_test", num_workers=2, num_iterations=4000, seed=7, runs_dir=tmp_path
        )
        run_dir = tmp_path / out.run_id
        metadata = load_run_metadata(run_dir)
        solver, _ = build_blueprint_for(run_dir, metadata, metadata.card_abstraction_hash, None)
        return solver, metadata, run_dir

    def _score(self, solver, metadata, run_dir, workers: int) -> tuple[float, int]:
        import functools

        from src.pipeline.evaluation.public_tree_br import compute_public_tree_br
        from src.pipeline.services.evaluation import _load_blueprint

        config = PublicBRConfig(num_flops=1, num_turns=1, num_rivers=1, num_workers=workers)
        factory = (
            functools.partial(
                _load_blueprint,
                metadata.config,
                run_dir,
                metadata.card_abstraction_hash,
                None,
            )
            if workers > 1
            else None
        )
        result = compute_public_tree_br(
            solver,
            config,
            starting_stack=metadata.config.game.starting_stack,
            blueprint_factory=factory,
        )
        return result.exploitability_mbb, result.nodes_visited

    def test_same_value_and_same_node_count(self, tmp_path):
        # ONE blueprint, scored two ways. Training twice would compare two
        # different blueprints and measure training variance, not scheduling.
        solver, metadata, run_dir = self._blueprint(tmp_path)
        serial, serial_nodes = self._score(solver, metadata, run_dir, 1)
        parallel, parallel_nodes = self._score(solver, metadata, run_dir, 4)
        # EXACT, not approximate: the walks are deterministic and disjoint, so
        # any drift means work was duplicated, dropped, or reordered.
        assert parallel == serial
        assert parallel_nodes == serial_nodes
