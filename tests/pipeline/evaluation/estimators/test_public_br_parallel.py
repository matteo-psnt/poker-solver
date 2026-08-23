"""Parallel exact-BR must agree with serial exactly, not approximately.

Exact BR is the zero-variance instrument: its whole value is that two
checkpoints under one board plan are exactly paired, so a difference is pure
signal. A parallel path that shifted the number even slightly would destroy
that property while still looking plausible.
"""

from __future__ import annotations

from functools import partial

import pytest

from src.pipeline.evaluation.estimators.public_tree_br import (
    PublicBRConfig,
    compute_public_tree_br,
)
from tests.test_helpers import build_trained_test_solver

STACK = 400


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


@pytest.mark.timeout(120)
class TestForkJoinMatchesSerial:
    """The fork-join splits BELOW the preflop max, where the walk is a weighted
    sum, and joins in the serial order -- so the number must be bit-identical,
    not close. Two checkpoints scored at different worker counts are still
    exactly paired only if this holds. No abstraction needed: the test solver
    is seeded, so the factory rebuilds the same blueprint in every worker."""

    @pytest.mark.parametrize("iterations", [0, 4])
    def test_same_number_same_telemetry(self, iterations):
        solver = build_trained_test_solver(iterations, starting_stack=STACK)
        factory = partial(build_trained_test_solver, iterations, starting_stack=STACK)

        def tier(workers: int) -> PublicBRConfig:
            return PublicBRConfig(
                num_flops=2, num_turns=1, num_rivers=1, board_seed=3, num_workers=workers
            )

        serial = compute_public_tree_br(solver, tier(1), starting_stack=STACK)
        forked = compute_public_tree_br(
            solver, tier(3), starting_stack=STACK, blueprint_factory=factory
        )

        assert forked.exploitability_mbb == serial.exploitability_mbb
        assert forked.seat_results == serial.seat_results
        assert forked.nodes_visited == serial.nodes_visited
        assert forked.missing_policy_mass == serial.missing_policy_mass


@pytest.mark.slow
@pytest.mark.timeout(600)
class TestParallelMatchesSerial:
    def _blueprint(self, tmp_path):
        from src.pipeline.services.runs import load_run_metadata
        from src.pipeline.services.scoring import build_blueprint_for
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

        from src.pipeline.evaluation.estimators.public_tree_br import compute_public_tree_br
        from src.pipeline.services.scoring._shared import load_blueprint

        config = PublicBRConfig(num_flops=1, num_turns=1, num_rivers=1, num_workers=workers)
        factory = (
            functools.partial(
                load_blueprint,
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

    def test_same_value_and_same_node_count(self, tmp_path, requires_card_abstraction):
        # ONE blueprint, scored two ways. Training twice would compare two
        # different blueprints and measure training variance, not scheduling.
        solver, metadata, run_dir = self._blueprint(tmp_path)
        serial, serial_nodes = self._score(solver, metadata, run_dir, 1)
        parallel, parallel_nodes = self._score(solver, metadata, run_dir, 4)
        # EXACT, not approximate: the walks are deterministic and disjoint, so
        # any drift means work was duplicated, dropped, or reordered.
        assert parallel == serial
        assert parallel_nodes == serial_nodes


class TestResponderAndTransformKnobs:
    """A constrained responder or a transformed blueprint is a different number,
    so it must split the tier -- and only when set, so old rows keep theirs."""

    def test_defaults_add_nothing(self):
        from src.pipeline.evaluation import ledger as eval_ledger

        knobs = eval_ledger.build_exact_br_knobs_from_params(
            num_flops=4, num_turns=2, num_rivers=2, board_seed=7
        )
        assert set(knobs) == {"num_flops", "num_turns", "num_rivers", "base_seed"}

    def test_each_variant_splits_the_tier(self):
        from src.pipeline.evaluation import ledger as eval_ledger
        from src.pipeline.evaluation.ledger.tiers import tier_key

        def knobs_with(**variant):
            return eval_ledger.build_exact_br_knobs_from_params(
                num_flops=4, num_turns=2, num_rivers=2, board_seed=7, **variant
            )

        plain = {"method": "exact_br", "knobs": knobs_with()}
        variants = [
            knobs_with(in_abstraction=True),
            knobs_with(policy_threshold=0.05),
            knobs_with(purify=True),
        ]
        keys = {tier_key({"method": "exact_br", "knobs": knobs}) for knobs in variants}
        assert len(keys) == 3
        assert tier_key(plain) not in keys

    def test_purify_overrides_threshold(self):
        from src.pipeline.evaluation import ledger as eval_ledger

        knobs = eval_ledger.build_exact_br_knobs_from_params(
            num_flops=4, num_turns=2, num_rivers=2, board_seed=7, purify=True, policy_threshold=0.1
        )
        assert knobs.get("purify") is True
        assert "policy_threshold" not in knobs
