"""The board-free leg's contract, which differs from a scalar leg in one way.

``universe_boards`` is not a tuning knob. The bucket-transition and showdown
matrices are estimated from those boards and they ARE the chance layer, so two
legs with different universes solve different games. Nothing downstream can
detect that: the tree fingerprint covers node layout and bucket counts, and the
abstraction hash covers bucket assignment; neither covers the matrices.

So the spec refuses a board-free leg that does not say what universe it means.
"""

from __future__ import annotations

import pytest

from src.interfaces.cloud import spec


def _leg(
    *,
    config: str = "production",
    run_id: str = "",
    to: int = 400,
    universe_boards: int = 2000,
    universe_seed: int = 7,
) -> spec.LegSpec:
    return spec.LegSpec(
        code_snapshot="snap-1",
        op=spec.TRAIN_VECTOR,
        config=config,
        run_id=run_id,
        to=to,
        universe_boards=universe_boards,
        universe_seed=universe_seed,
    )


class TestValidation:
    def test_a_board_free_leg_is_accepted(self):
        _leg().validate()

    def test_a_board_free_leg_must_name_its_universe(self):
        with pytest.raises(ValueError, match="universe-boards"):
            _leg(universe_boards=0).validate()

    def test_the_absolute_target_rule_applies_to_both_kernels(self):
        """Retry convergence is the reason ``to`` is absolute; it is not
        scalar-specific, so the same guard must cover the vector op."""
        with pytest.raises(ValueError, match="ABSOLUTE"):
            _leg(to=0).validate()

    def test_a_board_free_leg_still_needs_a_config_or_a_run(self):
        with pytest.raises(ValueError, match="--config"):
            _leg(config="", run_id="").validate()

    def test_a_scalar_leg_needs_no_universe(self):
        spec.LegSpec(code_snapshot="s", op=spec.TRAIN, config="production", to=1000).validate()


class TestEnvironment:
    def test_the_universe_reaches_the_node(self):
        env = _leg().environment()
        assert env["RUN_OP"] == "train-vector"
        assert env["RUN_UNIVERSE_BOARDS"] == "2000"
        assert env["RUN_UNIVERSE_SEED"] == "7"

    def test_a_scalar_leg_leaves_the_vector_keys_empty(self):
        """Empty, not absent: ``run_leg.sh`` tests every key with ``-n``, so an
        empty value is how a knob says 'not mine'."""
        env = spec.LegSpec(
            code_snapshot="s", op=spec.TRAIN, config="production", to=1000
        ).environment()
        assert env["RUN_UNIVERSE_BOARDS"] == ""
        assert env["RUN_UNIVERSE_SEED"] == ""
        assert env["RUN_DTYPE"] == ""

    def test_every_key_is_a_string(self):
        """Batch environment settings are strings; an int would fail at submit
        time on a node rather than here."""
        assert all(isinstance(v, str) for v in _leg().environment().values())
