"""Tests for the duplicate-deal resolver-vs-blueprint match harness."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.core.game.actions import call, check
from src.core.game.state import Card, Street
from src.pipeline.evaluation.estimators.resolver_match import (
    _allin_payoff,
    _chunk_bounds,
    _remaining_deck,
    complete_board,
    deal_from_stack,
    play_resolver_match,
)
from tests.test_helpers import build_trained_test_solver

if TYPE_CHECKING:
    from src.engine.solver.mccfr.static_solver import StaticTreeSolver


def _build_solver(iterations: int) -> StaticTreeSolver:
    return build_trained_test_solver(iterations)


# How many CFR iterations one resolver decision gets. See
# `TestParallelDealsAreTheSameExperiment` for why it is a count and not a clock.
RESOLVER_ITERATIONS = 200


def _pinned_solver(
    iterations: int, resolver_iterations: int = RESOLVER_ITERATIONS
) -> StaticTreeSolver:
    """A blueprint whose resolver stops on an iteration count, not a stopwatch.

    Module-level so the worker processes can pickle it as a factory. The depth
    is an argument because only the reproducibility test needs a solve worth
    comparing; a structural test wants it cheap.
    """
    return build_trained_test_solver(iterations, **{"resolver.max_iterations": resolver_iterations})


class TestDuplicateDealing:
    """Board cards must come off fixed deck positions, street by street."""

    def test_streets_deal_from_fixed_positions(self):
        solver = _build_solver(0)
        state = solver.deal_initial_state()
        # Limp, then the big blind checks its option: the flop chance node.
        state = state.apply_action(call(), solver.rules)
        state = state.apply_action(check(), solver.rules)
        assert solver.is_chance_node(state)

        deck = Card.get_full_deck()
        known = {c.mask for hand in state.hole_cards for c in hand}
        stack = [c for c in deck if c.mask not in known][:5]

        flop_state = deal_from_stack(state, stack)
        assert flop_state.street == Street.FLOP
        assert flop_state.board == tuple(stack[:3])
        assert not solver.is_chance_node(flop_state)


class TestResolverMatch:
    """End-to-end: the gate runs, pairs deals, and reports sane statistics."""

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_match_runs_and_reports(self):
        solver = _build_solver(3)
        result = play_resolver_match(solver, num_deals=2, time_budget_ms=20, seed=5)

        assert result.num_deals == 2
        assert result.num_hands == 4
        assert len(result.pair_samples_mbb) == 2
        assert result.resolver_decisions > 0
        assert result.resolver_fallbacks <= result.resolver_decisions
        assert np.isfinite(result.resolver_mbb_per_hand)
        assert np.isfinite(result.se_mbb)
        lo, hi = result.confidence_95_mbb
        assert lo <= result.resolver_mbb_per_hand <= hi


class TestParallelDealsAreTheSameExperiment:
    """Splitting deals across processes must not change a single number.

    MEASURED why this exists: 100 deals took 62 minutes single-threaded on a
    16-vCPU node, and se_mbb=1078 there puts ~50 mbb resolution near 46,500
    deals -- ~480 node-hours serially. Parallelism is what makes the gate
    runnable, and it is only allowed because every deal is a pure function of
    ``(seed, deal)``. This test is what keeps that true.
    """

    DEALS = 4
    SEED = 5
    # The budget is an ITERATION count, not a clock. Under `time_budget_ms` the
    # parallel arm -- three processes on the same cores -- gets through fewer
    # solve iterations than the serial one in the same 20ms, and lands on a
    # different `resolver_decisions`. Every SCORE still matched exactly, so the
    # test failed about once in three on the one number that measured how busy
    # the machine was. `resolver.py` names this as the way out: "pass
    # `config.max_iterations` to remove the wall-clock variability".
    #
    # The clock is left far above the iteration cost on purpose, so it is never
    # the binding constraint and cannot creep back in as the flake.
    BUDGET_MS = 60_000

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_three_workers_reproduce_the_serial_numbers_exactly(self):
        serial = play_resolver_match(
            _pinned_solver(3),
            num_deals=self.DEALS,
            time_budget_ms=self.BUDGET_MS,
            seed=self.SEED,
        )
        parallel = play_resolver_match(
            _pinned_solver(3),
            num_deals=self.DEALS,
            time_budget_ms=self.BUDGET_MS,
            seed=self.SEED,
            workers=3,
            blueprint_factory=functools.partial(_pinned_solver, 3),
        )

        assert parallel.pair_samples_mbb == serial.pair_samples_mbb
        assert parallel.resolver_mbb_per_hand == serial.resolver_mbb_per_hand
        assert parallel.se_mbb == serial.se_mbb
        assert parallel.resolver_decisions == serial.resolver_decisions
        assert parallel.resolver_fallbacks == serial.resolver_fallbacks
        # Not vacuous: a match that never resolved would agree trivially.
        assert serial.resolver_decisions > 0

    def test_without_a_factory_it_stays_serial_rather_than_failing(self):
        """`workers` alone cannot split: a solver cannot cross a process boundary."""
        result = play_resolver_match(
            # Pinned like its neighbour, and shallow: `BUDGET_MS` is a ceiling
            # the iteration count reaches first, so an unpinned resolver would
            # sit under it for the full minute. Nothing here reads a number --
            # the claim is that eight workers without a factory stay serial.
            _pinned_solver(3, resolver_iterations=5),
            num_deals=self.DEALS,
            time_budget_ms=self.BUDGET_MS,
            seed=self.SEED,
            workers=8,
        )
        assert result.num_deals == self.DEALS

    def test_chunks_cover_every_deal_exactly_once(self):
        for num_deals, workers in ((100, 16), (5, 16), (1, 4), (17, 3)):
            bounds = _chunk_bounds(num_deals, workers)
            covered = [deal for start, stop in bounds for deal in range(start, stop)]
            assert covered == list(range(num_deals)), (num_deals, workers)


class TestAllInRunoutAveraging:
    """Averaging an all-in board must move the VARIANCE and not the expectation.

    The measured per-deal spread after duplicate-deal pairing was 9.2 BB, which
    puts ~32,000 deals between us and a 100 mbb effect. Averaging over board
    completions at all-in terminals is the standard fix (LBR already does it
    with `allin_runouts`), and it is only sound if it leaves the expectation
    alone -- which is what these pin.
    """

    BOARD = (Card.new("Kh"), Card.new("8d"), Card.new("3c"), Card.new("Qs"))

    def _state(self, solver):
        """An all-in state one card short of a complete board.

        The stack comes from the solver's OWN config: a hand-picked 20 went
        negative against this config's 50/100 blinds, which is the engine
        correctly refusing a fixture that could not occur in play.
        """
        rules = solver.rules
        state = rules.create_initial_state(
            starting_stack=solver.config.game.starting_stack,
            hole_cards=((Card.new("As"), Card.new("Ad")), (Card.new("7h"), Card.new("7c"))),
            button=0,
        )
        # Engine-built, then only the BOARD and street moved: hand-setting pot
        # and stacks produced a state the engine rejects, and a fixture that
        # violates a production invariant tests nothing about production.
        # TURN, not RIVER: a 4-card board IS the turn, and the engine rejects
        # the mismatch. Completing it to a 5-card river is the thing under test.
        return state.replace(street=Street.TURN, board=self.BOARD, to_call=0)

    def test_one_runout_is_the_dealt_board_unchanged(self):
        """The default must reproduce the shipped number exactly, not approximately."""
        solver = _build_solver(50)
        rules = solver.rules
        state = self._state(solver)
        stack = [*self.BOARD, Card.new("2d")]
        dealt = complete_board(state, stack).get_payoff(0, rules)
        averaged = _allin_payoff(state, rules, stack, 0, runouts=1, rng=None)
        assert averaged == pytest.approx(float(dealt))

    def test_a_one_card_runout_is_enumerated_exactly(self):
        """44 completions is cheap, so this is the exact expectation -- zero
        variance -- rather than a sample of it. Checked against an independent
        enumeration, not against the implementation's own arithmetic."""
        solver = _build_solver(50)
        rules = solver.rules
        state = self._state(solver)
        deck = _remaining_deck(state)
        expected = sum(
            float(
                state.replace(
                    street=Street.RIVER, board=(*self.BOARD, card), is_terminal=True, to_call=0
                ).get_payoff(0, rules)
            )
            for card in deck
        ) / len(deck)

        got = _allin_payoff(
            state, rules, [*self.BOARD, deck[0]], 0, runouts=8, rng=np.random.default_rng(0)
        )
        assert got == pytest.approx(expected)
        # Exactness is the claim: a different rng must not move it at all.
        again = _allin_payoff(
            state, rules, [*self.BOARD, deck[0]], 0, runouts=8, rng=np.random.default_rng(99)
        )
        assert again == pytest.approx(got)

    def test_sampling_without_a_generator_is_refused(self):
        """An irreproducible eval number is worse than no eval number."""
        solver = _build_solver(50)
        rules = solver.rules
        state = self._state(solver).replace(board=(), street=Street.PREFLOP, validate=False)
        with pytest.raises(ValueError, match="reproducible"):
            _allin_payoff(state, rules, [], 0, runouts=16, rng=None)
