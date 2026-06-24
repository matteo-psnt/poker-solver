"""Tests for the deterministic public-tree exact best response.

The load-bearing check is scalar equivalence: the vectorized engine must
reproduce, per hero combo, the exact ``best_response_value`` of the
:class:`RestrictedHUNL` reference game — the same betting tree, board plan,
and annulled measure walked one scalar state at a time. That pins the measure
handling (public chance weights, card-removal masking, annulment, fold and
showdown valuation) against the generic BR implementation already validated
on Kuhn/Leduc. Property tests add BR-dominates-on-policy and determinism.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from src.core.game.state import Card
from src.engine.search.range_inference import NUM_COMBOS, combo_index_for
from src.pipeline.evaluation.estimators.public_tree_br import (
    PublicBRConfig,
    PublicTreeBestResponse,
    compute_public_tree_br,
    transform_policy_rows,
)
from src.pipeline.evaluation.reference.best_response import best_response_value, on_policy_value
from tests.pipeline.evaluation.restricted_hunl import RestrictedHUNL, blueprint_policy
from tests.test_helpers import build_trained_test_solver

CONFIG = PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=3)
STACK = 400
_DEAL = {Card.new("As"), Card.new("Kd"), Card.new("7c"), Card.new("2h")}
_FLOP = tuple(Card.new(c) for c in ("Qs", "Jh", "3d"))


def _combo(a: str, b: str) -> tuple[Card, Card]:
    return (Card.new(a), Card.new(b))


OPP_COMBOS = [
    _combo("Ah", "Kh"),
    _combo("Ac", "Qd"),
    _combo("Kd", "Kc"),
    _combo("Qs", "Qh"),
    _combo("Js", "Tc"),
    _combo("Td", "9c"),
    _combo("9s", "8s"),
    _combo("8h", "7d"),
    _combo("6c", "6d"),
    _combo("5h", "4h"),
    _combo("4c", "3s"),
    _combo("2h", "2c"),
    _combo("Ad", "5s"),
    _combo("Kh", "8c"),
    _combo("Qc", "7h"),
    _combo("3d", "2s"),
]
HERO_COMBOS = [_combo("As", "Ks"), _combo("7c", "2d"), _combo("Th", "9h")]


@pytest.fixture(scope="module")
def trained_solver():
    return build_trained_test_solver(4, starting_stack=STACK)


@pytest.fixture(scope="module")
def uniform_solver():
    return build_trained_test_solver(0, starting_stack=STACK)


def _assert_engine_matches_scalar(solver, hero_seat: int, button: int) -> None:
    engine = PublicTreeBestResponse(solver, CONFIG, starting_stack=STACK)
    for hero_combo in HERO_COMBOS:
        game = RestrictedHUNL(
            solver,
            engine._plan,
            hero_seat=hero_seat,
            hero_combo=hero_combo,
            opp_combos=OPP_COMBOS,
            button=button,
            starting_stack=STACK,
        )
        scalar = best_response_value(game, hero_seat, blueprint_policy(solver))
        reach = np.zeros(NUM_COMBOS)
        for combo in game.opp_combos:
            reach[combo_index_for(combo)] = 1.0
        values = engine.responder_values(hero_seat, button, reach)
        vectorized = values[combo_index_for(hero_combo)] / len(game.opp_combos)
        assert vectorized == pytest.approx(scalar, abs=1e-9)


@pytest.mark.timeout(30)
class TestScalarEquivalence:
    def test_uniform_blueprint_seat0(self, uniform_solver):
        _assert_engine_matches_scalar(uniform_solver, hero_seat=0, button=0)

    def test_uniform_blueprint_seat1(self, uniform_solver):
        _assert_engine_matches_scalar(uniform_solver, hero_seat=1, button=0)

    def test_trained_blueprint_seat0(self, trained_solver):
        _assert_engine_matches_scalar(trained_solver, hero_seat=0, button=0)

    def test_trained_blueprint_seat1_button1(self, trained_solver):
        _assert_engine_matches_scalar(trained_solver, hero_seat=1, button=1)


@pytest.mark.timeout(30)
def test_br_dominates_on_policy(trained_solver):
    """Per hero combo, exact BR value >= the blueprint's own on-policy value."""
    engine = PublicTreeBestResponse(trained_solver, CONFIG, starting_stack=STACK)
    policy = blueprint_policy(trained_solver)
    for hero_seat, hero_combo in ((0, HERO_COMBOS[0]), (1, HERO_COMBOS[2])):
        game = RestrictedHUNL(
            trained_solver,
            engine._plan,
            hero_seat=hero_seat,
            hero_combo=hero_combo,
            opp_combos=OPP_COMBOS,
            button=0,
            starting_stack=STACK,
            full_state_keys=True,
        )
        on_policy = on_policy_value(game, hero_seat, policy)
        reach = np.zeros(NUM_COMBOS)
        for combo in game.opp_combos:
            reach[combo_index_for(combo)] = 1.0
        values = engine.responder_values(hero_seat, 0, reach)
        best = values[combo_index_for(hero_combo)] / len(game.opp_combos)
        assert best >= on_policy - 1e-9


@pytest.mark.timeout(30)
def test_deterministic_and_nonnegative(trained_solver):
    result_a = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
    result_b = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
    assert result_a.exploitability_mbb == result_b.exploitability_mbb
    assert [r.value_mbb for r in result_a.seat_results] == [
        r.value_mbb for r in result_b.seat_results
    ]
    assert result_a.exploitability_mbb >= -1e-6
    assert result_a.nodes_visited > 0
    assert 0.0 <= result_a.missing_policy_mass <= 1.0


@pytest.mark.timeout(30)
def test_untrained_blueprint_is_all_fallback(uniform_solver):
    result = compute_public_tree_br(uniform_solver, CONFIG, starting_stack=STACK)
    assert result.missing_policy_mass == pytest.approx(1.0)
    assert result.exploitability_mbb >= -1e-6


class TestConditionalChance:
    """The annulled measure voids a branch a deal collides with, and a void pays
    zero -- the whole hand refunded -- so every postflop terminal is worth less
    than a preflop fold by the street's compatible fraction. The conditional
    plan divides that fraction back out; at full enumeration the branch
    weights then sum to exactly one over the boards a deal can reach."""

    def _compatible_mass(self, plan, board) -> float:
        return sum(w for cards, w in plan.deal_options(board) if not _DEAL & set(cards))

    @pytest.mark.timeout(30)
    def test_weights_sum_to_one_over_compatible_boards(self):
        from src.pipeline.evaluation.estimators.public_tree_br import _BoardPlan

        plan = _BoardPlan(
            PublicBRConfig(num_flops=1755, num_turns=49, num_rivers=48, conditional_chance=True)
        )
        # A canonical flop stands for its whole suit class, so "compatible with
        # this deal" is only defined per class in expectation: the total is
        # scaled so a deal's C(48,3)/C(52,3) share of it is exactly one.
        total = sum(w for _, w in plan.deal_options(()))
        assert total * math.comb(48, 3) / math.comb(52, 3) == pytest.approx(1.0)
        assert self._compatible_mass(plan, _FLOP) == pytest.approx(1.0)
        assert self._compatible_mass(plan, (*_FLOP, Card.new("9c"))) == pytest.approx(1.0)

    @pytest.mark.timeout(30)
    def test_annulled_weights_sum_below_one(self):
        from src.pipeline.evaluation.estimators.public_tree_br import _BoardPlan

        plan = _BoardPlan(PublicBRConfig(num_flops=1, num_turns=49, num_rivers=48))
        assert self._compatible_mass(plan, _FLOP) == pytest.approx(45 / 49)
        # The RIVER's own fraction, not the turn's. The two differ by 2%, and
        # applying one street's factor at the other's depth is a silent 2% on
        # the correction the conditional ladder's limit is read off.
        river = self._compatible_mass(plan, (*_FLOP, Card.new("9c")))
        assert river == pytest.approx(44 / 48)

    @pytest.mark.timeout(60)
    def test_it_is_a_different_game(self, trained_solver):
        annulled = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
        conditional = compute_public_tree_br(
            trained_solver,
            PublicBRConfig(
                num_flops=2, num_turns=1, num_rivers=1, board_seed=3, conditional_chance=True
            ),
            starting_stack=STACK,
        )
        assert conditional.exploitability_mbb != annulled.exploitability_mbb
        assert conditional.exploitability_mbb >= 0.0


class TestBranchProgress:
    """The bar an evaluation reports, and the one thing it may not cost.

    Scoring one checkpoint is ~10 minutes in which nothing about the process is
    observable from outside it. Flop branches are the outermost thing the walk
    counts — four walks of `--br-flops` each — so counting them is free, where a
    counter in the node recursion would not be.
    """

    def _engine(self, solver, on_branch=None):
        return PublicTreeBestResponse(solver, CONFIG, starting_stack=STACK, on_branch=on_branch)

    def test_progress_only_moves_forward(self, uniform_solver):
        seen: list[tuple[int, int]] = []
        engine = self._engine(uniform_solver, lambda done, total: seen.append((done, total)))
        engine.evaluate()

        assert seen, "nothing reported at all"
        assert [d for d, _ in seen] == sorted(d for d, _ in seen), "done went backwards"
        assert len({t for _, t in seen}) == 1, "the total must not move under the bar"

    def test_the_denominator_is_known_before_anything_is_walked(self, uniform_solver):
        """It used to be MEASURED, so the first report was the honest 1-of-4 and
        everything before it was blind. Under `--workers` — which is every
        evaluation on the pool — that was fatal: the four walks that could
        measure it all finish together."""
        engine = self._engine(uniform_solver, lambda done, total: None)
        assert engine.branch_total > 0

        seen: list[tuple[int, int]] = []
        watched = self._engine(uniform_solver, lambda done, total: seen.append((done, total)))
        watched.evaluate()
        assert seen[0][0] == 1, "the FIRST branch reports, not the first walk"

    def test_it_finishes_on_its_own_total(self, uniform_solver):
        """A branch skipped for having no reach is still a branch DONE — the
        earlier `continue` would have left the bar permanently short."""
        seen: list[tuple[int, int]] = []
        engine = self._engine(uniform_solver, lambda done, total: seen.append((done, total)))
        engine.evaluate()

        done, total = seen[-1]
        assert done == total == engine.branch_total

    def test_the_denominator_is_far_bigger_than_the_rung_it_replaced(self, uniform_solver):
        """The point of the change, stated as a number: this is a tiny 2-flop
        config and it still beats `1 rung` by an order of magnitude."""
        engine = self._engine(uniform_solver, lambda done, total: None)
        engine.evaluate()
        assert engine.branch_total >= 8

    def test_the_value_is_identical_with_and_without_a_bar(self, uniform_solver):
        """A progress hook that changes the number it measures is worthless."""
        watched = self._engine(uniform_solver, lambda done, total: None).evaluate()
        plain = self._engine(uniform_solver).evaluate()
        assert watched.exploitability_mbb == plain.exploitability_mbb


@pytest.mark.slow
@pytest.mark.timeout(300)
class TestBranchProgressAcrossProcesses:
    """The bar the NODE actually draws, which is not the one above.

    Every evaluation on the pool runs `--workers 16`, so it takes the parallel
    path, and there the walks are in other processes. Reporting as each walk
    RETURNED gave four steps that all land within seconds of the end: measured
    on the pool, a 573-second score sat at 0% for the whole of it, which is
    indistinguishable from a hung one -- the exact thing a bar exists to rule
    out.
    """

    PARALLEL = PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=3, num_workers=4)

    def test_the_walks_report_while_they_are_still_walking(self, uniform_solver):
        """Each walk reports from its FIRST branch, not on the way out — on any
        node, with nothing remembered from an earlier evaluation."""
        seen: list[tuple[int, int]] = []
        result = compute_public_tree_br(
            uniform_solver,
            self.PARALLEL,
            starting_stack=STACK,
            blueprint_factory=partial(build_trained_test_solver, 0, starting_stack=STACK),
            on_branch=lambda done, total: seen.append((done, total)),
        )

        assert result.exploitability_mbb is not None
        # A quarter of the total is ONE walk, so anything strictly below it can
        # only have come from a walk that had not finished.
        one_walk = seen[-1][1] // 4
        assert any(0 < done < one_walk for done, _ in seen), (
            f"nothing reported mid-walk, only {sorted({d for d, _ in seen})}"
        )
        assert [d for d, _ in seen] == sorted(d for d, _ in seen), "done went backwards"
        assert seen[-1][0] == seen[-1][1], "the bar must finish on its own total"


class TestTheCountedDenominator:
    """The count is READ off the betting tree, not measured by walking it.

    That is what lets the bar have a denominator before anything has been
    walked, which under `--workers` is the only moment it could get one: the
    four walks that could measure it all finish together.

    The count is structural and the walk is not — it PRUNES an action the
    blueprint never takes — so the two agreeing is the property that matters,
    and it is the one thing a structural count can silently get wrong.
    """

    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("iterations", [0, 4])
    @pytest.mark.parametrize("stack", [200, 400])
    def test_the_count_is_exactly_what_the_walk_reaches(self, iterations, stack):
        """The guard on the whole idea. A structural count that disagrees with
        the walk leaves a bar that stops short or runs past its end, and neither
        fails anything by itself."""
        solver = build_trained_test_solver(iterations, starting_stack=stack)
        engine = PublicTreeBestResponse(solver, CONFIG, starting_stack=stack)
        predicted = engine.branch_total

        engine.evaluate()

        assert predicted > 0
        assert engine._branches_done == predicted

    @pytest.mark.timeout(30)
    def test_an_action_the_blueprint_never_takes_still_counts(self, trained_solver, monkeypatch):
        """A pruned subtree is work that will NOT happen, so its branches are
        DONE. Counted any other way the bar stops short of a denominator read
        off the tree, which cannot know the policy.

        Preflop CALL specifically, and the credit is asserted rather than only
        the total: FOLD and ALL_IN have no flop deal under them, so pruning
        either one credits nothing and would make this test vacuous. Killing
        CALL loses 8 of these 16 branches — a bar that stops at 50%.
        """
        engine = PublicTreeBestResponse(trained_solver, CONFIG, starting_stack=STACK)
        policy, credit = engine._policy_matrix, engine._skip_branches
        credited: list[int] = []

        def never_call(state, legal):
            sigma, missing = policy(state, legal)
            if not state.board and len(legal) > 1:
                sigma = sigma.copy()
                sigma[:, 0] += sigma[:, 1]  # onto FOLD, which deals no flop
                sigma[:, 1] = 0.0
            return sigma, missing

        monkeypatch.setattr(engine, "_policy_matrix", never_call)
        monkeypatch.setattr(engine, "_skip_branches", lambda n: credited.append(n) or credit(n))
        engine.evaluate()

        assert sum(credited) > 0, "nothing was pruned, so this proves nothing"
        assert engine._branches_done == engine.branch_total


def _reach_for(game: RestrictedHUNL) -> np.ndarray:
    reach = np.zeros(NUM_COMBOS)
    for combo in game.opp_combos:
        reach[combo_index_for(combo)] = 1.0
    return reach


@pytest.mark.timeout(60)
class TestDecomposition:
    """The gain attribution is exact, not approximate.

    Self-play values are checked against the reference ``on_policy_value`` on
    the scalar game the way the BR values are, and the per-node terms must sum
    to ``BR - self-play`` walk by walk: prefix reach under the blueprint,
    suffix values under the best response, every term non-negative.
    """

    DECOMPOSED = PublicBRConfig(
        num_flops=2, num_turns=1, num_rivers=1, board_seed=3, decompose=True
    )

    @pytest.mark.parametrize(("hero_seat", "button"), [(0, 0), (1, 1)])
    def test_self_play_values_match_the_scalar_oracle(self, trained_solver, hero_seat, button):
        engine = PublicTreeBestResponse(trained_solver, CONFIG, starting_stack=STACK)
        policy = blueprint_policy(trained_solver)
        for hero_combo in HERO_COMBOS:
            game = RestrictedHUNL(
                trained_solver,
                engine._plan,
                hero_seat=hero_seat,
                hero_combo=hero_combo,
                opp_combos=OPP_COMBOS,
                button=button,
                starting_stack=STACK,
                full_state_keys=True,
            )
            scalar = on_policy_value(game, hero_seat, policy)
            _, self_play = engine.walk_values(hero_seat, button, _reach_for(game))
            vectorized = self_play[combo_index_for(hero_combo)] / len(game.opp_combos)
            assert vectorized == pytest.approx(scalar, abs=1e-9)

    @pytest.mark.parametrize("conditional", [False, True], ids=["annulled", "conditional"])
    def test_terms_sum_to_the_gain_on_every_walk(self, trained_solver, conditional):
        """Under BOTH chance measures: the conditional one rescales every
        street's weights, and the identity must survive that rescaling."""
        config = replace(self.DECOMPOSED, conditional_chance=conditional)
        result = compute_public_tree_br(trained_solver, config, starting_stack=STACK)
        decomposition = result.decomposition
        assert decomposition is not None
        for walk in decomposition["identity"]["per_walk"]:
            assert walk["gain_mbb"] == pytest.approx(walk["terms_mbb"], abs=1e-9)
            assert walk["gain_mbb"] == pytest.approx(
                walk["br_value_mbb"] - walk["self_play_mbb"], abs=1e-9
            )
        assert decomposition["identity"]["max_abs_gap_mbb"] < 1e-9
        assert sum(decomposition["by_street"].values()) == pytest.approx(
            result.exploitability_mbb, abs=1e-9
        )

    def test_self_play_is_zero_sum_across_seats_at_one_button(self, trained_solver):
        result = compute_public_tree_br(trained_solver, self.DECOMPOSED, starting_stack=STACK)
        by_button: dict[int, float] = {}
        for seat in result.seat_results:
            by_button[seat.button] = by_button.get(seat.button, 0.0) + seat.self_play_mbb
        for total in by_button.values():
            assert total == pytest.approx(0.0, abs=1e-9)

    def test_every_term_is_nonnegative_and_the_number_is_unchanged(self, trained_solver):
        plain = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
        decomposed = compute_public_tree_br(trained_solver, self.DECOMPOSED, starting_stack=STACK)
        assert decomposed.exploitability_mbb == plain.exploitability_mbb
        decomposition = decomposed.decomposition
        assert decomposition is not None
        assert all(v >= -1e-12 for v in decomposition["by_street"].values())
        assert decomposition["top_nodes"], "no responder node recorded a term"
        for node in decomposition["top_nodes"]:
            assert node["gain_mbb"] >= -1e-12
            assert sum(node["blueprint"].values()) == pytest.approx(1.0, abs=1e-9)
            assert sum(node["best_response"].values()) == pytest.approx(1.0, abs=1e-9)
        assert decomposition["nodes_with_gain"] > 0


@pytest.mark.timeout(60)
class TestInAbstraction:
    """A responder that must act per (node, bucket) is never stronger than one
    told its cards, and with one bucket for a hero set it IS the card-blind
    best response the scalar oracle computes."""

    CONSTRAINED = PublicBRConfig(
        num_flops=2, num_turns=1, num_rivers=1, board_seed=3, in_abstraction=True
    )

    def test_bucket_constrained_never_exceeds_per_combo(self, trained_solver):
        free = PublicTreeBestResponse(trained_solver, CONFIG, starting_stack=STACK)
        bound = PublicTreeBestResponse(trained_solver, self.CONSTRAINED, starting_stack=STACK)
        reach = np.ones(NUM_COMBOS)
        for hero_seat, button in ((0, 0), (1, 0)):
            unconstrained = free.responder_values(hero_seat, button, reach)
            constrained = bound.responder_values(hero_seat, button, reach)
            assert np.all(constrained <= unconstrained + 1e-9)
            assert constrained.sum() < unconstrained.sum(), "the constraint never bound"

    @pytest.mark.parametrize(("hero_seat", "button"), [(0, 0), (1, 1)])
    def test_one_bucket_for_the_hero_set_is_the_card_blind_oracle(
        self, trained_solver, hero_seat, button, monkeypatch
    ):
        """Hero combos share bucket 0 on every board, every other combo sits in
        bucket 1: the constrained responder must pick one action per public
        node for the set, which is exactly a hero dealt from that set whose
        information key carries no cards."""
        engine = PublicTreeBestResponse(trained_solver, self.CONSTRAINED, starting_stack=STACK)
        hero_set = {combo_index_for(combo) for combo in HERO_COMBOS}
        real_buckets = engine._bucket_vector

        def coarse(board, street):
            vector = np.where(real_buckets(board, street) >= 0, 1, -1)
            for index in hero_set:
                if vector[index] >= 0:
                    vector[index] = 0
            return vector

        monkeypatch.setattr(engine, "_responder_buckets", coarse)
        game = RestrictedHUNL(
            trained_solver,
            engine._plan,
            hero_seat=hero_seat,
            hero_combos=HERO_COMBOS,
            opp_combos=OPP_COMBOS,
            button=button,
            starting_stack=STACK,
            hero_public_key=True,
        )
        assert len(game.opp_combos) == len(OPP_COMBOS), "fixture: a hero blocks an opponent"
        scalar = best_response_value(game, hero_seat, blueprint_policy(trained_solver))
        values = engine.responder_values(hero_seat, button, _reach_for(game))
        vectorized = sum(values[index] for index in hero_set) / (
            len(hero_set) * len(game.opp_combos)
        )
        assert vectorized == pytest.approx(scalar, abs=1e-9)

    @pytest.mark.parametrize("conditional", [False, True], ids=["annulled", "conditional"])
    def test_decomposition_still_sums_under_the_constraint(self, trained_solver, conditional):
        config = PublicBRConfig(
            num_flops=2,
            num_turns=1,
            num_rivers=1,
            board_seed=3,
            in_abstraction=True,
            decompose=True,
            conditional_chance=conditional,
        )
        result = compute_public_tree_br(trained_solver, config, starting_stack=STACK)
        assert result.decomposition is not None
        assert result.decomposition["identity"]["max_abs_gap_mbb"] < 1e-9


@pytest.mark.timeout(60)
class TestPolicyTransforms:
    """Eval-time purification and thresholding reshape the fielded rows only."""

    ROWS = np.array(
        [
            [0.50, 0.30, 0.15, 0.05],
            [0.97, 0.01, 0.01, 0.01],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    MISSING = np.array([False, False, False, True])

    def test_thresholding_zeroes_the_tail_and_renormalises(self):
        out = transform_policy_rows(self.ROWS, self.MISSING, PublicBRConfig(policy_threshold=0.1))
        assert np.allclose(out[0], np.array([0.50, 0.30, 0.15, 0.0]) / 0.95)
        assert np.allclose(out[1], [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(out[2], 0.25), "nothing below the cut, nothing changes"
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_a_row_cut_to_nothing_falls_back_to_its_argmax(self):
        out = transform_policy_rows(self.ROWS, self.MISSING, PublicBRConfig(policy_threshold=0.3))
        assert np.allclose(out[2], [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(out[0], [0.5 / 0.8, 0.3 / 0.8, 0.0, 0.0])

    def test_purification_is_the_argmax(self):
        out = transform_policy_rows(self.ROWS, self.MISSING, PublicBRConfig(purify=True))
        assert np.array_equal(out[:3].max(axis=1), [1.0, 1.0, 1.0])
        assert np.array_equal(out[0], [1.0, 0.0, 0.0, 0.0])

    def test_fallback_rows_are_untouched_and_the_input_is_not_mutated(self):
        before = self.ROWS.copy()
        for config in (PublicBRConfig(purify=True), PublicBRConfig(policy_threshold=0.2)):
            out = transform_policy_rows(self.ROWS, self.MISSING, config)
            assert np.allclose(out[3], 0.25)
        assert np.array_equal(self.ROWS, before)

    def test_no_transform_returns_the_rows_themselves(self):
        assert transform_policy_rows(self.ROWS, self.MISSING, PublicBRConfig()) is self.ROWS

    def test_the_transformed_number_is_deterministic_and_different(self, trained_solver):
        purified = PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=3, purify=True)
        first = compute_public_tree_br(trained_solver, purified, starting_stack=STACK)
        second = compute_public_tree_br(trained_solver, purified, starting_stack=STACK)
        plain = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
        assert first.exploitability_mbb == second.exploitability_mbb
        assert first.exploitability_mbb != plain.exploitability_mbb
