"""The resolver gate must measure what the resolver was RETIRED on.

`--no-resolver` was fielded on 09-12 for two numbers, and this gate carried
neither: the resolver doubled the stack-off rate (5.25% vs 2.37% over ~5,500
decisions) while its chip edge read -203.6 +/- 133.1, i.e. nothing. In an
elimination format a stack-off that loses ends the match, so the rate is the
disqualifying number and mbb/hand is the one that cannot resolve -- se_mbb is
~80 at 4,000 deals, where a 5% rate carries ~0.2 points at the same cost.

The mechanism was diagnosed too: regret matching starts from zero regrets, so at
the fielded budget the root row measured 1.604 nats of a 1.609 maximum -- uniform
over {check, three sizes, all-in}, which ships stacks at random. `root_prior_weight`
is the built answer to that, and it had never been read on an instrument that
could see either quantity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from src.core.game.actions import Action, ActionType
from src.engine.search.resolver import _normalised_entropy
from src.engine.solver.policy.source import TreePolicySource
from src.pipeline.evaluation.estimators.blueprint_match import play_blueprint_match
from src.pipeline.evaluation.estimators.resolver_match import MatchTelemetry
from src.pipeline.evaluation.ledger import records, tiers
from src.pipeline.services.scoring import matches
from tests.test_helpers import build_trained_test_solver

if TYPE_CHECKING:
    from src.core.game.state import GameState
    from src.engine.solver.betting_tree import BettingTree


class TestAStackOffIsAShippedStack:
    """Counted off the stack the action LEFT, not off `Action.amount`."""

    @staticmethod
    def _tally(action: Action, stack_after: int, *, is_resolver: bool = True) -> MatchTelemetry:
        telemetry = MatchTelemetry()
        telemetry.note_action(action, stack_after, is_resolver=is_resolver)
        return telemetry

    def test_an_all_in_that_empties_the_stack_counts(self):
        telemetry = self._tally(Action(ActionType.ALL_IN, 200), 0)

        assert telemetry.resolver_stack_offs == 1
        assert telemetry.resolver_aggressions == 1

    def test_a_raise_that_leaves_chips_behind_is_an_aggression_only(self):
        telemetry = self._tally(Action(ActionType.RAISE, 50), 150)

        assert telemetry.resolver_stack_offs == 0
        assert telemetry.resolver_aggressions == 1

    def test_a_raise_that_happens_to_empty_the_stack_still_counts(self):
        """The wire is what matters: a size meant as a big bet can BE a stack-off."""
        telemetry = self._tally(Action(ActionType.RAISE, 200), 0)

        assert telemetry.resolver_stack_offs == 1

    def test_calling_an_all_in_is_not_a_stack_off(self):
        """The seat's own definition -- it is the aggressor who shipped, not the caller."""
        telemetry = self._tally(Action(ActionType.CALL), 0)

        assert telemetry.resolver_stack_offs == 0
        assert telemetry.resolver_aggressions == 0

    def test_folding_is_neither(self):
        telemetry = self._tally(Action(ActionType.FOLD), 200)

        assert telemetry.resolver_aggressions == 0

    def test_the_two_seats_are_counted_apart(self):
        """The cost was never the absolute rate -- it was DOUBLE the blueprint's."""
        telemetry = MatchTelemetry()
        telemetry.note_action(Action(ActionType.ALL_IN, 200), 0, is_resolver=True)
        telemetry.note_action(Action(ActionType.RAISE, 20), 180, is_resolver=False)

        assert (telemetry.resolver_stack_offs, telemetry.blueprint_stack_offs) == (1, 0)
        assert (telemetry.resolver_aggressions, telemetry.blueprint_aggressions) == (1, 1)


class TestRatesArePerAggression:
    """A per-DECISION rate moves with the fold rate, so two arms are not comparable."""

    def test_the_rate_divides_by_aggressions_not_decisions(self):
        telemetry = MatchTelemetry(resolver_stack_offs=1, resolver_aggressions=4, decisions=100)

        assert telemetry.resolver_stack_offs / telemetry.resolver_aggressions == 0.25


class TestAbsorbSumsEveryField:
    """A field added later must be aggregated without editing this."""

    def test_every_counter_is_summed(self):
        left = MatchTelemetry()
        right = MatchTelemetry()
        for index, name in enumerate(MatchTelemetry.__dataclass_fields__, start=1):
            setattr(left, name, index)
            setattr(right, name, index * 10)

        left.absorb(right)

        for index, name in enumerate(MatchTelemetry.__dataclass_fields__, start=1):
            assert getattr(left, name) == index * 11, name


class TestNormalisedEntropy:
    """1.0 is uniform whatever the menu size -- raw nats cannot be averaged."""

    def test_uniform_is_one(self):
        assert _normalised_entropy(np.full(6, 1 / 6)) == pytest.approx(1.0)

    def test_a_pure_row_is_zero(self):
        assert _normalised_entropy(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)

    def test_a_forced_action_is_decided_rather_than_undefined(self):
        """log(1) is zero, so a one-action menu must not divide by it."""
        assert _normalised_entropy(np.array([1.0])) == 0.0

    def test_two_menus_of_different_size_are_comparable(self):
        """The whole reason for normalising: uniform over 3 == uniform over 6."""
        assert _normalised_entropy(np.full(3, 1 / 3)) == pytest.approx(
            _normalised_entropy(np.full(6, 1 / 6))
        )

    def test_the_fielded_failure_reads_near_one(self):
        """1.604 nats of a 1.609 maximum, the diagnosed uniform root row."""
        five = np.full(5, 0.2)
        assert _normalised_entropy(five) > 0.99


class TestTheLedgerCanReadThemBack:
    """A metric that stays in the document is one no later reader can test.

    MEASURED: the first run of this gate recorded three stack-off ratios and
    `ledger` projected none of them, so the numbers existed only in a task log.
    The index row has to carry the rates AND their denominators.
    """

    @staticmethod
    def _document(**results: object) -> dict[str, object]:
        return {"run_id": "run-a", "method": "resolver_match", "results": dict(results)}

    def test_the_rates_and_their_counts_reach_the_index_row(self):
        row = records.ledger_row(
            self._document(
                resolver_mbb_per_hand=12.5,
                se_mbb=170.0,
                resolver_stack_off_rate=0.0892,
                blueprint_stack_off_rate=0.0838,
                resolver_stack_offs=178,
                blueprint_stack_offs=166,
                resolver_aggressions=1995,
                blueprint_aggressions=1981,
                resolver_root_entropy=0.478,
                played_root_entropy=0.408,
            )
        )

        assert row["results"]["resolver_stack_off_rate"] == 0.0892
        assert row["results"]["resolver_aggressions"] == 1995
        assert row["results"]["played_root_entropy"] == 0.408

    def test_a_document_without_them_projects_without_them(self):
        """Every row written before 09-30 has none of these keys, and `ledger
        --rebuild` regenerates its index row from the same document."""
        row = records.ledger_row(self._document(resolver_mbb_per_hand=12.5, se_mbb=170.0))

        assert "resolver_stack_off_rate" not in row["results"]
        assert row["results"]["edge_mbb_per_hand"] == 12.5


class TestADeployedSideA:
    """A chip match where side A plays blueprint+resolver, not the bare table.

    The resolver's claimed benefit is entirely OFF-TREE, and `resolver_match`
    plays a blueprint against ITSELF -- every line it walks is one its own tree
    has a node for, so no self-duel can ever see that benefit. This is the seam
    that lets a DIFFERENT action tree sit on the other side.
    """

    def test_the_bare_path_is_untouched(self):
        """A self-match with no resolver is still provably all-zero, and carries
        no telemetry -- so a plain match keeps the result shape it already had."""
        result = play_blueprint_match(
            build_trained_test_solver(40), build_trained_test_solver(40), num_deals=4, seed=3
        )

        assert result.telemetry is None
        assert all(sample == 0.0 for sample in result.pair_samples_mbb)

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_routing_side_a_through_the_resolver_changes_play_and_counts_it(self):
        result = play_blueprint_match(
            build_trained_test_solver(40),
            build_trained_test_solver(40),
            num_deals=4,
            seed=3,
            resolver_a=True,
        )

        assert result.telemetry is not None
        # Not vacuous: identical tables would pair to zero without the resolver,
        # so a non-zero pair is the resolver's decision and nothing else.
        assert any(sample != 0.0 for sample in result.pair_samples_mbb)
        assert result.telemetry.decisions > 0
        assert result.telemetry.blueprint_decisions > 0
        assert result.telemetry.entropy_decisions > 0

    def test_a_match_without_a_deployed_side_records_no_resolver_fields(self):
        """`_resolver_fields` keeps a plain match's results -- and so its tier -- as is."""
        assert matches._resolver_fields(None) == {}

    def test_a_deployed_side_records_the_rates_with_their_counts(self):
        telemetry = MatchTelemetry(
            decisions=10,
            blueprint_decisions=9,
            resolver_stack_offs=3,
            resolver_aggressions=6,
            blueprint_stack_offs=1,
            blueprint_aggressions=5,
            resolver_entropy_sum=2.0,
            played_entropy_sum=1.0,
            entropy_decisions=4,
        )

        fields = matches._resolver_fields(telemetry)

        assert fields["resolver_stack_off_rate"] == 0.5
        assert fields["blueprint_stack_off_rate"] == 0.2
        assert fields["resolver_root_entropy"] == 0.5
        assert fields["resolver_aggressions"] == 6


class TestADeployedRowSaysSo:
    """A row that does not name its player is a measurement nobody can identify.

    MEASURED: the first deployed chip match recorded its root entropy and its
    shove counts and NOTHING that said the resolver had been in the hand -- and
    its knob tier was identical to the bare match that chose the rung, so the two
    would have pooled as one instrument.
    """

    @staticmethod
    def _results(**extra: object) -> dict[str, object]:
        base = {"run_b": "run-b", "rung_b": 4600, "num_deals": 50, "seed": 1}
        return base | extra

    def test_a_deployed_arm_does_not_pair_with_a_bare_one(self):
        bare = tiers.build_blueprint_match_knobs(self._results(opponent="blueprint"))
        deployed = tiers.build_blueprint_match_knobs(
            self._results(opponent="deployed", resolver_max_iterations=64, root_prior_weight=200.0)
        )

        assert "opponent" not in bare
        assert deployed["opponent"] == "deployed"
        assert deployed["root_prior_weight"] == 200.0
        assert tiers.tier_digest({"method": "blueprint_match", "knobs": bare}) != tiers.tier_digest(
            {"method": "blueprint_match", "knobs": deployed}
        )

    def test_two_deployed_arms_at_different_priors_are_different_tiers(self):
        """The knob under test must split the tier, or an A/B pools into one row."""
        low, high = (
            tiers.build_blueprint_match_knobs(
                self._results(
                    opponent="deployed", resolver_max_iterations=64, root_prior_weight=prior
                )
            )
            for prior in (0.0, 200.0)
        )

        assert tiers.tier_digest({"method": "blueprint_match", "knobs": low}) != tiers.tier_digest(
            {"method": "blueprint_match", "knobs": high}
        )

    def test_the_label_names_the_player(self):
        knobs = tiers.build_blueprint_match_knobs(
            self._results(opponent="deployed", resolver_max_iterations=64, root_prior_weight=200.0)
        )

        label = tiers.tier_label({"method": "blueprint_match", "knobs": knobs})

        assert "opponent=deployed" in label
        assert "root_prior_weight=200.0" in label

    def test_the_index_row_carries_the_player(self):
        row = records.ledger_row(
            {
                "run_id": "run-a",
                "method": "blueprint_match",
                "results": {
                    "a_mbb_per_hand": 610.0,
                    "se_mbb": 493.6,
                    "opponent": "deployed",
                    "root_prior_weight": 200.0,
                    "resolver_max_iterations": 64,
                },
            }
        )

        assert row["results"]["opponent"] == "deployed"
        assert row["results"]["root_prior_weight"] == 200.0


class TestABareSideACannotBeTheControlOffTree:
    """MEASURED 09-30: against `gears@4600`, a bare side A raised on all four
    attempts -- `KeyError: State is off-tree ... 'c-x-x-b0.25-r0.40'`. The
    opponent's different action tree makes off-menu sizes by construction, and
    `TreePolicySource` refuses an off-tree state by design: a uniform fallback
    there would average a missing strategy into a plausible-looking score.

    So the control for an off-menu opponent is `resolver_blend_alpha=0` -- side A
    still reaches the blueprint through the resolver's lookup path, including the
    shadow that translates the off-tree state, but plays the blueprint's own
    decisions. Any gap is then the resolver's DECISIONS, not the path around them.
    """

    def test_zero_alpha_is_a_different_tier_from_the_shipped_one(self):
        control, shipped = (
            tiers.build_blueprint_match_knobs(
                {
                    "run_b": "gears",
                    "rung_b": 4600,
                    "num_deals": 4000,
                    "seed": 1,
                    "opponent": "deployed",
                    "resolver_max_iterations": 64,
                    "root_prior_weight": 200.0,
                    "resolver_blend_alpha": alpha,
                }
            )
            for alpha in (0.0, 0.35)
        )

        assert control["resolver_blend_alpha"] == 0.0
        assert tiers.tier_digest(
            {"method": "blueprint_match", "knobs": control}
        ) != tiers.tier_digest({"method": "blueprint_match", "knobs": shipped})

    def test_an_off_tree_miss_is_not_swallowed(self):
        """The seam that raised. A `None` here instead would have let the bare arm
        play a uniform fallback and report a plausible-looking chip edge."""
        solver = build_trained_test_solver(1)
        # Only the TREE is stubbed: the storage and abstraction are the real ones,
        # so the refusal comes from the node lookup and not from a broken stand-in.
        source = TreePolicySource(
            cast("BettingTree", _TreeThatHasNoNode()),
            solver.storage,
            solver.card_abstraction,
        )
        state = cast("GameState", object())

        with pytest.raises(KeyError):
            source.infoset_at(state, 0)

        assert source.covers(state) is False


class _TreeThatHasNoNode:
    """A betting tree that covers nothing, standing in for an off-menu line."""

    def node_id(self, state: object) -> int:
        raise KeyError("State is off-tree: betting_sequence='c-x-x-b0.25-r0.40'")
