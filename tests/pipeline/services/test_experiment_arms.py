"""Grouping an experiment's evaluations by instrument, then differencing arms.

The rule under test is the one this project has broken by hand before: two
numbers measured with different instruments are never subtracted. Everything
else here exists to make the difference READABLE, which is why it is computed
directly -- `exact_br` is deterministic, and the paired-sample machinery that
used to own this question refuses those rows outright.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.services.experiments import experiment_arms

TIER: dict[str, Any] = {
    "method": "exact_br",
    "card_abstraction_hash": "a1542e88",
    "action_config_hash": "45a36854",
    "eval_tree_fingerprint": "82379b16",
}


def row(arm: str, iteration: int, mbb: float, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        **TIER,
        "experiment_id": "pcs-weighting",
        "arm": arm,
        "run_id": f"run-{arm}",
        "checkpoint_iteration": iteration,
        "knobs": {"num_flops": 4, "num_turns": 16, "num_rivers": 16},
        "results": {"exploitability_mbb": mbb},
    }
    knobs = overrides.pop("knobs", None)
    if knobs:
        record["knobs"] = {**record["knobs"], **knobs}
    record.update(overrides)
    return record


class TestDifferences:
    def test_the_difference_is_direct_and_carries_no_invented_error(self):
        """`exact_br` has zero evaluation variance, so a standard error on the
        difference would be a number nothing measured."""
        out = experiment_arms(
            [row("linplus", 2000, 900.0), row("dcfr", 2000, 780.0)],
            "pcs-weighting",
            control="linplus",
        )
        (tier,) = out.tiers
        challenger = next(p for p in tier.points if p.arm == "dcfr")
        assert challenger.vs_control_mbb == -120.0
        assert challenger.vs_control_stderr_mbb is None
        assert next(p for p in tier.points if p.arm == "linplus").vs_control_mbb is None

    def test_a_sampled_row_keeps_the_combined_standard_error(self):
        out = experiment_arms(
            [
                row(
                    "linplus",
                    2000,
                    900.0,
                    results={"exploitability_mbb": 900.0, "std_error_mbb": 30.0},
                ),
                row(
                    "dcfr",
                    2000,
                    780.0,
                    results={"exploitability_mbb": 780.0, "std_error_mbb": 40.0},
                ),
            ],
            "pcs-weighting",
            control="linplus",
        )
        challenger = next(p for p in out.tiers[0].points if p.arm == "dcfr")
        assert challenger.vs_control_mbb == -120.0
        assert challenger.vs_control_stderr_mbb == 50.0  # sqrt(30^2 + 40^2)

    def test_an_arm_the_control_does_not_reach_is_left_undifferenced(self):
        out = experiment_arms(
            [row("linplus", 2000, 900.0), row("dcfr", 2000, 780.0), row("dcfr", 4000, 700.0)],
            "pcs-weighting",
            control="linplus",
        )
        (tier,) = out.tiers
        assert next(p for p in tier.points if p.iteration == 4000).vs_control_mbb is None
        assert tier.unmatched_iterations == [4000]


class TestTiers:
    def test_two_board_budgets_are_never_subtracted(self):
        rows = [
            row("linplus", 2000, 900.0),
            row("dcfr", 2000, 780.0, knobs={"num_turns": 2, "num_rivers": 2}),
        ]
        assert len(experiment_arms(rows, "pcs-weighting").tiers) == 2
        # Under a control the dcfr row is not merely undifferenced, it is not
        # RENDERED: its tier cannot answer the question that was asked.
        out = experiment_arms(rows, "pcs-weighting", control="linplus")
        assert [t.arms for t in out.tiers] == [["linplus"]]
        assert out.tiers_without_control == 1

    def test_a_different_tree_fingerprint_splits_the_tier(self):
        """The limp fix changed the tree under a fixed action config, so two rows
        agreeing on every knob can still describe different games."""
        rows = [
            row("linplus", 2000, 900.0),
            row("dcfr", 2000, 780.0, eval_tree_fingerprint="37e51fce"),
        ]
        assert len(experiment_arms(rows, "pcs-weighting").tiers) == 2
        assert experiment_arms(rows, "pcs-weighting", control="linplus").tiers_without_control == 1

    def test_the_best_covered_tier_comes_first(self):
        out = experiment_arms(
            [
                row("a", 2000, 900.0, knobs={"num_turns": 2}),
                row("linplus", 2000, 900.0),
                row("dcfr", 2000, 780.0),
            ],
            "pcs-weighting",
        )
        assert out.tiers[0].arms == ["dcfr", "linplus"]

    def test_a_reevaluation_supersedes_its_predecessor(self):
        out = experiment_arms([row("dcfr", 2000, 999.0), row("dcfr", 2000, 780.0)], "pcs-weighting")
        assert out.tiers[0].points[0].exploitability_mbb == 780.0


class TestWhatAControlSelects:
    def test_a_tier_without_the_control_is_counted_not_rendered(self):
        """`cfr-br` holds 47 tiers -- avg_gamma sweeps, mixtures, thresholds,
        three seeds -- and exactly one contains both weighting arms. Printing the
        other 46 buried the answer."""
        out = experiment_arms(
            [
                row("linplus", 2000, 900.0),
                row("dcfr", 2000, 780.0),
                row("other", 3000, 500.0, knobs={"num_turns": 2}),
                row("other", 3000, 510.0, knobs={"avg_gamma": 3.0}),
            ],
            "pcs-weighting",
            control="linplus",
        )
        assert [t.arms for t in out.tiers] == [["dcfr", "linplus"]]
        assert out.tiers_without_control == 2

    def test_without_a_control_every_tier_is_kept(self):
        out = experiment_arms(
            [row("linplus", 2000, 900.0), row("other", 3000, 500.0, knobs={"num_turns": 2})],
            "pcs-weighting",
        )
        assert len(out.tiers) == 2
        assert out.tiers_without_control == 0


class TestWhatItRefusesToPlace:
    def test_rows_with_no_checkpoint_iteration_are_counted_not_plotted(self):
        out = experiment_arms(
            [row("dcfr", 2000, 780.0), row("dcfr", 0, 900.0, checkpoint_iteration=None)],
            "pcs-weighting",
        )
        assert out.unplaceable_records == 1
        assert len(out.tiers[0].points) == 1

    def test_another_experiment_is_not_included(self):
        out = experiment_arms(
            [row("dcfr", 2000, 780.0), row("x", 2000, 1.0, experiment_id="turnbr")],
            "pcs-weighting",
        )
        assert out.tiers[0].arms == ["dcfr"]

    def test_an_unknown_control_leaves_nothing_to_render(self):
        """The command turns this into "no tier holds an arm named X", which is a
        different refusal from "nothing is scored" and needs a different fix."""
        out = experiment_arms([row("dcfr", 2000, 780.0)], "pcs-weighting", control="nope")
        assert out.tiers == []
        assert out.tiers_without_control == 1
