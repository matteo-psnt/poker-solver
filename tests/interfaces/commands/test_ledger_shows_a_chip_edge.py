"""A match row must show its number, and say which number it is.

`ledger` reached only for `exploitability_mbb`, so the head-to-head gate --
the one estimator folding cannot win -- rendered as an em dash: the row was
recorded, tiered and unreadable. The two are also different UNITS (a chip edge
is mbb per hand, an exploitability is mbb per game) and opposite in sign
convention, so they cannot share an unlabelled column.
"""

from __future__ import annotations

from src.interfaces.commands.ledger import _score
from src.pipeline.evaluation.ledger import records, tiers


class TestTheRowShowsItsNumber:
    def test_an_exploitability_reads_as_a_distance(self):
        assert _score({"exploitability_mbb": 1469.3, "std_error_mbb": 0.0}) == "1469.3±0.0"

    def test_a_chip_edge_reads_as_a_signed_edge(self):
        """Signed on purpose: which way it went IS the result."""
        assert _score({"edge_mbb_per_hand": 94.97, "edge_se_mbb": 27.7}) == "+95.0±28"
        assert _score({"edge_mbb_per_hand": -313.4, "edge_se_mbb": 97.1}) == "-313.4±97"

    def test_a_row_carrying_neither_still_renders(self):
        assert _score({}) == "—"


class TestTheIndexKeepsTheEdge:
    """It kept four keys, none of which a match or a resolver gate writes."""

    def test_a_blueprint_match_document_keeps_its_edge(self):
        row = records.ledger_row(
            {"results": {"a_mbb_per_hand": 94.97, "se_mbb": 27.7, "pair_samples_mbb": [1.0]}}
        )
        assert row["results"]["edge_mbb_per_hand"] == 94.97
        assert row["results"]["edge_se_mbb"] == 27.7

    def test_a_resolver_gate_document_keeps_its_edge_too(self):
        row = records.ledger_row({"results": {"resolver_mbb_per_hand": -46.9, "se_mbb": 159.2}})
        assert row["results"]["edge_mbb_per_hand"] == -46.9

    def test_the_bulk_samples_stay_in_the_document(self):
        row = records.ledger_row({"results": {"pair_samples_mbb": [1.0, 2.0, 3.0]}})
        assert "pair_samples_mbb" not in row["results"]
        assert row["results"]["n"] == 3


class TestDealCountSplitsTheTier:
    """MEASURED 09-21: a 2,000-deal probe and a 20,000-deal run of the same
    pairing hashed to one tier (`182b114e` for both). Both match builders had
    always declared `num_deals` in `knobs`; `tier_key` reads only
    `CONDITIONAL_TIER_KNOBS` and it was not in there. Every reader pools
    unweighted, so that averaged a +-153 mbb row into +-49 mbb ones."""

    def _digest(self, num_deals: int) -> str:
        return tiers.tier_digest(
            {
                "method": "blueprint_match",
                "knobs": {
                    "run_b": "run-b",
                    "rung_b": 4000,
                    "num_deals": num_deals,
                    "base_seed": 7,
                },
                "results": {},
                "card_abstraction_hash": "h",
                "action_config_hash": "a",
            }
        )

    def test_a_probe_does_not_pair_with_the_real_run(self):
        assert self._digest(2000) != self._digest(20000)

    def test_the_same_deal_count_still_pairs(self):
        assert self._digest(20000) == self._digest(20000)


class TestTheRungGapIsTheInstrument:
    """An average strategy moves less the wider the base it covers, so drift over
    400 rungs and drift over 2,000 are not two readings of one quantity. A
    settling curve is only a curve at a FIXED gap."""

    def _digest(self, from_iteration: int, gap: int) -> str:
        return tiers.tier_digest(
            {
                "method": "settle",
                "knobs": {"from_iteration": from_iteration, "rung_gap": gap},
                "results": {},
                "card_abstraction_hash": "h",
                "action_config_hash": "a",
            }
        )

    def test_a_different_gap_does_not_pair(self):
        assert self._digest(3600, 400) != self._digest(2000, 2000)

    def test_the_same_gap_at_a_later_rung_does_pair(self):
        """What makes a curve: same instrument, walked along the ladder."""
        assert self._digest(3600, 400) != self._digest(7600, 400)

    def test_the_knobs_are_exactly_the_gap(self):
        knobs = tiers.build_settling_knobs({"from_iteration": 3600, "rung_gap": 400})
        assert knobs == {"from_iteration": 3600, "rung_gap": 400}
