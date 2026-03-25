"""Placement and weighting invariants for the average-strategy update.

Under external sampling the average strategy must accumulate at OPPONENT
nodes, unweighted: an opponent node is visited exactly when the sampled
opponent/chance actions lead there, so visit frequency already supplies the
acting player's own reach — Zinkevich's weight. The historical defect
(docs/AVERAGE_STRATEGY_WEIGHTING.md) accumulated at the traverser's own nodes
with a dead reach weight of 1.0, converging to a pi_{-i}-weighted average.
These tests pin the corrected placement so a regression to either the old
site or an explicit reach term (option C's trap: threading pi_i in place
yields full reach, still wrong) is loud.
"""

from src.engine.solver.mccfr import traversal
from tests.test_helpers import DummyCardAbstraction, build_test_solver, make_test_config


def _spy_accumulations(monkeypatch):
    """Record, for every average-strategy accumulation, the acting player of the
    node it ran at, the traversing player, and the reach weight passed."""
    node_player: dict[int, int] = {}
    calls: list[dict] = []

    real_context = traversal._infoset_context
    real_accumulate = traversal._accumulate_average_strategy

    def context_spy(self, state, current_player):
        result = real_context(self, state, current_player)
        # The accumulate call (if any) follows immediately at the same node, so
        # the last-recorded player for this infoset object is the acting player.
        node_player[id(result[0])] = current_player
        return result

    def accumulate_spy(self, infoset, valid_indices, strategy, reach_weight):
        calls.append(
            {
                "node_player": node_player.get(id(infoset)),
                "traversing_player": self.iteration % 2,
                "reach_weight": reach_weight,
            }
        )
        return real_accumulate(self, infoset, valid_indices, strategy, reach_weight)

    monkeypatch.setattr(traversal, "_infoset_context", context_spy)
    monkeypatch.setattr(traversal, "_accumulate_average_strategy", accumulate_spy)
    return calls


def _build_solver(sampling_method: str):
    config = make_test_config(seed=42, sampling_method=sampling_method)
    return build_test_solver(config, DummyCardAbstraction())


def test_external_sampling_accumulates_only_at_opponent_nodes(monkeypatch):
    calls = _spy_accumulations(monkeypatch)
    solver, _ = _build_solver("external")

    for _ in range(10):
        solver.train_iteration()

    assert calls, "average strategy must accumulate somewhere"
    at_traverser = [c for c in calls if c["node_player"] == c["traversing_player"]]
    assert not at_traverser, (
        f"{len(at_traverser)}/{len(calls)} accumulations ran at the traverser's own "
        "nodes — that placement is pi_{-i}-weighted (the pre-fix defect)"
    )


def test_external_sampling_accumulation_is_unweighted(monkeypatch):
    calls = _spy_accumulations(monkeypatch)
    solver, _ = _build_solver("external")

    for _ in range(10):
        solver.train_iteration()

    weights = {c["reach_weight"] for c in calls}
    assert weights == {1.0}, (
        f"external sampling must pass reach_weight=1.0 (visit frequency already "
        f"supplies pi_i); saw {weights}"
    )


def test_both_players_averages_update_across_iterations(monkeypatch):
    """Alternating traversal still updates both seats' averages."""
    calls = _spy_accumulations(monkeypatch)
    solver, _ = _build_solver("external")

    for _ in range(10):
        solver.train_iteration()

    assert {c["traversing_player"] for c in calls} == {0, 1}


def test_outcome_sampling_placement_unchanged(monkeypatch):
    """Outcome sampling keeps its traverser-node update with a live reach weight,
    pending the outcome-sampling audit — pinned so the external-sampling fix
    cannot silently leak into this path."""
    calls = _spy_accumulations(monkeypatch)
    solver, _ = _build_solver("outcome")

    for _ in range(20):
        solver.train_iteration()

    assert calls
    assert all(c["node_player"] == c["traversing_player"] for c in calls)
    assert any(c["reach_weight"] != 1.0 for c in calls), (
        "outcome sampling threads the traverser's own reach; it should not be a dead 1.0"
    )
