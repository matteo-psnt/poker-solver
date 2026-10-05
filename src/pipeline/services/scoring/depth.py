"""Cross-depth gates: two runs at one table depth, and ladder against ladder.

Unlike `blueprint_match` these ALLOW the runs' starting stacks to differ -- that
is the point -- and refuse only differing blinds, the one thing the replay
cannot convert.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.pipeline.evaluation.estimators.cross_depth import CrossDepthPlayer
from src.pipeline.evaluation.estimators.depth_duel import play_depth_duel
from src.pipeline.evaluation.estimators.elimination import LadderSeat, play_elimination
from src.pipeline.services.runs import checkpoint_iteration_of, load_run_metadata
from src.pipeline.services.scoring._shared import (
    EvaluationOutput,
    build_blueprint_for,
    effective_abstraction_hash,
)
from src.pipeline.services.scoring.matches import threshold_side

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.game.rules import GameRules
    from src.engine.solver.mccfr.static_solver import StaticTreeSolver
    from src.pipeline.training.run_tracker import RunMetadata
    from src.shared.ports.record import RecordSource

logger = logging.getLogger(__name__)

DEPTH_DUEL_ESTIMATOR_LABEL = (
    "depth_duel (duplicate-deal chip edge at one TABLE depth between blueprints cut "
    "for any depths; not a bound)"
)

ELIMINATION_ESTIMATOR_LABEL = (
    "elimination (whole matches, ladder vs ladder, escalation as shrinking stacks, "
    "duplicate pairs; a match win rate, not a bound)"
)

# One rung of a ladder as the service takes it: the run, the rung that plays
# (None = its latest) and the threshold it is served at.
Rung = tuple[Path, int | None, float]


@dataclass(frozen=True)
class _Loaded:
    """A blueprint ready to play, and what it was loaded as."""

    solver: StaticTreeSolver
    metadata: RunMetadata
    rung: int | None
    rows_thresholded: int


def _load(
    run_dir: Path, at: int | None, threshold: float, side: str, source: RecordSource | None
) -> _Loaded:
    """Play-only: a player never reads regrets, and a ladder holds many at once."""
    metadata = load_run_metadata(run_dir, source)
    solver, storage, _ = build_blueprint_for(
        run_dir,
        metadata,
        effective_abstraction_hash(run_dir, metadata, None),
        at,
        play_only=True,
    )
    changed = threshold_side(storage, solver, threshold, side)
    rung = at if at is not None else checkpoint_iteration_of(run_dir)
    return _Loaded(solver, metadata, rung, changed)


def _table_rules(loaded: Sequence[_Loaded]) -> GameRules:
    """The one blind level every player shares, or a refusal naming who differs."""
    blinds = {
        (
            item.metadata.run_id,
            item.solver.config.game.small_blind,
            item.solver.config.game.big_blind,
        )
        for item in loaded
    }
    if len({(small, big) for _, small, big in blinds}) > 1:
        raise ValueError(
            f"Blinds differ between runs ({sorted(blinds)}). The replay converts "
            "STACKS, never currency, so these cannot share a table."
        )
    return loaded[0].solver.rules


def _refuse_shared_tables(loaded: Sequence[_Loaded]) -> None:
    """Thresholding is in place: two players over one table would both be cut."""
    tables = [item.solver.storage.strategy_sum for item in loaded]
    for i, first in enumerate(tables):
        for second in tables[i + 1 :]:
            if np.shares_memory(first, second):
                raise ValueError(
                    "Two players share one strategy table, so a per-side policy "
                    "threshold would apply to both."
                )


def evaluate_depth_duel(
    run_dir_a: Path,
    run_dir_b: Path,
    *,
    table_bb: float,
    num_deals: int = 2000,
    seed: int = 1,
    at_a: int | None = None,
    at_b: int | None = None,
    policy_threshold_a: float = 0.0,
    policy_threshold_b: float = 0.0,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    """A's chip edge over B at a ``table_bb``-deep table, whatever depth each was cut for.

    Positive ``a_mbb_per_hand`` means A wins, in the shared blinds.
    """
    a = _load(run_dir_a, at_a, policy_threshold_a, "A", record_source)
    b = _load(run_dir_b, at_b, policy_threshold_b, "B", record_source)
    _refuse_shared_tables((a, b))
    rules = _table_rules((a, b))
    starting_stack = round(table_bb * rules.big_blind)
    if starting_stack < 2 * rules.big_blind:
        raise ValueError(f"A {table_bb} bb table cannot post both blinds.")
    player_a, player_b = CrossDepthPlayer(a.solver), CrossDepthPlayer(b.solver)
    result = play_depth_duel(
        player_a,
        player_b,
        rules=rules,
        starting_stack=starting_stack,
        num_deals=num_deals,
        seed=seed,
    )
    results = {
        "run_a": a.metadata.run_id,
        "run_b": b.metadata.run_id,
        "rung_b": b.rung,
        # The table as played, in big blinds: `round` may have moved it a chip.
        "table_bb": starting_stack / rules.big_blind,
        "depth_a_bb": player_a.depth_in_blinds,
        "depth_b_bb": player_b.depth_in_blinds,
        "a_mbb_per_hand": result.a_mbb_per_hand,
        "se_mbb": result.se_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "p_value": result.p_value,
        "num_deals": result.num_deals,
        "num_hands": result.num_hands,
        "seed": seed,
        "policy_threshold_a": policy_threshold_a,
        "policy_threshold_b": policy_threshold_b,
        "rows_thresholded_a": a.rows_thresholded,
        "rows_thresholded_b": b.rows_thresholded,
        # Pairs where the sides played differently: the edge's real sample size.
        "divergent_pairs": sum(1 for sample in result.pair_samples_mbb if sample != 0.0),
        # What the fiction cost each side. Truncation is a replay that could not
        # follow the line at the player's own depth and fell to check/fold.
        **player_a.tally.as_rates("a"),
        **player_b.tally.as_rates("b"),
        "card_abstraction_hash_b": b.metadata.card_abstraction_hash,
        "action_config_hash_b": b.metadata.action_config_hash,
        "config_name_b": b.metadata.config_name,
        "git_commit_b": b.metadata.git_commit,
        "pair_samples_mbb": result.pair_samples_mbb,
    }
    return EvaluationOutput(
        infosets=a.solver.storage.num_infosets(),
        results=results,
        checkpoint_iteration=a.rung,
    )


def evaluate_elimination(
    run_dir: Path,
    *,
    ladder_a: Sequence[Rung],
    ladder_b: Sequence[Rung],
    num_pairs: int = 500,
    starting_bb: float = 100.0,
    escalate_every: int = 10,
    escalate_by: float = 1.5,
    select_a: str = "up",
    select_b: str = "up",
    seed: int = 1,
    at_iteration: int | None = None,
    record_source: RecordSource | None = None,
) -> EvaluationOutput:
    """Ladder A's elimination-match win rate over ladder B.

    Filed under ``run_dir``, which must be one of ladder A's rungs: the row's own
    columns then describe a run that actually played.
    """
    anchor = [rung for rung in ladder_a if rung[0].name == run_dir.name]
    if not anchor:
        raise ValueError(f"{run_dir.name} is not a rung of ladder A; file the row under one.")
    if at_iteration is not None and anchor[0][1] not in (None, at_iteration):
        raise ValueError(
            f"--at {at_iteration} disagrees with ladder A's {run_dir.name}:{anchor[0][1]}."
        )
    # One load per (run, rung, threshold): the ladders usually share most rungs,
    # and a 100 bb blueprint is gigabytes. Each side still gets its OWN player,
    # so the tallies stay per ladder.
    cache: dict[tuple[str, int | None, float], _Loaded] = {}

    def side(ladder: Sequence[Rung], name: str, rule: str) -> tuple[LadderSeat, list[str]]:
        players, specs = [], []
        for run, at, threshold in ladder:
            key = (run.name, at, threshold)
            if key not in cache:
                cache[key] = _load(run, at, threshold, f"{name} {run.name}", record_source)
            loaded = cache[key]
            players.append(CrossDepthPlayer(loaded.solver))
            specs.append(f"{run.name}:{loaded.rung}:{threshold:g}")
        return LadderSeat(players, rule), specs

    seat_a, specs_a = side(ladder_a, "A", select_a)
    seat_b, specs_b = side(ladder_b, "B", select_b)
    loaded = list(cache.values())
    _refuse_shared_tables(loaded)
    rules = _table_rules(loaded)
    result = play_elimination(
        seat_a,
        seat_b,
        rules=rules,
        starting_stack=round(starting_bb * rules.big_blind),
        num_pairs=num_pairs,
        seed=seed,
        escalate_every=escalate_every,
        escalate_by=escalate_by,
    )
    anchor_rung = cache[(run_dir.name, anchor[0][1], anchor[0][2])].rung
    results = {
        "run_a": run_dir.name,
        "ladder_a": specs_a,
        "ladder_b": specs_b,
        "a_win_rate": result.a_win_rate,
        "se": result.se,
        "confidence_95": result.confidence_95,
        "p_value": result.p_value,
        "num_pairs": result.num_pairs,
        "num_matches": result.num_matches,
        "mean_hands": result.mean_hands,
        "starting_bb": starting_bb,
        "escalate_every": escalate_every,
        "escalate_by": escalate_by,
        "select_a": select_a,
        "select_b": select_b,
        "seed": seed,
        "switches_a": seat_a.switches,
        "switches_b": seat_b.switches,
        "truncation_rate_a": seat_a.truncation_rate,
        "truncation_rate_b": seat_b.truncation_rate,
        "pair_samples": result.pair_samples,
    }
    return EvaluationOutput(
        infosets=sum(item.solver.storage.num_infosets() for item in loaded),
        results=results,
        checkpoint_iteration=anchor_rung,
    )
