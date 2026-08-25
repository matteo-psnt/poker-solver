"""Which strategy of a checkpoint gets scored: the DCFR average, the current
iterate, or the average over a late WINDOW of training.

``strategy_sum`` accumulates ``t^gamma * reach * sigma_t`` with the weight
applied at add time and never revisited, so the window sum over ``(t0, T]`` is
exactly ``SS(T) - SS(t0)``. That identity is why a windowed average is
measurable from two retained rungs rather than a retrained run.

Every variant is written INTO ``strategy_sum`` after the load, so nothing
downstream changes: the same normaliser turns a stored row into a distribution,
and ``max(regret, 0)`` under it IS regret matching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.engine.solver.storage.static_checkpoint import (
    StaticCheckpointManifest,
    read_strategy_sum,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.engine.solver.storage.static_array import StaticArrayStorage
    from src.shared.config.schema import SolverConfig

logger = logging.getLogger(__name__)

PolicyIterate = Literal["average", "current"]

# Weight on the pre-window average, added back so a row the window never
# visited keeps the full average instead of collapsing to the uniform fallback.
# A row with typical window mass is contaminated ~4e-4 by it; a row with almost
# no window evidence shrinks back toward the average, which is the honest
# answer where the window has nothing to say.
WINDOW_SHRINKAGE = 1e-3


def source_gamma_of(solver: SolverConfig) -> float:
    """The exponent a run actually weighted its ``strategy_sum`` contributions by.

    ``iteration_weighting`` decides it, not ``dcfr_gamma``: a linear run adds
    `t^1` and never reads the gamma field, so reweighting a linear run from
    gamma=2 would be correcting an exponent it never used. The PCS flagship is
    a linear run.
    """
    if solver.iteration_weighting == "dcfr":
        return float(solver.dcfr_gamma)
    return 1.0 if solver.iteration_weighting == "linear" else 0.0


def assemble_policy(
    storage: StaticArrayStorage,
    checkpoint_dir: Path,
    *,
    iterate: PolicyIterate = "average",
    window_from: int | None = None,
    avg_gamma: float | None = None,
    source_gamma: float = 2.0,
    loaded_iteration: int | None = None,
) -> dict[str, Any]:
    """Rewrite ``storage.strategy_sum`` as the requested strategy. In place.

    Returns what the record needs to tell two rows apart, empty for the
    identity. ``window_from`` must be a retained rung of the same run.
    """
    if iterate == "current" and (window_from is not None or avg_gamma is not None):
        raise ValueError(
            "policy_iterate='current' has no averaging weight: the current iterate is "
            "one iteration's strategy, not an average over any range of them."
        )
    if iterate == "current":
        np.maximum(storage.regrets, 0, out=storage.strategy_sum)
        return {"policy_iterate": "current"}
    if avg_gamma is not None:
        if loaded_iteration is None:
            raise ValueError("avg_gamma needs the iteration that was loaded to bound the ladder")
        return _apply_gamma(
            storage, checkpoint_dir, avg_gamma, source_gamma, loaded_iteration, window_from
        )
    if window_from is not None:
        return _apply_window(storage, checkpoint_dir, window_from)
    return {}


def _apply_window(
    storage: StaticArrayStorage, checkpoint_dir: Path, window_from: int
) -> dict[str, Any]:
    """``SS(T) - SS(t0)``, shrunk back toward ``SS(t0)`` where the window is empty."""
    earlier = read_strategy_sum(storage, checkpoint_dir, window_from)
    window = storage.strategy_sum
    had = earlier > 0
    np.subtract(window, earlier, out=window)
    np.maximum(window, 0, out=window)
    trained = int(np.count_nonzero(had))
    empty = int(np.count_nonzero(had & (window <= 0)))
    window += WINDOW_SHRINKAGE * earlier
    return {
        "avg_window_from": int(window_from),
        "avg_window_shrinkage": WINDOW_SHRINKAGE,
        # How much of the trained table the window has nothing to say about,
        # and so falls back to the full average. A large share means the arm
        # measures windowing PLUS a coverage change, not windowing alone.
        "avg_window_empty_slot_fraction": (empty / trained) if trained else 0.0,
    }


def _window_coefficients(
    rungs: list[int], target: float, source: float, floor: int = 0
) -> list[float]:
    """Per-window rescaling that turns a `t^source` average into a `t^target` one.

    Piecewise: within one window the weight ratio still varies as
    `t^(target-source)`, so this is exact only in the limit of a dense ladder.
    At the retained spacing it is snapshot averaging -- one point per band,
    reweighted -- which is what Pluribus does instead of a running average, and
    is the honest description of what gets scored.
    """

    def mass(exponent: float, low: int, high: int) -> float:
        return (high ** (exponent + 1) - low ** (exponent + 1)) / (exponent + 1)

    edges = [floor, *rungs]
    coefficients = [
        mass(target, edges[k], edges[k + 1]) / mass(source, edges[k], edges[k + 1])
        for k in range(len(rungs))
    ]
    scale = max(abs(c) for c in coefficients)
    return [c / scale for c in coefficients]


def _apply_gamma(
    storage: StaticArrayStorage,
    checkpoint_dir: Path,
    target_gamma: float,
    source_gamma: float,
    loaded_iteration: int,
    window_from: int | None = None,
) -> dict[str, Any]:
    """Reweight the average from `t^source_gamma` to `t^target_gamma`.

    With ``window_from`` the reweighting runs over ``(window_from, loaded]``
    only. `gamma=0` over a FIXED-WIDTH window is the one comparison that holds
    the averaging noise constant while moving the endpoint, which is what
    separates "the solver is still learning" from "averaging more iterates
    keeps shrinking the average's variance" -- two mechanisms that both make
    the all-history gamma=0 curve fall.

    Each retained rung's band is a difference of two rungs, so the reweighted
    average is a positive combination of the rungs themselves (Abel summation)
    -- read one at a time, never the whole ladder at once. Run it at
    ``--workers 1``: every fork-join worker would otherwise re-read the ladder
    off the share.
    """
    manifest = StaticCheckpointManifest.read(checkpoint_dir)
    if manifest is None:
        raise FileNotFoundError(f"No static checkpoint manifest in {checkpoint_dir}")
    floor = window_from or 0
    rungs = [rung for rung in manifest.ladder() if floor < rung <= loaded_iteration]
    if len(rungs) < 2 or rungs[-1] != loaded_iteration:
        raise ValueError(
            f"Reweighting the average needs a retained ladder from {floor:,} to "
            f"{loaded_iteration:,}; this run has {manifest.ladder()}."
        )
    coefficients = _window_coefficients(rungs, target_gamma, source_gamma, floor)
    # Abel: sum_k c_k (S_k - S_{k-1}) = c_n S_n + sum_{k<n} (c_k - c_{k+1}) S_k.
    weights = [coefficients[k] - coefficients[k + 1] for k in range(len(rungs) - 1)]
    logger.info(
        "reweighting the average from gamma=%g to gamma=%g over %d bands; band coefficients %s",
        source_gamma,
        target_gamma,
        len(rungs),
        ", ".join(
            f"{rung // 1_000_000}M:{c:.4g}" for rung, c in zip(rungs, coefficients, strict=True)
        ),
    )
    storage.strategy_sum *= np.float32(coefficients[-1])
    for rung, weight in zip(rungs[:-1], weights, strict=True):
        _accumulate(storage, checkpoint_dir, rung, weight)
    record: dict[str, Any] = {"avg_gamma": float(target_gamma), "avg_gamma_rungs": len(rungs)}
    if window_from is not None:
        # The Abel sum's S_0 term: everything before the window leaves with the
        # first band's coefficient. Shrunk back in afterwards on the same terms
        # as the plain window, so the two window arms stay comparable.
        base = read_strategy_sum(storage, checkpoint_dir, window_from)
        storage.strategy_sum -= np.float32(coefficients[0]) * base
        np.maximum(storage.strategy_sum, 0, out=storage.strategy_sum)
        had = base > 0
        trained = int(np.count_nonzero(had))
        empty = int(np.count_nonzero(had & (storage.strategy_sum <= 0)))
        storage.strategy_sum += WINDOW_SHRINKAGE * base
        record["avg_window_from"] = int(window_from)
        record["avg_window_shrinkage"] = WINDOW_SHRINKAGE
        record["avg_window_empty_slot_fraction"] = (empty / trained) if trained else 0.0
    # A target above the source makes some weight negative; a negative stored
    # row is not a strategy, and the normaliser reads it as one.
    np.maximum(storage.strategy_sum, 0, out=storage.strategy_sum)
    return record


def _accumulate(
    storage: StaticArrayStorage, checkpoint_dir: Path, rung: int, weight: float
) -> None:
    """``strategy_sum += weight * SS(rung)``."""
    storage.strategy_sum += np.float32(weight) * read_strategy_sum(storage, checkpoint_dir, rung)


__all__ = ("WINDOW_SHRINKAGE", "PolicyIterate", "assemble_policy", "source_gamma_of")
