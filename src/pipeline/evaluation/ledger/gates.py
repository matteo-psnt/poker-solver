"""Head-to-head gates: per-seed `blueprint_match` rows pooled into one result.

A gate is one comparison -- run A at rung a against run B at rung b, at fixed
purification thresholds and B's side (bare blueprint or as deployed) -- scored
over several deal seeds. Each seed is a separate ledger row; the reader wants
the pooled edge, its error, and how many seeds stand behind it.

Seeds are independent deal sets, so the pooled standard error is the root sum
of squares over n, and a seed scored twice keeps its newest row: a re-run is a
replacement, not extra evidence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable


class GateSeed(BaseModel):
    """One deal seed's result inside a gate."""

    seed: int | None
    edge_mbb: float
    se_mbb: float | None
    task_id: str | None = None
    timestamp: str = ""


class Gate(BaseModel):
    """One pooled comparison. `edge_mbb` is A's chips per hand against B."""

    run_id: str
    at: int | None
    run_b: str | None
    at_b: int | None
    thresholds: list[float | None]
    b_side: str | None
    experiment_id: str | None = None
    arm: str | None = None
    edge_mbb: float
    se_mbb: float
    sigma: float | None
    deals_per_seed: int | None
    duplicates: int = 0  # rows dropped because the same seed was scored again
    last: str
    seeds: list[GateSeed] = Field(default_factory=list)


def _key(record: dict[str, Any]) -> tuple:
    knobs = record.get("knobs") or {}
    return (
        record.get("run_id"),
        record.get("checkpoint_iteration"),
        knobs.get("run_b"),
        knobs.get("rung_b"),
        knobs.get("policy_threshold_a"),
        knobs.get("policy_threshold_b"),
        (record.get("results") or {}).get("opponent"),
    )


def pool_matches(records: Iterable[dict[str, Any]]) -> list[Gate]:
    """Every gate the records contain, newest first.

    Records that are not a scored `blueprint_match` (no edge) are skipped, so a
    caller may pass a whole ledger page.
    """
    groups: dict[tuple, dict[Any, dict[str, Any]]] = {}
    duplicates: dict[tuple, int] = {}
    for record in records:
        if record.get("method") != "blueprint_match":
            continue
        if not isinstance((record.get("results") or {}).get("edge_mbb_per_hand"), (int, float)):
            continue
        key = _key(record)
        seeds = groups.setdefault(key, {})
        seed = (record.get("knobs") or {}).get("base_seed")
        held = seeds.get(seed)
        if held is not None:
            duplicates[key] = duplicates.get(key, 0) + 1
            if str(held.get("timestamp", "")) >= str(record.get("timestamp", "")):
                continue
        seeds[seed] = record

    gates = [
        _gate(key, list(seeds.values()), duplicates.get(key, 0)) for key, seeds in groups.items()
    ]
    return sorted(gates, key=lambda g: g.last, reverse=True)


def _gate(key: tuple, rows: list[dict[str, Any]], duplicates: int) -> Gate:
    run_id, at, run_b, at_b, thr_a, thr_b, b_side = key
    rows.sort(
        key=lambda r: (
            (r.get("knobs") or {}).get("base_seed") is None,
            (r.get("knobs") or {}).get("base_seed"),
        )
    )
    seeds = [
        GateSeed(
            seed=(r.get("knobs") or {}).get("base_seed"),
            edge_mbb=float(r["results"]["edge_mbb_per_hand"]),
            se_mbb=_float((r.get("results") or {}).get("edge_se_mbb")),
            task_id=r.get("task_id"),
            timestamp=str(r.get("timestamp", "")),
        )
        for r in rows
    ]
    n = len(seeds)
    edge = sum(s.edge_mbb for s in seeds) / n
    se = math.sqrt(sum((s.se_mbb or 0.0) ** 2 for s in seeds)) / n
    first = rows[0]
    return Gate(
        run_id=str(run_id),
        at=at,
        run_b=run_b,
        at_b=at_b,
        thresholds=[thr_a, thr_b],
        b_side=b_side,
        experiment_id=first.get("experiment_id"),
        arm=first.get("arm"),
        edge_mbb=edge,
        se_mbb=se,
        sigma=edge / se if se > 0 else None,
        deals_per_seed=(first.get("knobs") or {}).get("num_deals"),
        duplicates=duplicates,
        last=max(s.timestamp for s in seeds),
        seeds=seeds,
    )


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None
