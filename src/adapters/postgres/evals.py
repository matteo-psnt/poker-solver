"""The Postgres implementation of `EvalSink`.

Synchronous and unbatched, unlike `sink.py`: an evaluation is one row produced
by hours of compute, so there is nothing to amortise and no loop to keep out of
the way. What it shares with the sink is the policy -- it never raises, because
the document is already on the share and the share is the source of truth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.dialects.postgresql import insert

from src.adapters.postgres import models

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)


def eval_values(eval_id: str, document: Mapping[str, Any], tier_digest: str) -> dict[str, Any]:
    """One evaluation document as the columns `evals` holds.

    Shared by the live writer and `backfill-record`, so an eval that arrives
    live and the same eval re-imported from the share are the same row -- which
    is exactly what `--verify` compares.
    """
    results = document.get("results") or {}
    knobs = document.get("knobs") or {}
    return {
        "eval_id": eval_id,
        "run_id": document.get("run_id"),
        "checkpoint_iteration": document.get("checkpoint_iteration"),
        "method": str(document.get("method") or document.get("estimator") or ""),
        # `base_seed`, which is where the seed actually lives -- not `board_seed`.
        "base_seed": knobs.get("base_seed"),
        "knobs": dict(knobs),
        "tier_digest": tier_digest,
        "exploitability_mbb": results.get("exploitability_mbb"),
        "std_error_mbb": results.get("std_error_mbb"),
        "num_hands": results.get("num_hands"),
        "recorded_at": document.get("timestamp"),
        "payload": dict(document),
    }


class PostgresEvalSink:
    """An `EvalSink` backed by Postgres."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def scored(self, eval_id: str, document: Mapping[str, Any], tier_digest: str) -> None:
        """Store one evaluation, or log and carry on.

        `eval_id` is the document's slug -- timestamp, knob hash and a random
        suffix -- so it is unique by construction and a conflict means this
        exact eval was already recorded. Doing nothing on one is what makes a
        retried task idempotent rather than a duplicate row.
        """
        try:
            with self._engine.begin() as connection:
                connection.execute(
                    insert(models.Eval)
                    .values(eval_values(eval_id, document, tier_digest))
                    .on_conflict_do_nothing(index_elements=["eval_id"])
                )
        except Exception:
            log.warning("eval %s not recorded to the database", eval_id, exc_info=True)
