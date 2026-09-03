"""The Postgres implementation of `EvalSink`.

Synchronous and unbatched, unlike `sink.py`: an evaluation is one row produced
by hours of compute, so there is nothing to amortise and no loop to keep out of
the way.

IT RAISES, and that is the change that came with retiring the share's copy. It
used to swallow everything "because the document is already on the share" --
true then, and the whole justification. Now the row IS the evaluation, and an
eval that ran for hours and recorded nothing is worse than one that says so.
Retried first, because a dropped connection is not a lost evaluation.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy.dialects.postgresql import insert

from src.adapters.postgres import models

if TYPE_CHECKING:
    from collections.abc import Mapping

log = logging.getLogger(__name__)

# A dropped connection is not a lost evaluation; a database that is gone is.
ATTEMPTS = 3
BACKOFF_SECONDS = 0.5


def eval_values(
    eval_id: str, run_id: str, document: Mapping[str, Any], tier_digest: str
) -> dict[str, Any]:
    """One evaluation document as the columns `evals` holds.

    The ONE place a document becomes columns. It was shared by the live writer
    and the importer so the two agreed; the importer is gone and the invariants
    stay, because they are what makes a row PAIRABLE, not what made two writers
    match.

    `run_id` is PASSED rather than read from the document. The caller knows it
    from the run being scored, and that is the authoritative answer; taking it
    from the payload instead would have let the two writers
    disagree about which run an eval belongs to on any document where the two
    differ. They agree on all 2,139 rows today, which is the data agreeing, not
    the code.
    """
    results = document.get("results") or {}
    knobs = document.get("knobs") or {}
    return {
        "eval_id": eval_id,
        "run_id": run_id,
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

    def scored(
        self, eval_id: str, run_id: str, document: Mapping[str, Any], tier_digest: str
    ) -> None:
        """Store one evaluation. RAISES when it cannot.

        `eval_id` is the document's slug -- timestamp, knob hash and a random
        suffix -- so it is unique by construction and a conflict means this
        exact eval was already recorded. Doing nothing on one is what makes a
        retried task idempotent rather than a duplicate row, and what makes the
        retries below safe.
        """
        for attempt in range(ATTEMPTS):
            try:
                with self._engine.begin() as connection:
                    connection.execute(
                        insert(models.Eval)
                        .values(eval_values(eval_id, run_id, document, tier_digest))
                        .on_conflict_do_nothing(index_elements=["eval_id"])
                    )
            except Exception:
                if attempt + 1 == ATTEMPTS:
                    log.exception("eval %s could not be recorded", eval_id)
                    raise
                time.sleep(BACKOFF_SECONDS * (2**attempt))
            else:
                return
