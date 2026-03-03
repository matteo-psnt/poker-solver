"""Append-only per-run training-metrics history.

``MetricsTracker`` computes rich convergence signals every batch (utility,
speed, and regret/entropy health) but only ever rendered them to a tqdm
postfix. ``.run.json`` keeps a *final snapshot* only, so questions like "where
did positive regret plateau?" or "when did strategy entropy collapse?" are
unanswerable after a run ends.

This writer persists one JSON line per batch to ``run_dir/metrics.jsonl`` — a
durable convergence curve you can plot or diff across runs. It is append-mode so
a resumed run continues the same curve rather than truncating it, and every
write is best-effort: a metrics failure must never take down training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MetricsHistoryWriter:
    """Appends batch-level metric rows to ``run_dir/metrics.jsonl`` (one JSON line each).

    Rows are written at batch granularity (coarse), so a plain open-append-close
    per row is cheap and needs no handle lifecycle. Writes are best-effort: the
    first error disables the writer rather than propagating into the training loop.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._disabled = False

    def append(self, row: dict[str, Any]) -> None:
        """Write one metrics row. Never raises — logs once and disables on error."""
        if self._disabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception as exc:  # pragma: no cover - defensive; metrics must not kill a run
            logger.warning(f"[metrics-history] disabled after write error: {exc}")
            self._disabled = True
