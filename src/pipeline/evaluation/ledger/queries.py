"""Reading the ledger back: selection, rebuild, and curve series."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.ledger.records import DEFAULT_LEDGER_PATH, record_instant
from src.pipeline.evaluation.ledger.tiers import tier_key, tier_label
from src.shared.jsonio import json_default

logger = logging.getLogger(__name__)


def read_records(ledger_path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    """Read all ledger rows, oldest first. Missing ledger → empty list.

    Sorted by recorded instant rather than file order: rows written by different
    machines arrive interleaved, so append order is "whose write landed last", not
    "when the eval happened".

    A torn line is skipped rather than raised on. An unterminated final write
    would otherwise make the whole ledger unreadable -- including by
    ``ledger --rebuild``, the one command able to repair it.
    """
    if not ledger_path.exists():
        return []
    records = []
    with ledger_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                logger.warning(
                    "Skipping unparseable ledger line %d in %s; `ledger --rebuild` "
                    "can regenerate rows that have a per-run record.",
                    number,
                    ledger_path,
                )
    return sorted(records, key=record_instant)


def rebuild_ledger(runs_dir: Path, ledger_path: Path = DEFAULT_LEDGER_PATH) -> tuple[int, int]:
    """Regenerate the ledger cache from the per-run records on disk.

    Forward-only by necessity: rows written before :func:`write_record` existed have
    no record file to rebuild from, and their ``eval_git_commit``/``knobs``/
    ``timestamp`` exist nowhere else. Those rows are preserved verbatim rather than
    dropped, so a rebuild is always non-destructive.

    Returns ``(recovered, preserved)``.
    """
    existing = read_records(ledger_path)

    recovered: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*/evals/record-*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        key = record.get("result_path")
        if key:
            recovered[key] = record

    # Every row without a record file is kept verbatim, duplicates included. The
    # historical ledger genuinely contains rows sharing a result_path (an eval
    # recorded twice during the 07-18 clobber recovery); collapsing them here would
    # make a command whose whole purpose is not losing rows lose rows.
    preserved = [r for r in existing if r.get("result_path") not in recovered]
    merged = sorted([*preserved, *recovered.values()], key=record_instant)

    # Write to a temp file and rename, rather than truncating in place. The
    # `preserved` rows are by definition the ones with no per-run record to
    # regenerate them from, so a crash midway through an in-place rewrite would
    # destroy exactly the rows this function exists to protect. Same tmp+replace
    # pattern as the checkpoint manifest in engine/solver/storage/helpers.py.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=json_default) + "\n" for r in merged))
    tmp.replace(ledger_path)
    return len(recovered), len(preserved)


def latest_record_for_run(
    run_id: str,
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    checkpoint_iteration: int | None = None,
) -> dict[str, Any] | None:
    """Most recent ledger row for a run id (by append order), or None.

    ``checkpoint_iteration`` selects a specific checkpoint of that run. A run id
    alone is ambiguous once a run has been evaluated at more than one iteration --
    the common case for a long run scored at successive checkpoints -- and without
    this the newest row silently wins, so two checkpoints of one run cannot be
    compared at all.

    "Most recent" is by recorded timestamp (:func:`read_records` sorts), not by
    position in the file: with several machines appending, file order says whose
    write landed last, which is not the same question.
    """
    match = None
    for record in read_records(ledger_path):
        if record.get("run_id") != run_id:
            continue
        if (
            checkpoint_iteration is not None
            and record.get("checkpoint_iteration") != checkpoint_iteration
        ):
            continue
        match = record
    return match


def curve_series(
    records: list[dict[str, Any]], run_id: str
) -> list[tuple[str, dict[int, dict[str, Any]]]]:
    """Group one run's rows into per-tier convergence series, best-covered first.

    A convergence curve is only meaningful within a single tier -- mixing a
    depth-2 and a depth-4 lookahead scorer plots two different instruments on one
    axis -- so this never merges tiers. Within a tier the *last* row for a given
    checkpoint wins (a re-evaluation supersedes its predecessor).
    """
    series: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    labels: dict[tuple[Any, ...], str] = {}
    for record in records:
        if record.get("run_id") != run_id:
            continue
        iteration = record.get("checkpoint_iteration")
        if iteration is None:
            continue
        key = tier_key(record)
        labels.setdefault(key, tier_label(record))
        series.setdefault(key, {})[int(iteration)] = record
    return [(labels[k], points) for k, points in sorted(series.items(), key=_series_rank)]


def _series_rank(item: tuple[tuple[Any, ...], dict[int, dict[str, Any]]]) -> tuple[int, int]:
    """Most-covered series first; ties broken by the deepest checkpoint reached."""
    _, points = item
    return (-len(points), -max(points, default=0))
