"""Reading the ledger back: selection, rebuild, and curve series."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.ledger.records import (
    DEFAULT_LEDGER_PATH,
    ledger_row,
    payload_pointer,
    record_instant,
)
from src.pipeline.evaluation.ledger.tiers import tier_key, tier_label
from src.shared import records as record_store
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

    def _report(number: int) -> None:
        # The substrate skips a torn line; only this caller can name the repair.
        logger.warning(
            "Skipping unparseable ledger line %d in %s; `ledger --rebuild` "
            "can regenerate rows that have a per-run record.",
            number,
            ledger_path,
        )

    rows = record_store.read_log(ledger_path, on_bad_line=_report)
    return sorted(rows, key=record_instant)


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
    # ANY eval document, not just the ones that happened to have a record file.
    # Provenance and samples used to live in different files, so 59 of the 78
    # evals on disk could not be rebuilt at all.
    for path in sorted(runs_dir.glob("*/evals/*.json")):
        # Legacy names are skipped, not read. `migrate_eval_files` is
        # deliberately non-destructive, so during the migration window both
        # layouts sit in one directory -- and a legacy record points at the OLD
        # filename, so reading both enters the same evaluation twice under two
        # pointers. Measured: 63 ledger rows became 110.
        if path.name.startswith(("eval-", "record-")):
            continue
        document = record_store.read_snapshot(path)
        if document is None or "run_id" not in document:
            continue
        # The ledger is the COMPARISON index, so a row that cannot be tiered has
        # no business in it: with no knobs and no timestamp it hashes into a tier
        # of (method, None, ...) and sorts to year 1 AD -- and since tiers rank
        # by coverage, a pile of them would become the DEFAULT curve. The
        # document stays on disk; only the index is withheld.
        if not document.get("knobs") or not document.get("timestamp"):
            continue
        key = document.get("result_path")
        if key:
            recovered[key] = ledger_row(document)

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


def migrate_eval_files(runs_dir: Path, ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, int]:
    """Convert the old three-shape eval layout into one document per evaluation.

    An evaluation used to be a payload (samples, almost no provenance), a record
    (provenance, a four-key summary), and a ledger row identical to the record.
    Provenance and samples lived in different files, so most evals could not be
    rebuilt from either alone -- on the tree this was written against, 59 of 78
    payloads had no matching record.

    The LEDGER is the fullest history, not the records, so it is the key: each
    row names a payload, and the row supplies the provenance that payload lacks.
    Merging the two yields a document strictly richer than either.

    Forward-only and non-destructive: the originals are left in place for an
    operator to delete once satisfied. Returns counts, so a caller can report
    what happened rather than claim success.
    """
    rows_by_pointer = {
        r.get("result_path"): r for r in read_records(ledger_path) if r.get("result_path")
    }
    counts = {"merged": 0, "payload_only": 0, "record_only": 0, "skipped": 0}

    for payload_path in sorted(runs_dir.glob("*/evals/eval-*.json")):
        slug = payload_path.name.removeprefix("eval-").removesuffix(".json")
        run_dir = payload_path.parent.parent
        target = payload_path.parent / f"{slug}.json"
        if target.exists():
            counts["skipped"] += 1
            continue

        payload = record_store.read_snapshot(payload_path)
        if payload is None:
            counts["skipped"] += 1
            continue

        record = record_store.read_snapshot(payload_path.parent / f"record-{slug}.json")
        if record is None:
            record = rows_by_pointer.get(payload_pointer(payload_path, run_dir.name))
        if record is None:
            # No provenance anywhere. Kept rather than dropped: the measurement
            # is still real, and a document that says less is better than one
            # that invents what it cannot know.
            document = dict(payload)
            counts["payload_only"] += 1
        else:
            document = {**record, "results": payload.get("results", record.get("results", {}))}
            counts["merged"] += 1
        document["result_path"] = payload_pointer(target, run_dir.name)
        record_store.write_snapshot(target, document, record_store.REGISTRY["evals/*.json"])

    # The move renames the file a row points at, so every row naming an old
    # payload must be re-pointed. Without this the rebuilt ledger holds the same
    # evaluation twice -- once under the old pointer, once under the new -- which
    # is exactly the silent duplication the ledger exists to prevent. Measured on
    # the real tree: 63 rows became 110.
    _repoint_ledger(ledger_path)

    # Records whose payload is gone still carry provenance and a summary.
    for record_path in sorted(runs_dir.glob("*/evals/record-*.json")):
        slug = record_path.name.removeprefix("record-").removesuffix(".json")
        target = record_path.parent / f"{slug}.json"
        if target.exists():
            continue
        record = record_store.read_snapshot(record_path)
        if record is None:
            counts["skipped"] += 1
            continue
        record["result_path"] = payload_pointer(target, record_path.parent.parent.name)
        record_store.write_snapshot(target, record, record_store.REGISTRY["evals/*.json"])
        counts["record_only"] += 1

    return counts


def _repoint_ledger(ledger_path: Path) -> None:
    """Re-point ledger rows at the consolidated document they now live in.

    ``<run>/evals/eval-<slug>.json`` becomes ``<run>/evals/<slug>.json``. Written
    through a temporary file and replaced, so a kill cannot leave the ledger
    half-repointed.
    """
    rows = read_records(ledger_path)
    if not rows:
        return
    for row in rows:
        pointer = str(row.get("result_path") or "")
        head, _, name = pointer.rpartition("/")
        if name.startswith("eval-"):
            row["result_path"] = f"{head}/{name.removeprefix('eval-')}"
    tmp = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, default=json_default) + "\n" for r in rows))
    tmp.replace(ledger_path)
