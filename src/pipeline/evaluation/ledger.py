"""Append-only ledger of evaluation results.

Cross-run comparison used to live entirely in a human notebook: scores, seeds,
scorer/opponent tiers, commit hashes, and p-values all tracked by discipline. The
recurring failure modes of that discipline are well known — mixing scorer or
opponent tiers in one comparison, or pairing two evals that did not share a base
seed (which silently invalidates the common-random-numbers variance cancellation
behind every p-value).

This ledger turns those rules into structure. Every evaluation appends one
compact row (provenance + the exact knob tier + summary result + a pointer to the
full per-eval payload) to ``data/eval_ledger.jsonl``; the full payload — including
the per-hand ``pair_samples_mbb`` needed for paired comparison — is written to a
non-overwriting file under the run dir. The ``compare`` command then reads two
rows and *refuses* to pair mismatched seeds or knob tiers, so the guardrail is
mechanical rather than remembered.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.hunl_local_best_response import LBRConfig
from src.shared.gitinfo import get_git_commit, is_git_dirty
from src.shared.jsonio import json_default

logger = logging.getLogger(__name__)

LEDGER_SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = Path("data/eval_ledger.jsonl")


@dataclass(frozen=True)
class RunProvenance:
    """Provenance of the evaluated run, recorded verbatim in each ledger row.

    Plain fields rather than the training layer's ``RunMetadata`` — the ledger
    only needs these scalars, and taking them directly keeps evaluation from
    importing training.
    """

    run_id: str
    git_commit: str | None
    git_dirty: bool | None
    config_name: str
    card_abstraction_hash: str | None
    action_config_hash: str | None
    # Experiment lineage, copied onto the row so a report can group and attribute
    # arms without opening every run's .run.json. None on unaffiliated runs.
    experiment_id: str | None = None
    arm: str | None = None
    parent_run_id: str | None = None
    # `config_name` is not identity (it comes from system.config_name in the YAML),
    # so an override-variant is only distinguishable from its base by this.
    config_hash: str | None = None


# Knobs that define an eval's comparison tier. Two evals may only be paired if these
# match (plus a shared base_seed) — otherwise the comparison mixes exploiters or
# measured strategies and the number is meaningless. Kept as data so `compare` and
# the record builder agree on exactly what "same tier" means.
TIER_KNOBS = ("scorer", "opponent", "include_off_tree")

# Knobs that also change what is being measured, but only when they apply to the
# method in question — a myopic eval has no lookahead depth, a blueprint opponent
# has no resolver iterations. Checked when present on either side, so a depth-2 and
# a depth-4 lookahead can no longer pair silently (they are different exploiters).
CONDITIONAL_TIER_KNOBS = (
    "runouts",
    "resolver_iterations",
    "lookahead_depth",
    "lookahead_top_k",
    "num_flops",
    "num_turns",
    "num_rivers",
)


def _knob_hash(knobs: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()
    return digest[:8]


def build_lbr_knobs_from_params(
    *,
    scorer: str,
    opponent: str,
    hands: int,
    runouts: int,
    include_off_tree: bool,
    base_seed: Any,
    resolver_iterations: int | None = None,
    lookahead_depth: int | None = None,
    lookahead_top_k: int | None = None,
) -> dict[str, Any]:
    """Canonical LBR knob tier, built from explicit values.

    ``base_seed`` is the seed the deals were actually drawn from (LBR resolves a random
    seed when none is passed and reports it back), which is the value paired comparison
    must match on. Tier-specific knobs are included only when they apply, so a
    blueprint+myopic eval and a deployed+lookahead eval never collide on knob shape.
    """
    knobs: dict[str, Any] = {
        "scorer": scorer,
        "opponent": opponent,
        "hands": hands,
        "runouts": runouts,
        "include_off_tree": bool(include_off_tree),
        "base_seed": base_seed,
    }
    if opponent == "deployed":
        knobs["resolver_iterations"] = resolver_iterations
    if scorer == "lookahead":
        knobs["lookahead_depth"] = lookahead_depth
        knobs["lookahead_top_k"] = lookahead_top_k
    return knobs


def build_lbr_knobs(config: LBRConfig, results: dict[str, Any]) -> dict[str, Any]:
    """Canonical LBR knob tier for an eval that ran under ``config``.

    Deriving the tier from the same :class:`LBRConfig` the eval consumed makes
    "every transport records identical tiers" structural — the guardrail in
    :func:`tier_mismatches` only works if all surfaces agree on exactly what
    "same tier" means. ``base_seed`` and the deployed resolver's pinned
    ``resolver_iterations`` come from the effective ``results`` because both are
    resolved during the eval, not fixed by the config object.
    """
    return build_lbr_knobs_from_params(
        scorer=config.scorer,
        opponent=config.opponent,
        hands=config.num_hands,
        runouts=config.equity_runouts,
        include_off_tree=config.include_off_tree,
        base_seed=results.get("base_seed"),
        resolver_iterations=results.get("resolver_iterations"),
        lookahead_depth=config.lookahead_depth,
        lookahead_top_k=config.lookahead_top_k,
    )


def build_exact_br_knobs_from_params(
    *, num_flops: int, num_turns: int, num_rivers: int, board_seed: int
) -> dict[str, Any]:
    """Canonical exact-BR knob tier: the board plan IS the comparison tier.

    ``base_seed`` mirrors ``board_seed`` so the pairing guard applies unchanged:
    two exact-BR evals are comparable iff they scored the same sampled boards.
    Values are deterministic points — within a matched tier a difference is
    exact, with no paired samples or p-value involved.
    """
    return {
        "num_flops": num_flops,
        "num_turns": num_turns,
        "num_rivers": num_rivers,
        "base_seed": board_seed,
    }


def build_record(
    *,
    provenance: RunProvenance,
    method: str,
    estimator: str,
    infosets: int,
    knobs: dict[str, Any],
    results: dict[str, Any],
    result_path: Path,
    timestamp: str,
    checkpoint_iteration: int | None = None,
) -> dict[str, Any]:
    """Compose the compact ledger row (no per-hand samples — those live in the payload)."""
    samples = results.get("pair_samples_mbb") or []
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "timestamp": timestamp,
        "run_id": provenance.run_id,
        "method": method,
        "estimator": estimator,
        # Two commits matter and mean different things: the code that produced the
        # checkpoint, and the code that measured it (LBR methodology changes across
        # commits). Both are recorded so neither has to be reconstructed later.
        "train_git_commit": provenance.git_commit,
        "train_git_dirty": provenance.git_dirty,
        "eval_git_commit": get_git_commit(),
        "eval_git_dirty": is_git_dirty(),
        "config_name": provenance.config_name,
        "config_hash": provenance.config_hash,
        "card_abstraction_hash": provenance.card_abstraction_hash,
        "action_config_hash": provenance.action_config_hash,
        "experiment_id": provenance.experiment_id,
        "arm": provenance.arm,
        "parent_run_id": provenance.parent_run_id,
        # WHICH checkpoint produced this number. A run id alone does not identify
        # one: the same run is evaluated at successive iterations, so without this
        # two rows for one run are indistinguishable and a stale read looks like a
        # real result. ``infosets`` was already being passed in and silently dropped.
        "checkpoint_iteration": checkpoint_iteration,
        "infosets": infosets,
        "knobs": knobs,
        "results": {
            "exploitability_mbb": results.get("exploitability_mbb"),
            "std_error_mbb": results.get("std_error_mbb"),
            "num_hands": results.get("num_hands"),
            "n": len(samples),
        },
        "result_path": payload_pointer(result_path, provenance.run_id),
    }


def record_evaluation(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    provenance: RunProvenance,
    method: str,
    estimator: str,
    knobs: dict[str, Any],
    ledger_path: Path = DEFAULT_LEDGER_PATH,
    timestamp: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Persist one evaluation: non-clobbering payload under the run dir + a ledger row.

    The single recording path shared by every caller (local CLI and Modal), so a
    cloud eval and a local eval produce the same on-disk provenance and can be paired
    by :func:`tier_mismatches` without either surface reimplementing the schema.

    ``payload`` must carry ``results`` (with the per-hand ``pair_samples_mbb``) and
    ``infosets``. Returns the payload path and the appended record.
    """
    results = payload["results"]
    slug = eval_slug(knobs)
    result_path = write_payload(run_dir, payload, slug)
    record = build_record(
        provenance=provenance,
        method=method,
        estimator=estimator,
        infosets=payload["infosets"],
        knobs=knobs,
        results=results,
        result_path=result_path,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        checkpoint_iteration=payload.get("checkpoint_iteration"),
    )
    write_record(run_dir, record, slug)
    append_record(record, ledger_path)
    return result_path, record


def eval_slug(knobs: dict[str, Any]) -> str:
    """Unique per-eval filename stem: UTC stamp + knob hash + random suffix.

    The random suffix is what makes this safe for concurrent writers. Microsecond
    stamps alone collide when several boxes evaluate the same run with the same
    knobs -- exactly the fan-out shape a noise-floor sweep produces -- and the repo
    already learned this for ``run_id`` (``uuid4().hex[:6]`` in session.py). UTC so
    names from boxes in different timezones still sort.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}-{_knob_hash(knobs)}-{uuid.uuid4().hex[:6]}"


def write_payload(run_dir: Path, payload: dict[str, Any], slug: str) -> Path:
    """Write the full eval payload to a non-overwriting per-eval file under the run dir.

    Named by timestamp + knob hash + random suffix so re-evaluating a run under
    different (or the same) settings never clobbers a prior result — the pre-ledger
    ``evaluate_result.json`` was overwritten on every eval, discarding history.
    """
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    path = evals_dir / f"eval-{slug}.json"
    path.write_text(json.dumps(payload, indent=2, default=json_default))
    return path


def write_record(run_dir: Path, record: dict[str, Any], slug: str) -> Path:
    """Write the complete ledger row beside its payload, under the run directory.

    This is the durable copy. The shared ``eval_ledger.jsonl`` is an append to a
    file every writer shares, which is the one shared-mutable-state in the system
    and the one thing that has actually lost data (12/14 rows in a parallel sweep).
    A uniquely-named file under the run being evaluated has no such contention: a
    run directory has a single writer, so this write cannot race anything.

    The full row is written, not just the payload -- ``eval_git_commit``, ``knobs``
    and ``timestamp`` are captured at record time and exist nowhere else, so a
    payload alone cannot reconstruct a row.
    """
    records_dir = run_dir / "evals"
    records_dir.mkdir(parents=True, exist_ok=True)
    path = records_dir / f"record-{slug}.json"
    path.write_text(json.dumps(record, indent=2, default=json_default))
    return path


def append_record(record: dict[str, Any], ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append one row to the ledger JSONL cache, creating it if needed.

    Best-effort convenience so the common single-machine case needs no rebuild.
    The durable copy is :func:`write_record`; anything lost here is recoverable
    with ``ledger --rebuild``.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default) + "\n")


def record_instant(record: dict[str, Any]) -> datetime:
    """When an eval happened, as an aware datetime, for ordering.

    Two timestamp vintages coexist and must not be compared as strings. Rows
    written before the UTC switch are naive *local* time -- that is what
    ``datetime.now()`` produced at write time -- while new rows carry an explicit
    ``+00:00``. Lexicographic comparison would skew the two apart by the writer's
    UTC offset, so naive values are attached to the local zone (the zone that
    actually produced them) rather than reinterpreted as UTC.

    Unparseable or missing timestamps sort first, keeping them visible rather
    than dropping them.
    """
    raw = str(record.get("timestamp") or "")
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    # astimezone() on a naive value interprets it as local time, which is the
    # assumption we want here and the reason this is not `replace(tzinfo=UTC)`.
    return stamp.astimezone() if stamp.tzinfo is None else stamp


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


def tier_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Identity of the comparison tier a row belongs to.

    The same rule :func:`tier_mismatches` enforces pairwise, expressed as a
    groupable key. Both must cover the SAME knobs or they contradict each other:
    without the conditional ones a depth-2 and a depth-4 lookahead eval hash into
    one tier and get plotted on a single axis, which is exactly the silent
    instrument-mixing a tier is supposed to prevent. Same for exact_br rows scored
    over different board budgets.
    """
    knobs = record.get("knobs") or {}
    return (
        record.get("method"),
        *(knobs.get(k) for k in TIER_KNOBS),
        *(knobs.get(k) for k in CONDITIONAL_TIER_KNOBS),
        knobs.get("base_seed"),
    )


def tier_label(record: dict[str, Any]) -> str:
    """Human-readable one-line description of a row's tier.

    Must name every knob :func:`tier_key` splits on, or two genuinely different
    tiers render as identical strings -- and the operator sees "also recorded, not
    mixed in: <the same text>" with no way to tell what ``--tier 1`` would select.
    Conditional knobs are shown only when present, so a myopic row is not padded
    with lookahead fields that do not apply to it.
    """
    knobs = record.get("knobs") or {}
    parts = [str(record.get("method") or "?")]
    parts += [f"{k}={knobs[k]}" for k in TIER_KNOBS if knobs.get(k) is not None]
    parts += [f"{k}={knobs[k]}" for k in CONDITIONAL_TIER_KNOBS if knobs.get(k) is not None]
    if knobs.get("base_seed") is not None:
        parts.append(f"seed={knobs['base_seed']}")
    return " ".join(parts)


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


def payload_pointer(result_path: Path, run_id: str) -> str:
    """Portable, run-relative pointer to a payload: ``<run_id>/evals/<file>``.

    Rows used to store a working-directory-relative path, which resolves to nothing
    on a machine that mounts its data elsewhere -- or, worse, to a *different*
    machine's local ``data/``. Anchoring at the run id makes the pointer mean the
    same thing wherever the runs directory happens to live.
    """
    return f"{run_id}/{result_path.parent.name}/{result_path.name}"


def load_payload(record: dict[str, Any], runs_dir: Path | None = None) -> dict[str, Any]:
    """Load the full per-eval payload a ledger row points at.

    Tries, in order: the run-relative pointer under ``runs_dir`` (current format),
    the stored path as-is (legacy CWD-relative rows), and finally the payload's
    basename under the run's ``evals/`` — the filename is unique by construction, so
    that last one recovers a row whose pointer was written by an older layout.
    """
    stored = str(record["result_path"])
    run_id = record.get("run_id") or ""
    candidates: list[Path] = []
    if runs_dir is not None:
        candidates.append(runs_dir / stored)
    candidates.append(Path(stored))
    if runs_dir is not None and run_id:
        candidates.append(runs_dir / run_id / "evals" / Path(stored).name)

    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())

    raise FileNotFoundError(f"Eval payload for run '{run_id}' not found at {stored}")


def tier_mismatches(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return human-readable reasons two ledger rows must not be paired (empty if OK).

    Enforces the two rules that were previously discipline-only: a *shared, non-null
    base seed* (paired common-random-numbers requires hand-for-hand identical deals)
    and *identical comparison-tier knobs* (never mix scorer/opponent/off-tree). Equal
    hand counts are required too, since paired stats need equal-length sequences.

    Also refuses to pair across ``method``, and to pair rows whose payloads carry no
    per-hand samples: two ``exact_br`` rows used to pass every check vacuously (their
    knobs have no scorer/opponent/off-tree keys, so ``None == None``) and then fail
    downstream with a bare ``KeyError: 'pair_samples_mbb'``.
    """
    reasons: list[str] = []
    ka, kb = a.get("knobs", {}), b.get("knobs", {})

    method_a, method_b = a.get("method"), b.get("method")
    if method_a != method_b:
        reasons.append(
            f"method differs ({method_a!r} vs {method_b!r}): these are different "
            "estimators, not two measurements of the same thing."
        )
    elif method_a == "exact_br":
        reasons.append(
            "exact_br rows carry no per-hand samples, so there is nothing to pair. "
            "Compare their exploitability_mbb directly — within a matched board tier "
            "the difference is exact and needs no p-value."
        )

    for knob in ("card_abstraction_hash", "action_config_hash"):
        if a.get(knob) != b.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({a.get(knob)!r} vs {b.get(knob)!r}): the two runs are "
                "bucketed differently, so their exploitability numbers are not on one scale."
            )

    seed_a, seed_b = ka.get("base_seed"), kb.get("base_seed")
    if seed_a is None or seed_b is None:
        reasons.append(
            "base_seed missing on one side: paired CRN comparison needs both evals run "
            "with the same explicit --seed so hand i is the same deal in both."
        )
    elif seed_a != seed_b:
        reasons.append(
            f"base_seed differs ({seed_a} vs {seed_b}): the deals are not paired, so the "
            "variance cancellation behind the p-value does not hold."
        )

    for knob in TIER_KNOBS:
        if ka.get(knob) != kb.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({ka.get(knob)!r} vs {kb.get(knob)!r}): mixing tiers "
                "compares two different exploiters/strategies, not two runs."
            )

    for knob in CONDITIONAL_TIER_KNOBS:
        if (knob in ka or knob in kb) and ka.get(knob) != kb.get(knob):
            reasons.append(  # noqa: PERF401 - multi-line message reads worse as a genexp
                f"{knob} differs ({ka.get(knob)!r} vs {kb.get(knob)!r}): the exploiter "
                "searched to a different depth/width, so the two numbers are not comparable."
            )

    na = a.get("results", {}).get("num_hands")
    nb = b.get("results", {}).get("num_hands")
    if na != nb:
        reasons.append(f"num_hands differs ({na} vs {nb}): paired samples must be equal-length.")

    return reasons
