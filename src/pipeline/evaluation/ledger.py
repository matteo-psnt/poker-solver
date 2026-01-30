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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.pipeline.evaluation.hunl_local_best_response import LBRConfig
from src.shared.gitinfo import get_git_commit, is_git_dirty
from src.shared.jsonio import json_default

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
    representation_version: int


# Knobs that define an eval's comparison tier. Two evals may only be paired if these
# match (plus a shared base_seed) — otherwise the comparison mixes exploiters or
# measured strategies and the number is meaningless. Kept as data so `compare` and
# the record builder agree on exactly what "same tier" means.
TIER_KNOBS = ("scorer", "opponent", "include_off_tree")


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


def build_rollout_knobs_from_params(
    *, samples: int, rollouts: int, use_current: bool, base_seed: Any
) -> dict[str, Any]:
    """Canonical rollout knob tier, built from explicit values (see LBR counterpart)."""
    return {
        "samples": samples,
        "rollouts": rollouts,
        "use_current": bool(use_current),
        "base_seed": base_seed,
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
        "card_abstraction_hash": provenance.card_abstraction_hash,
        "action_config_hash": provenance.action_config_hash,
        "representation_version": provenance.representation_version,
        "knobs": knobs,
        "results": {
            "exploitability_mbb": results.get("exploitability_mbb"),
            "std_error_mbb": results.get("std_error_mbb"),
            "num_hands": results.get("num_hands"),
            "n": len(samples),
        },
        "result_path": str(result_path),
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
    result_path = write_payload(run_dir, payload, knobs)
    record = build_record(
        provenance=provenance,
        method=method,
        estimator=estimator,
        infosets=payload["infosets"],
        knobs=knobs,
        results=results,
        result_path=result_path,
        timestamp=timestamp or datetime.now().isoformat(),
    )
    append_record(record, ledger_path)
    return result_path, record


def write_payload(run_dir: Path, payload: dict[str, Any], knobs: dict[str, Any]) -> Path:
    """Write the full eval payload to a non-overwriting per-eval file under the run dir.

    Named by timestamp + a knob hash so re-evaluating a run under different (or the
    same) settings never clobbers a prior result — the pre-ledger ``evaluate_result.json``
    was overwritten on every eval, silently discarding history.
    """
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = evals_dir / f"eval-{stamp}-{_knob_hash(knobs)}.json"
    path.write_text(json.dumps(payload, indent=2, default=json_default))
    return path


def append_record(record: dict[str, Any], ledger_path: Path = DEFAULT_LEDGER_PATH) -> None:
    """Append one row to the ledger JSONL, creating it if needed."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=json_default) + "\n")


def read_records(ledger_path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    """Read all ledger rows in append order. Missing ledger → empty list."""
    if not ledger_path.exists():
        return []
    records = []
    with open(ledger_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_record_for_run(
    run_id: str, ledger_path: Path = DEFAULT_LEDGER_PATH
) -> dict[str, Any] | None:
    """Most recent ledger row for a run id (by append order), or None."""
    match = None
    for record in read_records(ledger_path):
        if record.get("run_id") == run_id:
            match = record
    return match


def load_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Load the full per-eval payload a ledger row points at."""
    path = Path(record["result_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Eval payload for run '{record.get('run_id')}' not found at {path}"
        )
    return json.loads(path.read_text())


def tier_mismatches(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Return human-readable reasons two ledger rows must not be paired (empty if OK).

    Enforces the two rules that were previously discipline-only: a *shared, non-null
    base seed* (paired common-random-numbers requires hand-for-hand identical deals)
    and *identical comparison-tier knobs* (never mix scorer/opponent/off-tree). Equal
    hand counts are required too, since paired stats need equal-length sequences.
    """
    reasons: list[str] = []
    ka, kb = a.get("knobs", {}), b.get("knobs", {})

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
            reasons.append(
                f"{knob} differs ({ka.get(knob)!r} vs {kb.get(knob)!r}): mixing tiers "
                "compares two different exploiters/strategies, not two runs."
            )

    na = a.get("results", {}).get("num_hands")
    nb = b.get("results", {}).get("num_hands")
    if na != nb:
        reasons.append(f"num_hands differs ({na} vs {nb}): paired samples must be equal-length.")

    return reasons
