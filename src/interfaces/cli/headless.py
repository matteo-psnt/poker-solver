"""Non-interactive (headless) entrypoint for training and evaluation.

Unlike the questionary menu in :mod:`src.interfaces.cli.app`, every operation here
is fully specified by CLI flags and emits a machine-readable summary. This is the
surface used by scripts and cloud execution — where an interactive prompt is not an
option. Every long-running operation (train, resume, evaluate, precompute) is
reachable here, so a cloud job is a shell invocation of this module rather than a
provider-specific reimplementation.

Cloud callers should prefer importing :func:`src.pipeline.services.train`
directly (it returns a ``TrainingOutput`` object); this module is the local /
subprocess transport around the same function and additionally writes a
``result.json`` into the run directory.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from src.interfaces.cli.headless_render import print_human
from src.pipeline import services
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.hunl_local_best_response import LBRConfig
from src.pipeline.evaluation.public_tree_br import PublicBRConfig
from src.pipeline.evaluation.statistics import compare_paired_samples
from src.pipeline.services import RolloutParams
from src.shared import checkpoint_profile
from src.shared.jsonio import json_default
from src.shared.log import configure_logging


def _write_result(run_dir: Path, payload: dict[str, Any]) -> None:
    """Persist a per-operation result file (e.g. ``train_result.json``) in the run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{payload['op']}_result.json").write_text(
        json.dumps(payload, indent=2, default=json_default)
    )


def _resolve_run_dir(run: str, runs_dir: str) -> Path:
    """Resolve a run identifier (name under ``runs_dir``) or an explicit path."""
    as_path = Path(run)
    if as_path.is_dir():
        return as_path
    candidate = Path(runs_dir) / run
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"Run not found: '{run}' (looked at {as_path} and {candidate})")


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set key__path=value`` into the config loader's override kwargs.

    Values go through JSON so ``1000``/``true``/``null`` arrive as the types the
    strict config models require; anything JSON rejects stays a plain string, which
    is what bare names like ``--set system__config_name=probe`` want.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got '{pair}'")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def _cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    out = services.train(
        args.config,
        num_workers=args.workers,
        num_iterations=args.iterations,
        seed=args.seed,
        config_overrides=_parse_overrides(args.overrides),
        experiment=services.ExperimentTag(
            experiment_id=args.experiment,
            arm=args.arm,
            parent_run_id=args.parent,
        ),
    )
    payload: dict[str, Any] = {"op": "train", **dataclasses.asdict(out)}
    _write_result(Path(out.runs_dir) / out.run_id, payload)
    return payload


def _cmd_train_static(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.train_static`."""
    out = services.train_static(
        args.config,
        num_workers=args.workers,
        num_iterations=args.iterations,
        seed=args.seed,
        config_overrides=_parse_overrides(args.overrides),
        experiment=services.ExperimentTag(
            experiment_id=args.experiment,
            arm=args.arm,
            parent_run_id=args.parent,
        ),
        checkpoint_every=args.checkpoint_every,
        run_id=args.run,
    )
    payload: dict[str, Any] = {"op": "train-static", **dataclasses.asdict(out)}
    _write_result(Path(out.runs_dir) / out.run_id, payload)
    return payload


def _cmd_resume(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.resume`."""
    run_dir = _resolve_run_dir(args.run, args.runs_dir)
    out = services.resume(
        run_dir,
        args.to_iteration,
        num_workers=args.workers,
        capacity_override=args.capacity,
    )
    payload: dict[str, Any] = {
        "op": "resume",
        "runs_dir": args.runs_dir,
        **dataclasses.asdict(out),
    }
    _write_result(run_dir, payload)
    return payload


def _cmd_precompute(args: argparse.Namespace) -> dict[str, Any]:
    """Precompute a combo abstraction into ``data/combo_abstraction/<name>``."""
    out = services.precompute_abstraction(
        args.config,
        num_workers=args.workers,
        overwrite=args.overwrite,
    )
    return {
        "op": "precompute",
        "abstraction_config": args.config,
        "output_dir": str(out),
    }


def _cmd_curve(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.exploitability_curve`."""
    run_dir = _resolve_run_dir(args.run, args.runs_dir)
    out = services.exploitability_curve(
        run_dir,
        ledger_path=Path(args.ledger),
        tier_index=args.tier,
    )
    return {"op": "curve", "decay_ratio": out.decay_ratio, **dataclasses.asdict(out)}


def _cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.experiment_report`."""
    out = services.experiment_report(
        args.experiment,
        ledger_path=Path(args.ledger),
        runs_dir=Path(args.runs_dir),
        baseline_path=Path(args.baseline),
    )
    return {"op": "report", **dataclasses.asdict(out)}


def _cmd_promote(args: argparse.Namespace) -> dict[str, Any]:
    """Point the baseline at a run, closing one turn of the base-fork loop."""
    run_dir = _resolve_run_dir(args.run, args.runs_dir)
    baseline = services.promote_baseline(
        run_dir.name,
        args.rationale,
        path=Path(args.baseline),
        checkpoint_iteration=services.checkpoint_iteration_of(run_dir),
    )
    return {"op": "promote", "baseline": str(args.baseline), **dataclasses.asdict(baseline)}


def _cmd_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Argparse transport around :func:`services.evaluate_and_record`.

    All dispatch, payload shaping, and ledger recording live in the orchestrator;
    this function only maps flags onto the params objects. The orchestrator's
    ledger warning prints to stdout, which under ``--json`` is redirected to
    stderr — keeping the machine-readable payload clean.
    """
    run_dir = _resolve_run_dir(args.run, args.runs_dir)
    payload = services.evaluate_and_record(
        run_dir,
        method=args.method,
        lbr=LBRConfig(
            num_hands=args.hands,
            equity_runouts=args.runouts,
            include_off_tree=args.include_off_tree,
            seed=args.seed,
            num_workers=args.workers,
            allin_runouts=args.allin_runouts,
            opponent=args.opponent,
            scorer=args.scorer,
            lookahead_depth=args.lookahead_depth,
            lookahead_top_k=args.lookahead_top_k,
        ),
        rollout=RolloutParams(
            num_samples=args.samples,
            num_rollouts=args.rollouts,
            use_average_strategy=not args.current,
            seed=args.seed,
        ),
        exact_br=PublicBRConfig(
            num_flops=args.br_flops,
            num_turns=args.br_turns,
            num_rivers=args.br_rivers,
            board_seed=args.br_board_seed,
            # --workers is shared with lbr; exact_br splits its four independent
            # (seat, button) walks over it, so 4 saturates the useful range.
            num_workers=args.workers,
        ),
        resolver_iterations=args.resolver_iterations,
        abstraction_hash=args.abstraction_hash,
        at_iteration=args.at,
        ledger_path=Path(args.ledger),
    )
    _write_result(run_dir, payload)
    return payload


def _cmd_ledger(args: argparse.Namespace) -> dict[str, Any]:
    """List recent eval-ledger rows as a compact table, optionally rebuilding first."""
    ledger_path = Path(args.ledger)
    rebuilt = None
    if args.rebuild:
        recovered, preserved = eval_ledger.rebuild_ledger(Path(args.runs_dir), ledger_path)
        rebuilt = {"recovered": recovered, "preserved": preserved}

    records = eval_ledger.read_records(ledger_path)
    if args.run:
        records = [r for r in records if r.get("run_id") == args.run]
    if args.experiment:
        records = [r for r in records if r.get("experiment_id") == args.experiment]
    if args.method:
        records = [r for r in records if r.get("method") == args.method]
    if args.since:
        # Instants, not strings: the ledger holds naive-local legacy rows beside
        # UTC-aware new ones, so a lexicographic cutoff skews by the writer's
        # offset — the exact defect `record_instant` exists to remove.
        cutoff = eval_ledger.record_instant({"timestamp": args.since})
        records = [r for r in records if eval_ledger.record_instant(r) >= cutoff]
    # `records[-0:]` is the whole list, so a 0 limit already meant "all" by accident.
    # Made deliberate: `--limit 0` is how a rebuild shows everything it recovered.
    if args.limit > 0:
        records = records[-args.limit :]
    return {
        "op": "ledger",
        "ledger": str(args.ledger),
        "rebuilt": rebuilt,
        "rows": records,
    }


def _cmd_checkpoint_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Summarize a run's per-checkpoint phase timings and the Volume commit."""
    run_dir = Path(args.runs_dir) / args.run
    path = run_dir / checkpoint_profile.PROFILE_FILENAME
    if not path.exists():
        raise SystemExit(
            f"No checkpoint profile at {path}. It is written per checkpoint, so the "
            "run must have checkpointed at least once with profiling in place."
        )

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    checkpoints = [r for r in rows if r.get("event") != "volume_commit"]
    commits = [r for r in rows if r.get("event") == "volume_commit"]

    phase_totals: dict[str, float] = {}
    for row in checkpoints:
        for name, secs in row.get("phases", {}).items():
            phase_totals[name] = phase_totals.get(name, 0.0) + secs

    checkpoint_seconds = sum(r["total_seconds"] for r in checkpoints)
    commit_seconds = sum(r["total_seconds"] for r in commits)
    # storage_write wraps the engine-level phases, so counting it alongside them
    # would double-count; collect_keys and storage_write are the top-level split.
    top_level = {k: v for k, v in phase_totals.items() if k in ("collect_keys", "storage_write")}

    return {
        "op": "checkpoint-profile",
        "run": args.run,
        "num_checkpoints": len(checkpoints),
        "checkpoint_seconds": round(checkpoint_seconds, 2),
        "volume_commit_seconds": round(commit_seconds, 2),
        "total_seconds": round(checkpoint_seconds + commit_seconds, 2),
        "commit_share": (
            round(commit_seconds / (checkpoint_seconds + commit_seconds), 3)
            if checkpoint_seconds + commit_seconds > 0
            else None
        ),
        "top_level_phases": {k: round(v, 2) for k, v in sorted(top_level.items())},
        "write_phases": {
            k: round(v, 2)
            for k, v in sorted(phase_totals.items(), key=lambda kv: -kv[1])
            if k not in ("collect_keys", "storage_write")
        },
        "checkpoints": checkpoints,
        "volume_commits": commits,
    }


def _cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    """Paired (common-random-numbers) comparison of two runs' latest evals."""
    ledger_path = Path(args.ledger)
    rec_a = eval_ledger.latest_record_for_run(args.a, ledger_path, args.a_at)
    rec_b = eval_ledger.latest_record_for_run(args.b, ledger_path, args.b_at)
    if rec_a is None or rec_b is None:
        missing, at = (args.a, args.a_at) if rec_a is None else (args.b, args.b_at)
        at_note = f" at checkpoint iteration {at}" if at is not None else ""
        raise SystemExit(f"No ledger entry found for run '{missing}'{at_note} in {ledger_path}")

    reasons = eval_ledger.tier_mismatches(rec_a, rec_b)
    if reasons and not args.force:
        joined = "\n".join(f"  - {r}" for r in reasons)
        raise SystemExit(
            "Refusing to compare: the two evals are not a valid paired comparison:\n"
            f"{joined}\n"
            "Re-run both evals with matching knobs and the same --seed, or pass --force "
            "to override (the resulting p-value will not be trustworthy)."
        )

    runs_dir = Path(args.runs_dir)
    payload_a = eval_ledger.load_payload(rec_a, runs_dir)
    payload_b = eval_ledger.load_payload(rec_b, runs_dir)

    # Checked AFTER --force, deliberately: --force overrides a judgement about
    # whether a comparison is meaningful, but it cannot conjure per-hand samples
    # that were never recorded. exact_br is deterministic and stores none, so the
    # forced path would otherwise die on a bare KeyError.
    samples_a = payload_a["results"].get("pair_samples_mbb")
    samples_b = payload_b["results"].get("pair_samples_mbb")
    if not samples_a or not samples_b:
        missing, rec_missing = (args.a, rec_a) if not samples_a else (args.b, rec_b)
        raise SystemExit(
            f"Cannot pair: the eval for '{missing}' recorded no per-hand samples "
            f"(method '{rec_missing.get('method')}'). Deterministic estimators like "
            "exact_br have nothing to pair — within a matched board tier compare "
            "their exploitability_mbb directly; no p-value applies. --force does "
            "not help here."
        )

    comparison = compare_paired_samples(samples_a, samples_b)
    return {
        "op": "compare",
        "run_a": args.a,
        "run_b": args.b,
        # Which checkpoints were actually compared: a run id alone does not say.
        "checkpoint_iteration_a": rec_a.get("checkpoint_iteration"),
        "checkpoint_iteration_b": rec_b.get("checkpoint_iteration"),
        "forced": bool(reasons and args.force),
        "tier_warnings": reasons,
        "comparison": comparison,
    }


# One builder per subcommand below. Each takes the shared `--json` parent so the
# flag stays defined once, and ends in `set_defaults(func=...)` binding the command
# to its transport — the parser and the function it dispatches to sit together.
# argparse exports no public name for what add_subparsers() returns, so the private
# one is the only way to type these builders at all.
_SubParsers = argparse._SubParsersAction  # noqa: SLF001


def _add_train_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run train`."""
    p_train = sub.add_parser(
        "train", parents=[common], help="Train a solver from a named training config."
    )
    p_train.add_argument("--config", required=True, help="Config stem under config/training/.")
    p_train.add_argument(
        "--workers", type=int, default=None, help="Parallel workers (default: all CPUs)."
    )
    p_train.add_argument(
        "--iterations", type=int, default=None, help="Override the config iteration count."
    )
    p_train.add_argument("--seed", type=int, default=None, help="Override system.seed.")
    p_train.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Nested config override, `__` as the separator — e.g. "
        "--set storage__checkpoint_retain_every=1000. Repeatable.",
    )
    p_train.add_argument("--experiment", default=None, help="Experiment id this run is an arm of.")
    p_train.add_argument(
        "--arm",
        default=None,
        help="Arm within the experiment, e.g. 'control' or 'variant:pruning'. A variant's "
        "score is uninterpretable without a paired control — the extra training a fork "
        "gets moves the number on its own.",
    )
    p_train.add_argument(
        "--parent", default=None, help="Run id this was forked from (base-fork lineage)."
    )
    p_train.set_defaults(func=_cmd_train)


def _add_train_static_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run train-static`."""
    p_ts = sub.add_parser(
        "train-static",
        parents=[common],
        help="Train over the statically-enumerated tree (fixed memory, no key maps).",
    )
    p_ts.add_argument("--config", required=True, help="Config stem under config/training/.")
    p_ts.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker processes. Unlike `train`, this does NOT raise memory: the table "
        "is shared and there are no per-worker key maps, so it is a pure throughput knob.",
    )
    p_ts.add_argument(
        "--iterations", type=int, default=None, help="Override the config iteration count."
    )
    p_ts.add_argument("--seed", type=int, default=None, help="Override system.seed.")
    p_ts.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Nested config override, `__` as the separator. Repeatable.",
    )
    p_ts.add_argument("--experiment", default=None, help="Experiment id this run is an arm of.")
    p_ts.add_argument("--arm", default=None, help="Arm within the experiment.")
    p_ts.add_argument("--parent", default=None, help="Run id this was forked from.")
    p_ts.add_argument(
        "--checkpoint-every",
        type=int,
        default=1_000_000,
        dest="checkpoint_every",
        help="Checkpoint every N iterations (0 = only at the end). This is the bound "
        "on what a killed run loses, traded against disk and write time: a full table "
        "is written each time, and at 250k the writes were ~17%% of a 30M run's wall "
        "clock and left 120 snapshots on the share.",
    )
    p_ts.add_argument(
        "--run",
        default=None,
        help="Continue an EXISTING run instead of starting one. --iterations is an "
        "ABSOLUTE target, so re-running past it is a no-op and a retry converges.",
    )
    p_ts.set_defaults(func=_cmd_train_static)


def _add_resume_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run resume`."""
    p_resume = sub.add_parser(
        "resume",
        parents=[common],
        help="Resume a run and train up to an absolute iteration target.",
    )
    p_resume.add_argument("--run", required=True, help="Run id (dir name) or path to a run dir.")
    p_resume.add_argument(
        "--runs-dir", default="data/runs", help="Base runs dir for id resolution."
    )
    p_resume.add_argument(
        "--to-iteration",
        type=int,
        required=True,
        help="ABSOLUTE target iteration (not an increment) — retry-safe under scheduler restarts.",
    )
    p_resume.add_argument(
        "--workers", type=int, default=None, help="Parallel workers (default: all CPUs)."
    )
    p_resume.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Pre-allocate shared storage above the checkpoint's capacity (avoids mid-run resize).",
    )
    p_resume.set_defaults(func=_cmd_resume)


def _add_precompute_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run precompute`."""
    p_precompute = sub.add_parser(
        "precompute",
        parents=[common],
        help="Precompute a combo abstraction into data/combo_abstraction/.",
    )
    p_precompute.add_argument(
        "--config", required=True, help="Abstraction config stem (e.g. production)."
    )
    p_precompute.add_argument(
        "--workers", type=int, default=None, help="Parallel workers (default: config value)."
    )
    p_precompute.add_argument(
        "--overwrite", action="store_true", help="Recompute even if a complete abstraction exists."
    )
    p_precompute.set_defaults(func=_cmd_precompute)


def _add_eval_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run evaluate`."""
    p_eval = sub.add_parser(
        "evaluate",
        parents=[common],
        help="Evaluate a run's exploitability (Local Best Response by default).",
    )
    p_eval.add_argument("--run", required=True, help="Run id (dir name) or path to a run dir.")
    p_eval.add_argument("--runs-dir", default="data/runs", help="Base runs dir for id resolution.")
    p_eval.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Append-only eval ledger path (records provenance + knobs + result).",
    )
    p_eval.add_argument(
        "--at",
        type=int,
        default=None,
        help="Score a RETAINED checkpoint at this iteration instead of the run's latest "
        "(requires storage.checkpoint_retain_every at train time). Repeat over the ladder "
        "under one fixed config to build a within-run convergence curve.",
    )
    p_eval.add_argument(
        "--method",
        choices=["lbr", "rollout", "exact_br"],
        default="lbr",
        help="lbr = Local Best Response (trustworthy, default); rollout = legacy diagnostic; "
        "exact_br = deterministic exact BR on a sampled public tree (zero eval variance; "
        "compare within a matched board tier).",
    )
    # LBR options (--method lbr).
    p_eval.add_argument("--hands", type=int, default=1000, help="[lbr] Number of hands.")
    p_eval.add_argument("--runouts", type=int, default=12, help="[lbr] Equity runouts per node.")
    p_eval.add_argument("--workers", type=int, default=1, help="[lbr] Parallel workers over hands.")
    p_eval.add_argument(
        "--include-off-tree",
        action="store_true",
        help="[lbr] Add off-tree bet/raise sizes to the exploiter's menu (rigorous via "
        "shadow-state translation; changes the measured completion — re-baseline).",
    )
    p_eval.add_argument(
        "--allin-runouts",
        type=int,
        default=50,
        help="[lbr] Board runouts averaged at all-in showdown terminals "
        "(variance reduction; same expectation).",
    )
    p_eval.add_argument(
        "--abstraction-hash",
        default=None,
        help="Pin the card abstraction to this hash (see the abstraction's metadata.json "
        "'config_hash'). Default: the hash recorded on the run.",
    )
    p_eval.add_argument(
        "--opponent",
        choices=["blueprint", "deployed"],
        default="blueprint",
        help="[lbr] Strategy under measurement: raw table, or blueprint+resolver as deployed.",
    )
    p_eval.add_argument(
        "--resolver-iterations",
        type=int,
        default=64,
        help="[lbr] Pinned subgame-CFR iterations per deployed-opponent solve.",
    )
    p_eval.add_argument(
        "--scorer",
        choices=["myopic", "lookahead"],
        default="myopic",
        help="[lbr] Exploiter action selection: myopic one-step arithmetic, or a "
        "depth-limited best-response lookahead vs the blueprint (stronger exploiter).",
    )
    p_eval.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        help="[lbr] Opponent-response levels the lookahead scorer expands.",
    )
    p_eval.add_argument(
        "--lookahead-top-k",
        type=int,
        default=3,
        help="[lbr] Lookahead-rescore only the top-k myopic candidates (<=0: all).",
    )
    # Exact-BR options (--method exact_br). The board plan defines the comparison
    # tier: evals pair iff flops/turns/rivers and the board seed all match.
    p_eval.add_argument(
        "--br-flops", type=int, default=8, help="[exact_br] Sampled canonical flops (>=1755: all)."
    )
    p_eval.add_argument(
        "--br-turns", type=int, default=2, help="[exact_br] Turn cards per board node."
    )
    p_eval.add_argument(
        "--br-rivers", type=int, default=2, help="[exact_br] River cards per board node."
    )
    p_eval.add_argument(
        "--br-board-seed", type=int, default=7, help="[exact_br] Seed pinning the board sample."
    )
    # Rollout options (--method rollout).
    p_eval.add_argument("--samples", type=int, default=500, help="[rollout] Number of samples.")
    p_eval.add_argument("--rollouts", type=int, default=50, help="[rollout] Rollouts per infoset.")
    p_eval.add_argument(
        "--current",
        action="store_true",
        help="[rollout] Evaluate the current strategy instead of the average.",
    )
    p_eval.add_argument("--seed", type=int, default=None, help="Random seed (default: random).")
    p_eval.set_defaults(func=_cmd_evaluate)


def _add_ledger_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run ledger`."""
    p_ledger = sub.add_parser(
        "ledger", parents=[common], help="List recorded evaluations from the eval ledger."
    )
    p_ledger.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Eval ledger path to read.",
    )
    p_ledger.add_argument("--run", default=None, help="Filter to a single run id.")
    p_ledger.add_argument("--experiment", default=None, help="Filter to one experiment id.")
    p_ledger.add_argument(
        "--method", default=None, choices=["lbr", "rollout", "exact_br"], help="Filter by method."
    )
    p_ledger.add_argument(
        "--since", default=None, metavar="ISO8601", help="Only rows at or after this timestamp."
    )
    p_ledger.add_argument(
        "--limit", type=int, default=25, help="Show only the last N rows (0 = all)."
    )
    p_ledger.add_argument("--runs-dir", default="data/runs", help="Runs dir scanned by --rebuild.")
    p_ledger.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate the ledger from the per-run records on disk before listing. "
        "Recovers rows lost to concurrent writers; rows predating per-run records are "
        "preserved as-is, never dropped.",
    )
    p_ledger.set_defaults(func=_cmd_ledger)


def _add_curve_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run curve`."""
    p_curve = sub.add_parser(
        "curve",
        parents=[common],
        help="Within-run exploitability vs iteration, from the retained checkpoint ladder.",
    )
    p_curve.add_argument("--run", required=True, help="Run id (dir name) or path to a run dir.")
    p_curve.add_argument("--runs-dir", default="data/runs", help="Base runs dir for id resolution.")
    p_curve.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Eval ledger path to read.",
    )
    p_curve.add_argument(
        "--tier",
        type=int,
        default=0,
        help="Which comparison tier to plot when a run was scored by more than one "
        "(0 = best-covered). Tiers are never merged — see the listing in the output.",
    )
    p_curve.set_defaults(func=_cmd_curve)


def _add_report_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run report`."""
    p_report = sub.add_parser(
        "report",
        parents=[common],
        help="Score every arm of an experiment, each attributed against its control.",
    )
    p_report.add_argument("--experiment", required=True, help="Experiment id to report on.")
    p_report.add_argument(
        "--runs-dir", default="data/runs", help="Runs dir, for resolving eval payloads."
    )
    p_report.add_argument(
        "--ledger", default=str(eval_ledger.DEFAULT_LEDGER_PATH), help="Eval ledger path."
    )
    p_report.add_argument(
        "--baseline", default=str(services.DEFAULT_BASELINE_PATH), help="Baseline pointer file."
    )
    p_report.set_defaults(func=_cmd_report)


def _add_promote_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run promote`."""
    p_promote = sub.add_parser(
        "promote",
        parents=[common],
        help="Make a run the new baseline (closes one turn of the base-fork loop).",
    )
    p_promote.add_argument("--run", required=True, help="Run id to promote.")
    p_promote.add_argument(
        "--rationale",
        required=True,
        help="Why this run becomes the baseline. Required — a lineage that moved for "
        "an unrecorded reason cannot be audited later.",
    )
    p_promote.add_argument(
        "--runs-dir", default="data/runs", help="Base runs dir for id resolution."
    )
    p_promote.add_argument(
        "--baseline", default=str(services.DEFAULT_BASELINE_PATH), help="Baseline pointer file."
    )
    p_promote.set_defaults(func=_cmd_promote)


def _add_profile_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run checkpoint-profile`."""
    p_profile = sub.add_parser(
        "checkpoint-profile",
        parents=[common],
        help="Per-checkpoint phase timings and Volume-commit cost for a run.",
    )
    p_profile.add_argument("--run", required=True, help="Run id to summarize.")
    p_profile.add_argument(
        "--runs-dir", default="data/runs", help="Directory containing run directories."
    )
    p_profile.set_defaults(func=_cmd_checkpoint_profile)


def _add_compare_parser(sub: _SubParsers, common: argparse.ArgumentParser) -> None:
    """Arguments for `poker-solver-run compare`."""
    p_compare = sub.add_parser(
        "compare",
        parents=[common],
        help="Paired (CRN) comparison of two runs' latest evals; refuses mismatched tiers.",
    )
    p_compare.add_argument("--a", required=True, help="First run id (baseline).")
    p_compare.add_argument("--b", required=True, help="Second run id (candidate).")
    p_compare.add_argument(
        "--runs-dir", default="data/runs", help="Runs dir, for resolving eval payloads."
    )
    p_compare.add_argument(
        "--a-at",
        type=int,
        default=None,
        help=(
            "Checkpoint iteration to select for --a. Needed when a run has been "
            "evaluated at more than one checkpoint; otherwise the newest row wins."
        ),
    )
    p_compare.add_argument(
        "--b-at",
        type=int,
        default=None,
        help="Checkpoint iteration to select for --b (see --a-at).",
    )
    p_compare.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Eval ledger path to read.",
    )
    p_compare.add_argument(
        "--force",
        action="store_true",
        help="Compare even if seeds/knob tiers differ (p-value will not be trustworthy).",
    )
    p_compare.set_defaults(func=_cmd_compare)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        help="Emit the result payload as JSON only (no human-readable summary).",
    )

    parser = argparse.ArgumentParser(
        prog="poker-solver-run",
        description="Headless training/evaluation entrypoint for scripts and cloud runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(sub, common)
    _add_train_static_parser(sub, common)
    _add_resume_parser(sub, common)
    _add_precompute_parser(sub, common)
    _add_eval_parser(sub, common)
    _add_ledger_parser(sub, common)
    _add_curve_parser(sub, common)
    _add_report_parser(sub, common)
    _add_promote_parser(sub, common)
    _add_profile_parser(sub, common)
    _add_compare_parser(sub, common)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.json:
        # Library layers log to stderr, but third-party writers (numba, zarr) can still
        # print to stdout; redirect so the JSON blob is the ONLY thing on stdout and
        # machine consumers can parse it directly.
        with contextlib.redirect_stdout(sys.stderr):
            payload = args.func(args)
        print(json.dumps(payload, indent=2, default=json_default))
    else:
        payload = args.func(args)
        print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
