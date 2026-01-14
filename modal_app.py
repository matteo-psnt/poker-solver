"""Modal app: run headless MCCFR training/evaluation in the cloud.

The container mirrors the local repository layout so the existing
CWD/``__file__``-relative config and data resolution works unchanged:

  * ``/root/src`` and ``/root/config`` are mounted from the local tree.
  * The ``poker-data`` Volume is mounted at ``/root/data`` and holds both the
    ``combo_abstraction/`` inputs and the ``runs/`` checkpoints, so training
    output survives the ephemeral container.
  * ``workdir`` is ``/root`` so ``data/combo_abstraction`` and ``data/runs``
    (both CWD-relative in the codebase) resolve onto the Volume.

Run the in-container smoke test with::

    uv run modal run modal_app.py

Dependencies come from ``uv.lock`` via ``uv_sync`` (deps only; the project source
is live-mounted for fast iteration).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import modal

REPO = Path(__file__).parent
DATA_MOUNT = "/root/data"
data_volume = modal.Volume.from_name("poker-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-install-project --no-default-groups")
    .env({"PYTHONPATH": "/root"})
    .workdir("/root")
    .add_local_dir(str(REPO / "config"), "/root/config")
    .add_local_dir(str(REPO / "src"), "/root/src")
)

app = modal.App("poker-solver")

# Modal RESERVES `cpu` cores but does NOT cap the container to them: inside the
# container mp.cpu_count(), os.sched_getaffinity, AND the cgroup CPU quota all report
# the HOST's cores (measured: 24 in a cpu=8 container), not the reservation. There is
# therefore no reliable in-container signal for the allocation, so num_workers must be
# pinned by the caller. If it fell through to services.train's default
# (mp.cpu_count()), a run would spawn ~24 workers regardless of the reservation — 3x the
# per-worker memory and a real OOM risk. Callers using with_options(cpu=n) must pass a
# matching num_workers=n (the calibrate sweep does).
DEFAULT_CPU = 8
DEFAULT_MEMORY_MB = 8192


@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    cpu=32,
    memory=16384,
    timeout=10800,
)
def precompute(
    abstraction_config: str,
    num_workers: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Precompute a combo abstraction on a big-core box and persist it to the Volume.

    Output lands under /root/data/combo_abstraction (the Volume). num_workers is pinned
    to the cpu reservation (the container reports host cores, not the reservation).
    """
    from src.pipeline.training import services

    out = services.precompute_abstraction(
        abstraction_config,
        num_workers=num_workers if num_workers is not None else 32,
        overwrite=overwrite,
    )
    data_volume.commit()
    return {"abstraction_config": abstraction_config, "output_dir": str(out)}


@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=3600,
)
def train(
    config_name: str,
    num_workers: int | None = None,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Run a headless training session in the container and (optionally) persist checkpoints."""
    from src.pipeline.training import services

    out = services.train(
        config_name,
        num_workers=num_workers if num_workers is not None else DEFAULT_CPU,
        num_iterations=num_iterations,
        seed=seed,
        config_overrides=config_overrides,
    )
    # Checkpoints were written under /root/data/runs (the Volume); commit so they
    # survive the container teardown. Skipped for throughput calibration runs.
    if commit:
        data_volume.commit()
    return dataclasses.asdict(out)


# LBR parallelizes over hands (embarrassingly parallel), so evaluate reserves
# DEFAULT_CPU cores and pins num_workers to match (same reason as train — the
# container reports host cores, not the reservation). The long timeout covers the
# minutes-scale default (num_hands=2000).
@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=3600,
)
def evaluate(
    run_id: str,
    method: str = "lbr",
    num_hands: int = 2000,
    equity_runouts: int = 24,
    num_workers: int | None = None,
    num_samples: int = 500,
    num_rollouts: int = 50,
    use_average_strategy: bool = True,
    seed: int | None = None,
    allin_runouts: int = 50,
) -> dict[str, Any]:
    """Evaluate a run stored on the Volume (Local Best Response by default)."""
    from src.pipeline.training import services
    from src.pipeline.training.services import LBR_ESTIMATOR_LABEL, ROLLOUT_ESTIMATOR_LABEL

    # Pick up runs committed by earlier train() calls in other containers.
    data_volume.reload()
    run_dir = Path("data/runs") / run_id
    if method == "rollout":
        out = services.evaluate_run_rollout(
            run_dir=run_dir,
            num_samples=num_samples,
            num_rollouts=num_rollouts,
            use_average_strategy=use_average_strategy,
            seed=seed,
        )
        estimator = ROLLOUT_ESTIMATOR_LABEL
    else:  # "lbr" (default, trustworthy)
        out = services.evaluate_run_lbr(
            run_dir=run_dir,
            num_hands=num_hands,
            equity_runouts=equity_runouts,
            seed=seed,
            num_workers=num_workers if num_workers is not None else DEFAULT_CPU,
            allin_runouts=allin_runouts,
        )
        estimator = LBR_ESTIMATOR_LABEL
    return {
        "run_id": run_id,
        "method": method,
        "estimator": estimator,
        "infosets": out.infosets,
        "results": out.results,
    }


# Serial (single-process) match: one blueprint copy in memory, resolver decisions
# dominate the runtime. cpu=4 covers numpy/leaf-rollout parallelism within a decision.
@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    cpu=4,
    memory=24576,
    timeout=10800,
)
def resolver_gate(
    run_id: str,
    num_deals: int = 1000,
    time_budget_ms: int = 100,
    seed: int = 1,
) -> dict[str, Any]:
    """Duplicate-deal head-to-head: blueprint+resolver vs bare blueprint."""
    from src.pipeline.training import services

    data_volume.reload()
    out = services.evaluate_run_resolver_gate(
        run_dir=Path("data/runs") / run_id,
        num_deals=num_deals,
        time_budget_ms=time_budget_ms,
        seed=seed,
    )
    return {"run_id": run_id, "infosets": out.infosets, "results": out.results}


@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_MB,
    timeout=3600,
)
def resume(
    run_id: str,
    additional_iterations: int,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Resume an existing Volume run in a fresh container and persist the result.

    Loads the latest checkpoint the trainer committed and continues training. Passing
    a ``num_workers`` different from the original run also exercises key re-partitioning.
    """
    from src.pipeline.training import services

    data_volume.reload()
    run_dir = Path("data/runs") / run_id
    session, resumed_from = services.create_resumed_session(run_dir)
    services.run_training(
        session,
        num_workers=num_workers if num_workers is not None else DEFAULT_CPU,
        num_iterations=additional_iterations,
    )
    data_volume.commit()

    metadata = services.load_run_metadata(run_dir)
    return {
        "run_id": run_id,
        "resumed_from_iteration": resumed_from,
        "final_iterations": metadata.iterations,
        "num_infosets": metadata.num_infosets,
        "status": metadata.status,
    }


@app.local_entrypoint()
def resume_test(
    config: str = "quick_test",
    seed: int = 42,
) -> None:
    """Prove resume-after-teardown: train (container A), then resume in a fresh container B.

    Trains exactly to a checkpoint boundary, then resumes with a *different* worker count
    so both cross-container checkpoint recovery and key re-partitioning are exercised.
    """
    print("Container A: training quick_test to iter=1000 (checkpoint boundary)...")
    train_result = train.remote(config_name=config, num_workers=2, num_iterations=1000, seed=seed)
    run_id = train_result["run_id"]
    print(
        f"  run_id={run_id} iterations={train_result['iterations']} "
        f"infosets={train_result['num_infosets']:,}"
    )

    print(f"\nContainer B (fresh, 4 workers): resuming {run_id} for +1500 iterations...")
    resume_result = resume.remote(run_id=run_id, additional_iterations=1500, num_workers=4)
    print(f"  resumed_from_iteration={resume_result['resumed_from_iteration']}")
    print(f"  final_iterations={resume_result['final_iterations']}")
    print(f"  num_infosets={resume_result['num_infosets']:,}")

    ok = (
        resume_result["resumed_from_iteration"] == train_result["iterations"]
        and resume_result["final_iterations"] == train_result["iterations"] + 1500
    )
    print(
        f"\nRESUME TEST: {'PASS' if ok else 'FAIL'} "
        f"(expected final={train_result['iterations'] + 1500}, "
        f"got {resume_result['final_iterations']})"
    )


@app.local_entrypoint()
def main(
    config: str = "quick_test",
    workers: int | None = None,
    iterations: int = 1200,
    seed: int = 42,
) -> None:
    """Full-loop smoke test: train in one container, evaluate the run in another.

    Exercises multiprocessing/SharedMemory training, checkpoint persistence to the
    Volume, and cross-container reads (evaluate reloads the Volume the trainer committed).
    Leaving ``workers`` unset uses the pinned DEFAULT_CPU so the safe default is exercised.
    """
    train_result = train.remote(
        config_name=config,
        num_workers=workers,
        num_iterations=iterations,
        seed=seed,
    )
    print("REMOTE TRAINING RESULT:")
    for key, value in train_result.items():
        print(f"  {key}: {value}")

    run_id = train_result["run_id"]
    print(f"\nEvaluating run {run_id} in a fresh container (LBR, small sample)...")
    eval_result = evaluate.remote(run_id=run_id, num_hands=100, seed=1)
    results = eval_result["results"]
    print("REMOTE EVALUATION RESULT:")
    print(f"  estimator:      {eval_result['estimator']}")
    print(f"  infosets:       {eval_result['infosets']:,}")
    print(f"  exploitability: {results['exploitability_mbb']:.2f} mbb/g")


@app.local_entrypoint()
def run_train(
    config: str = "production",
    cpu: int = 32,
    iterations: int = 0,
    seed: int = 42,
    capacity: int = 0,
    memory: int = 24576,
    timeout: int = 10800,
    eval_hands: int = 2000,
    eval_cpu: int = 6,
    eval_memory: int = 32768,
) -> None:
    """Train a named config and evaluate the result with LBR.

    ``iterations=0`` uses the config's own count. For large/long runs:
      --capacity above the expected final infoset count (avoids mid-run storage
        resize, which doubles capacity and briefly holds old+new arrays);
      --memory to cover arrays + the coordinator's key dict + workers;
      --timeout above the projected wall-clock (train's own default is only 1h);
      low --eval-cpu with high --eval-memory (each LBR worker rebuilds the full
        blueprint, so eval RAM ~= workers * blueprint size).
    """
    overrides = {"storage__initial_capacity": capacity} if capacity > 0 else None
    train_result = train.with_options(cpu=cpu, memory=memory, timeout=timeout).remote(
        config_name=config,
        num_workers=cpu,
        num_iterations=iterations or None,
        seed=seed,
        config_overrides=overrides,
    )
    print("TRAINING RESULT:")
    for key, value in train_result.items():
        print(f"  {key}: {value}")

    run_id = train_result["run_id"]
    print(f"\nEvaluating {run_id} with LBR ({eval_hands} hands, {eval_cpu} workers)...")
    eval_result = evaluate.with_options(cpu=eval_cpu, memory=eval_memory, timeout=timeout).remote(
        run_id=run_id, num_hands=eval_hands, num_workers=eval_cpu, seed=1
    )
    results = eval_result["results"]
    print("\nEXPLOITABILITY (LBR — rigorous lower bound):")
    print(
        f"  {results['exploitability_mbb']:.1f} mbb/g "
        f"(± {results['std_error_mbb']:.1f}; 95% CI {results['confidence_95_mbb']})"
    )
    print(f"  infosets: {eval_result['infosets']:,}")


def _print_variance_decomposition(results: dict[str, Any]) -> None:
    """Print the terminal-type variance decomposition of an LBR eval, if present."""
    decomposition = results.get("variance_decomposition")
    if not decomposition:
        return
    print("  variance by terminal type (within-group share of total):")
    for label, group in decomposition["groups"].items():
        print(
            f"    {label:>9}: {group['variance_share']:>5.1%} of variance "
            f"({group['n']} deals, {group['share_of_samples']:.1%})"
        )
    print(f"    (between-group: {decomposition['between_group_share']:.1%})")


@app.local_entrypoint()
def run_eval(
    run_id: str,
    hands: int = 2000,
    cpu: int = 6,
    memory: int = 32768,
    seed: int = 1,
) -> None:
    """LBR-evaluate an existing Volume run. Fewer workers + more memory for large
    blueprints, since each parallel worker rebuilds the full blueprint."""
    eval_result = evaluate.with_options(cpu=cpu, memory=memory).remote(
        run_id=run_id, num_hands=hands, num_workers=cpu, seed=seed
    )
    results = eval_result["results"]
    print("\nEXPLOITABILITY (LBR — rigorous lower bound):")
    print(
        f"  {results['exploitability_mbb']:.1f} mbb/g "
        f"(± {results['std_error_mbb']:.1f}; 95% CI {results['confidence_95_mbb']})"
    )
    print(f"  infosets: {eval_result['infosets']:,}")
    print(f"  base_seed: {results['base_seed']} (reuse for paired run_compare)")
    _print_variance_decomposition(results)


@app.local_entrypoint()
def run_compare(
    run_a: str,
    run_b: str,
    hands: int = 2000,
    cpu: int = 6,
    memory: int = 32768,
    seed: int = 1,
    timeout: int = 10800,
) -> None:
    """Paired LBR comparison of two Volume runs under common random numbers.

    Both evals run with the same seed, so hand ``i`` sees the identical deal in
    both; the confidence interval is computed on the per-hand *differences*, which
    cancels the shared deal luck and resolves far smaller gaps than comparing two
    independent CIs. Positive ``diff`` means ``run_b`` is less exploitable (better).
    """
    from src.pipeline.evaluation.statistics import compare_paired_samples

    fn = evaluate.with_options(cpu=cpu, memory=memory, timeout=timeout)
    call_a = fn.spawn(run_id=run_a, num_hands=hands, num_workers=cpu, seed=seed)
    call_b = fn.spawn(run_id=run_b, num_hands=hands, num_workers=cpu, seed=seed)
    result_a, result_b = call_a.get(), call_b.get()

    results_a, results_b = result_a["results"], result_b["results"]
    if results_a["base_seed"] != results_b["base_seed"]:
        raise RuntimeError(
            f"base_seed mismatch ({results_a['base_seed']} vs {results_b['base_seed']}): "
            "the evals are not paired."
        )

    def _mbb_samples(results: dict[str, Any]) -> list[float]:
        factor = 1000.0 / results["big_blind"]
        return [(r["u0"] + r["u1"]) / 2.0 * factor for r in results["hand_records"]]

    comparison = compare_paired_samples(_mbb_samples(results_a), _mbb_samples(results_b))

    for name, result in ((run_a, result_a), (run_b, result_b)):
        results = result["results"]
        print(f"\n{name} ({result['infosets']:,} infosets):")
        print(f"  {results['exploitability_mbb']:.1f} mbb/g (± {results['std_error_mbb']:.1f})")
        _print_variance_decomposition(results)

    print(f"\nPAIRED DIFFERENCE (A - B, {comparison['n']} common deals):")
    print(
        f"  {comparison['mean_diff']:.1f} mbb/g (± {comparison['se_diff']:.1f}; "
        f"95% CI [{comparison['ci_lower']:.1f}, {comparison['ci_upper']:.1f}])"
    )
    print(f"  p-value: {comparison['p_value']:.4f} | correlation: {comparison['correlation']:.3f}")
    print(
        f"  pairing gain: SE {comparison['se_diff']:.1f} vs {comparison['se_unpaired']:.1f} "
        f"unpaired ({comparison['se_unpaired'] / max(comparison['se_diff'], 1e-12):.1f}x tighter)"
    )
    if comparison["is_significant"]:
        better = run_b if comparison["mean_diff"] > 0 else run_a
        print(f"  VERDICT: {better} is significantly less exploitable (95% level).")
    else:
        print("  VERDICT: no significant difference at the 95% level.")


@app.local_entrypoint()
def resume_eval(
    run_id: str,
    additional: int = 9_000_000,
    cpu: int = 32,
    eval_hands: int = 2000,
    eval_cpu: int = 16,
) -> None:
    """Resume an existing Volume run for more iterations, then re-evaluate with LBR.

    Warm-start from a base checkpoint (no retrain from scratch); compare the LBR
    number before and after to see whether more training helped.
    """
    resume_result = resume.with_options(cpu=cpu, memory=16384).remote(
        run_id=run_id, additional_iterations=additional, num_workers=cpu
    )
    print("RESUME RESULT:")
    for key, value in resume_result.items():
        print(f"  {key}: {value}")

    print(f"\nEvaluating {run_id} with LBR ({eval_hands} hands, {eval_cpu} workers)...")
    eval_result = evaluate.with_options(cpu=eval_cpu, memory=16384).remote(
        run_id=run_id, num_hands=eval_hands, num_workers=eval_cpu, seed=1
    )
    results = eval_result["results"]
    print("\nEXPLOITABILITY (LBR — rigorous lower bound):")
    print(
        f"  {results['exploitability_mbb']:.1f} mbb/g "
        f"(± {results['std_error_mbb']:.1f}; 95% CI {results['confidence_95_mbb']})"
    )
    print(f"  infosets: {eval_result['infosets']:,}")


@app.local_entrypoint()
def run_resolver_gate(
    run_id: str,
    deals: int = 1000,
    budget_ms: int = 100,
    seed: int = 1,
) -> None:
    """Resolver gate on a Volume run: does blueprint+resolver beat the bare blueprint?

    Positive edge = the search path is alive; negative = the MVP resolver still
    hurts on this blueprint (leaf values). Duplicate seat-swapped deals cancel
    card luck, so ~1k deals resolve sub-bb/hand edges.
    """
    gate_result = resolver_gate.remote(
        run_id=run_id, num_deals=deals, time_budget_ms=budget_ms, seed=seed
    )
    results = gate_result["results"]
    print(f"\nRESOLVER GATE on {run_id} ({gate_result['infosets']:,} infosets):")
    print(
        f"  resolver edge: {results['resolver_mbb_per_hand']:+.1f} mbb/hand "
        f"(± {results['se_mbb']:.1f}; 95% CI {results['confidence_95_mbb']})"
    )
    print(
        f"  p-value: {results['p_value']:.4f} | deals: {results['num_deals']} "
        f"(paired, seat-swapped) | budget: {results['time_budget_ms']}ms"
    )
    print(
        f"  resolver decisions: {results['resolver_decisions']:,} "
        f"({results['resolver_fallbacks']} fell back to blueprint)"
    )
    if results["p_value"] < 0.05:
        verdict = "WINS" if results["resolver_mbb_per_hand"] > 0 else "LOSES"
        print(f"  VERDICT: resolver {verdict} significantly (95% level).")
    else:
        print("  VERDICT: no significant edge either way at this sample size.")


@app.local_entrypoint()
def run_precompute(config: str = "production", workers: int = 32, overwrite: bool = False) -> None:
    """Precompute an abstraction on Modal and persist it to the poker-data Volume."""
    result = precompute.remote(abstraction_config=config, num_workers=workers, overwrite=overwrite)
    print("PRECOMPUTE RESULT:")
    for key, value in result.items():
        print(f"  {key}: {value}")


@app.local_entrypoint()
def calibrate(
    config: str = "quick_test",
    iterations: int = 40000,
    capacity: int = 8_000_000,
    memory: int = 16384,
    cpus: str = "8,16,32,64",
    ipw: int = 0,
) -> None:
    """Sweep worker/core counts to measure throughput scaling and the $/run curve.

    Forces ``capacity`` (default 8M ≈ 1.4GB SharedMemory) to test the production
    ``/dev/shm`` risk at every core count. Checkpointing is disabled so timing
    reflects pure solver throughput. Note: reported it/s includes a fixed ~3.5s
    worker-pool startup, so high-core numbers are conservative.

    ``ipw`` overrides ``iterations_per_worker``: a run must span several batches
    (batch = num_workers*ipw) to reach steady state, so for configs with a large
    native ipw (e.g. production=5000) set a smaller value plus enough ``iterations``
    or high-core counts never fill a batch and the measurement is startup-dominated.
    """
    import time

    cpu_list = [int(c) for c in cpus.split(",")]
    overrides = {
        "storage__initial_capacity": capacity,
        "storage__checkpoint_enabled": False,
    }
    if ipw > 0:
        overrides["training__iterations_per_worker"] = ipw
    rows: list[tuple[int, float, float, float]] = []
    for n in cpu_list:
        print(f"\n--- cpu={n}, memory={memory}MB, capacity={capacity:,} ---")
        try:
            t0 = time.time()
            result = train.with_options(cpu=n, memory=memory).remote(
                config_name=config,
                num_workers=n,
                num_iterations=iterations,
                seed=42,
                config_overrides=overrides,
                commit=False,
            )
            wall = time.time() - t0
            ips = result["iterations_per_second"]
            rows.append((n, ips, result["runtime_seconds"], wall))
            print(f"  {ips:.0f} it/s (train {result['runtime_seconds']:.1f}s, wall {wall:.1f}s)")
        except Exception as exc:
            print(f"  FAILED at cpu={n}: {type(exc).__name__}: {str(exc)[:200]}")

    print("\n=== CALIBRATION SUMMARY ===")
    print(f"  config={config} iterations={iterations:,} capacity={capacity:,}")
    print(f"{'cpu':>5} {'it/s':>9} {'train_s':>9} {'wall_s':>8} {'it/s/cpu':>9}")
    for n, ips, train_s, wall in rows:
        print(f"{n:>5} {ips:>9.0f} {train_s:>9.1f} {wall:>8.1f} {ips / n:>9.1f}")
