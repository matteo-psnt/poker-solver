"""Modal app: run headless MCCFR training/evaluation in the cloud.

Phase 1.2 walking skeleton. The container mirrors the local repository layout so
the existing CWD/``__file__``-relative config and data resolution works unchanged:

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
        )
        estimator = LBR_ESTIMATOR_LABEL
    return {
        "run_id": run_id,
        "method": method,
        "estimator": estimator,
        "infosets": out.infosets,
        "results": out.results,
    }


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
    eval_hands: int = 2000,
    eval_cpu: int = 16,
) -> None:
    """Train a real run and evaluate it with LBR — the improvement-loop primitive.

    ``iterations=0`` uses the config's own count (production = 1M). Training runs at
    the ~32-core sweet spot; LBR eval uses fewer workers (each rebuilds the blueprint).
    """
    train_result = train.with_options(cpu=cpu, memory=16384).remote(
        config_name=config,
        num_workers=cpu,
        num_iterations=iterations or None,
        seed=seed,
    )
    print("TRAINING RESULT:")
    for key, value in train_result.items():
        print(f"  {key}: {value}")

    run_id = train_result["run_id"]
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
