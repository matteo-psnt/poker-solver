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


@app.function(image=image, volumes={DATA_MOUNT: data_volume}, cpu=2, timeout=3600)
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
        num_workers=num_workers,
        num_iterations=num_iterations,
        seed=seed,
        config_overrides=config_overrides,
    )
    # Checkpoints were written under /root/data/runs (the Volume); commit so they
    # survive the container teardown. Skipped for throughput calibration runs.
    if commit:
        data_volume.commit()
    return dataclasses.asdict(out)


@app.function(image=image, volumes={DATA_MOUNT: data_volume}, cpu=2, timeout=1800)
def evaluate(
    run_id: str,
    num_samples: int = 500,
    num_rollouts: int = 50,
    use_average_strategy: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a run stored on the Volume and return its (rollout-estimator) metrics."""
    from src.pipeline.training import services
    from src.pipeline.training.services import ROLLOUT_ESTIMATOR_LABEL

    # Pick up runs committed by earlier train() calls in other containers.
    data_volume.reload()
    run_dir = Path("data/runs") / run_id
    out = services.evaluate_run(
        run_dir=run_dir,
        num_samples=num_samples,
        num_rollouts=num_rollouts,
        use_average_strategy=use_average_strategy,
        seed=seed,
    )
    return {
        "run_id": run_id,
        "estimator": ROLLOUT_ESTIMATOR_LABEL,
        "infosets": out.infosets,
        "results": out.results,
    }


@app.function(image=image, volumes={DATA_MOUNT: data_volume}, cpu=4, timeout=3600)
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
    services.run_training(session, num_workers=num_workers, num_iterations=additional_iterations)
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
    workers: int = 2,
    iterations: int = 1200,
    seed: int = 42,
) -> None:
    """Full-loop smoke test: train in one container, evaluate the run in another.

    Exercises multiprocessing/SharedMemory training, checkpoint persistence to the
    Volume, and cross-container reads (evaluate reloads the Volume the trainer committed).
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
    print(f"\nEvaluating run {run_id} in a fresh container...")
    eval_result = evaluate.remote(run_id=run_id, num_samples=100, num_rollouts=20, seed=1)
    results = eval_result["results"]
    print("REMOTE EVALUATION RESULT:")
    print(f"  estimator:      {eval_result['estimator']}")
    print(f"  infosets:       {eval_result['infosets']:,}")
    print(f"  exploitability: {results['exploitability_mbb']:.2f} mbb/g")


@app.local_entrypoint()
def calibrate(
    config: str = "quick_test",
    iterations: int = 40000,
    capacity: int = 8_000_000,
    memory: int = 16384,
    cpus: str = "8,16,32,64",
) -> None:
    """Sweep worker/core counts to measure throughput scaling and the $/run curve.

    Forces ``capacity`` (default 8M ≈ 1.4GB SharedMemory) to test the production
    ``/dev/shm`` risk at every core count. Checkpointing is disabled so timing
    reflects pure solver throughput. Note: reported it/s includes a fixed ~3.5s
    worker-pool startup, so high-core numbers are conservative.
    """
    import time

    cpu_list = [int(c) for c in cpus.split(",")]
    overrides = {
        "storage__initial_capacity": capacity,
        "storage__checkpoint_enabled": False,
    }
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
