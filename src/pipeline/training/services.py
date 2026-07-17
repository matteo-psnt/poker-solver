"""Service-layer APIs for training and evaluation orchestration."""

import functools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.core.game.state import Street
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.hunl_local_best_response import (
    LBRConfig,
    LBRResult,
    compute_lbr_exploitability,
    dominant_terminal,
)
from src.pipeline.evaluation.resolver_match import play_resolver_match
from src.pipeline.evaluation.statistics import variance_decomposition
from src.pipeline.training.abstraction_resolver import AbstractionHashMismatchError
from src.pipeline.training.components import (
    build_evaluation_solver,
    evaluate_solver_exploitability,
)
from src.pipeline.training.run_tracker import RunMetadata, RunTracker
from src.pipeline.training.trainer import TrainingSession
from src.shared.config import Config
from src.shared.config_loader import load_training_config
from src.shared.units import pair_mean_mbb

# Local Best Response: a rigorous lower bound on exploitability (LBR <= exact BR,
# validated on Kuhn/Leduc). This is the trustworthy default metric.
LBR_ESTIMATOR_LABEL = "local_best_response (rigorous lower bound on exploitability)"

# The legacy `evaluate_run` metric is a one-ply rollout that both understates the
# structure it explores AND is upward-biased by a recursive max over noisy MC
# estimates — it is not a valid bound in either direction. Kept as an explicit
# opt-in for diagnostics/comparison only; do NOT treat it as exploitability.
ROLLOUT_ESTIMATOR_LABEL = "rollout_one_ply (uninformative; not a valid bound — diagnostic only)"


@dataclass(frozen=True)
class EvaluationOutput:
    """Container for run evaluation output."""

    infosets: int
    results: dict[str, Any]


@dataclass(frozen=True)
class RolloutParams:
    """Settings for the legacy one-ply rollout estimator (diagnostic opt-in only)."""

    num_samples: int = 500
    num_rollouts: int = 50
    use_average_strategy: bool = True
    seed: int | None = None


@dataclass(frozen=True)
class TrainingOutput:
    """Machine-readable summary of a completed training run.

    ``run_id`` is a portable identifier (the run directory's name relative to
    ``runs_dir``), never an absolute path, so a follow-up evaluate/resume call
    can locate the run regardless of where a volume is mounted.
    """

    run_id: str
    runs_dir: str
    config_name: str
    iterations: int
    num_infosets: int
    runtime_seconds: float
    iterations_per_second: float
    storage_capacity: int
    status: str


def list_runs(runs_dir: Path) -> list[str]:
    """List available training runs in the provided base directory."""
    return RunTracker.list_runs(runs_dir)


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Load run metadata from an existing run directory."""
    tracker = RunTracker.load(run_dir)
    return tracker.metadata


def create_training_session(config: Config) -> TrainingSession:
    """Create a new training session."""
    return TrainingSession(config)


def create_resumed_session(run_dir: Path) -> tuple[TrainingSession, int]:
    """Create a resumed session and return it with latest completed iteration."""
    metadata = load_run_metadata(run_dir)
    latest_iteration = metadata.iterations
    return TrainingSession.resume(run_dir), latest_iteration


def run_training(
    session: TrainingSession,
    *,
    num_workers: int | None = None,
    num_iterations: int | None = None,
) -> None:
    """Execute training for an existing session."""
    session.train(
        num_workers=num_workers,
        num_iterations=num_iterations,
    )


def start_training(config: Config, num_workers: int) -> TrainingSession:
    """Start a new training session and run it."""
    session = create_training_session(config)
    run_training(session, num_workers=num_workers)
    return session


def train(
    config_name: str,
    *,
    num_workers: int | None = None,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> TrainingOutput:
    """Run a full training session from a named config and return a portable summary.

    This is the headless, non-interactive training entrypoint used by scripts and
    cloud (Modal) execution. It loads the config, verifies the card abstraction is
    present, trains, and returns a :class:`TrainingOutput` — no stdout parsing required.

    Args:
        config_name: Stem of a config under ``config/training`` (e.g. ``"quick_test"``).
        num_workers: Parallel worker count; defaults to all available CPUs when ``None``.
        num_iterations: Overrides the config's iteration count when provided.
        seed: Overrides ``system.seed`` for reproducibility when provided.
        config_overrides: Extra nested config overrides (``__`` separator), e.g.
            ``{"storage__initial_capacity": 8_000_000}`` for calibration sweeps.

    Raises:
        FileNotFoundError: If the card abstraction for the config is missing (precompute it).
        ValueError: If the card abstraction is stale (config hash mismatch — recompute it).
    """
    overrides: dict[str, Any] = dict(config_overrides or {})
    if seed is not None:
        overrides["system__seed"] = seed
    config = load_training_config(config_name, **overrides)

    # TrainingSession.__init__ builds the card abstraction before creating the run
    # directory and cleans up on failure, so we surface its errors here with an
    # actionable message rather than pre-loading the (large) abstraction pickle twice.
    try:
        session = create_training_session(config)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Card abstraction '{config.card_abstraction.config}' for training config "
            f"'{config_name}' is missing. Precompute it before training. ({e})"
        ) from e
    except AbstractionHashMismatchError as e:
        raise AbstractionHashMismatchError(
            f"Card abstraction '{config.card_abstraction.config}' for training config "
            f"'{config_name}' is stale (config hash mismatch). Recompute it. ({e})"
        ) from e

    run_training(session, num_workers=num_workers, num_iterations=num_iterations)

    metadata = load_run_metadata(session.run_dir)
    ips = metadata.iterations / metadata.runtime_seconds if metadata.runtime_seconds > 0 else 0.0
    return TrainingOutput(
        run_id=metadata.run_id,
        runs_dir=config.training.runs_dir,
        config_name=metadata.config_name,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        runtime_seconds=metadata.runtime_seconds,
        iterations_per_second=ips,
        storage_capacity=metadata.storage_capacity,
        status=metadata.status,
    )


def resume_training(run_dir: Path, additional_iterations: int) -> tuple[TrainingSession, int]:
    """Resume training from run_dir and execute additional_iterations."""
    session, latest_iteration = create_resumed_session(run_dir)
    run_training(session, num_iterations=additional_iterations)
    return session, latest_iteration


def precompute_abstraction(
    abstraction_config: str,
    *,
    num_workers: int | None = None,
    base_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Headless precompute of a combo abstraction; return the output directory.

    Output goes to ``<base_dir>/data/combo_abstraction/<name>`` (``base_dir`` defaults
    to the working directory, matching the resolver's lookup). Skips work if a complete
    abstraction already exists there unless ``overwrite`` is set.
    """
    config = PrecomputeConfig.from_yaml(abstraction_config)
    if num_workers is not None:
        config = config.model_copy(update={"num_workers": num_workers})
    out = abstraction_output_path(base_dir or Path.cwd(), config)
    if not overwrite and (out / "metadata.json").exists():
        return out
    precomputer = PostflopPrecomputer(config)
    precomputer.precompute_all(streets=[Street.FLOP, Street.TURN, Street.RIVER])
    precomputer.save(out)
    return out


def _load_blueprint(
    config: Config, checkpoint_dir: Path, abstraction_hash: str | None = None
) -> object:
    """Build a fresh evaluation blueprint (solver) from a checkpoint.

    Used as a picklable factory (via ``functools.partial``) so parallel-LBR worker
    processes each construct their own solver — the solver holds a non-picklable
    Cython member and cannot be sent across a process boundary.
    """
    solver, _ = build_evaluation_solver(
        config, checkpoint_dir=checkpoint_dir, abstraction_hash=abstraction_hash
    )
    return solver


def _lbr_results_dict(result: LBRResult, big_blind: int) -> dict[str, Any]:
    """Map an LBRResult into the portable results dict.

    Per-hand records, ready-made paired samples, and the base seed travel with
    the aggregate so a later paired (common-random-numbers) comparison against
    another run — or an offline variance decomposition — never requires
    re-running the eval or re-deriving the sample definition.
    """
    samples_mbb = [pair_mean_mbb(o0.value, o1.value, big_blind) for o0, o1 in result.hand_outcomes]
    groups = [dominant_terminal(o0.terminal, o1.terminal) for o0, o1 in result.hand_outcomes]
    return {
        "exploitability_mbb": result.exploitability_mbb,
        "exploitability_bb": result.exploitability_bb,
        "std_error_mbb": result.std_error_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "lbr_utility_p0": result.lbr_utility_p0,
        "lbr_utility_p1": result.lbr_utility_p1,
        "num_hands": result.num_hands,
        "base_seed": result.base_seed,
        "big_blind": big_blind,
        "pair_samples_mbb": samples_mbb,
        "hand_records": [
            {
                "u0": o0.value,
                "u1": o1.value,
                "terminal_p0": o0.terminal,
                "terminal_p1": o1.terminal,
                "pot_p0": o0.pot,
                "pot_p1": o1.pot,
            }
            for o0, o1 in result.hand_outcomes
        ],
        "variance_decomposition": (
            variance_decomposition(samples_mbb, groups) if samples_mbb else None
        ),
    }


def evaluate_run_lbr(
    run_dir: Path,
    config: LBRConfig | None = None,
    *,
    resolver_iterations: int = 64,
    abstraction_hash: str | None = None,
) -> EvaluationOutput:
    """Evaluate a run's exploitability via Local Best Response (trustworthy default).

    LBR is a rigorous lower bound on true exploitability (LBR <= exact BR, validated
    on Kuhn/Leduc). Every eval knob — hand count, scorer, opponent model, off-tree
    menu, parallelism — travels in ``config`` (:class:`LBRConfig`), so transports
    construct one object instead of relisting knobs; see the LBRConfig field docs
    for the semantics and comparison-tier rules of each knob.

    Only two knobs stay outside ``config`` because they are resolved against the
    run itself: ``abstraction_hash`` (pin the card abstraction; defaults to the
    run's recorded hash) and ``resolver_iterations``. For ``config.opponent ==
    "deployed"`` the resolver settings come from the run's own config with
    ``max_iterations`` pinned to ``resolver_iterations`` — iteration-pinned (not
    wall-clock) so the measured strategy is machine-independent and CRN pairing
    stays valid.

    The results dict carries per-hand records plus the base seed; evaluate two runs
    with the same explicit ``config.seed`` and feed the per-hand samples to
    :func:`~src.pipeline.evaluation.statistics.compare_paired_samples` for a paired
    comparison that resolves far smaller gaps than two independent intervals.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    config = config or LBRConfig()
    metadata = load_run_metadata(run_dir)
    effective_hash = abstraction_hash or metadata.card_abstraction_hash
    if effective_hash is None:
        raise ValueError(
            f"Run '{run_dir.name}' does not record which card abstraction it was trained "
            "against, so it cannot be evaluated faithfully: resolving by config name alone "
            "would silently rebucket the checkpoint under whatever abstraction that name "
            "now points at, yielding plausible but invalid numbers.\n"
            "Pass abstraction_hash explicitly if you know it (see the abstraction's "
            "metadata.json 'config_hash')."
        )
    solver, storage = build_evaluation_solver(
        metadata.config,
        checkpoint_dir=run_dir,
        abstraction_hash=effective_hash,
    )
    # For parallel LBR each worker rebuilds its own solver from the checkpoint (the
    # solver is not picklable across processes); the factory captures only picklable
    # args (config + checkpoint dir).
    factory = (
        functools.partial(_load_blueprint, metadata.config, run_dir, effective_hash)
        if config.num_workers > 1
        else None
    )
    if config.opponent == "deployed":
        config = replace(
            config,
            resolver=metadata.config.resolver.model_copy(
                update={"max_iterations": resolver_iterations}
            ),
        )
    result = compute_lbr_exploitability(solver, config, blueprint_factory=factory)
    results = _lbr_results_dict(result, big_blind=metadata.config.game.big_blind)
    results["opponent_model"] = config.opponent
    results["scorer"] = config.scorer
    if config.scorer == "lookahead":
        results["lookahead_depth"] = config.lookahead_depth
        results["lookahead_top_k"] = config.lookahead_top_k
    if config.resolver is not None:
        results["resolver_iterations"] = config.resolver.max_iterations
        results["resolver_blend_alpha"] = config.resolver.policy_blend_alpha
    return EvaluationOutput(infosets=storage.num_infosets(), results=results)


def evaluate_run_resolver_gate(
    run_dir: Path,
    *,
    num_deals: int = 1000,
    time_budget_ms: int = 100,
    seed: int = 1,
) -> EvaluationOutput:
    """Head-to-head resolver gate on a run: blueprint+resolver vs bare blueprint.

    Duplicate deals (seat-swapped pairs off a fixed deck) cancel card luck, so the
    resolver's chip edge is measurable in ~1k deals. Positive edge means the
    resolver improves on the blueprint it wraps — the go/no-go signal for
    investing in the search path.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    metadata = load_run_metadata(run_dir)
    solver, storage = build_evaluation_solver(metadata.config, checkpoint_dir=run_dir)
    result = play_resolver_match(
        solver,
        num_deals=num_deals,
        time_budget_ms=time_budget_ms,
        seed=seed,
    )
    results = {
        "resolver_mbb_per_hand": result.resolver_mbb_per_hand,
        "se_mbb": result.se_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "p_value": result.p_value,
        "num_deals": result.num_deals,
        "num_hands": result.num_hands,
        "resolver_decisions": result.resolver_decisions,
        "resolver_fallbacks": result.resolver_fallbacks,
        "time_budget_ms": time_budget_ms,
        "seed": seed,
        "pair_samples_mbb": result.pair_samples_mbb,
    }
    return EvaluationOutput(infosets=storage.num_infosets(), results=results)


def evaluate_run_rollout(
    run_dir: Path,
    params: RolloutParams | None = None,
) -> EvaluationOutput:
    """Evaluate a run with the legacy one-ply rollout estimator (diagnostic opt-in only).

    NOT a valid exploitability bound (see ``ROLLOUT_ESTIMATOR_LABEL``); prefer
    :func:`evaluate_run_lbr`. Kept for comparison/diagnostics.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state.
    """
    params = params or RolloutParams()
    metadata = load_run_metadata(run_dir)
    config = metadata.config

    solver, storage = build_evaluation_solver(
        config,
        checkpoint_dir=run_dir,
    )
    results = evaluate_solver_exploitability(
        solver,
        num_samples=params.num_samples,
        use_average_strategy=params.use_average_strategy,
        num_rollouts_per_infoset=params.num_rollouts,
        seed=params.seed,
    )
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
    )


def evaluate_and_record(
    run_dir: Path,
    *,
    method: str = "lbr",
    lbr: LBRConfig | None = None,
    rollout: RolloutParams | None = None,
    resolver_iterations: int = 64,
    abstraction_hash: str | None = None,
    ledger_path: Path = eval_ledger.DEFAULT_LEDGER_PATH,
) -> dict[str, Any]:
    """Evaluate a run and persist the result to the eval ledger (best-effort).

    The single evaluate orchestrator shared by every transport (headless CLI,
    Modal): method dispatch, payload shape, knob-tier derivation, and the
    best-effort ledger recording live here once, so a cloud eval and a local
    eval cannot drift in what they run or record.

    Returns the portable evaluate payload; when recording succeeded it carries
    ``ledger_result_path``. Recording failures print a warning but never fail
    the evaluation itself — the ledger is a research convenience.
    """
    if method == "rollout":
        params = rollout or RolloutParams()
        out = evaluate_run_rollout(run_dir, params)
        estimator = ROLLOUT_ESTIMATOR_LABEL
        knobs = eval_ledger.build_rollout_knobs_from_params(
            samples=params.num_samples,
            rollouts=params.num_rollouts,
            use_current=not params.use_average_strategy,
            base_seed=out.results.get("base_seed", params.seed),
        )
    else:  # "lbr" (default, trustworthy)
        config = lbr or LBRConfig()
        out = evaluate_run_lbr(
            run_dir,
            config,
            resolver_iterations=resolver_iterations,
            abstraction_hash=abstraction_hash,
        )
        estimator = LBR_ESTIMATOR_LABEL
        knobs = eval_ledger.build_lbr_knobs(config, out.results)
    payload: dict[str, Any] = {
        "op": "evaluate",
        "run_id": run_dir.name,
        "method": method,
        "estimator": estimator,
        "infosets": out.infosets,
        "results": out.results,
    }
    try:
        result_path, _ = eval_ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload,
            method=method,
            estimator=estimator,
            knobs=knobs,
            ledger_path=ledger_path,
        )
        payload["ledger_result_path"] = str(result_path)
        print(f"  Ledger:        appended to {ledger_path} (payload: {result_path})")
    except Exception as exc:  # recording must never break the eval
        print(f"  Ledger:        skipped ({type(exc).__name__}: {exc})")
    return payload
