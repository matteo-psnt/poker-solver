"""Service-layer APIs for training and evaluation orchestration."""

import dataclasses
import functools
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.core.game.state import Street
from src.engine.solver.protocols import Blueprint
from src.engine.solver.storage.helpers import (
    CHECKPOINT_MANIFEST_FILE,
    read_checkpoint_manifest,
    retained_checkpoint_iterations,
)
from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.abstraction.paths import abstraction_output_path
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.blueprint_match import play_blueprint_match
from src.pipeline.evaluation.hunl_local_best_response import (
    LBRConfig,
    LBRResult,
    compute_lbr_exploitability,
    dominant_terminal,
)
from src.pipeline.evaluation.public_tree_br import PublicBRConfig, compute_public_tree_br
from src.pipeline.evaluation.resolver_match import play_resolver_match
from src.pipeline.evaluation.statistics import compare_paired_samples, variance_decomposition
from src.pipeline.training.abstraction_resolver import AbstractionHashMismatchError
from src.pipeline.training.components import (
    build_evaluation_solver,
    evaluate_solver_exploitability,
)
from src.pipeline.training.run_tracker import ExperimentTag, RunMetadata, RunTracker
from src.pipeline.training.trainer import TrainingSession
from src.pipeline.training.versioning import REPRESENTATION_VERSION
from src.shared.config import Config
from src.shared.config_loader import load_training_config
from src.shared.gitinfo import commits_ahead_of
from src.shared.units import pair_mean_mbb

# Local Best Response: a rigorous lower bound on exploitability (LBR <= exact BR,
# validated on Kuhn/Leduc). This is the trustworthy default metric.

logger = logging.getLogger(__name__)
LBR_ESTIMATOR_LABEL = "local_best_response (rigorous lower bound on exploitability)"

# The legacy `evaluate_run` metric is a one-ply rollout that both understates the
# structure it explores AND is upward-biased by a recursive max over noisy MC
# estimates — it is not a valid bound in either direction. Kept as an explicit
# opt-in for diagnostics/comparison only; do NOT treat it as exploitability.
ROLLOUT_ESTIMATOR_LABEL = "rollout_one_ply (uninformative; not a valid bound — diagnostic only)"

# Exact best response on a deterministic sampled public tree: zero evaluation
# variance, exactly paired across checkpoints under one board plan. The absolute
# value is exploitability of the board-restricted game, not full HUNL — compare
# within a tier, don't quote as a bound on the full game.
EXACT_BR_ESTIMATOR_LABEL = "public_tree_exact_br (deterministic exact BR on sampled public tree)"
BLUEPRINT_MATCH_ESTIMATOR_LABEL = (
    "blueprint_match (duplicate-deal head-to-head chip edge; abstraction-safe, not a bound)"
)


@dataclass(frozen=True)
class EvaluationOutput:
    """Container for run evaluation output."""

    infosets: int
    results: dict[str, Any]
    # Which checkpoint was actually scored, from the manifest committed atomically
    # with the arrays. Without it a stale read -- evaluating a checkpoint written
    # before a still-running leg's newer one -- is indistinguishable from a real
    # result, which is exactly how a 10M-iteration checkpoint was once silently
    # reported as the score of a 16M-iteration run. None only for pre-manifest runs.
    checkpoint_iteration: int | None = None


def checkpoint_iteration_of(run_dir: Path, at_iteration: int | None = None) -> int | None:
    """Iteration of the checkpoint an evaluator will actually score.

    Read from the manifest rather than run metadata: the manifest is committed in
    the same atomic replace as the arrays it names, so it describes exactly what an
    evaluator loading this directory will see. None for pre-manifest runs.

    With ``at_iteration`` the evaluator loads that ladder rung instead, so the
    reported iteration must be the rung -- reporting the published one would
    relabel every point of a convergence curve with the run's final iteration,
    which is the exact mislabelling this field exists to prevent.
    """
    if at_iteration is not None:
        return at_iteration
    manifest = read_checkpoint_manifest(run_dir)
    return int(manifest["iteration"]) if manifest is not None else None


@dataclass(frozen=True)
class CurvePoint:
    """One measured rung of a within-run convergence curve."""

    iteration: int
    exploitability_mbb: float
    std_error_mbb: float
    num_hands: int
    eval_git_commit: str | None


@dataclass(frozen=True)
class CurveOutput:
    """A within-run exploitability-vs-iteration curve, plus what is still missing.

    ``tier`` names the single instrument every point was measured with. Points from
    different tiers are never merged: a curve mixing two scorers measures two
    different things and its shape means nothing.

    ``missing_iterations`` are ladder rungs on disk with no evaluation in this tier
    -- the gaps to fill with ``evaluate --at N`` to complete the curve.
    """

    run_id: str
    tier: str | None
    points: list[CurvePoint]
    missing_iterations: list[int]
    other_tiers: list[str]
    retained_iterations: list[int]
    # Rows for this run that predate `checkpoint_iteration` being recorded. They
    # cannot be placed on an axis -- an unlabelled point is not a point -- but an
    # empty curve beside a non-empty ledger otherwise reads as a bug.
    unplaceable_records: int = 0

    @property
    def decay_ratio(self) -> float | None:
        """First point's exploitability divided by the last. O(1/sqrt(T)) predicts
        ~sqrt of the iteration ratio, so this is the number to read against theory."""
        if len(self.points) < 2 or self.points[-1].exploitability_mbb == 0:
            return None
        return self.points[0].exploitability_mbb / self.points[-1].exploitability_mbb


def exploitability_curve(
    run_dir: Path,
    *,
    ledger_path: Path = eval_ledger.DEFAULT_LEDGER_PATH,
    tier_index: int = 0,
) -> CurveOutput:
    """Join the retained checkpoint ladder to recorded evaluations, as a curve.

    Pure reader -- it never evaluates. Rungs without a recorded eval come back in
    ``missing_iterations`` rather than being silently skipped, because a curve with
    holes in it and a curve that stops early look identical once plotted.
    """
    # The directory name IS the run id (RunTracker defines it that way), so this
    # reads nothing that a legacy or torn .run.json could make it fail on.
    run_id = run_dir.name
    try:
        retained = retained_checkpoint_iterations(run_dir)
    except (OSError, ValueError, KeyError):
        # Legacy or torn manifest. A reporting command must still render the
        # evaluations it can find rather than dying on the ladder it cannot.
        retained = []
    records = eval_ledger.read_records(ledger_path)
    series = eval_ledger.curve_series(records, run_id)
    unplaceable = sum(
        1 for r in records if r.get("run_id") == run_id and r.get("checkpoint_iteration") is None
    )

    if not series:
        return CurveOutput(
            run_id=run_id,
            tier=None,
            points=[],
            missing_iterations=retained,
            other_tiers=[],
            retained_iterations=retained,
            unplaceable_records=unplaceable,
        )

    # Negative rejected as well as out-of-range: Python would happily index from
    # the end, quietly plotting a tier the caller did not ask for.
    if not 0 <= tier_index < len(series):
        raise IndexError(
            f"--tier {tier_index} is out of range: this run has {len(series)} "
            f"recorded tier(s), selectable as 0-{len(series) - 1}."
        )
    label, by_iteration = series[tier_index]
    points = [
        CurvePoint(
            iteration=iteration,
            exploitability_mbb=record["results"]["exploitability_mbb"],
            std_error_mbb=record["results"].get("std_error_mbb", 0.0),
            num_hands=record["results"].get("num_hands", 0),
            eval_git_commit=record.get("eval_git_commit"),
        )
        for iteration, record in sorted(by_iteration.items())
    ]
    return CurveOutput(
        run_id=run_id,
        tier=label,
        points=points,
        missing_iterations=[i for i in retained if i not in by_iteration],
        other_tiers=[lbl for idx, (lbl, _) in enumerate(series) if idx != tier_index],
        retained_iterations=retained,
        unplaceable_records=unplaceable,
    )


DEFAULT_BASELINE_PATH = Path("data/baseline.json")
CONTROL_ARM = "control"


@dataclass(frozen=True)
class Baseline:
    """The run the base-fork loop currently treats as the thing to beat."""

    run_id: str
    rationale: str
    promoted_at: str
    checkpoint_iteration: int | None = None


def load_baseline(path: Path = DEFAULT_BASELINE_PATH) -> Baseline | None:
    """Current baseline pointer, or None if none has been promoted."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Baseline(
            run_id=data["run_id"],
            rationale=data.get("rationale", ""),
            promoted_at=data.get("promoted_at", ""),
            checkpoint_iteration=data.get("checkpoint_iteration"),
        )
    except (OSError, ValueError, KeyError):
        return None


def promote_baseline(
    run_id: str,
    rationale: str,
    *,
    path: Path = DEFAULT_BASELINE_PATH,
    checkpoint_iteration: int | None = None,
) -> Baseline:
    """Point the baseline at ``run_id``, closing one turn of the base-fork loop.

    ``rationale`` is required by the caller rather than optional: a baseline that
    moved for a reason nobody wrote down is how a lineage becomes unauditable.
    Written via a temp file and an atomic replace so a kill mid-write cannot leave
    the pointer unreadable.
    """
    baseline = Baseline(
        run_id=run_id,
        rationale=rationale,
        promoted_at=datetime.now(UTC).isoformat(),
        checkpoint_iteration=checkpoint_iteration,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(baseline), indent=2))
    tmp.replace(path)
    return baseline


@dataclass(frozen=True)
class ArmResult:
    """One arm of an experiment, scored and attributed against its control."""

    arm: str
    run_id: str
    checkpoint_iteration: int | None
    exploitability_mbb: float
    std_error_mbb: float
    # Variant minus control, in mbb/g. NEGATIVE means the variant is less
    # exploitable, i.e. the idea helped. None when the pairing was refused.
    vs_control_mbb: float | None = None
    vs_control_p_value: float | None = None
    # Why the paired comparison could not be made, if it could not. Reported rather
    # than silently omitted: a missing delta and an invalid one look identical.
    vs_control_blocked: list[str] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class ExperimentReport:
    """Every arm of one experiment, each attributed against the control arm.

    A variant's raw exploitability is not evidence on its own: a fork receives extra
    training on top of its base, and that alone moves the number. The control arm is
    the same fork with the same extra training and no idea, so the variant-minus-
    control delta is the part attributable to the idea.
    """

    experiment_id: str
    control_run_id: str | None
    baseline_run_id: str | None
    arms: list[ArmResult]
    notes: list[str]


def _latest_by_arm(
    records: list[dict[str, Any]], tier: tuple[Any, ...] | None = None
) -> dict[str, dict[str, Any]]:
    """Last record per arm in timestamp order, optionally restricted to one tier.

    Restricting matters: an arm re-scored under different knobs would otherwise put
    its newest — but incomparable — eval into the table, and every arm would report
    "not attributable" against a control it was never measured beside. Pinning to the
    control's tier picks the eval that actually pairs.
    """
    by_arm: dict[str, dict[str, Any]] = {}
    for record in records:
        arm = record.get("arm")
        if not arm:
            continue
        if tier is not None and eval_ledger.tier_key(record) != tier:
            continue
        by_arm[arm] = record
    return by_arm


def experiment_report(
    experiment_id: str,
    *,
    ledger_path: Path = eval_ledger.DEFAULT_LEDGER_PATH,
    runs_dir: Path = Path("data/runs"),
    baseline_path: Path = DEFAULT_BASELINE_PATH,
) -> ExperimentReport:
    """Score every arm of an experiment and attribute each variant to its control."""
    records = [
        r for r in eval_ledger.read_records(ledger_path) if r.get("experiment_id") == experiment_id
    ]
    notes: list[str] = []
    if not records:
        notes.append(f"No evaluations recorded for experiment '{experiment_id}'.")

    # Pick the tier that ACTUALLY COMPARES THE MOST ARMS, among tiers containing a
    # control. Using the control's newest eval instead would let one stray
    # re-score of the control -- an operator sanity-checking it at a deeper
    # lookahead, say -- empty the whole report and blame every variant for
    # "no evaluation in the control's tier", while a fully matched set sat in the
    # older tier. The control is the row that sets the tier, so it is exactly the
    # row whose strays do the most damage.
    tiers: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("arm"):
            tiers.setdefault(eval_ledger.tier_key(record), []).append(record)
    with_control = {
        key: rows for key, rows in tiers.items() if any(r.get("arm") == CONTROL_ARM for r in rows)
    }

    if records and not with_control:
        notes.append(
            f"No '{CONTROL_ARM}' arm: without it a variant's score cannot be separated "
            "from the extra training its fork received."
        )

    tier = None
    if with_control:
        # Most arms wins; ties broken by the most recent activity in the tier.
        tier = max(
            with_control,
            key=lambda k: (
                len({r.get("arm") for r in with_control[k]}),
                max(eval_ledger.record_instant(r) for r in with_control[k]),
            ),
        )
    by_arm = _latest_by_arm(records, tier)
    control = by_arm.get(CONTROL_ARM)
    if control is not None:
        notes.append(f"Tier: {eval_ledger.tier_label(control)}")
        for arm in {r.get("arm") for r in records if r.get("arm")} - set(by_arm):
            notes.append(f"Arm '{arm}' has no evaluation in the control's tier; omitted.")

    control_samples: list[float] | None = None
    if control is not None:
        try:
            control_samples = eval_ledger.load_payload(control, runs_dir)["results"].get(
                "pair_samples_mbb"
            )
        except (FileNotFoundError, KeyError):
            notes.append("Control payload is missing, so no arm can be attributed.")

    arms: list[ArmResult] = []
    for arm, record in sorted(by_arm.items()):
        results = record.get("results", {})
        blocked: list[str] = []
        delta = p_value = None

        if arm != CONTROL_ARM and control is not None:
            blocked = eval_ledger.tier_mismatches(control, record)
            if not blocked and control_samples:
                try:
                    samples = eval_ledger.load_payload(record, runs_dir)["results"][
                        "pair_samples_mbb"
                    ]
                    stats = compare_paired_samples(samples, control_samples)
                    delta, p_value = stats["mean_diff"], stats["p_value"]
                except (FileNotFoundError, KeyError):
                    blocked = ["payload missing, cannot pair"]
            elif not control_samples and not blocked:
                blocked = ["control has no per-hand samples to pair against"]

        arms.append(
            ArmResult(
                arm=arm,
                run_id=record.get("run_id", ""),
                checkpoint_iteration=record.get("checkpoint_iteration"),
                exploitability_mbb=results.get("exploitability_mbb", 0.0),
                std_error_mbb=results.get("std_error_mbb", 0.0),
                vs_control_mbb=delta,
                vs_control_p_value=p_value,
                vs_control_blocked=blocked,
            )
        )

    baseline = load_baseline(baseline_path)
    return ExperimentReport(
        experiment_id=experiment_id,
        control_run_id=control.get("run_id") if control else None,
        baseline_run_id=baseline.run_id if baseline else None,
        arms=arms,
        notes=notes,
    )


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


@dataclass(frozen=True)
class ResumeOutput:
    """Machine-readable summary of a resume leg.

    ``no_op`` marks a leg that found the checkpoint already at or past its target
    and changed nothing — what a retried attempt sees.
    """

    run_id: str
    resumed_from_iteration: int
    target_iteration: int
    iterations: int
    num_infosets: int
    status: str
    no_op: bool


def list_runs(runs_dir: Path) -> list[str]:
    """List available training runs in the provided base directory."""
    return RunTracker.list_runs(runs_dir)


@dataclass(frozen=True)
class RunSummary:
    """A run annotated with how old it is and whether it can still be loaded.

    ``commits_ago`` is the number of commits HEAD is ahead of the run's train
    commit (0 == trained on the current HEAD, None == commit unknown to this
    checkout). ``loadable`` is False when the run cannot be opened at HEAD --
    either it never checkpointed or its on-disk format predates the current
    ``REPRESENTATION_VERSION`` -- with ``blocker`` naming the reason for the UI.
    """

    name: str
    commits_ago: int | None
    git_dirty: bool | None
    representation_version: int | None
    current_version: int
    has_checkpoint: bool
    loadable: bool
    blocker: str | None
    # Descriptive metadata for the picker (None when metadata is unreadable).
    iterations: int | None
    num_infosets: int | None
    config_name: str | None
    status: str | None


def _has_checkpoint(run_dir: Path) -> bool:
    """Whether ``run_dir`` holds a checkpoint the loader can open.

    Covers both the versioned manifest and the legacy fixed-name/iteration-suffixed
    zarr layouts (``checkpoint.zarr`` / ``checkpoint-N.zarr``).
    """
    return (run_dir / CHECKPOINT_MANIFEST_FILE).exists() or any(run_dir.glob("checkpoint*.zarr"))


def _summarize_run(runs_dir: Path, name: str) -> RunSummary:
    run_dir = runs_dir / name
    try:
        metadata = load_run_metadata(run_dir)
    except (OSError, ValueError, KeyError):
        return RunSummary(
            name=name,
            commits_ago=None,
            git_dirty=None,
            representation_version=None,
            current_version=REPRESENTATION_VERSION,
            has_checkpoint=_has_checkpoint(run_dir),
            loadable=False,
            blocker="unreadable metadata",
            iterations=None,
            num_infosets=None,
            config_name=None,
            status=None,
        )

    has_checkpoint = _has_checkpoint(run_dir)
    version = metadata.representation_version
    if not has_checkpoint:
        blocker: str | None = "no checkpoint"
    elif version != REPRESENTATION_VERSION:
        blocker = f"format v{version} ≠ v{REPRESENTATION_VERSION}"
    else:
        blocker = None

    return RunSummary(
        name=name,
        commits_ago=commits_ahead_of(metadata.git_commit),
        git_dirty=metadata.git_dirty,
        representation_version=version,
        current_version=REPRESENTATION_VERSION,
        has_checkpoint=has_checkpoint,
        loadable=blocker is None,
        blocker=blocker,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        config_name=metadata.config_name,
        status=metadata.status,
    )


def describe_runs(runs_dir: Path) -> list[RunSummary]:
    """Summarize every run, newest first, with age and loadability annotations."""
    names = list_runs(runs_dir)
    return [_summarize_run(runs_dir, name) for name in reversed(names)]


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Load run metadata from an existing run directory."""
    tracker = RunTracker.load(run_dir)
    return tracker.metadata


def create_training_session(
    config: Config, experiment: ExperimentTag | None = None
) -> TrainingSession:
    """Create a new training session."""
    return TrainingSession(config, experiment=experiment)


def create_resumed_session(
    run_dir: Path, capacity_override: int | None = None
) -> tuple[TrainingSession, int]:
    """Create a resumed session and return it with latest completed iteration."""
    metadata = load_run_metadata(run_dir)
    latest_iteration = metadata.iterations
    session = TrainingSession.resume(run_dir, capacity_override=capacity_override)
    return session, latest_iteration


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


def train(
    config_name: str,
    *,
    num_workers: int | None = None,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, Any] | None = None,
    experiment: ExperimentTag | None = None,
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
        experiment: Experiment/arm/parent this run belongs to, recorded in run
            metadata so arms can later be grouped and attributed against controls.

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
        session = create_training_session(config, experiment=experiment)
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


def resume(
    run_dir: Path,
    to_iteration: int,
    *,
    num_workers: int | None = None,
    capacity_override: int | None = None,
) -> ResumeOutput:
    """Resume an existing run and train up to an ABSOLUTE iteration target.

    The single resume orchestrator shared by every transport (headless CLI, Modal),
    so a cloud resume and a local resume cannot drift.

    Absolute, not "train N more": a scheduler-retried attempt re-reads a *newer*
    checkpoint, so a relative target compounds — a leg aimed at 25.5M retried after
    committing 21.5M would chase 30.6M.

    Args:
        run_dir: Directory of the run to resume.
        to_iteration: Absolute iteration to train up to; at or below the committed
            checkpoint this is a no-op.
        num_workers: Parallel worker count; defaults to all available CPUs when ``None``.
        capacity_override: Pre-allocate shared storage above the checkpoint's capacity
            so the leg never has to resize mid-run.
    """
    session, resumed_from = create_resumed_session(run_dir, capacity_override=capacity_override)

    remaining = to_iteration - resumed_from
    if remaining > 0:
        run_training(session, num_workers=num_workers, num_iterations=remaining)
    else:
        # Nothing forks here, so the bootstrap shared memory training would have
        # released during the worker handoff would leak for the process lifetime.
        session.release_bootstrap_storage()
        # `create_resumed_session` already called `mark_resumed`, which set the run
        # to "running" and opened an attempt. On this branch nothing runs and
        # nothing else closes it, so the run would be left looking live with a
        # dangling attempt -- the exact shape `mark_resumed` treats as evidence of
        # a death. Under a scheduler this is the COMMON case (every retry past the
        # target lands here), so it must be closed, not tolerated.
        if session.run_tracker is not None:
            session.run_tracker.mark_completed()

    metadata = load_run_metadata(run_dir)
    return ResumeOutput(
        run_id=metadata.run_id,
        resumed_from_iteration=resumed_from,
        target_iteration=to_iteration,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        status=metadata.status,
        no_op=remaining <= 0,
    )


def sweep_bucket_counts(
    abstraction_config: str,
    street: Street,
    bucket_counts: Sequence[int],
    *,
    num_workers: int | None = None,
    board_limit: int | None = None,
) -> list[dict]:
    """Quality metrics for one street bucketed at several bucket counts.

    Answers "how much resolution does bucket count k actually buy?" without
    paying for k separate precomputes: the equity pass — which is essentially
    all of the cost — runs once, and each bucket count is then a cheap k-means
    over the same matrices. ``PostflopPrecomputer`` was already factored for
    this; ``compute_street_matrices`` and ``bucket_street``'s ``num_buckets``
    override exist precisely so a sweep can reuse one pass.

    Nothing is written to disk. This measures an abstraction rather than
    producing one, and a sweep that also emitted artifacts would invite loading
    a bucketing that no run's ``card_abstraction_hash`` accounts for.

    Returns one quality dict per entry of ``bucket_counts``, each carrying the
    ``num_buckets`` it was measured at.
    """
    config = PrecomputeConfig.from_yaml(abstraction_config)
    if num_workers is not None:
        config = config.model_copy(update={"num_workers": num_workers})

    precomputer = PostflopPrecomputer(config)
    board_ids, equity, weights, histograms = precomputer.compute_street_matrices(
        street, board_limit=board_limit
    )

    results: list[dict] = []
    for count in bucket_counts:
        quality = precomputer.bucket_street(
            street, board_ids, equity, weights, histograms, num_buckets=count
        )
        results.append({"requested_buckets": int(count), **quality})
        logger.info(
            f"{street.name} @ {count} buckets: "
            f"occupied {quality['occupied_buckets']}/{quality['num_buckets']}, "
            f"variance explained {quality['variance_explained']:.6f}, "
            f"within-bucket std {quality['within_bucket_std']:.6f}"
        )
    return results


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


def _effective_abstraction_hash(
    run_dir: Path, metadata: RunMetadata, abstraction_hash: str | None
) -> str:
    """The abstraction hash an eval must pin to, refusing unpinnable runs."""
    effective = abstraction_hash or metadata.card_abstraction_hash
    if effective is None:
        raise ValueError(
            f"Run '{run_dir.name}' does not record which card abstraction it was trained "
            "against, so it cannot be evaluated faithfully: resolving by config name alone "
            "would silently rebucket the checkpoint under whatever abstraction that name "
            "now points at, yielding plausible but invalid numbers.\n"
            "Pass abstraction_hash explicitly if you know it (see the abstraction's "
            "metadata.json 'config_hash')."
        )
    return effective


def _load_blueprint(
    config: Config,
    checkpoint_dir: Path,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> Blueprint:
    """Build a fresh evaluation blueprint (solver) from a checkpoint.

    Used as a picklable factory (via ``functools.partial``) so parallel-LBR worker
    processes each construct their own solver — the solver holds a non-picklable
    Cython member and cannot be sent across a process boundary.
    """
    solver, _ = build_evaluation_solver(
        config,
        checkpoint_dir=checkpoint_dir,
        abstraction_hash=abstraction_hash,
        at_iteration=at_iteration,
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
    at_iteration: int | None = None,
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
    effective_hash = _effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    solver, storage = build_evaluation_solver(
        metadata.config,
        checkpoint_dir=run_dir,
        abstraction_hash=effective_hash,
        at_iteration=at_iteration,
    )
    # For parallel LBR each worker rebuilds its own solver from the checkpoint (the
    # solver is not picklable across processes); the factory captures only picklable
    # args (config + checkpoint dir).
    factory = (
        functools.partial(_load_blueprint, metadata.config, run_dir, effective_hash, at_iteration)
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
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir, at_iteration),
    )


def evaluate_run_exact_br(
    run_dir: Path,
    config: PublicBRConfig | None = None,
    *,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> EvaluationOutput:
    """Exact best response on the sampled public tree (deterministic point value).

    Zero evaluation variance: the same checkpoint under the same
    :class:`PublicBRConfig` always scores identically, so two checkpoints in one
    tier are exactly paired — a difference is pure signal, with no hand budget
    or p-value involved. The value is the exploitability of the board-sampled
    restricted game (see :mod:`~src.pipeline.evaluation.public_tree_br`), not of
    full HUNL: compare within a tier, don't quote it as a bound.

    ``at_iteration`` scores a retained ladder rung instead of the published
    snapshot. Because the value is deterministic, scoring several rungs of one run
    under an identical config gives an exactly-paired within-run convergence curve
    — the intended use. Hold the board plan fixed across rungs; shrinking the
    sample for cheap early points destroys the pairing.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid configuration or checkpoint state, or ``at_iteration``
            is not a retained rung (the error lists the available ones).
    """
    config = config or PublicBRConfig()
    metadata = load_run_metadata(run_dir)
    effective_hash = _effective_abstraction_hash(run_dir, metadata, abstraction_hash)
    solver, storage = build_evaluation_solver(
        metadata.config,
        checkpoint_dir=run_dir,
        abstraction_hash=effective_hash,
        at_iteration=at_iteration,
    )
    result = compute_public_tree_br(
        solver, config, starting_stack=metadata.config.game.starting_stack
    )
    results: dict[str, Any] = {
        "exploitability_mbb": result.exploitability_mbb,
        "std_error_mbb": 0.0,
        "num_hands": 0,
        "seat_values_mbb": [
            {
                "br_seat": r.br_seat,
                "button": r.button,
                "value_mbb": r.value_mbb,
                "missing_policy_mass": r.missing_policy_mass,
            }
            for r in result.seat_results
        ],
        "missing_policy_mass": result.missing_policy_mass,
        "nodes_visited": result.nodes_visited,
        "num_flops": result.num_flops,
        "num_turns": config.num_turns,
        "num_rivers": config.num_rivers,
        "board_seed": config.board_seed,
        "elapsed_s": result.elapsed_s,
        "big_blind": metadata.config.game.big_blind,
    }
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir, at_iteration),
    )


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
    return EvaluationOutput(
        infosets=storage.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir),
    )


def evaluate_blueprint_match(
    run_dir_a: Path,
    run_dir_b: Path,
    *,
    num_deals: int = 2000,
    seed: int = 1,
) -> EvaluationOutput:
    """Head-to-head match between two runs' blueprints on duplicate deals.

    Positive ``a_mbb_per_hand`` means run A's blueprint wins chips off run B's.
    Each blueprint is pinned to the card abstraction its run trained against.

    Raises:
        FileNotFoundError: Missing run metadata/checkpoint or abstraction file.
        ValueError: Invalid checkpoint state, or mismatched game configurations.
    """
    metadata_a = load_run_metadata(run_dir_a)
    metadata_b = load_run_metadata(run_dir_b)
    if metadata_a.config.game != metadata_b.config.game:
        raise ValueError(
            f"Game configs differ between runs ({metadata_a.config.game} vs "
            f"{metadata_b.config.game}); a chip match would be meaningless."
        )

    solver_a, storage_a = build_evaluation_solver(
        metadata_a.config,
        checkpoint_dir=run_dir_a,
        abstraction_hash=metadata_a.card_abstraction_hash,
    )
    solver_b, storage_b = build_evaluation_solver(
        metadata_b.config,
        checkpoint_dir=run_dir_b,
        abstraction_hash=metadata_b.card_abstraction_hash,
    )
    result = play_blueprint_match(solver_a, solver_b, num_deals=num_deals, seed=seed)
    results = {
        "run_a": metadata_a.run_id,
        "run_b": metadata_b.run_id,
        "a_mbb_per_hand": result.a_mbb_per_hand,
        "se_mbb": result.se_mbb,
        "confidence_95_mbb": result.confidence_95_mbb,
        "p_value": result.p_value,
        "num_deals": result.num_deals,
        "num_hands": result.num_hands,
        "infosets_a": storage_a.num_infosets(),
        "infosets_b": storage_b.num_infosets(),
        "seed": seed,
        "pair_samples_mbb": result.pair_samples_mbb,
    }
    return EvaluationOutput(
        infosets=storage_a.num_infosets(),
        results=results,
        checkpoint_iteration=checkpoint_iteration_of(run_dir_a),
    )


def record_blueprint_match(
    run_dir_a: Path,
    run_dir_b: Path,
    *,
    num_deals: int = 2000,
    seed: int = 1,
) -> dict[str, Any]:
    """Run a head-to-head match and persist a durable, self-describing payload.

    A blueprint match is inherently PAIRWISE, so it does not fit the single-run eval
    ledger (``evaluate_and_record``); instead the full result is written non-clobbering
    under run A's dir with BOTH runs' provenance embedded — the card-abstraction hashes
    especially, since a chip edge is uninterpretable later without recording which two
    abstractions were compared. The payload (not any ledger row) is the durable record,
    so a client guillotine after the match commits loses nothing.

    Returns the payload; the caller (Modal) commits the Volume so it survives the
    container.
    """
    metadata_a = load_run_metadata(run_dir_a)
    metadata_b = load_run_metadata(run_dir_b)
    out = evaluate_blueprint_match(run_dir_a, run_dir_b, num_deals=num_deals, seed=seed)

    def _provenance(meta: RunMetadata) -> dict[str, Any]:
        return {
            "run_id": meta.run_id,
            "git_commit": meta.git_commit,
            "git_dirty": meta.git_dirty,
            "config_name": meta.config_name,
            "card_abstraction_hash": meta.card_abstraction_hash,
            "action_config_hash": meta.action_config_hash,
            "representation_version": meta.representation_version,
        }

    payload: dict[str, Any] = {
        "op": "blueprint_match",
        "run_a": metadata_a.run_id,
        "run_b": metadata_b.run_id,
        "estimator": BLUEPRINT_MATCH_ESTIMATOR_LABEL,
        "provenance_a": _provenance(metadata_a),
        "provenance_b": _provenance(metadata_b),
        "checkpoint_iteration_a": checkpoint_iteration_of(run_dir_a),
        "checkpoint_iteration_b": checkpoint_iteration_of(run_dir_b),
        "infosets": out.infosets,
        "results": out.results,
    }
    knobs = {"run_b": metadata_b.run_id, "num_deals": num_deals, "base_seed": seed}
    try:
        result_path = eval_ledger.write_payload(run_dir_a, payload, eval_ledger.eval_slug(knobs))
        payload["result_path"] = str(result_path)
    except OSError as exc:  # recording is a research convenience; never fail the match
        logger.warning("Blueprint-match payload not written: %s", exc)
    return payload


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
        checkpoint_iteration=checkpoint_iteration_of(run_dir),
    )


def evaluate_and_record(
    run_dir: Path,
    *,
    method: str = "lbr",
    lbr: LBRConfig | None = None,
    rollout: RolloutParams | None = None,
    exact_br: PublicBRConfig | None = None,
    resolver_iterations: int = 64,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
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

    ``at_iteration`` scores a retained ladder rung rather than the published
    snapshot; each rung records its own ``checkpoint_iteration``, so a run's
    convergence curve arrives as ordinary ledger rows.
    """
    if method == "rollout":
        if at_iteration is not None:
            # Refuse rather than silently score the published snapshot and label the
            # row with a rung that was never loaded.
            raise ValueError(
                "--at is not supported for method 'rollout' (a diagnostic estimator, "
                "not a curve tool); use exact_br for within-run convergence curves."
            )
        params = rollout or RolloutParams()
        out = evaluate_run_rollout(run_dir, params)
        estimator = ROLLOUT_ESTIMATOR_LABEL
        knobs = eval_ledger.build_rollout_knobs_from_params(
            samples=params.num_samples,
            rollouts=params.num_rollouts,
            use_current=not params.use_average_strategy,
            base_seed=out.results.get("base_seed", params.seed),
        )
    elif method == "exact_br":
        br_config = exact_br or PublicBRConfig()
        out = evaluate_run_exact_br(
            run_dir, br_config, abstraction_hash=abstraction_hash, at_iteration=at_iteration
        )
        estimator = EXACT_BR_ESTIMATOR_LABEL
        knobs = eval_ledger.build_exact_br_knobs_from_params(
            num_flops=br_config.num_flops,
            num_turns=br_config.num_turns,
            num_rivers=br_config.num_rivers,
            board_seed=br_config.board_seed,
        )
    else:  # "lbr" (default, trustworthy)
        config = lbr or LBRConfig()
        out = evaluate_run_lbr(
            run_dir,
            config,
            resolver_iterations=resolver_iterations,
            abstraction_hash=abstraction_hash,
            at_iteration=at_iteration,
        )
        estimator = LBR_ESTIMATOR_LABEL
        knobs = eval_ledger.build_lbr_knobs(config, out.results)
    payload: dict[str, Any] = {
        "op": "evaluate",
        "run_id": run_dir.name,
        "method": method,
        "estimator": estimator,
        "infosets": out.infosets,
        "checkpoint_iteration": out.checkpoint_iteration,
        "results": out.results,
    }
    try:
        metadata = load_run_metadata(run_dir)
        result_path, _ = eval_ledger.record_evaluation(
            run_dir=run_dir,
            payload=payload,
            provenance=eval_ledger.RunProvenance(
                run_id=metadata.run_id,
                git_commit=metadata.git_commit,
                git_dirty=metadata.git_dirty,
                config_name=metadata.config_name,
                card_abstraction_hash=metadata.card_abstraction_hash,
                action_config_hash=metadata.action_config_hash,
                representation_version=metadata.representation_version,
                experiment_id=metadata.experiment_id,
                arm=metadata.arm,
                parent_run_id=metadata.parent_run_id,
                config_hash=metadata.config_hash,
            ),
            method=method,
            estimator=estimator,
            knobs=knobs,
            ledger_path=ledger_path,
        )
        payload["ledger_result_path"] = str(result_path)
        logger.info(f"  Ledger:        appended to {ledger_path} (payload: {result_path})")
    except Exception as exc:  # recording must never break the eval
        logger.warning(f"  Ledger:        skipped ({type(exc).__name__}: {exc})")
    return payload
