"""Reading the experiment record: convergence curves, baselines, arm attribution.

Pure readers over the eval ledger and the retained checkpoint ladder — nothing
here trains or evaluates. The one rule the whole module enforces is that
numbers measured with different instruments are never combined: curves never
merge comparison tiers, and an arm is only reported against a control it was
actually measured beside.
"""

import dataclasses
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.engine.solver.storage.helpers import retained_checkpoint_iterations
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.statistics import compare_paired_samples


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
