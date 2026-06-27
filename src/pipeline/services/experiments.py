"""Reading the run record: convergence curves, and everything known about a run.

Pure readers over the eval ledger and the retained checkpoint ladder — nothing
here trains or evaluates. The one rule the module enforces is that numbers
measured with different instruments are never combined: a curve never merges
comparison tiers.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, computed_field

from src.engine.solver.storage.static_checkpoint import StaticCheckpointManifest
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.services.runs import load_run_metadata
from src.pipeline.training.run_tracker import RunMetadata
from src.shared import records, run_events, task_history
from src.shared.cloudtask.node import archive


class CurvePoint(BaseModel):
    """One measured rung of a within-run convergence curve."""

    iteration: int
    exploitability_mbb: float
    std_error_mbb: float
    num_hands: int
    eval_git_commit: str | None


class CurveOutput(BaseModel):
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

    @computed_field
    @property
    def decay_ratio(self) -> float | None:
        """First point's exploitability divided by the last. O(1/sqrt(T)) predicts
        ~sqrt of the iteration ratio, so this is the number to read against theory.

        ``computed_field`` and not a bare property, because a bare one does not
        SERIALISE: `model_dump` skips it, exactly as `dataclasses.asdict` did.
        `curve` used to work around that by spreading the value in by hand, which
        meant `runinfo` -- which embeds one of these -- never carried it at all.
        Declared here, both surfaces get it and so does the OpenAPI schema.
        """
        if len(self.points) < 2 or self.points[-1].exploitability_mbb == 0:
            return None
        return self.points[0].exploitability_mbb / self.points[-1].exploitability_mbb


def exploitability_curve(
    run_dir: Path,
    *,
    ledger_path: Path,
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
        manifest = StaticCheckpointManifest.read(run_dir)
        retained = manifest.ladder() if manifest is not None else []
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


class RunDigest(BaseModel):
    """Everything recorded about one run, joined into a single view.

    A run's evidence is spread across five artifacts written by four subsystems
    -- identity in ``.run.json``, the curve in ``progress.jsonl``, scores in the
    eval ledger, the ladder in the checkpoint manifest, deaths in the share's
    task records. Answering "is this run trustworthy yet" meant opening all of
    them and holding the joins in your head.

    ``gaps`` is the part that matters: what this run cannot yet support a
    conclusion about. A missing rung and an unscored run look identical in a
    plot, and a run whose coverage stopped climbing is a different object from
    one still opening up the tree.
    """

    run_id: str
    config_name: str
    status: str
    experiment_id: str | None
    arm: str | None
    parent_run_id: str | None
    git_commit: str | None
    git_dirty: bool | None
    card_abstraction_hash: str | None
    iterations: int
    runtime_seconds: float
    attempts: int
    progress: list[dict[str, Any]]
    coverage_flat_from: int | None
    curve: CurveOutput
    tasks: list[task_history.TaskRow]
    gaps: list[str]


def run_digest(
    run_dir: Path,
    *,
    ledger_path: Path,
    tier_index: int = 0,
    tasks_dir: Path | None = None,
) -> RunDigest:
    """Join every record this run left behind. Pure reader.

    ``tasks_dir`` points at a local copy of the share's ``legs/`` (``just fetch``
    brings one down). Omitted for a purely local run, which has no tasks.
    """
    metadata = load_run_metadata(run_dir)
    progress = run_events.checkpoints(run_events.read(run_dir))
    curve = exploitability_curve(run_dir, ledger_path=ledger_path, tier_index=tier_index)
    tasks = (
        [row for row in task_history.read_tasks(tasks_dir) if row.run_id == run_dir.name]
        if tasks_dir
        else []
    )

    return RunDigest(
        run_id=metadata.run_id,
        config_name=metadata.config_name,
        status=metadata.status,
        experiment_id=metadata.experiment_id,
        arm=metadata.arm,
        parent_run_id=metadata.parent_run_id,
        git_commit=metadata.git_commit,
        git_dirty=metadata.git_dirty,
        card_abstraction_hash=metadata.card_abstraction_hash,
        iterations=metadata.iterations,
        runtime_seconds=metadata.runtime_seconds,
        attempts=len(metadata.attempts),
        progress=progress,
        coverage_flat_from=run_events.plateau_iteration(progress),
        curve=curve,
        tasks=tasks,
        gaps=_digest_gaps(metadata, progress, curve, tasks, run_dir),
    )


def _digest_gaps(
    metadata: RunMetadata,
    progress: list[dict[str, Any]],
    curve: CurveOutput,
    tasks: list[task_history.TaskRow],
    run_dir: Path,
) -> list[str]:
    """What this run cannot yet support a conclusion about.

    Stated rather than left to be noticed. Every entry here has been an actual
    source of a wrong reading in this project's history: an unscored ladder read
    as a flat curve, a single point read as convergence, a dirty tree read as a
    reproducible result.
    """
    gaps = []
    if not progress:
        gaps.append(
            "no progress history — the run predates it, or died before its first checkpoint"
        )
    if not curve.points:
        gaps.append(
            f"no evaluations recorded — `evaluate --run {metadata.run_id}` to start the curve"
        )
    elif len(curve.points) == 1:
        gaps.append("one curve point: a single score cannot show convergence, only a level")
    if curve.missing_iterations:
        rungs = ", ".join(f"{i:,}" for i in curve.missing_iterations[:6])
        more = (
            ""
            if len(curve.missing_iterations) <= 6
            else f" (+{len(curve.missing_iterations) - 6} more)"
        )
        gaps.append(f"unscored ladder rungs: {rungs}{more}")
    gaps += _ladder_gaps(run_dir)
    if metadata.card_abstraction_hash is None:
        gaps.append("no abstraction hash recorded — this run cannot be evaluated faithfully")
    if metadata.git_dirty:
        gaps.append("trained from a dirty working tree — the commit does not identify the code")
    if metadata.status != "completed":
        gaps.append(f"status is '{metadata.status}', not 'completed'")
    unresolved = [row for row in tasks if row.cause not in task_history.TERMINAL_CAUSES]
    if unresolved:
        gaps.append(
            f"{len(unresolved)} task(s) with no terminal record — `poker-solver tasks` to reconcile"
        )
    return gaps


def _ladder_gaps(run_dir: Path) -> list[str]:
    """What the published ladder cannot supply, said here instead of on a node.

    Answered off the completion MARKERS, which `pull_metadata` recreates from
    the share's own listing: a marker is written only once a rung has fully
    landed, so its absence is exactly what a fetch refuses on. Without this a
    run reads `completed` and every score of it dies minutes into a task with
    no diagnostic anywhere (gamma3-s103). A local run has no markers and no
    share, so it reports nothing.
    """
    published = {name[len(archive.MARKER_PREFIX) :] for name in _marker_names(run_dir)}
    if not published:
        return []
    # `archive.read_manifest`, not `StaticCheckpointManifest.read`: a torn
    # manifest must degrade to "nothing to say" here, not take `runinfo` down.
    manifest = archive.read_manifest(run_dir / records.STATIC_CHECKPOINT)
    if not manifest:
        return [
            (
                f"{len(published)} rung(s) are published but no manifest names them — "
                f"scoring will refuse. Re-publish from the node that trained the run."
            )
        ]
    entries = [manifest, *(e for e in manifest.get("retained") or [] if isinstance(e, dict))]
    named = {str(e["zarr"]) for e in entries if isinstance(e.get("zarr"), str)}
    unusable = sorted(named - published)
    if not unusable:
        return []
    return [
        (
            f"the manifest names {', '.join(unusable)}, which the share cannot supply "
            f"(no completion marker) — scoring those rungs will be refused. Re-publish "
            f"from the node that trained them."
        )
    ]


def _marker_names(run_dir: Path) -> list[str]:
    return [path.name for path in run_dir.glob(f"{archive.MARKER_PREFIX}*")]
