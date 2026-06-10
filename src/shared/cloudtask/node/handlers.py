"""What each KIND of task actually does on the node.

One executor per kind, and a registry keyed by the kind's own name -- so a kind
with no executor is a ``KeyError`` naming it rather than a task that silently
does nothing. The lifecycle around them (the guard, the tee, the exit account)
is deliberately not here: it is identical for all three, and mixing the two is
what made the shell version impossible to reason about.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from src.shared.cloudtask import kinds, task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import archive, progress
from src.shared.cloudtask.node.paths import NodePaths
from src.shared.cloudtask.node.plan import TaskPlan
from src.shared.cloudtask.node.process import TaskLogger, run_guarded

# How often a running sweep copies its curve to the share. Two minutes against
# arms that run for hours: frequent enough to see the first rung land, rare
# enough that the copy is invisible next to the work.
PUBLISH_EVERY_SECONDS = 120


# The second element is the OUTCOME an exit code cannot carry: an evaluation that
# scored some rungs and failed others exits 0 for Batch's retry economics.
Handler = Callable[[TaskPlan, NodePaths, TaskLogger], tuple[int, str | None]]


def _cli(argv: list[str]) -> list[str]:
    return ["uv", "run", "poker-solver", *argv]


def _reporting(plan: TaskPlan, paths: NodePaths) -> TaskPlan:
    """Fill in where this task reports its progress, when its kind takes one.

    Only the node knows its scratch directory, so the path is filled in here and
    the KIND puts it on the command line. A kind that declares no file is handed
    back unchanged -- ``paths.work / ""`` is the scratch DIRECTORY, which a
    command would then be asked to write a JSON document to.
    """
    declared = kinds.kind(plan.op).progress_file
    return replace(plan, progress_path=str(paths.work / declared)) if declared else plan


def _train(plan: TaskPlan, paths: NodePaths, log: TaskLogger) -> tuple[int, str | None]:
    # The prior lives on the share like any other run, and fetching it is the
    # node's job -- without this the trainer resolves a run directory that was
    # never brought down.
    #
    # FATAL when it is missing, not a warning. This warned and trained on, which
    # is how four 30M arms -- eight node-hours -- came back as four identical
    # controls: the prior had been pruned from the share, every arm quietly
    # became its own control, and nothing said so until the coverage ladders
    # turned out identical. For a warm-started task the prior IS the experiment,
    # so a missing one is a task that cannot do its job, not one that can do
    # less of it.
    if getattr(plan, "warm_start_from", ""):
        prior = paths.archive / plan.warm_start_from
        if not prior.is_dir():
            log(
                f"FATAL warm-start prior {plan.warm_start_from} is not on the share; "
                "refusing to train an unseeded arm that would look like a control"
            )
            return 1, "missing-prior"
        log(f"fetching warm-start prior {plan.warm_start_from}")
        # The rung the task will SEED FROM, not the manifest's current one. A
        # prior is a ladder and the best rung is rarely the last: asking for
        # rung 100 while fetching rung 200 left the trainer with no such
        # checkpoint, the first attempt died, and the Batch retry -- finding a
        # populated run directory -- skipped seeding and trained a control. Two
        # 30M sweeps were lost that way before the cause was visible.
        wanted = getattr(plan, "warm_start_at", 0)
        destination = paths.runs / plan.warm_start_from
        if wanted:
            archive.fetch_metadata(prior, destination)
            name = f"static-{wanted}.zarr"
            if not (prior / name).is_dir():
                log(f"FATAL warm-start prior has no rung {wanted} ({name} absent on the share)")
                return 1, "missing-rung"
            archive.require_complete(prior, name)
            archive.fetch_snapshot(prior, destination, name)
            log(f"fetched warm-start rung {name}")
        else:
            archive.fetch_current_rung(prior, destination, log)
    run_id = plan.train_run_id
    published = paths.archive / run_id
    if published.is_dir():
        log(f"fetching published checkpoint for {run_id}")
        archive.fetch_current_rung(published, paths.runs / run_id, log)

    plan = _reporting(plan, paths)
    progress.note_baseline(paths, plan)
    watcher = progress.LadderWatcher(paths, log, plan=plan)
    watcher.start()
    try:
        log(
            f"train-static: config={plan.config} run={run_id} to={plan.to} "
            f"(timeout {plan.timeout_seconds}s)"
        )
        code = run_guarded(
            _cli(plan.commands[0]),
            cwd=paths.code,
            timeout=plan.timeout_seconds,
            log=log,
        )
    finally:
        watcher.stop()
    if code == 137:
        log("KILLED (SIGKILL, not the guard -- suspect OOM); published rungs are on the share")
    return code, None


def _evaluate(plan: TaskPlan, paths: NodePaths, log: TaskLogger) -> tuple[int, str | None]:
    """Scoring belongs on the node, not on a laptop: the share is a local mount
    here and a WAN download away from anywhere else -- one checkpoint is
    ~540 MB of small zarr chunks, ~20 minutes to pull over SMB.

    Rungs are scored in ONE task because the fetch dominates the cost: a whole
    convergence curve for the price of one.
    """
    published = paths.archive / plan.run_id
    if not published.is_dir():
        log(f"FATAL no such run on the share: {plan.run_id}")
        return 1, None
    rungs = _fetch_rungs(plan, paths, published, log)
    if rungs is None:
        return 1, None

    # WITHOUT THIS the evaluator is never told where to write, so `--progress-file`
    # never reaches its command line and the branch counter it keeps has nowhere
    # to go: every score fell back to counting rungs, which is 1 for a `score`
    # task, so the bar read 0% for the whole ten minutes and then vanished.
    plan = _reporting(plan, paths)
    progress.note_baseline(paths, plan)
    # Progress ONLY, no ladder: this task has just fetched rungs onto the node,
    # and a ladder tick would push ~540 MB of them straight back to the share.
    watcher = progress.ProgressWatcher(paths, log, plan=plan)
    watcher.start()
    ok, bad = 0, 0
    # Built from the rungs that FETCHED, not the ones requested: `fetch_for_evaluation`
    # drops what is not on the share, and a command list built from the request
    # would score the wrong rung for every drop before it.
    commands = replace(plan, eval_rungs=tuple(rungs)).commands
    try:
        for rung, argv in zip(rungs or [""], commands, strict=True):
            log(
                f"evaluate: run={plan.run_id} method={plan.eval_method}"
                + (f" at={rung}" if rung else "")
            )
            code = run_guarded(_cli(argv), cwd=paths.code, timeout=plan.timeout_seconds, log=log)
            # One bad rung must not abandon the rest: a partial curve beats none,
            # and the failure is visible in this log and absent from the ledger.
            if code == 0:
                ok += 1
            else:
                bad += 1
                log(f"WARN rung {rung or 'latest'} failed (rc={code})")
            # The rung counter the running evaluator cannot know: its file
            # describes the walk in front of it, not the ones already scored.
            watcher.note(scored=ok + bad)
            log.publish()
    finally:
        watcher.stop()

    log(f"evaluate complete: {ok} scored, {bad} failed")
    # Exit 0 when ANYTHING scored: a non-zero exit retries the WHOLE task, and
    # one bad rung once turned a 30-minute job into nearly four hours of
    # re-scoring, writing every record twice. Only a clean sweep is worth a
    # retry. But exit 0 is not a claim of success -- 1 of 30 rungs would read
    # as `completed`, so the outcome carries what the code drops.
    if ok:
        return 0, (task_log.CAUSE_PARTIAL if bad else None)
    return 1, None


def _fetch_rungs(
    plan: TaskPlan, paths: NodePaths, published: Path, log: TaskLogger
) -> list[str] | None:
    """The rungs to score, pulled down; ``None`` when there is nothing to score.

    An empty request means "the latest checkpoint", so the manifest's current
    rung is exactly what has to come down. The shell had no branch for this and
    fell to a catch-all that copied the WHOLE published directory -- the entire
    ladder, to score one rung of it.
    """
    requested = list(plan.eval_rungs)
    if not requested:
        if archive.fetch_current_rung(published, paths.runs / plan.run_id, log):
            return []
        log(f"FATAL {plan.run_id} has no published checkpoint to score")
        return None
    fetched = archive.fetch_for_evaluation(published, paths.runs / plan.run_id, requested, log)
    if not fetched:
        log("FATAL none of the requested rungs could be fetched")
        return None
    return fetched


def _precompute(plan: TaskPlan, paths: NodePaths, log: TaskLogger) -> tuple[int, str | None]:
    """Build a card abstraction on a node and publish it once.

    THE GUARD IS THE POINT. The invariant is "computed ONCE, never
    recomputed" -- not "computed locally", which is what the old local-only
    workflow confused it with. Bucket ASSIGNMENT is not pinned by
    ``card_abstraction_hash``, so republishing under the same name can silently
    change which bucket a hand lands in, while every run trained against the
    old copy keeps a provenance check that now passes over different buckets.
    Refusing to overwrite is what makes running this in the cloud as safe as
    running it on a laptop.
    """
    payload = paths.work / "precompute.json"
    plan = _reporting(plan, paths)
    log(f"precompute: config={plan.config} (timeout {plan.timeout_seconds}s)")
    progress.note_baseline(paths, plan)
    watcher = progress.ProgressWatcher(paths, log, plan=plan)
    watcher.start()
    try:
        code = run_guarded(
            _cli(plan.commands[0]),
            cwd=paths.code,
            timeout=plan.timeout_seconds,
            log=log,
            stdout_to=payload,
        )
    finally:
        watcher.stop()
    if code != 0:
        log(f"precompute failed rc={code}")
        return code, None

    # The command reports where it wrote; do not re-derive the directory name.
    try:
        output = Path(json.loads(payload.read_text())["output_dir"])
    except (OSError, ValueError, KeyError) as error:
        log(f"FATAL precompute wrote no usable output_dir: {error}")
        return 1, None
    destination = paths.share / "combo_abstraction" / output.name
    log(f"precomputed {output.name} -> {output}")

    if destination.is_dir() and not plan.force_publish:
        log(f"REFUSING to republish: {output.name} already exists on the share.")
        log("  Bucket assignment is not pinned by the abstraction hash, so replacing")
        log("  it would silently invalidate every run trained against the old copy.")
        log("  Set RUN_FORCE_PUBLISH=1 only if no such run matters.")
        return 1, None

    try:
        # UNCONDITIONAL. Either the name is new or RUN_FORCE_PUBLISH asked to
        # replace it, and the update rule would skip every file the existing
        # copy has newer -- which is all of them. `cp -ru` had the same hole:
        # "force" left the old abstraction in place.
        archive.copy_tree(output, destination, update=False)
    except OSError as error:
        log(f"FATAL publish failed: {error}")
        return 1, None
    log(f"published {output.name} to the share")
    return 0, None


# Keyed by the kind's own name, so a kind with no executor is a KeyError naming
# it rather than a task that silently does nothing. ``str`` rather than
# ``TaskName`` because what arrives from the environment is the wire string.


def _vector_sweep(plan: TaskPlan, paths: NodePaths, log: TaskLogger) -> tuple[int, str | None]:
    """Measure one kernel on one abstraction, publishing the curve as it grows.

    The result is published on EVERY exit, not only success. A sweep that runs
    past its timeout is killed mid-checkpoint, and without this it would lose
    every checkpoint it had already scored -- the same failure this module's
    training path publishes rungs to avoid, in a different shape.

    Abstractions are read from the share in place: they are ~773 MB each and the
    node's disk is discarded when the task ends, so a copy would be paid for on
    every arm and thrown away.
    """
    plan = _reporting(plan, paths)
    result = Path(plan.progress_path)
    destination = paths.share / "vector-sweeps"
    # The task id is the node's, not the plan's -- the plan describes the WORK
    # and several retries of it would share one. Read the same way every other
    # node module reads it.
    published = f"{os.environ.get('AZ_BATCH_TASK_ID', 'local')}.json"

    def publish_partial() -> None:
        if result.is_file() and result.stat().st_size > 0:
            try:
                destination.mkdir(parents=True, exist_ok=True)
                archive.copy_file(result, destination / published)
            except OSError as error:
                log(f"could not publish partial result: {error}")

    def publish_as_it_lands(stop: threading.Event) -> None:
        """Copy the curve up whenever it grows, not only when the task ends.

        The sweep writes the whole result after every checkpoint, but that file
        is on the node's disk, which is discarded. Publishing only on exit means
        a six-hour arm shows NOTHING until it stops -- no way to tell a slow
        sweep from a wedged one, and no early read on a measurement whose first
        rung landed in minutes. The command prints its table at the end too, so
        the log is no help either.

        Cheap enough to be unconditional: the curve is a few kilobytes, and a
        copy is skipped entirely unless the file changed.
        """
        seen: tuple[int, float] | None = None
        while not stop.wait(PUBLISH_EVERY_SECONDS):
            if not result.is_file():
                continue
            stat = result.stat()
            current = (stat.st_size, stat.st_mtime)
            if current != seen and stat.st_size > 0:
                seen = current
                publish_partial()

    abstractions = paths.share / "combo_abstraction"
    if not (abstractions / plan.config).is_dir():
        log(f"FATAL no such abstraction on the share: {plan.config}")
        log("  publish one with `poker-solver push-data`, or build one with submit-precompute")
        return 1, None

    log(f"vector-sweep: {plan.arm or 'sweep'} on {plan.config} (timeout {plan.timeout_seconds}s)")
    progress.note_baseline(paths, plan)
    watcher = progress.ProgressWatcher(paths, log, plan=plan)
    watcher.start()
    stop = threading.Event()
    publisher = threading.Thread(
        target=publish_as_it_lands, args=(stop,), name="vector-sweep-publish", daemon=True
    )
    publisher.start()
    try:
        code = run_guarded(
            _cli([*plan.commands[0], "--abstractions-dir", str(abstractions)]),
            cwd=paths.code,
            timeout=plan.timeout_seconds,
            log=log,
        )
    finally:
        stop.set()
        publisher.join(timeout=30)
        watcher.stop()
        publish_partial()

    if code != 0:
        log(f"vector-sweep failed rc={code} (partial curve published if any was reached)")
        return code, None
    log(f"published vector-sweeps/{published}")
    return 0, None


HANDLERS: dict[str, Handler] = {
    TaskName.TRAIN: _train,
    # Same executor: the board-free kernel writes ordinary checkpoints, so
    # the fetch, the ladder watcher and the publish path are identical.
    TaskName.TRAIN_VECTOR: _train,
    TaskName.EVALUATE: _evaluate,
    TaskName.PRECOMPUTE: _precompute,
    TaskName.VECTOR_SWEEP: _vector_sweep,
}
