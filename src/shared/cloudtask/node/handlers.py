"""What each KIND of task actually does on the node.

One executor per kind, and a registry keyed by the kind's own name -- so a kind
with no executor is a ``KeyError`` naming it rather than a task that silently
does nothing. The lifecycle around them (the guard, the tee, the exit account)
is deliberately not here: it is identical for every kind, and mixing the two is
what made the shell version impossible to reason about.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from src.shared.cloudtask import kinds, task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import archive, blobstore, profile, progress
from src.shared.cloudtask.node.paths import NodePaths
from src.shared.cloudtask.node.plan import TaskPlan
from src.shared.cloudtask.node.process import TaskLogger, run_guarded

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


def _refresh_abstractions(paths: NodePaths, log: TaskLogger, sas: str = "") -> None:
    """Merge the share's abstractions onto this node before training.

    `infra/main.tf`'s START TASK is the only other thing that does this, and it
    runs once per node BOOT -- so a node that came up before an abstraction was
    precomputed can never see it, and the task dies in the resolver several
    minutes deep, after `uv sync`. That is a trap for exactly the case the
    precompute path exists to serve: build a new abstraction, then train on it.
    Merging here makes the two orderings equivalent.
    """
    if sas:
        # THE CONTAINER FIRST, and the share only while it still holds them.
        # Same order, and for the same reason, as a rung's fetch.
        try:
            fetched = archive.fetch_abstractions(
                blobstore.sibling_container(sas, archive.ABSTRACTIONS_CONTAINER),
                paths.data / "combo_abstraction",
                log,
            )
            if fetched:
                log(f"fetched {fetched} abstraction(s) from the container")
        except Exception as error:  # noqa: BLE001 -- the share may still answer
            log(f"WARN could not read the abstractions container: {error}")

    source = paths.share / "combo_abstraction"
    if not source.is_dir():
        return
    try:
        # update=True is `cp -u`: an abstraction already on the node is not
        # recopied, so the steady-state cost is a directory walk, not 400 MB.
        archive.copy_tree(source, paths.data / "combo_abstraction", update=True)
    except OSError as error:
        # Not fatal. The node may already hold the abstraction this task wants,
        # and the resolver says so precisely if it does not.
        log(f"WARN could not refresh abstractions from the share ({error})")


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
            # THE PRIOR'S MANIFEST NAMES ITS RUNGS, and asking the share for a
            # directory is a second opinion that fails for every migrated run:
            # the rung is in the container and there is no directory to find.
            name = dict(archive.manifest_entries(prior)).get(int(wanted), "")
            if not name:
                log(f"FATAL warm-start prior has no rung {wanted} (its manifest names none)")
                return 1, "missing-rung"
            try:
                archive.require_complete(prior, name, plan.checkpoint_sas)
            except archive.FetchRefusedError as refusal:
                log(f"FATAL warm-start rung {wanted}: {refusal}")
                return 1, "missing-rung"
            archive.fetch_snapshot(prior, destination, name, plan.checkpoint_sas)
            log(f"fetched warm-start rung {name}")
        else:
            archive.fetch_current_rung(prior, destination, log, plan.checkpoint_sas)
    _refresh_abstractions(paths, log, plan.checkpoint_sas)
    run_id = plan.train_run_id
    published = paths.archive / run_id
    if published.is_dir():
        log(f"fetching published checkpoint for {run_id}")
        archive.fetch_current_rung(published, paths.runs / run_id, log, plan.checkpoint_sas)

    plan = _reporting(plan, paths)
    progress.note_baseline(paths, plan)
    watcher = progress.LadderWatcher(
        paths, log, run_dir=paths.runs / run_id, plan=plan, publish_log=log.publish
    )
    watcher.start()
    try:
        # The command this actually runs, not a hardcoded one: `_train` is the
        # executor for the board-free kernel too, and said `train-static` for it.
        log(
            f"{plan.commands[0][0]}: config={plan.config} run={run_id} to={plan.to} "
            f"(timeout {plan.timeout_seconds}s)"
        )
        code = run_guarded(
            _cli(plan.commands[0]),
            cwd=paths.code,
            timeout=plan.timeout_seconds,
            log=log,
            # Training is the long one, and the only task anybody watches for
            # hours wondering where the time is going.
            profile_dir=paths.share / profile.PROFILES_DIRNAME,
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
    if not _fetch_mix_run(plan, paths, log):
        return 1, None
    # Same boot-order trap as training: evaluation resolves the abstraction the
    # checkpoint is PINNED to, so a node that predates that abstraction cannot
    # score the run at all.
    _refresh_abstractions(paths, log, plan.checkpoint_sas)

    # WITHOUT THIS the evaluator is never told where to write, so `--progress-file`
    # never reaches its command line and the branch counter it keeps has nowhere
    # to go: every score fell back to counting rungs, which is 1 for a `score`
    # task, so the bar read 0% for the whole ten minutes and then vanished.
    plan = _reporting(plan, paths)
    progress.note_baseline(paths, plan)
    # Progress ONLY, no ladder: this task has just fetched rungs onto the node,
    # and a ladder tick would push ~540 MB of them straight back to the share.
    watcher = progress.ProgressWatcher(paths, log, plan=plan, publish_log=log.publish)
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
        if archive.fetch_current_rung(
            published, paths.runs / plan.run_id, log, plan.checkpoint_sas
        ):
            return []
        log(f"FATAL {plan.run_id} has no published checkpoint to score")
        return None
    destination = paths.runs / plan.run_id
    fetched = archive.fetch_for_evaluation(
        published, destination, requested, log, plan.checkpoint_sas
    )
    if not fetched:
        log("FATAL none of the requested rungs could be fetched")
        return None
    support = _support_rungs(plan.eval_flags, destination, fetched)
    if support:
        log(f"eval also READS {len(support)} more rung(s): {', '.join(support)}")
        support_fetched = archive.fetch_for_evaluation(
            published, destination, support, log, plan.checkpoint_sas
        )
        if len(support_fetched) != len(support):
            log("FATAL a rung the reassembled average reads is missing")
            return None
    return fetched


def _fetch_mix_run(plan: TaskPlan, paths: NodePaths, log: TaskLogger) -> bool:
    """Pull down the OTHER run a `--mix-run` evaluation blends in.

    Rung fetching is per-run and this one is a different run entirely, so
    without it the mixture dies in the loader the same way a windowed average
    did before its support rungs were fetched.
    """
    options = dict(itertools.pairwise(plan.eval_flags))
    other = options.get("--mix-run")
    if not other:
        return True
    source = paths.archive / other
    if not source.is_dir():
        log(f"FATAL --mix-run names no run on the share: {other}")
        return False
    wanted = [options["--mix-at"]] if "--mix-at" in options else []
    destination = paths.runs / other
    if wanted:
        fetched = archive.fetch_for_evaluation(
            source, destination, wanted, log, plan.checkpoint_sas
        )
    else:
        fetched = (
            ["current"]
            if archive.fetch_current_rung(source, destination, log, plan.checkpoint_sas)
            else []
        )
    if not fetched:
        log(f"FATAL could not fetch the mixture partner {other}")
        return False
    log(f"mixing in {other} at rung {wanted[0] if wanted else 'current'}")
    return True


def _support_rungs(flags: tuple[str, ...], destination: Path, scored: list[str]) -> list[str]:
    """Rungs the eval READS but does not score, from its own flags.

    `--avg-window-from` subtracts one earlier rung; `--avg-gamma` recombines
    every retained rung below the one being scored. Selective fetching means a
    rung nobody asked to SCORE is simply absent on the node, which surfaces as
    a zarr path error inside the loader -- so the flags have to be read here,
    where the fetch happens.
    """
    top = max(int(rung) for rung in scored)
    options = dict(itertools.pairwise(flags))
    floor = int(options["--avg-window-from"]) if "--avg-window-from" in options else 0
    wanted: set[int] = set()
    if floor:
        wanted.add(floor)
    if "--avg-gamma" in options:
        # Only the bands the reweighting actually spans. Without the floor a
        # 1.2B run would drag its whole 24-rung ladder onto the node to
        # recombine the last quarter of it.
        wanted.update(rung for rung in _retained_ladder(destination) if floor <= rung < top)
    return [str(rung) for rung in sorted(wanted) if str(rung) not in scored]


def _retained_ladder(destination: Path) -> list[int]:
    """Every rung the run's manifest still points at."""
    return [iteration for iteration, _name in archive.manifest_entries(destination)]


def _publish_abstraction(plan: TaskPlan, output: Path, log: TaskLogger) -> int:
    """Pack the built abstraction and put it in the container. 0 when it landed.

    REFUSES TO REPLACE, as the share publish does and for the same reason:
    bucket ASSIGNMENT is not pinned by `card_abstraction_hash`, so republishing
    under a name that exists would silently change which bucket a hand lands in
    for every run already trained against it.
    """
    sas = blobstore.sibling_container(plan.checkpoint_sas, archive.ABSTRACTIONS_CONTAINER)
    name = archive.abstraction_object(output.name)
    if blobstore.exists(sas, name) and not plan.force_publish:
        log(f"REFUSING to republish: {name} is already in the container.")
        log("  Set RUN_FORCE_PUBLISH=1 only if no run trained against it matters.")
        return 1
    packed = output.parent / name
    try:
        size = archive.pack_abstraction(output, packed)
        blobstore.put_object(sas, name, packed)
        log(f"published {name} ({size / 1024**2:.0f} MiB) to the container")
    except Exception as error:  # noqa: BLE001 -- the share publish below still runs
        log(f"FATAL could not publish {name}: {error}")
        return 1
    finally:
        packed.unlink(missing_ok=True)
    return 0


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
    watcher = progress.ProgressWatcher(paths, log, plan=plan, publish_log=log.publish)
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
    log(f"precomputed {output.name} -> {output}")
    if plan.checkpoint_sas:
        code = _publish_abstraction(plan, output, log)
        if code:
            return code, None

    destination = paths.share / "combo_abstraction" / output.name
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


def publish_own_run(plan: TaskPlan, paths: NodePaths, log: TaskLogger) -> None:
    """The end-of-task publish: a TRAINING task's own run, and nothing else.

    Every other kind either fetched runs onto this disk (evaluate) or wrote
    nowhere under ``runs/``; publishing the directory wholesale re-uploaded
    those fetched ladders on every task that ran on the node.
    """
    kind = kinds.kind_of(plan.op)
    if kind is None or not kind.publishes_run:
        return
    run_dir = paths.runs / plan.train_run_id
    if run_dir.is_dir():
        archive.publish_run(run_dir, paths.archive / run_dir.name, log, plan.checkpoint_sas)
        # BOTH STORES, while there are two. The container is where rungs are
        # going; the share is what still answers every fetch. Publishing to one
        # and reading from the other is the state this migration passes through,
        # not one it stops in.
        archive.publish_rungs_to_blob(run_dir, run_dir.name, plan.checkpoint_sas, log)


HANDLERS: dict[str, Handler] = {
    TaskName.TRAIN: _train,
    # Same executor: the board-free kernel writes ordinary checkpoints, so
    # the fetch, the ladder watcher and the publish path are identical.
    TaskName.TRAIN_PCS: _train,
    TaskName.EVALUATE: _evaluate,
    TaskName.PRECOMPUTE: _precompute,
}
