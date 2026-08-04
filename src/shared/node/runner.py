"""The process lifecycle around one leg: guard it, watch it, account for it.

What a Batch task actually runs. The shape is::

    fetch the last published checkpoint -> train/score to an ABSOLUTE target
    -> publish each retained rung as it appears -> publish on ANY exit

Four things replaced a shell construct that had already failed in production,
each argued where it happens: the signal handler RAISES (bash's EXIT trap read
``$?`` as zero on a signal death, so ``cancel`` logged clean completions);
124 and 137 are different deaths; the tee reads chunks because tqdm emits no
newlines; and the watcher is joined before the final publish.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from src.shared import cache, leg_log
from src.shared.node import archive
from src.shared.node.plan import (
    EVALUATE,
    PRECOMPUTE,
    REPAIR_LADDER,
    TRAIN,
    BadEnvironmentError,
    LegPlan,
    parse_environment,
)

"""`timeout`'s convention, kept because `leg_log` maps it to a distinct cause
and a wrong terminal cause is permanent -- it suppresses reconciliation."""
EXIT_TIMEOUT = 124

"""How long the trainer gets to flush after SIGTERM before SIGKILL. It exists
because the case that motivated the guard was a process that ignored TERM."""
GRACE_SECONDS = 120

WATCH_INTERVAL_SECONDS = 120

"""Its own, much shorter ceiling. A wedged dependency install is not a long
job running slowly -- it is a leg that will never start."""
SYNC_TIMEOUT_SECONDS = 30 * 60

"""The interesting part of a leg log is always the end, and a multi-hour tqdm
stream is mostly progress-bar repaints that cost more to copy than they
inform."""
PUBLISHED_LOG_BYTES = 2_000_000

CHUNK = 65536


class Killed(BaseException):
    """A signal reached the wrapper. ``BaseException`` so no broad ``except``
    can swallow the one event the exit record exists to report."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


@dataclass(frozen=True)
class NodePaths:
    """Where things live on a Batch node.

    Discovery and mounting are NOT done here -- ``infra/main.tf``'s start task
    formats and mounts the data disk before any task runs, and the share is
    mounted by the pool. This only names the results.
    """

    work: Path
    share: Path
    code: Path

    @property
    def data(self) -> Path:
        return self.work / "data"

    @property
    def runs(self) -> Path:
        return self.data / "runs"

    @property
    def archive(self) -> Path:
        return self.share / "archive"

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> NodePaths:
        env = dict(os.environ if environ is None else environ)
        work = Path(env.get("RUN_WORK_DIR") or "/mnt/work")
        mounts = env.get("AZ_BATCH_NODE_MOUNTS_DIR") or "/mnt/batch/tasks/fsmounts"
        return cls(
            work=work,
            share=Path(env.get("RUN_SHARE_DIR") or f"{mounts}/shared"),
            # Set by the task command line, which extracts there. Task-owned,
            # and unique per task, so concurrent legs on one node cannot share
            # a tree.
            code=Path(env.get("CODE_DIR") or str(work / "code")),
        )


class LegLogger:
    """Tees everything to node-local disk, then to the share on demand.

    Batch keeps a task's stdout ON THE NODE and the pool drains within minutes
    of a task ending, so anything only echoed is gone for exactly the legs
    worth reading later -- which is what happened to a 30M leg that died at
    ~720k iterations, leaving ``exit 1`` and nothing else. Node-local first
    because the training stream is chatty and writing every line straight to
    SMB would put the leg's throughput at the mercy of the share.
    """

    def __init__(self, path: Path, share: Path) -> None:
        self.path = path
        self.share = share
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("ab")
        self._lock = threading.Lock()

    def __call__(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        self.write(f"[run_leg {stamp}] {message}\n".encode())

    def write(self, chunk: bytes) -> None:
        with self._lock:
            self._handle.write(chunk)
            self._handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    def publish(self) -> None:
        """Copied on every publish, not only at exit, so a leg later killed
        outright still leaves its log behind."""
        task = os.environ.get("AZ_BATCH_TASK_ID", "leg")
        destination = self.share / "logs" / f"{task}.log"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("rb") as source:
                size = source.seek(0, os.SEEK_END)
                source.seek(max(0, size - PUBLISHED_LOG_BYTES))
                destination.write_bytes(source.read())
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class LadderWatcher:
    """Publishes mid-run, so a killed leg keeps everything up to its last rung.

    Polls the checkpoint manifest rather than hooking the trainer, for the same
    reason a cloud leg is a subprocess of the headless CLI and not a
    provider-specific reimplementation: the training layer stays unaware it is
    running in the cloud.
    """

    def __init__(
        self,
        paths: NodePaths,
        log: archive.Log,
        interval: float = WATCH_INTERVAL_SECONDS,
    ) -> None:
        self._paths = paths
        self._log = log
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ladder-watcher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Joined, not merely signalled -- see the module docstring."""
        self._stop.set()
        self._thread.join(timeout=GRACE_SECONDS)

    def _loop(self) -> None:
        # Starts EMPTY, so the first tick publishes whatever is already there
        # rather than treating it as seen. A resumed leg pays almost nothing for
        # that -- the completion markers skip every rung already on the share --
        # and it closes the window where a leg fetched a rung, died early, and
        # published nothing because nothing had changed since it started.
        seen = ""
        while not self._stop.wait(self._interval):
            state = archive.ladder_state(self._paths.runs)
            if state and state != seen:
                self._log(f"retained ladder changed -> {state}")
                archive.publish_all(self._paths.runs, self._paths.archive, self._log)
                seen = state


def run_guarded(
    argv: list[str],
    *,
    cwd: Path,
    timeout: int,
    log: LegLogger,
    stdout_to: Path | None = None,
) -> int:
    """Run a subprocess under a wall-clock ceiling, teeing its output.

    The task-level ``maxWallClockTime`` (P1D) is not a backstop for a hang: it
    is longer than any leg is meant to run, so a wedged process bills a full
    node-day before Batch acts. One leg proved this -- training died, the
    process could not exit, and the task stayed ``running`` indefinitely.

    ``stdout_to`` captures stdout to a file instead of teeing it, for the one
    command whose stdout is a JSON payload rather than a log.
    """
    sink = stdout_to.open("wb") if stdout_to else None
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=sink if sink else subprocess.PIPE,
            stderr=subprocess.PIPE if sink else subprocess.STDOUT,
            close_fds=True,
            # Its OWN group, so the guard can signal the whole tree.
            # `terminate()` reaches only `uv`; the trainer and its 16 workers
            # are grandchildren, and a deadline used to return 124 with them
            # still running, holding the /dev/shm segments that killed the NEXT
            # leg. Batch cleans up by cgroup, so a new session does not escape.
            start_new_session=True,
        )
    except OSError as error:
        log(f"FATAL could not start {argv[0]}: {error}")
        if sink:
            sink.close()
        return 1

    pump = threading.Thread(
        target=_pump,
        args=(process.stderr if sink else process.stdout, log),
        name="tee",
        daemon=True,
    )
    pump.start()
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            log(f"TIMEOUT after {timeout}s -- guard fired; published rungs are on the share")
            _terminate(process)
    except Killed:
        # The wrapper itself was signalled. Take the child down with it, then
        # let the exception carry the cause to the exit record.
        _terminate(process)
        raise
    finally:
        pump.join(timeout=GRACE_SECONDS)
        if sink:
            sink.close()

    if timed_out:
        return EXIT_TIMEOUT
    # A child killed by a signal reports a negative code. Restore the shell's
    # 128+n so 137 keeps meaning what `legs` reads it as: SIGKILL from outside,
    # which on a training node is the OOM killer.
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


def _pump(stream, log: LegLogger) -> None:
    """Chunked, never line-buffered: tqdm emits carriage returns, not newlines."""
    if stream is None:
        return
    try:
        while True:
            chunk = stream.read1(CHUNK) if hasattr(stream, "read1") else stream.read(CHUNK)
            if not chunk:
                return
            log.write(chunk)
    except (OSError, ValueError):
        return


def _terminate(process: subprocess.Popen) -> None:
    """TERM the whole group so the trainer's handlers can flush, KILL if it
    will not go -- the case that motivated the guard ignored TERM."""
    _signal_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait()


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    """Falls back to the child alone if the group is already gone."""
    try:
        os.killpg(os.getpgid(process.pid), signum)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            process.send_signal(signum)


def _cli(argv: list[str]) -> list[str]:
    return ["uv", "run", "poker-solver-run", *argv]


def _train(plan: LegPlan, paths: NodePaths, log: LegLogger) -> tuple[int, str | None]:
    run_id = plan.train_run_id
    published = paths.archive / run_id
    if published.is_dir():
        log(f"fetching published checkpoint for {run_id}")
        archive.fetch_current_rung(published, paths.runs / run_id, log)

    watcher = LadderWatcher(paths, log)
    watcher.start()
    try:
        log(
            f"train-static: config={plan.config} run={run_id} to={plan.to} "
            f"(timeout {plan.timeout_seconds}s)"
        )
        code = run_guarded(
            _cli(plan.train_argv()),
            cwd=paths.code,
            timeout=plan.timeout_seconds,
            log=log,
        )
    finally:
        watcher.stop()
    if code == 137:
        log("KILLED (SIGKILL, not the guard -- suspect OOM); published rungs are on the share")
    return code, None


def _evaluate(plan: LegPlan, paths: NodePaths, log: LegLogger) -> tuple[int, str | None]:
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
    rungs = list(plan.eval_rungs)
    if rungs:
        rungs = archive.fetch_for_evaluation(published, paths.runs / plan.run_id, rungs, log)
        if not rungs:
            log("FATAL none of the requested rungs could be fetched")
            return 1, None
    else:
        # `score --run X` with no `--at` means "the latest checkpoint", so the
        # manifest's current rung is exactly what has to come down. The shell
        # had no branch for this and fell to a catch-all that copied the WHOLE
        # published directory -- the entire ladder, to score one rung of it.
        current = archive.fetch_current_rung(published, paths.runs / plan.run_id, log)
        if not current:
            log(f"FATAL {plan.run_id} has no published checkpoint to score")
            return 1, None

    ok, bad = 0, 0
    for rung in rungs or [""]:
        log(
            f"evaluate: run={plan.run_id} method={plan.eval_method}"
            + (f" at={rung}" if rung else "")
        )
        code = run_guarded(
            _cli(plan.evaluate_argv(rung)),
            cwd=paths.code,
            timeout=plan.timeout_seconds,
            log=log,
        )
        # One bad rung must not abandon the rest: a partial curve beats none,
        # and the failure is visible in this log and absent from the ledger.
        if code == 0:
            ok += 1
        else:
            bad += 1
            log(f"WARN rung {rung or 'latest'} failed (rc={code})")
        log.publish()

    log(f"evaluate complete: {ok} scored, {bad} failed")
    # Exit 0 when ANYTHING scored: a non-zero exit retries the WHOLE task, and
    # one bad rung once turned a 30-minute job into nearly four hours of
    # re-scoring, writing every record twice. Only a clean sweep is worth a
    # retry. But exit 0 is not a claim of success -- 1 of 30 rungs would read
    # as `completed`, so the outcome carries what the code drops.
    if ok:
        return 0, (leg_log.CAUSE_PARTIAL if bad else None)
    return 1, None


def _precompute(plan: LegPlan, paths: NodePaths, log: LegLogger) -> tuple[int, str | None]:
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
    log(f"precompute: config={plan.config} (timeout {plan.timeout_seconds}s)")
    code = run_guarded(
        _cli(plan.precompute_argv()),
        cwd=paths.code,
        timeout=plan.timeout_seconds,
        log=log,
        stdout_to=payload,
    )
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


def _repair_ladder(plan: LegPlan, paths: NodePaths, log: LegLogger) -> tuple[int, str | None]:
    """Prove each published rung by loading it, and mark the ones that survive.

    Reads the share IN PLACE and needs nothing on the node -- which is why this
    op fetches nothing. Without that exemption it once fell to a catch-all copy
    and spent 25+ minutes duplicating a 16 GB ladder it then ignored.
    """
    published = paths.archive / plan.run_id
    if not published.is_dir():
        log(f"FATAL no such run on the share: {plan.run_id}")
        return 1, None
    log(f"verifying ladder for {plan.run_id} (reading the share in place)")
    code = run_guarded(
        [
            "uv",
            "run",
            "python",
            str(paths.code / "infra" / "verify_ladder.py"),
            plan.config,
            str(published),
        ],
        cwd=paths.code,
        timeout=plan.timeout_seconds,
        log=log,
    )
    log(f"repair finished rc={code}")
    return code, None


HANDLERS = {
    TRAIN: _train,
    EVALUATE: _evaluate,
    PRECOMPUTE: _precompute,
    REPAIR_LADDER: _repair_ladder,
}


def _stage(paths: NodePaths, log: LegLogger) -> int:
    """The code is ALREADY extracted -- the task command line untars it before
    invoking this, because this file lives inside that tarball.

    ``$CODE/data`` is symlinked to the node's data disk so that anything
    writing under ``<base>/data/`` lands there rather than in the throwaway
    code tree -- which is where ``precompute`` puts an abstraction, and where
    ``runs_dir`` resolves to.
    """
    log(f"code snapshot '{os.environ.get('CODE_SNAPSHOT', '?')}' staged at {paths.code}")
    paths.runs.mkdir(parents=True, exist_ok=True)
    link = paths.code / "data"
    if link.is_symlink() or link.exists():
        if link.is_symlink() or link.is_file():
            link.unlink()
        else:
            shutil.rmtree(link, ignore_errors=True)
    link.symlink_to(paths.data)

    # ON THE DATA DISK, not the task's HOME -- which is its working directory,
    # wiped with the task, so the `~/.cache` default would re-canonicalise the
    # river's 2.6M boards (~1 min) on every leg. /mnt/work is node-scoped.
    shared_cache = paths.work / "cache"
    os.environ[cache.ENV_OVERRIDE] = str(shared_cache)
    # OPENED UP HERE, like the start task's `chmod -R a+rwX /mnt/work`:
    # `submit_leg` sets no `user_identity`, so leg tasks run as Batch's default
    # auto-user, and a directory left with the first task's ownership is one the
    # next leg cannot write into -- which would undo the sharing entirely.
    try:
        shared_cache.mkdir(parents=True, exist_ok=True)
        shared_cache.chmod(0o777)
    except OSError as error:
        log(f"WARN could not prepare {shared_cache} ({error}); each leg will recompute")
    log(f"cache: {shared_cache}")

    # Through the guard, so an install failure explains ITSELF in the published
    # log rather than in Batch's node-local capture, which the pool destroys
    # minutes after the task ends. `--quiet` still writes failures to stderr,
    # which the tee catches, so this costs no diagnostic.
    log("syncing dependencies")
    return run_guarded(
        ["uv", "sync", "--quiet"], cwd=paths.code, timeout=SYNC_TIMEOUT_SECONDS, log=log
    )


def _install_signal_handlers() -> None:
    """Raise, rather than set a flag.

    A flag would only be noticed between subprocesses; raising interrupts the
    wait, so a cancelled leg publishes what it has and records `cancelled`
    instead of a clean completion.
    """

    def handler(signum: int, _frame: FrameType | None) -> None:
        raise Killed(signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _cause(code: int, outcome: str | None) -> str:
    if outcome:
        return outcome
    return {
        0: leg_log.CAUSE_COMPLETED,
        EXIT_TIMEOUT: leg_log.CAUSE_TIMEOUT,
        130: leg_log.CAUSE_CANCELLED,
        143: leg_log.CAUSE_CANCELLED,
        137: leg_log.CAUSE_KILLED,
    }.get(code, leg_log.CAUSE_FAILED)


def _eval_flags() -> tuple[str, ...]:
    """The leg's eval flags, and NEVER an exception.

    ``_record`` suppresses everything, so a raise in here would not surface --
    it would silently cost the whole exit account, which is the one thing the
    leg log exists to preserve. A malformed value is worth losing; the record
    around it is not.
    """
    try:
        parsed = json.loads(os.environ.get("RUN_EVAL_FLAGS_JSON") or "[]")
    except ValueError:
        return ()
    return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()


def _record(paths: NodePaths, event: str, *, code: int | None = None, cause: str | None = None):
    """Never fatal, and never allowed to be the reason a leg fails.

    The whole point is that a leg dying anywhere -- including during dependency
    install -- still leaves an account on the share.
    """
    with contextlib.suppress(Exception):
        leg_log.write_node_record(
            paths.share,
            task_id=os.environ.get("AZ_BATCH_TASK_ID", "local"),
            job_id=os.environ.get("AZ_BATCH_JOB_ID", ""),
            node_id=os.environ.get("AZ_BATCH_NODE_ID", ""),
            run_id=os.environ.get("RUN_ID", ""),
            op=os.environ.get("RUN_OP") or TRAIN,
            config=os.environ.get("RUN_CONFIG", ""),
            target_iteration=os.environ.get("RUN_TO", ""),
            # `RUN_TO` is a TRAIN target and an evaluate leg leaves it 0, so
            # without these two an evaluation records nothing about what it
            # actually scored -- which is how 38 evaluate legs came to be
            # indistinguishable in the record.
            eval_at=os.environ.get("RUN_EVAL_AT", ""),
            eval_flags=_eval_flags(),
            event=event,
            cause=cause,
            exit_code=code,
        )


def main() -> int:
    paths = NodePaths.from_environment()
    # BEFORE the logger. Opening the leg log touches /mnt/work, which can fail,
    # and the started record is the one guarantee this module exists for: a leg
    # that leaves nothing is indistinguishable from one that never ran.
    _record(paths, leg_log.EVENT_STARTED)
    task = os.environ.get("AZ_BATCH_TASK_ID", "local")
    log = LegLogger(paths.work / f"leg-{task}.log", paths.share)
    _install_signal_handlers()

    code, outcome = 1, None
    try:
        plan = parse_environment()
        log(f"code provenance: {plan.provenance}")
        sync = _stage(paths, log)
        if sync != 0:
            log(f"FATAL dependency sync failed rc={sync}")
            code = sync
        else:
            code, outcome = HANDLERS[plan.op](plan, paths, log)
    except Killed as killed:
        code = 128 + killed.signum
        log(f"signalled ({killed.signum}); publishing what this leg has")
    except (BadEnvironmentError, archive.FetchRefusedError) as refusal:
        log(f"FATAL {refusal}")
        code = 1
    finally:
        # Publish on ANY exit -- success, failure, or cancellation. An
        # operator-cancelled task still leaves its progress on the share.
        archive.publish_all(paths.runs, paths.archive, log)
        log.publish()
        _record(paths, leg_log.EVENT_FINISHED, code=code, cause=_cause(code, outcome))
        log.close()
    return code
