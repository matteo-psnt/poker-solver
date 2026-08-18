"""One task, start to finish: stage it, run it, account for however it ended.

What a Batch task actually runs. The shape is::

    record STARTED -> stage the code and sync deps -> hand off to the kind's
    handler -> publish on ANY exit -> record FINISHED with a cause

The exit account is the point. A run log cannot record a death -- the container
is gone first -- so this writes its own, and ``poker-solver tasks`` reconciles
what never landed against Batch's view. Which is why the signal handler RAISES
(bash's EXIT trap read ``$?`` as zero on a signal death, so ``cancel`` logged
clean completions), and why 124 and 137 are kept distinct: a wrong terminal
cause is permanent, because it suppresses reconciliation.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sys
import time
from typing import TYPE_CHECKING, Any

from src.shared import cache
from src.shared.cloudtask import kinds, task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import archive, legmirror, progress
from src.shared.cloudtask.node.handlers import HANDLERS, publish_own_run
from src.shared.cloudtask.node.paths import NodePaths
from src.shared.cloudtask.node.plan import BadEnvironmentError, parse_environment
from src.shared.cloudtask.node.process import EXIT_TIMEOUT, Killed, TaskLogger, run_guarded

if TYPE_CHECKING:
    from types import FrameType

"""Its own, much shorter ceiling. A wedged dependency install is not a long
job running slowly -- it is a task that will never start."""
SYNC_TIMEOUT_SECONDS = 30 * 60


def _stage(paths: NodePaths, log: TaskLogger) -> int:
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
    # river's 2.6M boards (~1 min) on every task. /mnt/work is node-scoped.
    shared_cache = paths.work / "cache"
    os.environ[cache.ENV_OVERRIDE] = str(shared_cache)
    # OPENED UP HERE, like the start task's `chmod -R a+rwX /mnt/work`:
    # `submit_task` sets no `user_identity`, so tasks run as Batch's default
    # auto-user, and a directory left with the first task's ownership is one the
    # next task cannot write into -- which would undo the sharing entirely.
    try:
        shared_cache.mkdir(parents=True, exist_ok=True)
        shared_cache.chmod(0o777)
    except OSError as error:
        log(f"WARN could not prepare {shared_cache} ({error}); each task will recompute")
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
    wait, so a cancelled task publishes what it has and records `cancelled`
    instead of a clean completion.
    """

    def handler(signum: int, _frame: FrameType | None) -> None:
        raise Killed(signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# Distinct from every signal and from `EXIT_TIMEOUT`, so a task that died
# unable to record itself is tellable apart from one that merely failed --
# through Batch's observation, which is the only account such a task leaves.
NO_RECORD_EXIT_CODE = 44


def _cause(code: int, outcome: str | None) -> str:
    if outcome:
        return outcome
    return {
        0: task_log.CAUSE_COMPLETED,
        EXIT_TIMEOUT: task_log.CAUSE_TIMEOUT,
        130: task_log.CAUSE_CANCELLED,
        143: task_log.CAUSE_CANCELLED,
        137: task_log.CAUSE_KILLED,
    }.get(code, task_log.CAUSE_FAILED)


def _eval_flags() -> tuple[str, ...]:
    """The task's eval flags, and NEVER an exception.

    ``_record`` suppresses everything, so a raise in here would not surface --
    it would silently cost the whole exit account, which is the one thing the
    task log exists to preserve. A malformed value is worth losing; the record
    around it is not.
    """
    try:
        parsed = json.loads(os.environ.get("RUN_EVAL_FLAGS_JSON") or "[]")
    except ValueError:
        return ()
    return tuple(str(item) for item in parsed) if isinstance(parsed, list) else ()


def _workers() -> int:
    """The RESOLVED count, not the requested one.

    `RUN_WORKERS` is empty to mean "all the CPUs this node has", so the number
    that predicts throughput is the one the plan worked out, not the blank.
    """
    with contextlib.suppress(Exception):
        return parse_environment().workers
    return 0


def _units_unit() -> str:
    """What `units` is counted IN, so a later reader can tell lineages apart.

    A count is not a measurement without its unit, and one has already changed:
    `evaluate` moved from rungs to flop branches. Averaging a rung-rate into a
    branch-rate does not fail, it predicts ~30x wrong -- so the unit travels with
    the number rather than being assumed from the op.
    """
    with contextlib.suppress(Exception):
        return kinds.kind(os.environ.get("RUN_OP") or TaskName.TRAIN).unit
    return ""


def _record(
    paths: NodePaths, event: str, *, code: int | None = None, cause: str | None = None
) -> int:
    """The task's account, in the database, which is the only place it lives now.

    THE STARTED RECORD IS FATAL AND THE TERMINAL ONE IS NOT, and the asymmetry
    is the whole design. Starting CLAIMS an attempt number, and every later
    record of this task -- its progress samples, its exit code -- belongs to
    that number. Inventing one because the write failed means overwriting a
    PREVIOUS attempt's account, which is the account of the failure that caused
    this retry. So the claim raises, and the task dies having recorded why.

    That is not a new failure mode: since the record moved off the share, a task
    that cannot reach the database fails at its first training event regardless.
    This names the reason minutes earlier, before a node spends an hour on work
    nothing will be able to describe.

    The terminal record keeps the old contract -- a task that survived its work
    must not die reporting it -- so it is best-effort and returns the attempt it
    used. Returns the attempt this record belongs to; 0 when there is none.
    """
    dsn = os.environ.get("POKER_SOLVER_RECORD_DSN", "")
    task_id = os.environ.get("AZ_BATCH_TASK_ID", "local")
    if event == task_log.EVENT_STARTED:
        return _record_start(paths, task_id, dsn)
    attempt = 0
    with contextlib.suppress(Exception):
        attempt = legmirror.latest_attempt(task_id, dsn=dsn) if dsn else 0
    with contextlib.suppress(Exception):
        legmirror.record(
            task_id,
            attempt,
            "exit",
            _node_fields(paths, task_id, event, attempt, code=code, cause=cause),
            dsn=dsn,
        )
    return attempt


def _record_start(paths: NodePaths, task_id: str, dsn: str) -> int:
    """Claim the attempt, or die saying so. See `_record`."""
    if not dsn:
        raise RuntimeError(
            "No POKER_SOLVER_RECORD_DSN: this task would run with no account of "
            "itself anywhere. The share no longer holds one."
        )
    return legmirror.claim_attempt(
        task_id, _node_fields(paths, task_id, task_log.EVENT_STARTED, 0), dsn=dsn
    )


def _node_fields(
    paths: NodePaths,
    task_id: str,
    event: str,
    attempt: int,
    *,
    code: int | None = None,
    cause: str | None = None,
) -> dict[str, Any]:
    """The record's body, straight from the environment."""
    return task_log.node_record(
        task_id=task_id,
        attempt=attempt,
        job_id=os.environ.get("AZ_BATCH_JOB_ID", ""),
        node_id=os.environ.get("AZ_BATCH_NODE_ID", ""),
        run_id=os.environ.get("RUN_ID", ""),
        op=os.environ.get("RUN_OP") or TaskName.TRAIN,
        config=os.environ.get("RUN_CONFIG", ""),
        target_iteration=os.environ.get("RUN_TO", ""),
        # `RUN_TO` is a TRAIN target and an evaluate task leaves it 0, so
        # without these two an evaluation records nothing about what it
        # actually scored -- which is how 38 evaluate tasks came to be
        # indistinguishable in the record.
        eval_at=os.environ.get("RUN_EVAL_AT", ""),
        eval_flags=_eval_flags(),
        # Straight from the environment rather than through the plan: this
        # record is written BEFORE the plan is parsed, and its whole purpose
        # is to survive a task that dies before anything else runs. A task
        # that fails during dependency install still has to say what code it
        # was going to run.
        code_snapshot=os.environ.get("CODE_SNAPSHOT", ""),
        git_commit=os.environ.get("RUN_GIT_COMMIT", ""),
        git_dirty=os.environ.get("RUN_GIT_DIRTY", ""),
        git_branch=os.environ.get("RUN_GIT_BRANCH", ""),
        workers=_workers(),
        units=progress.units_done(paths) if event == task_log.EVENT_FINISHED else 0.0,
        units_unit=_units_unit(),
        event=event,
        cause=cause,
        exit_code=code,
    )


def main() -> int:
    paths = NodePaths.from_environment()
    # BEFORE the logger. Opening the task log touches /mnt/work, which can fail,
    # and the started record is the one guarantee this module exists for: a task
    # that leaves nothing is indistinguishable from one that never ran.
    #
    # Which is exactly why the claim's failure is CAUGHT here rather than left
    # to propagate. It has to be fatal -- see `_record` -- but a bare traceback
    # at this point produces the very thing the line above is about: no log, no
    # row, nothing on the share. Batch's own observation is then the only
    # account, so this makes it a readable one: a dedicated exit code, and the
    # reason on stderr where the node captures it.
    try:
        _record(paths, task_log.EVENT_STARTED)
    except Exception as exc:  # noqa: BLE001 -- the message IS the deliverable here
        print(f"FATAL: cannot record this task: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "The database is the only record; a task that cannot claim its "
            "attempt would overwrite a previous one's account.",
            file=sys.stderr,
        )
        return NO_RECORD_EXIT_CODE
    task = os.environ.get("AZ_BATCH_TASK_ID", "local")
    log = TaskLogger(paths.work / f"task-{task}.log", paths.share)
    _install_signal_handlers()

    code, outcome = 1, None
    plan = None
    try:
        plan = parse_environment()
        log(f"code provenance: {plan.provenance}")
        sync = _stage(paths, log)
        if sync != 0:
            log(f"FATAL dependency sync failed rc={sync}")
            code = sync
        else:
            started = time.monotonic()
            code, outcome = HANDLERS[plan.op](plan, paths, log)
            # The ONE boundary this log had no line for, and it hides minutes.
            # A quick_test whose training reports 22.6s of work publishes five
            # minutes later, every time and long before any of the record work
            # -- so the gap is the handler returning, not the publish. Which
            # PART of the handler is the next question, and it cannot be asked
            # while the log jumps straight from the child's last line to
            # `publishing`.
            log(f"handler returned rc={code} after {time.monotonic() - started:.1f}s")
    except Killed as killed:
        code = 128 + killed.signum
        log(f"signalled ({killed.signum}); publishing what this task has")
    except (BadEnvironmentError, archive.FetchRefusedError) as refusal:
        log(f"FATAL {refusal}")
        code = 1
    finally:
        # Hand the signals back to the OS first: `Killed` is a BaseException and
        # `_record` suppresses only Exception, so a second SIGTERM arriving while
        # the publish below copies a ladder over SMB would escape this block and
        # leave the task `unresolved` forever -- exactly the zombie state the
        # cost screen then bills against.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        # Publish on ANY exit -- success, failure, or cancellation. An
        # operator-cancelled task still leaves its progress on the share. Its
        # OWN run only: the disk also holds what earlier tasks fetched.
        if plan is not None:
            publish_own_run(plan, paths, log)
        log.publish()
        _record(paths, task_log.EVENT_FINISHED, code=code, cause=_cause(code, outcome))
        log.close()
    return code
