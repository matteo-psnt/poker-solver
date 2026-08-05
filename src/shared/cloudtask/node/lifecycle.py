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
from types import FrameType

from src.shared import cache
from src.shared.cloudtask import task_log
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.node import archive, progress
from src.shared.cloudtask.node.handlers import HANDLERS
from src.shared.cloudtask.node.paths import NodePaths
from src.shared.cloudtask.node.plan import BadEnvironmentError, parse_environment
from src.shared.cloudtask.node.process import EXIT_TIMEOUT, Killed, TaskLogger, run_guarded

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


def _record(
    paths: NodePaths, event: str, *, code: int | None = None, cause: str | None = None
) -> None:
    """Never fatal, and never allowed to be the reason a task fails.

    The whole point is that a task dying anywhere -- including during dependency
    install -- still leaves an account on the share.
    """
    with contextlib.suppress(Exception):
        task_log.write_node_record(
            paths.share,
            task_id=os.environ.get("AZ_BATCH_TASK_ID", "local"),
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
            workers=_workers(),
            units=progress.units_done(paths) if event == task_log.EVENT_FINISHED else 0.0,
            event=event,
            cause=cause,
            exit_code=code,
        )


def main() -> int:
    paths = NodePaths.from_environment()
    # BEFORE the logger. Opening the task log touches /mnt/work, which can fail,
    # and the started record is the one guarantee this module exists for: a task
    # that leaves nothing is indistinguishable from one that never ran.
    _record(paths, task_log.EVENT_STARTED)
    task = os.environ.get("AZ_BATCH_TASK_ID", "local")
    log = TaskLogger(paths.work / f"task-{task}.log", paths.share)
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
        log(f"signalled ({killed.signum}); publishing what this task has")
    except (BadEnvironmentError, archive.FetchRefusedError) as refusal:
        log(f"FATAL {refusal}")
        code = 1
    finally:
        # Publish on ANY exit -- success, failure, or cancellation. An
        # operator-cancelled task still leaves its progress on the share.
        archive.publish_all(paths.runs, paths.archive, log)
        log.publish()
        _record(paths, task_log.EVENT_FINISHED, code=code, cause=_cause(code, outcome))
        log.close()
    return code
