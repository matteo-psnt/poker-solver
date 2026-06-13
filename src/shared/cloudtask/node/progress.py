"""How far along this task is, how fast, and how much of it is its own.

Three questions with one answer path. WHILE a task runs, a sample published to
the share is the only thing that distinguishes a long job from a hung one --
nothing else about a multi-hour build is observable from outside the process.
The RATE that sample carries is what a time estimate is built from while the
task is still running. And when it ENDS, the sample less the baseline taken at
entry is what the task ACHIEVED, which is what estimates for the NEXT one are
built from.

The kinds stay free of the filesystem: this module gathers the state and hands
it over, and :mod:`~src.shared.cloudtask.kinds` decides whether it is progress
and against what.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.shared.cloudtask import kinds, task_log
from src.shared.cloudtask.node import archive
from src.shared.cloudtask.node.plan import TaskPlan, parse_environment
from src.shared.cloudtask.node.process import GRACE_SECONDS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from src.shared.cloudtask.node.paths import NodePaths

"""How often the retained ladder is checked. Deliberately coarse: publishing is
a copy to SMB, and a task that has not reached a new rung has nothing to send."""
WATCH_INTERVAL_SECONDS = 120

"""How often progress is sampled. Much finer, because it is one small write and
a bar that moves twice an hour is not a bar. The 60k probe finished between two
ladder ticks and so published nothing at all."""
PROGRESS_INTERVAL_SECONDS = 15

"""How often the task log is copied to the share. Between the two: it rewrites a
tail of up to `PUBLISHED_LOG_BYTES`, so it does not belong on the progress
cadence, but the share copy is the ONLY one a reader can reach and training used
to send it just once, at exit."""
LOG_INTERVAL_SECONDS = 60

# A module global because this process runs exactly ONE task, and because the
# baseline cannot be taken at entry: a resumed run's checkpoint is not on the
# node until its handler fetches it, so anything earlier reads zero and credits
# this task with the whole run's work.
#
_BASELINE: dict[str, float] = {}

# The window the RATE is measured over, which is NOT the baseline above. It is
# re-opened for as long as nothing has moved, because a node spends minutes
# fetching a snapshot, running `uv sync` and loading a ~773 MB abstraction, and
# charging that to the first unit of work makes a task look several times slower
# than it is. `unit` guards against differencing two counts of different things,
# which `evaluate` does the moment it stops reporting rungs and starts reporting
# board branches.
_WINDOW: dict[str, Any] = {}


class ProgressWatcher:
    """Samples the running task and publishes where it has got to. Nothing else.

    Polls what the work has written rather than hooking it, for the same reason
    a cloud task is a subprocess of the headless CLI and not a
    provider-specific reimplementation: the training and evaluation layers stay
    unaware they are running in the cloud.
    """

    def __init__(
        self,
        paths: NodePaths,
        log: archive.Log,
        interval: float = WATCH_INTERVAL_SECONDS,
        plan: TaskPlan | None = None,
        publish_log: Callable[[], None] | None = None,
        log_interval: float = LOG_INTERVAL_SECONDS,
    ) -> None:
        self._paths = paths
        self._log = log
        self._interval = interval
        self._plan = plan
        self._noted: dict[str, object] = {}
        # Passed explicitly rather than reached for on `log`: every caller hands
        # in a `TaskLogger`, but the parameter is typed as the write-a-line
        # callable and a test may pass exactly that.
        self._publish_log = publish_log
        self._log_interval = log_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="task-watcher", daemon=True)

    def note(self, **fields: object) -> None:
        """State only the HANDLER knows, folded into every later sample.

        A running command reports itself into a file; how many of them have
        already finished is not in it, and an evaluation scoring a ladder is
        one command per rung.
        """
        self._noted.update(fields)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Joined, not merely signalled: a publish still in flight when the
        wrapper exits is a rung that never reaches the share.

        Then sampled once more, so the last thing a reader sees is where the
        work actually ended rather than wherever it was up to a tick before.
        """
        self._stop.set()
        self._thread.join(timeout=GRACE_SECONDS)
        self._sample()

    def _loop(self) -> None:
        # Progress goes out immediately and then on its OWN, much finer cadence
        # than whatever the coarse tick does. Sharing one interval meant a task
        # finishing between two ticks published nothing at all.
        # The SMALLEST of the three, so no cadence can gate another -- a coarse
        # interval below the progress one would otherwise never fire.
        waited, since_log = 0.0, 0.0
        step = min(PROGRESS_INTERVAL_SECONDS, self._interval, self._log_interval)
        self._sample()
        # Immediately, not after a first interval: by this point the wrapper has
        # already logged its provenance and what it is fetching, and that is
        # exactly what someone opening the log in the first minute wants.
        self._send_log()
        while not self._stop.wait(step):
            self._sample()
            since_log += step
            if since_log >= self._log_interval:
                since_log = 0.0
                self._send_log()
            waited += step
            if waited >= self._interval:
                waited = 0.0
                self._coarse()

    def _sample(self) -> None:
        if self._plan is not None:
            state = node_state(self._paths, self._plan)
            state.update(self._noted)
            publish(self._paths, self._plan, state)

    def _send_log(self) -> None:
        """Here rather than on `LadderWatcher`, which is training-only: an
        evaluation's log is exactly as unreadable while it runs, and it is the
        one a failed rung has to be explained from.

        NEVER fatal, for the same reason `publish` is not: a task must not die
        because the account of it could not be copied.
        """
        if self._publish_log is None:
            return
        with contextlib.suppress(Exception):
            self._publish_log()

    def _coarse(self) -> None:
        """The slow tick. Progress alone has nothing to do on it."""


class LadderWatcher(ProgressWatcher):
    """Also publishes mid-run, so a killed task keeps everything up to its last
    rung.

    Training only. Copying to SMB is what makes the coarse cadence coarse, and
    an evaluation -- which FETCHES rungs onto the node -- would spend the first
    tick pushing ~540 MB of somebody else's checkpoints back where they came
    from.
    """

    # Starts EMPTY, so the first tick publishes whatever is already there rather
    # than treating it as seen. A resumed task pays almost nothing for that --
    # the completion markers skip every rung already on the share -- and it
    # closes the window where a task fetched a rung, died early, and published
    # nothing because nothing had changed since it started.
    _seen = ""

    def _coarse(self) -> None:
        state = archive.ladder_state(self._paths.runs)
        if state and state != self._seen:
            self._log(f"retained ladder changed -> {state}")
            archive.publish_all(self._paths.runs, self._paths.archive, self._log)
            self._seen = state


def node_state(paths: NodePaths, plan: TaskPlan) -> dict[str, object]:
    """Where each kind's progress actually lives on this node.

    The one place that knows work reports through a file it writes itself, or
    through its checkpoint manifest, or -- for training -- both; the kind
    branches on nothing, it is simply handed what there is. BOTH are gathered
    rather than one chosen, so a task whose writer has not started yet still
    reads true, and a kind that reads only one of them simply ignores the other.
    """
    state = _training_state(paths, plan)
    declared = kinds.kind(plan.op).progress_file
    if declared:
        state.update(_published_state(paths, declared))
    return state


def note_baseline(paths: NodePaths, plan: TaskPlan) -> None:
    """Mark the starting point, once the work to continue is actually here."""
    with contextlib.suppress(Exception):
        # A NODE IS REUSED, and the work file is node-local and named for the
        # kind rather than the task. The previous task's residue would otherwise
        # be read as this one's baseline, and then as its progress.
        declared = kinds.kind(plan.op).progress_file
        if declared:
            (paths.work / declared).unlink(missing_ok=True)
        progress = kinds.kind(plan.op).sample(plan, node_state(paths, plan))
        _BASELINE["done"] = progress.done if progress is not None else 0.0


def _windowed(progress: kinds.Progress) -> kinds.Progress:
    """The sample, plus the window this task has measured it over.

    The window is anchored on the first sample that MOVED, and then left alone.
    Two failures either side of that:

    - Anchoring at entry charges the node's startup -- the snapshot fetch, `uv
      sync`, a ~773 MB abstraction load -- to the first unit of work, and the
      task reads several times slower than it is.
    - Anchoring at the last STATIONARY sample charges a whole jump to the
      interval that merely contained its tail. A kind that reports only when a
      checkpoint lands moves a million iterations at once, which read as a
      million in fifteen seconds: ~66,000 it/s against a real one to three
      thousand, and an ETA of nearly zero on hours of training.

    It opens again from scratch if the kind changes what it counts -- one count
    minus another in a different unit is not a small error, it is a number with
    no meaning.
    """
    now = time.monotonic()
    if _WINDOW.get("unit") != progress.unit or progress.done < _WINDOW.get("base", progress.done):
        _WINDOW.update({"unit": progress.unit, "open": False, "base": progress.done, "at": now})
    if not _WINDOW["open"]:
        # Still standing still, or moving for the FIRST time. Either way this
        # sample is where measuring starts, not something to measure across.
        _WINDOW.update({"open": progress.done > _WINDOW["base"], "base": progress.done, "at": now})
        return replace(progress, base=progress.done)
    return replace(progress, base=_WINDOW["base"], window_seconds=now - _WINDOW["at"])


def units_done(paths: NodePaths) -> float:
    """Work done BY THIS TASK: where it ended, less where it began."""
    with contextlib.suppress(Exception):
        plan = parse_environment()
        progress = kinds.kind(plan.op).sample(plan, node_state(paths, plan))
        if progress is not None:
            return max(0.0, progress.done - _BASELINE.get("done", 0.0))
    return 0.0


def publish(paths: NodePaths, plan: TaskPlan, state: Mapping[str, object]) -> None:
    """Sample the kind and write it to the share. NEVER fatal.

    A task must not die because the thing describing it could not be written --
    the share can be slow, a sample can be torn, and none of that is a reason to
    lose the work. The reader treats a missing sample as "no bar", which is what
    it was before this existed.
    """
    with contextlib.suppress(Exception):
        progress = kinds.kind(plan.op).sample(plan, state)
        if progress is not None:
            task_log.write_progress_record(
                paths.share,
                task_id=task_log.current_task_id("local"),
                progress=_windowed(progress),
            )


def _published_state(paths: NodePaths, name: str) -> dict[str, object]:
    """What the work has published about itself, if anything yet."""
    try:
        return json.loads((paths.work / name).read_text())
    except (OSError, ValueError):
        return {}


def _training_state(paths: NodePaths, plan: TaskPlan) -> dict[str, object]:
    """What THIS task's manifest says, for the kind to interpret.

    Named by `train_run_id`, never "the first run directory on the node". A node
    is reused: it ran a task to 40k, then one to 80k, and the first sorted run
    dir was still the earlier one -- so the bar read the OLD run's 40,000
    against the NEW task's 80,000 target and showed a plausible-looking 50%,
    while the baseline and the final reading were the same number and the task
    recorded zero work done.
    """
    for name in archive.MANIFESTS:
        manifest = archive.read_manifest(paths.runs / plan.train_run_id / name)
        if manifest and isinstance(manifest.get("iteration"), int):
            return {"iteration": manifest["iteration"]}
    return {}
