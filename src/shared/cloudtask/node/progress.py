"""How far along this task is, and how much of that it can claim as its own.

Two questions with one answer path. WHILE a task runs, a sample published to the
share is the only thing that distinguishes a long job from a hung one -- nothing
else about a multi-hour build is observable from outside the process. When it
ENDS, the same sample less the baseline taken at entry is what the task
ACHIEVED, which is what every later estimate is built from.

The kinds stay free of the filesystem: this module gathers the state and hands
it over, and :mod:`~src.shared.cloudtask.kinds` decides whether it is progress
and against what.
"""

from __future__ import annotations

import contextlib
import json
import threading
from typing import TYPE_CHECKING

from src.shared.cloudtask import kinds, task_log
from src.shared.cloudtask.node import archive
from src.shared.cloudtask.node.plan import TaskPlan, parse_environment
from src.shared.cloudtask.node.process import GRACE_SECONDS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from src.shared.cloudtask.node.paths import NodePaths

"""How often the retained ladder is checked. Deliberately coarse: publishing is
a copy to SMB, and a task that has not reached a new rung has nothing to send."""
WATCH_INTERVAL_SECONDS = 120

"""How often progress is sampled. Much finer, because it is one small write and
a bar that moves twice an hour is not a bar. The 60k probe finished between two
ladder ticks and so published nothing at all."""
PROGRESS_INTERVAL_SECONDS = 15

"""What this task had already done when it began.

A module global because this process runs exactly ONE task, and because the
baseline cannot be taken at entry: a resumed run's checkpoint is not on the node
until its handler fetches it, so anything measured before that reads zero and
would credit this task with the whole run's work.
"""
_BASELINE: dict[str, float] = {}


class LadderWatcher:
    """Publishes mid-run, so a killed task keeps everything up to its last rung.

    Polls the checkpoint manifest rather than hooking the trainer, for the same
    reason a cloud task is a subprocess of the headless CLI and not a
    provider-specific reimplementation: the training layer stays unaware it is
    running in the cloud.
    """

    def __init__(
        self,
        paths: NodePaths,
        log: archive.Log,
        interval: float = WATCH_INTERVAL_SECONDS,
        plan: TaskPlan | None = None,
    ) -> None:
        self._paths = paths
        self._log = log
        self._interval = interval
        self._plan = plan
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="ladder-watcher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Joined, not merely signalled: a publish still in flight when the
        wrapper exits is a rung that never reaches the share."""
        self._stop.set()
        self._thread.join(timeout=GRACE_SECONDS)

    def _loop(self) -> None:
        # Starts EMPTY, so the first tick publishes whatever is already there
        # rather than treating it as seen. A resumed task pays almost nothing for
        # that -- the completion markers skip every rung already on the share --
        # and it closes the window where a task fetched a rung, died early, and
        # published nothing because nothing had changed since it started.
        seen, waited = "", 0.0
        # Progress goes out immediately and then on its OWN, much finer cadence.
        # The ladder keeps the coarse one: it copies to SMB, and a task that has
        # reached no new rung has nothing to send. Sharing one interval meant a
        # task finishing between two ticks published nothing at all.
        # The SMALLER of the two, so neither cadence can gate the other -- a
        # ladder interval below the progress one would otherwise never fire.
        step = min(PROGRESS_INTERVAL_SECONDS, self._interval)
        self._sample()
        while not self._stop.wait(step):
            self._sample()
            waited += step
            if waited < self._interval:
                continue
            waited = 0.0
            state = archive.ladder_state(self._paths.runs)
            if state and state != seen:
                self._log(f"retained ladder changed -> {state}")
                archive.publish_all(self._paths.runs, self._paths.archive, self._log)
                seen = state

    def _sample(self) -> None:
        if self._plan is not None:
            publish(self._paths, self._plan, node_state(self._paths, self._plan))


def node_state(paths: NodePaths, plan: TaskPlan) -> dict[str, object]:
    """Where each kind's progress actually lives on this node.

    The one place that knows a training task publishes through its checkpoint
    manifest and a build through a file it writes itself; the kind branches on
    nothing, it is simply handed what there is.
    """
    declared = kinds.kind(plan.op).progress_file
    return _published_state(paths, declared) if declared else _training_state(paths, plan)


def note_baseline(paths: NodePaths, plan: TaskPlan) -> None:
    """Mark the starting point, once the work to continue is actually here."""
    with contextlib.suppress(Exception):
        progress = kinds.kind(plan.op).sample(plan, node_state(paths, plan))
        _BASELINE["done"] = progress.done if progress is not None else 0.0


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
                progress=progress,
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
