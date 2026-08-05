"""Where things live on a Batch node.

Pure addressing, so every other module in this package takes one of these rather
than reading the environment for itself -- which is what makes the whole node
side testable against a ``tmp_path``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NodePaths:
    """The results of the node's disk layout, not the act of creating it.

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
            # and unique per task, so concurrent tasks on one node cannot share
            # a tree.
            code=Path(env.get("CODE_DIR") or str(work / "code")),
        )
