"""Where a task decides things live, before it touches any of them."""

from __future__ import annotations

from src.shared.cloudtask.node.paths import NodePaths


class TestNodePaths:
    def test_the_defaults_are_the_batch_node_layout(self):
        resolved = NodePaths.from_environment({})
        assert str(resolved.runs) == "/mnt/work/data/runs"
        assert str(resolved.share) == "/mnt/batch/tasks/fsmounts/shared"
        assert str(resolved.archive) == "/mnt/batch/tasks/fsmounts/shared/archive"

    def test_the_mount_root_comes_from_batch(self):
        resolved = NodePaths.from_environment({"AZ_BATCH_NODE_MOUNTS_DIR": "/mnt/fs"})
        assert str(resolved.share) == "/mnt/fs/shared"

    def test_the_code_tree_is_task_owned(self):
        """Unique per task, so concurrent tasks on one node cannot share a tree."""
        resolved = NodePaths.from_environment({"CODE_DIR": "/mnt/work/code-task-7"})
        assert str(resolved.code) == "/mnt/work/code-task-7"
