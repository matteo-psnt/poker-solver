"""Two publishers of one run must not destroy each other's copy.

MEASURED 09-03: two scoring tasks scored correctly, logged "1 scored, 0 failed"
and exited 0, and their results never reached the share. A concurrent session
was publishing the same run; `<name>.partial` is a DETERMINISTIC staging path,
so both wrote it, one `replace()` moved it, and the other raised

    No such file or directory: '.../<eval>.json.partial' -> '.../<eval>.json'

which aborted the whole copy. Silent: the exit code stays 0.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from src.shared.cloudtask.node.archive import copy_tree

if TYPE_CHECKING:
    from pathlib import Path


def _tree(root: Path, n: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"eval_{i}.json").write_text(f'{{"i": {i}}}')
    return root


class TestConcurrentPublishersDoNotCollide:
    def test_both_copies_land(self, tmp_path):
        source = _tree(tmp_path / "src", 40)
        destination = tmp_path / "dst"

        def publish() -> int:
            return copy_tree(source, destination, update=False, atomic=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            # Four publishers of the SAME tree, exactly the shape that failed.
            list(pool.map(lambda _: publish(), range(4)))

        for i in range(40):
            assert (destination / f"eval_{i}.json").read_text() == f'{{"i": {i}}}'
        leftovers = list(destination.glob("*.partial"))
        assert not leftovers, f"staging files survived: {leftovers}"
