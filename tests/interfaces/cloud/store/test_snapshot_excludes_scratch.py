"""`scratch/` must not ride a dispatch up to the share.

A code snapshot seals the WORKING TREE, and `scratch/` is where this project
puts local-only working files -- decision records, probes, and anything a
session stages to read. Nothing in it belongs on a node.

MEASURED 2026-09-24: ~3 GB of staged checkpoints in `scratch/stage/` made every
`score` dispatch hang for minutes and return NO output at all, because
`build_code_snapshot` was gzipping them before the upload began. Two dispatches
were abandoned as argparse errors before the cause was found.

Built as a real tarball rather than an assertion about the exclusion set,
because the set is the mechanism and the tarball is the thing that leaves.
"""

from __future__ import annotations

import tarfile

from src.interfaces.cloud.store.blob import build_code_snapshot


def _sealed(tmp_path):
    root = tmp_path / "tree"
    (root / "src").mkdir(parents=True)
    (root / "src" / "solver.py").write_text("# real code\n")
    (root / "scratch" / "stage").mkdir(parents=True)
    (root / "scratch" / "stage" / "static-4000.ckpt.zst").write_bytes(b"\x00" * 4096)
    (root / "scratch" / "notes.md").write_text("a local-only decision record\n")

    destination = tmp_path / "snapshot.tar.gz"
    build_code_snapshot(root, destination)
    with tarfile.open(destination) as archive:
        return set(archive.getnames())


def test_scratch_never_enters_the_tarball(tmp_path):
    sealed = _sealed(tmp_path)
    assert not [name for name in sealed if name.split("/")[0] == "scratch"]


def test_the_code_still_does(tmp_path):
    """The exclusion must not be so broad it drops the thing being dispatched."""
    assert "src/solver.py" in _sealed(tmp_path)
