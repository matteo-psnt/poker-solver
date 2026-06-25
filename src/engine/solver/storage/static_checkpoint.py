"""Checkpointing for :class:`StaticArrayStorage`.

A checkpoint is the flat arrays plus a 16-byte tree fingerprint, and nothing
else: the tree already says which infoset each row is, and the tree is a pure
function of config. (Carrying that mapping explicitly was ~83% of the old
write cost.)

That fingerprint is load-bearing rather than defensive. A checkpoint carries no
self-describing row identity, so loading one against a different tree would not
fail — it would silently reinterpret every row as a different infoset and
continue training on scrambled regrets. Refusing the load is the only way that
failure is ever visible.

Layout on disk::

    <dir>/STATIC_CHECKPOINT.json     manifest: current + retained ladder
    <dir>/static-<iteration>.zarr    the five arrays

The manifest is published with an atomic ``Path.replace`` after the arrays are
fully written, so a snapshot is either current or absent, never half-current.
The retained ladder keeps at most one snapshot per ``retain_every`` band, which
is what makes a within-run exploitability curve computable after the fact — the
absence of that ladder is why no such curve has ever existed for this project.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numcodecs
import numpy as np
import zarr

from src.engine.solver.storage.static_array import _ARRAYS, StaticArrayStorage
from src.shared import records

logger = logging.getLogger(__name__)

FORMAT_VERSION = "static-1"

# clevel=1: benchmarked as both fastest and smallest for these arrays. The
# chunk size is set by FILE COUNT, not by the write: a snapshot is copied to
# the share one file at a time over SMB, and at 50k rows a production table
# was 5,507 files and 20+ minutes per rung. Measured 08-22 at production
# size with ~85%-coverage content (838 MiB either way):
#     50k  5,507 files  write 2.6 s  read 3.4 s
#     4M      79 files  write 0.6 s  read 1.7 s     <- this
#    16M      27 files  write 1.0 s  read 3.7 s
DEFAULT_COMPRESSION_LEVEL = 1
DEFAULT_CHUNK_SIZE = 4_000_000


class FingerprintMismatchError(RuntimeError):
    """The checkpoint was written against a different betting tree."""


class AbstractionMismatchError(RuntimeError):
    """The checkpoint was written under a different bucket ASSIGNMENT.

    Distinct from a fingerprint mismatch: the tree, layout and bucket counts can
    match exactly while the abstraction maps hands to different buckets. Nothing
    about the arrays reveals it, so it has to be recorded and checked.
    """


@dataclass(frozen=True)
class StaticCheckpointManifest:
    iteration: int
    zarr_name: str
    fingerprint: str
    retained: list[dict]
    abstraction_id: str | None = None

    @classmethod
    def read(cls, checkpoint_dir: Path) -> StaticCheckpointManifest | None:
        """Read the manifest, or None when there is none.

        Deliberately NOT tolerant of a malformed one, unlike the other
        snapshots: the loader resolves which arrays to mmap through this, so a
        damaged manifest silently read as "absent" would look like a run with no
        checkpoints rather than one whose pointer needs repair. Callers that can
        proceed without it catch the error themselves.
        """
        path = Path(checkpoint_dir) / records.STATIC_CHECKPOINT
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        for field in ("iteration", "zarr", "fingerprint"):
            if field not in raw:
                raise ValueError(f"Invalid static checkpoint manifest {path}: missing {field!r}")
        return cls(
            iteration=int(raw["iteration"]),
            zarr_name=raw["zarr"],
            fingerprint=raw["fingerprint"],
            retained=list(raw.get("retained", [])),
            abstraction_id=raw.get("abstraction_id"),
        )

    def entry_for(self, iteration: int | None) -> dict:
        """The manifest entry for ``iteration``, or the current one when None."""
        if iteration is None:
            return {"iteration": self.iteration, "zarr": self.zarr_name}
        for entry in [*self.retained, {"iteration": self.iteration, "zarr": self.zarr_name}]:
            if int(entry["iteration"]) == iteration:
                return entry
        raise KeyError(f"No retained checkpoint at iteration {iteration}; have {self.ladder()}")

    def ladder(self) -> list[int]:
        """Every iteration this run can still be evaluated at, ascending.

        The published snapshot is part of the ladder, not separate from it: a
        curve that omitted it would stop one rung short of the run's own final
        score.
        """
        return sorted({int(e["iteration"]) for e in self.retained} | {self.iteration})


def save_checkpoint(
    storage: StaticArrayStorage,
    checkpoint_dir: Path,
    iteration: int,
    *,
    retain_every: int = 0,
    abstraction_id: str | None = None,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """Write a snapshot and atomically publish it. Returns the zarr path.

    ``abstraction_id`` identifies the bucket ASSIGNMENT, which the tree
    fingerprint deliberately does not cover: the fingerprint pins node identity,
    layout and bucket COUNTS, so an abstraction that buckets the same hands
    differently under the same counts produces an identical fingerprint. Resuming
    or scoring across such a change silently trains on rebucketed hands. Pass the
    resolved abstraction hash to make that detectable.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = checkpoint_dir / f"static-{iteration}.zarr"

    compressor = numcodecs.Blosc(
        cname="zstd", clevel=compression_level, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    root = zarr.open(zarr.DirectoryStore(zarr_path), mode="w")
    for name in _ARRAYS:
        array = getattr(storage, name)
        root.create_dataset(
            name,
            data=np.asarray(array),
            chunks=(min(chunk_size, max(1, array.shape[0])),),
            compressor=compressor,
            dtype=array.dtype,
        )
    root.attrs["iteration"] = iteration
    root.attrs["fingerprint"] = storage.tree.fingerprint()
    root.attrs["format_version"] = FORMAT_VERSION
    root.attrs["num_rows"] = storage.tree.num_rows
    root.attrs["num_slots"] = storage.tree.num_slots

    previous = StaticCheckpointManifest.read(checkpoint_dir)
    if (
        previous is not None
        and previous.abstraction_id is not None
        and abstraction_id is not None
        and previous.abstraction_id != abstraction_id
    ):
        raise AbstractionMismatchError(
            f"{checkpoint_dir} holds snapshots bucketed by {previous.abstraction_id}, "
            f"but this run buckets by {abstraction_id}. Appending would leave a ladder "
            "whose rungs are not comparable."
        )

    manifest = {
        "iteration": iteration,
        "zarr": zarr_path.name,
        "fingerprint": storage.tree.fingerprint(),
        "abstraction_id": abstraction_id,
        "format_version": FORMAT_VERSION,
        "retained": _extend_ladder(previous, iteration, zarr_path.name, retain_every),
    }
    # Through the substrate, which keeps the atomic replace this has always
    # relied on and adds the envelope's schema_version beside `format_version`
    # -- the two describe different things: the field set here, and the layout
    # of the arrays the manifest points at.
    records.write_snapshot(
        checkpoint_dir / records.STATIC_CHECKPOINT,
        manifest,
        records.REGISTRY[records.STATIC_CHECKPOINT],
    )

    _prune(checkpoint_dir, manifest)
    return zarr_path


def _extend_ladder(
    previous: StaticCheckpointManifest | None,
    iteration: int,
    zarr_name: str,
    retain_every: int,
) -> list[dict]:
    """The retained ladder after committing ``iteration`` — one entry per band.

    Append-only, and preserved even when a later call passes ``retain_every=0``:
    a resume whose caller forgot the knob must not delete measurement points an
    earlier task was told to keep.
    """
    ladder: list[dict] = list(previous.retained) if previous else []
    if retain_every <= 0:
        return ladder
    occupied = {int(entry["iteration"]) // retain_every for entry in ladder}
    if iteration // retain_every not in occupied:
        ladder.append({"iteration": iteration, "zarr": zarr_name})
    return ladder


def _prune(checkpoint_dir: Path, manifest: dict) -> None:
    """Delete snapshots that are neither current nor retained."""
    keep = {manifest["zarr"]} | {entry["zarr"] for entry in manifest["retained"]}
    for path in checkpoint_dir.glob("static-*.zarr"):
        if path.name not in keep:
            shutil.rmtree(path, ignore_errors=True)


def load_checkpoint(
    storage: StaticArrayStorage,
    checkpoint_dir: Path,
    *,
    at_iteration: int | None = None,
    abstraction_id: str | None = None,
) -> int:
    """Load a snapshot into ``storage`` in place. Returns the iteration loaded.

    ``at_iteration`` selects a retained ladder rung instead of the current
    snapshot — sweeping it is how a within-run convergence curve is built.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest = StaticCheckpointManifest.read(checkpoint_dir)
    if manifest is None:
        raise FileNotFoundError(f"No static checkpoint manifest in {checkpoint_dir}")

    expected = storage.tree.fingerprint()
    # A v1 (node-major) checkpoint of the SAME tree is loadable: the values
    # per infoset are identical, only their addresses moved, so the load
    # permutes rather than refuses. Any other fingerprint still refuses.
    legacy = storage.tree.legacy_fingerprint()
    translate = manifest.fingerprint == legacy
    if manifest.fingerprint != expected and not translate:
        raise FingerprintMismatchError(
            f"Checkpoint in {checkpoint_dir} was written against betting tree "
            f"{manifest.fingerprint}, but this storage indexes tree {expected}. "
            "Loading it would reinterpret every row as a different infoset. "
            "Rebuild the tree from the config the checkpoint was trained under."
        )

    if (
        abstraction_id is not None
        and manifest.abstraction_id is not None
        and manifest.abstraction_id != abstraction_id
    ):
        raise AbstractionMismatchError(
            f"Checkpoint in {checkpoint_dir} was trained under abstraction "
            f"{manifest.abstraction_id}, but this storage buckets by {abstraction_id}. "
            "The tree matches, so every row is the right SHAPE while holding a "
            "different hand's strategy."
        )

    entry = manifest.entry_for(at_iteration)
    root = zarr.open(zarr.DirectoryStore(checkpoint_dir / entry["zarr"]), mode="r")

    stored = root.attrs.get("fingerprint")
    if stored != (legacy if translate else expected):
        raise FingerprintMismatchError(
            f"Snapshot {entry['zarr']} carries fingerprint {stored}, expected "
            f"{legacy if translate else expected}. "
            "The manifest and the arrays disagree; the directory is corrupt."
        )

    row_source, slot_source = _legacy_index_maps(storage.tree) if translate else (None, None)
    if translate:
        logger.info("Checkpoint is v1 node-major; permuting arrays into the bucket-major layout.")
    for name in _ARRAYS:
        target = getattr(storage, name)
        source = root[name][:]
        if source.shape != target.shape:
            raise ValueError(
                f"Checkpoint array {name!r} has shape {source.shape}, storage expects "
                f"{target.shape} — fingerprints matched, so this is a format bug."
            )
        if translate:
            gather = slot_source if name in ("regrets", "strategy_sum") else row_source
            target[:] = source[gather]
        else:
            target[:] = source

    loaded = int(entry["iteration"])
    logger.info(
        f"Loaded static checkpoint at iteration {loaded:,} "
        f"({storage.num_touched_infosets():,} rows touched, tree {expected})"
    )
    return loaded


def _legacy_index_maps(tree) -> tuple[np.ndarray, np.ndarray]:
    """For each NEW (bucket-major) index, the OLD (v1 node-major) index.

    v1 laid rows/slots out node-major: node ``n`` owned ``buckets`` contiguous
    rows from ``cumsum`` offsets, each row ``num_actions`` contiguous slots.
    Gathering ``old_array[map]`` therefore lands every infoset's values at its
    new address. Built per load; loads are rare and this is seconds.
    """
    buckets = tree.buckets_per_node
    widths = tree.num_actions
    row_offset_v1 = np.zeros(len(tree.nodes) + 1, dtype=np.int64)
    np.cumsum(buckets, out=row_offset_v1[1:])
    slot_offset_v1 = np.zeros(len(tree.nodes) + 1, dtype=np.int64)
    np.cumsum(buckets * widths, out=slot_offset_v1[1:])

    row_source = np.empty(tree.num_rows, dtype=np.int64)
    slot_source = np.empty(tree.num_slots, dtype=np.int64)
    for node in tree.nodes:
        n = node.node_id
        count = int(buckets[n])
        width = int(widths[n])
        bucket_axis = np.arange(count, dtype=np.int64)
        new_rows = int(tree.row_base[n]) + bucket_axis * int(tree.row_stride[n])
        row_source[new_rows] = row_offset_v1[n] + bucket_axis
        if width == 0:
            continue
        new_slots = (
            int(tree.slot_base[n])
            + bucket_axis[:, None] * int(tree.slot_stride[n])
            + np.arange(width, dtype=np.int64)[None, :]
        )
        old_slots = slot_offset_v1[n] + np.arange(count * width, dtype=np.int64).reshape(
            count, width
        )
        slot_source[new_slots] = old_slots
    return row_source, slot_source


def retained_iterations(checkpoint_dir: Path) -> list[int]:
    """Every iteration still on disk, oldest first — the ladder plus current."""
    manifest = StaticCheckpointManifest.read(Path(checkpoint_dir))
    if manifest is None:
        return []
    return sorted({int(e["iteration"]) for e in manifest.retained} | {manifest.iteration})


__all__ = (
    "FingerprintMismatchError",
    "StaticCheckpointManifest",
    "load_checkpoint",
    "retained_iterations",
    "save_checkpoint",
)
