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
    <dir>/static-<iteration>.ckpt.zst    the five arrays, one object

The manifest is published with an atomic ``Path.replace`` after the arrays are
fully written, so a snapshot is either current or absent, never half-current.
The retained ladder keeps at most one snapshot per ``retain_every`` band, which
is what makes a within-run exploitability curve computable after the fact — the
absence of that ladder is why no such curve has ever existed for this project.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.engine.solver.storage import snapshot_format
from src.engine.solver.storage.static_array import _ARRAYS, StaticArrayStorage
from src.shared import records

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

FORMAT_VERSION = "static-1"

# The chunk size these constants tuned is GONE with the format. It was set by
# FILE COUNT rather than by the write -- a snapshot was copied to the share one
# file at a time over SMB, and at 50k rows a production table was 5,507 files
# and 20+ minutes per rung, which 4M chunks cut to 79 files. One object per rung
# retires the whole question; compression now lives in `snapshot_format`.


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
) -> Path:
    """Write a snapshot and atomically publish it. Returns the snapshot path.

    ``abstraction_id`` identifies the bucket ASSIGNMENT, which the tree
    fingerprint deliberately does not cover: the fingerprint pins node identity,
    layout and bucket COUNTS, so an abstraction that buckets the same hands
    differently under the same counts produces an identical fingerprint. Resuming
    or scoring across such a change silently trains on rebucketed hands. Pass the
    resolved abstraction hash to make that detectable.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = checkpoint_dir / f"static-{iteration}{snapshot_format.SUFFIX}"

    # A trainer's own scaffolding rides along beside the five. It is not part of
    # the answer and nothing reading a checkpoint needs it, but a run reaching
    # its target in several tasks would otherwise restart it from zero at every
    # task boundary -- and CFR-BR's opponent lives in there.
    arrays = {name: np.asarray(getattr(storage, name)) for name in (*_ARRAYS, *storage.extra)}
    snapshot_format.write_snapshot(
        snapshot_path,
        arrays,
        {
            "iteration": iteration,
            "fingerprint": storage.tree.fingerprint(),
            "format_version": FORMAT_VERSION,
            "num_rows": storage.tree.num_rows,
            "num_slots": storage.tree.num_slots,
        },
    )

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
        # STILL SPELLED `zarr`, and deliberately: it is the field every
        # published manifest already carries and every reader already looks up.
        # Renaming it would make 1,081 existing rungs unreadable to buy a word.
        "zarr": snapshot_path.name,
        "fingerprint": storage.tree.fingerprint(),
        "abstraction_id": abstraction_id,
        "format_version": FORMAT_VERSION,
        "retained": _extend_ladder(previous, iteration, snapshot_path.name, retain_every),
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
    return snapshot_path


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
    """Delete snapshots that are neither current nor retained.

    THROUGH `object_name` ON BOTH SIDES. The manifest of a run that started
    before the format changed names `static-N.zarr` while the file beside it is
    `static-N.ckpt.zst`, so comparing the two literally kept nothing: the rung
    a resume had just fetched matched no entry and was deleted as surplus.
    """
    keep = {records.object_name(manifest["zarr"])} | {
        records.object_name(entry["zarr"]) for entry in manifest["retained"]
    }
    for path in checkpoint_dir.glob(f"static-*{snapshot_format.SUFFIX}"):
        if path.name not in keep:
            path.unlink(missing_ok=True)


#: What a PLAYER reads. `regrets`, `reach_counts` and `cumulative_utility` exist
#: so a run can RESUME, and nothing on the decision path touches them — measured
#: on the serve box, they are 1.23 GB of a 2.7 GB resident blueprint. `np.zeros`
#: is lazily backed, so an array that is never written never costs a page: not
#: loading them is the whole saving, and the storage class is unchanged.
PLAY_ARRAYS: tuple[str, ...] = ("strategy_sum", "visited")


def load_checkpoint(
    storage: StaticArrayStorage,
    checkpoint_dir: Path,
    *,
    arrays: Sequence[str] | None = None,
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
    if manifest.fingerprint not in (expected, legacy):
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
    # Push the restriction DOWN to the read. The rung is one compressed object
    # now, so filtering after the fact would decompress every array and throw
    # most of them away -- which is the whole cost a play-only load avoids.
    #
    # The rung decides its own vintage; the manifest check above only settles
    # that this run belongs to this tree at all.
    root, translate = _open_snapshot(checkpoint_dir, entry, expected, legacy, arrays)

    row_source, slot_source = _legacy_index_maps(storage.tree) if translate else (None, None)
    if translate:
        logger.info("Checkpoint is v1 node-major; permuting arrays into the bucket-major layout.")
    # `arrays` restricts WHAT is read; `storage.extra` widens it. They compose:
    # a play-only load wants two of the base arrays and none of a trainer's.
    wanted = set(arrays) if arrays is not None else None
    for name in (*_ARRAYS, *(name for name in storage.extra if name in root)):
        if wanted is not None and name not in wanted:
            continue
        target = getattr(storage, name)
        source = root[name]
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


def _open_snapshot(
    checkpoint_dir: Path,
    entry: dict,
    current: str,
    legacy: str,
    names: Iterable[str] | None = None,
) -> tuple[dict[str, np.ndarray], bool]:
    """One rung's arrays, and whether they need permuting into the current layout.

    THE VINTAGE IS THE SNAPSHOT'S OWN, NOT THE MANIFEST'S. A ladder written
    across the v1->v2 layout change carries rungs of both, and the manifest's
    fingerprint describes only the rung that was current when it was last
    written -- so deciding from it refused every OLDER rung of such a run.
    Measured 09-09: five published ladders, 18 rungs, and because the stranded
    ones are always the early rungs it is the left half of a within-run
    convergence curve, for the four 100M abstraction arms among others.

    THE NAME IS MAPPED, NOT TRUSTED. A manifest written before the format
    changed still spells `static-N.zarr` and is never repointed, so
    `records.object_name` is what turns the claim into the file that exists.
    """
    path = Path(checkpoint_dir) / records.object_name(entry["zarr"])
    arrays, attrs = snapshot_format.read_snapshot(path, names)
    stored = attrs.get("fingerprint")
    if stored == current:
        return arrays, False
    if stored == legacy:
        return arrays, True
    raise FingerprintMismatchError(
        f"Snapshot {entry['zarr']} carries fingerprint {stored}, which is neither this "
        f"tree ({current}) nor its v1 node-major layout ({legacy}). Loading it would "
        "reinterpret every row as a different infoset."
    )


def read_strategy_sum(storage: StaticArrayStorage, checkpoint_dir: Path, iteration: int):
    """One retained rung's ``strategy_sum``, in THIS storage's slot order.

    Exists because a second rung is a legitimate INPUT -- a windowed or
    reweighted average is a combination of rungs -- and every published run
    predates the bucket-major layout, so a reader that skipped the v1
    permutation would silently combine two different orderings.
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest = StaticCheckpointManifest.read(checkpoint_dir)
    if manifest is None:
        raise FileNotFoundError(f"No static checkpoint manifest in {checkpoint_dir}")
    expected = storage.tree.fingerprint()
    legacy = storage.tree.legacy_fingerprint()
    if manifest.fingerprint not in (expected, legacy):
        raise FingerprintMismatchError(
            f"Checkpoint in {checkpoint_dir} was written against betting tree "
            f"{manifest.fingerprint}, but this storage indexes tree {expected}."
        )
    root, translate = _open_snapshot(
        checkpoint_dir, manifest.entry_for(iteration), expected, legacy
    )
    values = root["strategy_sum"]
    if values.shape != storage.strategy_sum.shape:
        raise ValueError(
            f"Rung {iteration} holds {values.shape} slots, storage expects "
            f"{storage.strategy_sum.shape} — fingerprints matched, so this is a format bug."
        )
    if translate:
        values = values[_legacy_index_maps(storage.tree)[1]]
    return values


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
