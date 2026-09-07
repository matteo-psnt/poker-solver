"""One rung, one object: a JSON header and one zstd frame per array.

REPLACES ZARR, and the reason is that zarr's chunking was never bought for
reading. Every read here is a whole-array read -- `root[name][:]` -- nothing
slices and nothing uses `oindex`; `static_checkpoint.py`'s own comment said the
chunk size was "set by FILE COUNT, not by the write", chosen to make copying a
snapshot over SMB tolerable. The store it was a workaround for is being left.

MEASURED on a production rung (1,695 MB raw, 5 arrays):

    zarr dir, Blosc zstd(1)+BITSHUFFLE   787.7 MB   107 files   read 1.09s
    this, zstd(3), threaded              541.9 MB     1 file    read 1.38s

31% smaller for a comparable read, one object instead of 107, and it drops two
pinned dependencies (`zarr`, `numcodecs`) whose upper bounds block `uv lock
--upgrade`.

STREAMED BOTH WAYS, which is the property that decides the shape. Serialising
the whole snapshot into a buffer first -- `np.savez` into `BytesIO` -- costs
+1,208 MB of peak memory on that rung, on a node whose trainer is holding its
own tables at the same moment. One frame per array means peak memory is one
array, and it costs nothing: the same rung is 583.4 MB framed against 581.7 MB
buffered.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import zstandard

from src.shared import records

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

#: Level 3, measured. 1 -> 3 buys 38 MB per rung for +0.08 s; 3 -> 7 costs
#: twice the write for another 20 MB. Compression happens once and the bytes
#: are stored and fetched forever, so the trade favours the smaller side until
#: the write starts showing up next to a checkpoint interval measured in
#: minutes -- which 2.4 s does not.
COMPRESSION_LEVEL = 3

#: `threads=-1` lets zstd use every core for compression: 7.79 s -> 2.44 s on
#: the production rung. Decompression of a single frame cannot be threaded, so
#: the read side gets its parallelism from having one frame PER ARRAY instead.
COMPRESS_THREADS = -1

#: How many frames decode at once. Five arrays, so more workers buy nothing.
READ_WORKERS = 8

#: The extension lives in `shared.records`, not here: the node wrapper resolves
#: a rung's file name and is stdlib-only, so it cannot import this module.
SUFFIX = records.SNAPSHOT_SUFFIX

HEADER_LENGTH_BYTES = 8


def write_snapshot(path: Path, arrays: Mapping[str, np.ndarray], attrs: Mapping[str, Any]) -> int:
    """Write the arrays and their metadata as ONE object. Returns bytes written.

    The header is JSON and carries dtype, shape and frame length per array, so
    the file is self-describing: nothing outside it has to remember the layout,
    which is what `zarr`'s `.zarray` files were doing.
    """
    compressor = zstandard.ZstdCompressor(level=COMPRESSION_LEVEL, threads=COMPRESS_THREADS)
    header: dict[str, Any] = {"attrs": dict(attrs), "arrays": []}
    frames = []
    for name, array in arrays.items():
        contiguous = np.ascontiguousarray(array)
        frames.append(compressor.compress(contiguous.tobytes()))
        header["arrays"].append(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "frame_bytes": len(frames[-1]),
            }
        )
    encoded = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(len(encoded).to_bytes(HEADER_LENGTH_BYTES, "little"))
        handle.write(encoded)
        for frame in frames:
            handle.write(frame)
    return path.stat().st_size


def read_header(path: Path) -> dict[str, Any]:
    """The metadata alone, without decompressing a single array.

    What a manifest check or a fingerprint guard needs: refusing a snapshot for
    the wrong tree should cost a few hundred bytes, not 1.7 GB of decompression.
    """
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(HEADER_LENGTH_BYTES), "little")
        return json.loads(handle.read(length))


def read_snapshot(
    path: Path, names: Iterable[str] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The arrays and the attrs. Frames decode in PARALLEL.

    `names` decodes only what is asked for and SEEKS PAST the rest, which is
    what one frame per array buys over a single stream: the reweighted-average
    path opens many rungs for `strategy_sum` alone, and making it pay for all
    five would be 1.7 GB of decompression per rung to use 0.6 GB of it.

    One decompressor PER FRAME, never shared. `ZstdDecompressor` is not
    thread-safe and a shared instance raises "decompression error: Data
    corruption detected" under a pool -- intermittently, which is the worst way
    to find out. Do not hoist it out of the worker.
    """
    wanted = None if names is None else set(names)
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(HEADER_LENGTH_BYTES), "little")
        header = json.loads(handle.read(length))
        frames = []
        for spec in header["arrays"]:
            if wanted is not None and spec["name"] not in wanted:
                handle.seek(spec["frame_bytes"], 1)
                continue
            frames.append((spec, handle.read(spec["frame_bytes"])))

    def _decode(item: tuple[dict[str, Any], bytes]) -> tuple[str, np.ndarray]:
        spec, raw = item
        decompressor = zstandard.ZstdDecompressor()  # per thread; see above
        flat = np.frombuffer(decompressor.decompress(raw), dtype=np.dtype(spec["dtype"]))
        return spec["name"], flat.reshape(spec["shape"])

    with ThreadPoolExecutor(max_workers=min(READ_WORKERS, max(1, len(frames)))) as pool:
        arrays = dict(pool.map(_decode, frames))
    return arrays, header["attrs"]
