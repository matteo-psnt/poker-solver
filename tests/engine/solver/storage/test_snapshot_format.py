"""One rung, one object -- and the properties that decide the format.

Size and speed were settled by measurement on a production rung (in the
module docstring). What tests can hold is everything that would silently
corrupt or bloat a snapshot instead.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.engine.solver.storage import snapshot_format as fmt


def _arrays(rows: int = 5000):
    """The five a checkpoint holds, with their real dtypes."""
    rng = np.random.default_rng(7)
    return {
        "cumulative_utility": rng.normal(size=rows).astype(np.float64),
        "reach_counts": rng.integers(0, 1 << 30, size=rows).astype(np.int64),
        "regrets": rng.normal(size=rows * 3).astype(np.float32),
        "strategy_sum": rng.normal(size=rows * 3).astype(np.float32),
        "visited": rng.integers(0, 2, size=rows).astype(np.uint8),
    }


class TestItRoundTripsExactly:
    """A checkpoint carries no self-describing row identity, so a value that
    changes in transit does not fail -- it trains on a different number."""

    def test_every_array_is_bit_identical(self, tmp_path):
        arrays = _arrays()
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", arrays, {"iteration": 5})
        back, _attrs = fmt.read_snapshot(tmp_path / "s.ckpt.zst")
        for name, original in arrays.items():
            assert np.array_equal(original, back[name]), name
            assert original.dtype == back[name].dtype, f"{name} dtype drifted"

    def test_shapes_survive(self, tmp_path):
        arrays = {"a": np.arange(24, dtype=np.float32).reshape(6, 4)}
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", arrays, {})
        back, _ = fmt.read_snapshot(tmp_path / "s.ckpt.zst")
        assert back["a"].shape == (6, 4)
        assert np.array_equal(back["a"], arrays["a"])

    def test_attrs_survive(self, tmp_path):
        attrs = {"iteration": 5_000_000, "fingerprint": "ab" * 8, "num_rows": 1114482}
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", _arrays(64), attrs)
        _back, got = fmt.read_snapshot(tmp_path / "s.ckpt.zst")
        assert got == attrs

    def test_an_empty_array_survives(self, tmp_path):
        """A run that checkpoints before touching anything writes these."""
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", {"a": np.array([], dtype=np.float32)}, {})
        back, _ = fmt.read_snapshot(tmp_path / "s.ckpt.zst")
        assert back["a"].shape == (0,)


class TestItIsOneObject:
    def test_exactly_one_file(self, tmp_path):
        """107 files per rung was the thing being fixed."""
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", _arrays(), {})
        assert [p.name for p in tmp_path.iterdir()] == ["s.ckpt.zst"]

    def test_it_is_smaller_than_the_raw_arrays(self, tmp_path):
        arrays = _arrays()
        size = fmt.write_snapshot(tmp_path / "s.ckpt.zst", arrays, {})
        assert size < sum(a.nbytes for a in arrays.values())


class TestTheHeaderIsReadableAlone:
    def test_metadata_costs_no_decompression(self, tmp_path):
        """Refusing a snapshot for the wrong tree must cost a few hundred
        bytes, not 1.7 GB of decompression."""
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", _arrays(), {"fingerprint": "cafe"})
        header = fmt.read_header(tmp_path / "s.ckpt.zst")
        assert header["attrs"]["fingerprint"] == "cafe"
        assert [a["name"] for a in header["arrays"]] == list(_arrays())

    def test_the_header_is_json_and_length_prefixed(self, tmp_path):
        """Self-describing on purpose: nothing outside the file has to remember
        the layout, which is what zarr's `.zarray` files were doing."""
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", {"a": np.zeros(4)}, {})
        raw = (tmp_path / "s.ckpt.zst").read_bytes()
        length = int.from_bytes(raw[: fmt.HEADER_LENGTH_BYTES], "little")
        head = json.loads(raw[fmt.HEADER_LENGTH_BYTES : fmt.HEADER_LENGTH_BYTES + length])
        assert head["arrays"][0]["dtype"] == np.zeros(4).dtype.str


class TestConcurrencyIsSafe:
    """MEASURED FAULT: a `ZstdDecompressor` shared across a pool raises
    "decompression error: Data corruption detected". The read path decodes
    frames in parallel, so a shared instance would corrupt a checkpoint
    intermittently -- the worst way to find out."""

    def test_reading_the_same_snapshot_concurrently_is_exact(self, tmp_path):
        arrays = _arrays(20_000)
        fmt.write_snapshot(tmp_path / "s.ckpt.zst", arrays, {})
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda _i: fmt.read_snapshot(tmp_path / "s.ckpt.zst"), range(8))
            )
        for back, _attrs in results:
            for name, original in arrays.items():
                assert np.array_equal(original, back[name]), name

    def test_the_decompressor_is_built_inside_the_worker(self):
        """Pinned as source, because the failure it prevents is intermittent
        and would not reliably fail a test."""
        import inspect

        body = inspect.getsource(fmt.read_snapshot)
        decode = body[body.index("def _decode") :]
        assert "ZstdDecompressor()" in decode, "hoisting it out of the worker corrupts reads"


class TestTheLevelIsDeliberate:
    def test_level_three(self):
        """1 -> 3 buys 38 MB per production rung for +0.08s; 3 -> 7 costs twice
        the write for another 20 MB."""
        assert fmt.COMPRESSION_LEVEL == 3

    def test_compression_uses_every_core(self):
        """7.79s -> 2.44s on the production rung."""
        assert fmt.COMPRESS_THREADS == -1


def test_a_truncated_object_raises_rather_than_returning_short_arrays(tmp_path):
    """Half a snapshot that loads is a run trained on partial regrets."""
    fmt.write_snapshot(tmp_path / "s.ckpt.zst", _arrays(), {})
    raw = (tmp_path / "s.ckpt.zst").read_bytes()
    (tmp_path / "s.ckpt.zst").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(Exception):  # noqa: B017, PT011 -- any refusal beats a short read
        fmt.read_snapshot(tmp_path / "s.ckpt.zst")
