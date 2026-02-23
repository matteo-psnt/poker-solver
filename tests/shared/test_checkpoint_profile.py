"""Tests for per-checkpoint phase profiling.

The profiler sits inside the checkpoint path, so its contract is as much about
what it must *not* do (raise, or record when inactive) as what it records.
"""

from __future__ import annotations

import json
import threading

from src.shared import checkpoint_profile


def _rows(run_dir):
    path = run_dir / checkpoint_profile.PROFILE_FILENAME
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_phase_outside_recording_is_a_noop():
    with checkpoint_profile.phase("orphan"):
        pass
    checkpoint_profile.add_stats(ignored=1)


def test_emit_writes_one_row_per_recording(tmp_path):
    with checkpoint_profile.recording(100) as rec:
        with checkpoint_profile.phase("a"):
            pass
        checkpoint_profile.add_stats(num_infosets=7)
        checkpoint_profile.emit(tmp_path, rec, num_workers=4)

    (row,) = _rows(tmp_path)
    assert row["iteration"] == 100
    assert row["num_infosets"] == 7
    assert row["num_workers"] == 4
    assert "a" in row["phases"]
    assert row["total_seconds"] >= 0


def test_repeated_phase_names_accumulate(tmp_path):
    with checkpoint_profile.recording(1) as rec:
        for _ in range(3):
            with checkpoint_profile.phase("repeated"):
                pass
        checkpoint_profile.emit(tmp_path, rec)

    (row,) = _rows(tmp_path)
    assert len(row["phases"]) == 1, "same-named phases must merge, not overwrite"


def test_nested_recording_does_not_start_a_second_row(tmp_path):
    with checkpoint_profile.recording(1) as outer:
        with checkpoint_profile.recording(2) as inner:
            assert inner is outer
        checkpoint_profile.emit(tmp_path, outer)

    assert len(_rows(tmp_path)) == 1


def test_recording_is_cleared_even_if_the_body_raises():
    try:
        with checkpoint_profile.recording(1):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    # A leaked recording would silently attribute the next checkpoint's phases here.
    with checkpoint_profile.phase("after"):
        pass


def test_phase_records_even_when_the_body_raises(tmp_path):
    with checkpoint_profile.recording(1) as rec:
        try:
            with checkpoint_profile.phase("failing"):
                raise ValueError("boom")
        except ValueError:
            pass
        checkpoint_profile.emit(tmp_path, rec)

    (row,) = _rows(tmp_path)
    assert "failing" in row["phases"]


def test_emit_never_raises_on_a_bad_destination(tmp_path):
    bad = tmp_path / "not-a-dir"
    bad.write_text("i am a file")
    with checkpoint_profile.recording(1) as rec:
        checkpoint_profile.emit(bad, rec)

    checkpoint_profile.emit(None, None)


def test_recording_is_per_thread(tmp_path):
    """The checkpoint runs on a background executor thread; state must not leak."""
    seen = []

    def worker():
        seen.append(checkpoint_profile._current())

    with checkpoint_profile.recording(1):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert seen == [None]


def test_measure_tree_counts_files_and_bytes(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)

    assert checkpoint_profile.measure_tree(tmp_path) == {"files": 2, "bytes": 15}


def test_measure_tree_on_a_missing_path_is_zero(tmp_path):
    assert checkpoint_profile.measure_tree(tmp_path / "nope") == {"files": 0, "bytes": 0}


def test_record_volume_commit_appends_a_tagged_row(tmp_path):
    checkpoint_profile.record_volume_commit(tmp_path, 12.5, run_files=1900)

    (row,) = _rows(tmp_path)
    assert row["event"] == "volume_commit"
    assert row["total_seconds"] == 12.5
    assert row["run_files"] == 1900


def test_checkpoint_and_commit_rows_coexist(tmp_path):
    with checkpoint_profile.recording(5) as rec:
        checkpoint_profile.emit(tmp_path, rec)
    checkpoint_profile.record_volume_commit(tmp_path, 1.0)

    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert [r.get("event") for r in rows] == [None, "volume_commit"]
