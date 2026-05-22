"""Tests for run summarization (age + loadability annotations)."""

from types import SimpleNamespace

from src.pipeline.services import runs as services_runs
from src.shared.records import STATIC_CHECKPOINT


def _patch_metadata(monkeypatch, *, commit="abc123", dirty=False):
    monkeypatch.setattr(
        services_runs,
        "load_run_metadata",
        lambda _dir: SimpleNamespace(
            git_commit=commit,
            git_dirty=dirty,
            iterations=6000,
            num_infosets=43041,
            config_name="quick_test",
            status="completed",
        ),
    )
    monkeypatch.setattr(services_runs, "commits_ahead_of", lambda _commit: 4)


def test_missing_checkpoint_is_not_loadable(tmp_path, monkeypatch):
    (tmp_path / "run-x").mkdir()
    _patch_metadata(monkeypatch)

    summary = services_runs._summarize_run(tmp_path, "run-x")

    assert not summary.loadable
    assert summary.blocker == "no checkpoint"
    assert summary.commits_ago == 4


def test_current_run_is_loadable(tmp_path, monkeypatch):
    run = tmp_path / "run-cur"
    run.mkdir()
    (run / STATIC_CHECKPOINT).write_text("{}")
    _patch_metadata(monkeypatch, dirty=True)

    summary = services_runs._summarize_run(tmp_path, "run-cur")

    assert summary.loadable
    assert summary.blocker is None
    assert summary.git_dirty is True


def test_a_run_without_the_static_manifest_is_not_loadable(tmp_path, monkeypatch):
    """Loadability is decided by the manifest, not by a stray zarr directory.

    A bare ``*.zarr`` is not evidence a run can be opened: the manifest is what
    is committed atomically with the arrays, so a directory carrying only the
    zarr is exactly the torn-copy case the marker exists to reject.
    """
    run = tmp_path / "run-zarr-only"
    run.mkdir()
    (run / "static-6000.zarr").mkdir()
    _patch_metadata(monkeypatch)

    assert not services_runs._summarize_run(tmp_path, "run-zarr-only").loadable


def test_unreadable_metadata_is_blocked(tmp_path, monkeypatch):
    (tmp_path / "run-bad").mkdir()

    def _raise(_dir):
        raise ValueError("corrupt")

    monkeypatch.setattr(services_runs, "load_run_metadata", _raise)

    summary = services_runs._summarize_run(tmp_path, "run-bad")

    assert not summary.loadable
    assert summary.blocker == "unreadable metadata"
    assert summary.commits_ago is None
