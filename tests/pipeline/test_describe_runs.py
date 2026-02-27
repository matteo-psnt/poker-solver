"""Tests for run summarization (age + loadability annotations)."""

from types import SimpleNamespace

from src.pipeline import services
from src.pipeline.training.versioning import REPRESENTATION_VERSION


def _patch_metadata(monkeypatch, *, version, commit="abc123", dirty=False):
    monkeypatch.setattr(
        services,
        "load_run_metadata",
        lambda _dir: SimpleNamespace(
            representation_version=version,
            git_commit=commit,
            git_dirty=dirty,
            iterations=6000,
            num_infosets=43041,
            config_name="quick_test",
            status="completed",
        ),
    )
    monkeypatch.setattr(services, "commits_ahead_of", lambda _commit: 4)


def test_missing_checkpoint_is_not_loadable(tmp_path, monkeypatch):
    (tmp_path / "run-x").mkdir()
    _patch_metadata(monkeypatch, version=REPRESENTATION_VERSION)

    summary = services._summarize_run(tmp_path, "run-x")

    assert not summary.loadable
    assert summary.blocker == "no checkpoint"
    assert summary.commits_ago == 4


def test_stale_format_is_not_loadable(tmp_path, monkeypatch):
    run = tmp_path / "run-old"
    run.mkdir()
    (run / "CHECKPOINT.json").write_text("{}")
    _patch_metadata(monkeypatch, version=REPRESENTATION_VERSION - 1)

    summary = services._summarize_run(tmp_path, "run-old")

    assert not summary.loadable
    assert summary.blocker is not None and "format" in summary.blocker


def test_current_run_is_loadable(tmp_path, monkeypatch):
    run = tmp_path / "run-cur"
    run.mkdir()
    (run / "CHECKPOINT.json").write_text("{}")
    _patch_metadata(monkeypatch, version=REPRESENTATION_VERSION, dirty=True)

    summary = services._summarize_run(tmp_path, "run-cur")

    assert summary.loadable
    assert summary.blocker is None
    assert summary.git_dirty is True


def test_legacy_zarr_layout_counts_as_checkpoint(tmp_path, monkeypatch):
    run = tmp_path / "run-legacy"
    run.mkdir()
    (run / "checkpoint-6000.zarr").mkdir()
    _patch_metadata(monkeypatch, version=REPRESENTATION_VERSION)

    assert services._summarize_run(tmp_path, "run-legacy").loadable


def test_unreadable_metadata_is_blocked(tmp_path, monkeypatch):
    (tmp_path / "run-bad").mkdir()

    def _raise(_dir):
        raise ValueError("corrupt")

    monkeypatch.setattr(services, "load_run_metadata", _raise)

    summary = services._summarize_run(tmp_path, "run-bad")

    assert not summary.loadable
    assert summary.blocker == "unreadable metadata"
    assert summary.commits_ago is None
