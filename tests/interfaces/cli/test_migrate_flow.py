"""Tests for the migrate-run CLI flow wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.interfaces.cli.flows import migrate as migrate_flow
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.services import RunSummary
from src.pipeline.training.versioning import REPRESENTATION_VERSION


def _ctx(tmp_path) -> CliContext:
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def _summary(name, *, loadable, has_checkpoint, blocker) -> RunSummary:
    return RunSummary(
        name=name,
        commits_ago=5,
        git_dirty=False,
        representation_version=1,
        current_version=REPRESENTATION_VERSION,
        has_checkpoint=has_checkpoint,
        loadable=loadable,
        blocker=blocker,
        iterations=10000,
        num_infosets=270000,
        config_name="production",
        status="completed",
    )


def test_migrates_selected_run_into_versioned_dir(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    summaries = [
        _summary("run-fmt", loadable=False, has_checkpoint=True, blocker="format v1 ≠ v3"),
        _summary("run-nockpt", loadable=False, has_checkpoint=False, blocker="no checkpoint"),
        _summary("run-ok", loadable=True, has_checkpoint=True, blocker=None),
    ]
    monkeypatch.setattr(migrate_flow.services, "describe_runs", lambda _dir: summaries)

    captured = {}

    def _fake_select(_ctx, _msg, choices):
        # Only the format-blocked run with a checkpoint should be offered.
        captured["values"] = [c.value for c in choices]
        return "run-fmt"

    monkeypatch.setattr(migrate_flow.prompts, "select", _fake_select)

    def _fake_migrate(src, dst):
        captured["src"] = src
        captured["dst"] = dst
        return dst

    monkeypatch.setattr(migrate_flow, "migrate_run", _fake_migrate)
    monkeypatch.setattr(migrate_flow.ui, "pause", lambda *_a, **_k: None)
    monkeypatch.setattr(migrate_flow.ui, "info", lambda *_a, **_k: None)

    migrate_flow.migrate_run_flow(ctx)

    assert "run-fmt" in captured["values"]
    assert "run-nockpt" not in captured["values"]  # no checkpoint: not migratable here
    assert "run-ok" not in captured["values"]  # already loadable
    assert captured["src"] == ctx.runs_dir / "run-fmt"
    assert captured["dst"] == ctx.runs_dir / f"run-fmt-v{REPRESENTATION_VERSION}"


def test_no_migratable_runs_is_graceful(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(
        migrate_flow.services,
        "describe_runs",
        lambda _dir: [_summary("run-ok", loadable=True, has_checkpoint=True, blocker=None)],
    )
    called = SimpleNamespace(migrated=False)
    monkeypatch.setattr(
        migrate_flow, "migrate_run", lambda *_a, **_k: called.__setattr__("migrated", True)
    )
    monkeypatch.setattr(migrate_flow.ui, "pause", lambda *_a, **_k: None)
    monkeypatch.setattr(migrate_flow.ui, "info", lambda *_a, **_k: None)

    migrate_flow.migrate_run_flow(ctx)

    assert called.migrated is False  # nothing to migrate, migrate_run never called
