"""Queueing a precompute from the menu.

This flow is a cloud client: the leg it queues must be the same one
``submit-precompute`` builds, and it must make the same refusal when the target
name is already published. Both are pinned here, because the failure they guard
against -- republishing a name under an unchanged abstraction hash -- silently
invalidates the provenance of every run trained against it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import ClientAuthenticationError

from src.core.game.state import Street
from src.interfaces.cli.flows.combo_precompute import run
from src.interfaces.cli.ui.context import CliContext
from src.interfaces.cloud import spec
from src.interfaces.cloud.config import CloudConfigError
from src.pipeline.abstraction.config import PrecomputeConfig


def _make_ctx(tmp_path: Path) -> CliContext:
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def _config() -> PrecomputeConfig:
    return PrecomputeConfig.model_validate(
        {
            "config_name": "quick_test",
            "buckets": {"flop": 10, "turn": 20, "river": 30},
        }
    )


@pytest.fixture(autouse=True)
def _quiet_ui(monkeypatch):
    monkeypatch.setattr(run.ui, "header", lambda _t: None)
    monkeypatch.setattr(run.ui, "error", lambda _m: print(_m))
    monkeypatch.setattr(run.ui, "pause", lambda: None)


class TestGetConfigChoice:
    def test_no_configs_is_reported(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(run, "list_config_names", lambda _d: [])

        assert run._get_config_choice(_make_ctx(tmp_path)) is None
        assert "No configuration files found" in capsys.readouterr().out

    def test_cancelling_answers_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run, "list_config_names", lambda _d: ["quick_test"])
        monkeypatch.setattr(run.prompts, "select", lambda *_a, **_k: None)

        assert run._get_config_choice(_make_ctx(tmp_path)) is None

    def test_it_returns_the_stem_not_the_config_name(self, tmp_path, monkeypatch):
        """The stem is what a leg carries; the node resolves the YAML from it."""
        monkeypatch.setattr(run, "list_config_names", lambda _d: ["quick_test"])
        monkeypatch.setattr(run.prompts, "select", lambda *_a, **_k: "quick_test.yaml")
        monkeypatch.setattr(run.PrecomputeConfig, "from_yaml", staticmethod(lambda _s: _config()))

        chosen = run._get_config_choice(_make_ctx(tmp_path))

        assert chosen is not None
        assert chosen[0] == "quick_test"

    def test_an_unloadable_config_is_reported(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(run, "list_config_names", lambda _d: ["broken"])
        monkeypatch.setattr(run.prompts, "select", lambda *_a, **_k: "broken.yaml")

        def _boom(_stem):
            raise ValueError("bad yaml")

        monkeypatch.setattr(run.PrecomputeConfig, "from_yaml", staticmethod(_boom))

        assert run._get_config_choice(_make_ctx(tmp_path)) is None
        assert "Error loading config 'broken': bad yaml" in capsys.readouterr().out


class TestEstimateTime:
    def test_it_reports_every_street_and_a_total(self, capsys):
        run._estimate_time(_config(), workers=16)

        out = capsys.readouterr().out
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            assert street.name in out
        assert "TOTAL:" in out

    def test_more_workers_never_lengthens_the_estimate(self, capsys):
        """The estimate divides by the LEG's worker count, not this machine's cores."""
        run._estimate_time(_config(), workers=1)
        one = capsys.readouterr().out
        run._estimate_time(_config(), workers=32)
        many = capsys.readouterr().out

        assert "hours" in one
        assert one != many


class TestHandleComboPrecompute:
    def _wire(self, monkeypatch, *, published=(), workers=16, confirm=True):
        monkeypatch.setattr(run, "_get_config_choice", lambda _ctx: ("quick_test", _config()))
        monkeypatch.setattr(run.CloudConfig, "load", staticmethod(lambda: object()))
        monkeypatch.setattr(run, "published_abstractions", lambda _c: set(published))
        monkeypatch.setattr(run.prompts, "prompt_int", lambda *_a, **_k: workers)
        monkeypatch.setattr(run.prompts, "confirm", lambda *_a, **_k: confirm)
        queued: list[list[spec.LegSpec]] = []
        monkeypatch.setattr(run, "queue_legs", lambda make: queued.append(make("snap-1")))
        return queued

    def test_no_config_cancels(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(run, "_get_config_choice", lambda _ctx: None)
        monkeypatch.setattr(
            run.CloudConfig, "load", staticmethod(lambda: pytest.fail("must not reach the cloud"))
        )

        run.handle_combo_precompute(_make_ctx(tmp_path))

        assert "Cancelled." in capsys.readouterr().out

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(CloudConfigError("no terraform"), id="terraform-missing"),
            pytest.param(ClientAuthenticationError("expired"), id="bad-credential"),
        ],
    )
    def test_an_unreadable_share_is_reported_not_raised(self, tmp_path, monkeypatch, capsys, error):
        monkeypatch.setattr(run, "_get_config_choice", lambda _ctx: ("quick_test", _config()))

        def _boom():
            raise error

        monkeypatch.setattr(run.CloudConfig, "load", staticmethod(_boom))

        run.handle_combo_precompute(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        assert "Could not read the share" in out
        assert "az login" in out

    def test_an_already_published_target_is_refused(self, tmp_path, monkeypatch, capsys):
        target = run.target_name("quick_test")
        queued = self._wire(monkeypatch, published=(target,))

        run.handle_combo_precompute(_make_ctx(tmp_path))

        out = capsys.readouterr().out
        assert "already published" in out
        assert "invalidating the provenance" in out
        assert queued == []

    def test_cancelling_the_worker_prompt_queues_nothing(self, tmp_path, monkeypatch):
        queued = self._wire(monkeypatch, workers=None)

        run.handle_combo_precompute(_make_ctx(tmp_path))

        assert queued == []

    def test_declining_the_confirm_queues_nothing(self, tmp_path, monkeypatch, capsys):
        queued = self._wire(monkeypatch, confirm=False)

        run.handle_combo_precompute(_make_ctx(tmp_path))

        assert queued == []
        assert "Cancelled." in capsys.readouterr().out

    def test_it_queues_the_leg_submit_precompute_would_build(self, tmp_path, monkeypatch):
        queued = self._wire(monkeypatch, workers=8)

        run.handle_combo_precompute(_make_ctx(tmp_path))

        [(leg,)] = queued
        assert leg.op == spec.PRECOMPUTE
        assert leg.config == "quick_test"
        assert leg.workers == 8
        assert leg.code_snapshot == "snap-1"
        assert leg.timeout == run.PRECOMPUTE_TIMEOUT
