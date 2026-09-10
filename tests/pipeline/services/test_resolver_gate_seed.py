"""`--seed` must reach the resolver gate, which is what picks its deals."""

from __future__ import annotations

import inspect

import pytest

from src.interfaces.cli.headless import build_parser
from src.interfaces.commands import evaluate as evaluate_command
from src.pipeline.services import scoring
from src.pipeline.services.scoring.matches import evaluate_run_resolver_gate


class _StopError(Exception):
    """Cut the call short once the seed has been observed."""


class _NullSink:
    def scored(self, *args, **kwargs) -> None: ...


class TestTheGateSeedIsWired:
    """A MEASURED failure: six seeds fanned over six nodes returned six
    IDENTICAL numbers -- same edge, same CI, same decision count -- because
    `--seed` reached `LBRConfig` and nothing else, so every gate row ever
    recorded carries `base_seed=1`. A gate arm could not be replicated, and the
    fan-out read as six independent samples of the same knob tier.
    """

    def test_the_flag_reaches_the_service(self, monkeypatch, tmp_path):
        args = build_parser().parse_args(
            ["evaluate", "--run", "r", "--method", "resolver_match", "--seed", "17"]
        )
        assert args.seed == 17

        captured: dict[str, object] = {}

        def spy(run_dir, **kwargs):
            captured.update(kwargs)
            raise _StopError

        monkeypatch.setattr(scoring, "evaluate_run_resolver_gate", spy)
        with pytest.raises(_StopError):
            scoring.evaluate_and_record(
                run_dir=tmp_path,
                sink=_NullSink(),
                method="resolver_match",
                resolver_gate_seed=args.seed,
            )

        assert captured.get("seed") == 17

    def test_the_command_passes_a_seed_even_when_the_flag_is_absent(self):
        """`--seed` defaults to None (meaning "random") for lbr; the gate wants a
        deterministic default, so None must become 1 rather than reach the
        service and break the deal generator."""
        source = inspect.getsource(evaluate_command.run)
        assert "resolver_gate_seed=" in source, "the command must pass the gate seed at all"
        assert "if args.seed is not None else 1" in source

    def test_the_service_still_defaults_to_one(self):
        assert inspect.signature(evaluate_run_resolver_gate).parameters["seed"].default == 1
        assert (
            inspect.signature(scoring.evaluate_and_record).parameters["resolver_gate_seed"].default
            == 1
        )
