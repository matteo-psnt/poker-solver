"""Every command's payload must survive human-mode rendering.

Renderers are pure formatting, so the only failure they can have is a key the
payload does not carry -- the failure that shipped once, when a command with no
renderer of its own fell through to the evaluate branch and died on
``payload["results"]``. These drive each renderer with an instance of the
payload its own ``run`` declares.

The instance is BUILT from the model, not written here. A hand-written example
per command was a second declaration of every shape: it had to be extended for
each new command, it drifted whenever a field moved, and it was the fourth
most-read file in the repo. `example_of` reads the fields instead, so a new
command is covered the moment it is registered.

"Every subcommand HAS a renderer" is no longer tested: a ``Command`` cannot be
constructed without one, so it is a property of the type rather than a thing to
check.
"""

import pytest

from src.interfaces.commands import evaluate, load_all
from src.interfaces.commands.blueprint_serve import BlueprintServePayload
from src.interfaces.commands.progress import ProgressPayload, ProgressRow
from src.interfaces.commands.runinfo import RunInfoPayload
from src.interfaces.commands.serve import ServePayload
from src.pipeline.services import EvaluationPayload
from tests.test_helpers import example_of

BY_NAME = {command.name: command for command in load_all()}


# These two renderers ARE the server: `render` prints one line and then blocks
# in `uvicorn.run`. They are skipped by the loop below and driven individually
# in `TestTheServersStillFormatTheirPayload`, with the server stubbed -- which
# is the part that matters, and the part that was NOT covered when the comment
# here claimed `tests/interfaces/web/` did it. Nothing called either renderer,
# so both kept subscripting a payload that had become a model, and
# `poker-solver serve` died before uvicorn.
SIDE_EFFECTING = {"serve", "blueprint-serve"}


class TestEveryOpRenders:
    @pytest.mark.parametrize("name", sorted(set(BY_NAME) - SIDE_EFFECTING))
    def test_every_command_renders_its_payload(self, name, capsys):
        """The smallest instance of the declared payload, and the renderer
        must still say something: an empty answer is a state a human sees."""
        command = BY_NAME[name]
        command.render(example_of(command.payload_type()))
        assert capsys.readouterr().out, f"'{name}' rendered nothing"

    def test_a_new_command_does_not_borrow_the_evaluate_renderer(self, capsys):
        # The original regression's shape: a command with no renderer of its own
        # fell through to the evaluate branch and died on payload["results"].
        BY_NAME["runinfo"].render(example_of(RunInfoPayload, run_id="run-a"))
        out = capsys.readouterr().out
        assert "run-a" in out
        assert "Evaluation complete" not in out

    def test_progress_blanks_fields_a_legacy_row_predates(self, capsys):
        """A resumed run appends across tasks, so one log spans code versions."""
        payload = ProgressPayload(
            run_id="run-a",
            total_rows=2,
            schema_version_min=0,
            schema_version_max=1,
            rows=[
                ProgressRow(schema_version=1, iteration=1_000_000, coverage=0.1),
                # A row from before a field existed: the renderer must blank it,
                # not crash and not print a placeholder that reads as a measurement.
                ProgressRow(schema_version=0, iteration=2_000_000, coverage=0.101),
            ],
        )
        BY_NAME["progress"].render(payload)
        out = capsys.readouterr().out
        assert "2,000,000" in out, "the legacy row must still be shown"
        assert "10.1%" in out, "and the fields it does carry must render"


class TestTheServersStillFormatTheirPayload:
    """The two renderers the loop above cannot call, called anyway.

    `serve` and `blueprint-serve` block in `uvicorn.run`, so they are skipped by
    `TestEveryOpRenders` -- and that skip is why both went on subscripting a
    payload that had become a model. `poker-solver serve` (and `just console`,
    the only way to start it) died with `TypeError: 'ServePayload' object is not
    subscriptable` BEFORE uvicorn, and `--json` still worked, so nothing
    noticed. `blueprint-serve` is worse: `infra/serve/main.tf` invokes it
    without `--json`, so the unit failed and the box deallocated while
    `/api/box/start` reported success.

    Stubbing the server is the whole trick -- everything before it is ordinary
    formatting, and that is the part that broke.
    """

    def test_serve_prints_where_it_will_listen(self, monkeypatch, capsys):
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        payload = ServePayload(
            url="http://127.0.0.1:8765", host="127.0.0.1", port=8765, reload=False
        )
        BY_NAME["serve"].render(payload)
        assert "http://127.0.0.1:8765" in capsys.readouterr().out

    def test_blueprint_serve_reads_every_field_before_it_loads(self, monkeypatch, capsys):
        """It reads `run_dir` and `runs_dir` before touching a blueprint.

        Stopped at the load rather than the listen: building one is ~a minute
        off the share. Reaching the refusal proves the payload access above it.
        """
        import uvicorn

        from src.adapters.postgres import connect

        monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
        monkeypatch.setattr(connect, "record_source_from_environment", lambda: None)
        with pytest.raises(Exception, match=r"run-production|No such file|not found|Errno"):
            BY_NAME["blueprint-serve"].render(
                BlueprintServePayload(
                    run="run-production-025433-1095",
                    run_dir="/mnt/work/runs/run-production-025433-1095",
                    at_iteration=None,
                    url="http://127.0.0.1:8790",
                    host="127.0.0.1",
                    port=8790,
                )
            )


class TestResolverMatchRenderer:
    """The resolver gate reports a chip edge, not exploitability.

    Its payload carries no `exploitability_mbb` at all, and its
    `confidence_95_mbb` is an INTERVAL rather than a half-width. Both facts
    broke this renderer on first contact with a real run -- the second one as a
    `TypeError: unsupported format string passed to tuple.__format__`, six
    minutes into a paid box. A shape example here costs nothing and catches it
    in milliseconds, which is the whole argument for driving every renderer here.
    """

    PAYLOAD = EvaluationPayload(
        run_id="run-a",
        method="resolver_match",
        estimator="resolver_match (duplicate-deal chip edge)",
        infosets=32_240_608,
        results={
            "resolver_mbb_per_hand": 12.5,
            "se_mbb": 4.0,
            "confidence_95_mbb": (4.7, 20.3),
            "p_value": 0.0018,
            "num_deals": 1000,
            "num_hands": 2000,
            "resolver_decisions": 5000,
            "resolver_fallbacks": 3,
            "leaf_continuation_fraction": 0.5,
            "resolver_max_iterations": 64,
        },
    )

    def test_it_renders_the_interval_without_dying(self, capsys):
        evaluate.render(self.PAYLOAD)
        out = capsys.readouterr().out
        assert "+12.50 mbb/hand" in out
        assert "+4.70..+20.30" in out

    def test_it_shows_the_arm_and_the_fallback_count(self, capsys):
        """Two numbers that decide whether the result means anything: which arm
        produced it, and whether the resolver actually resolved."""
        evaluate.render(self.PAYLOAD)
        out = capsys.readouterr().out
        assert "0.5 pot" in out
        assert "fell back" in out
        assert "5,000" in out
