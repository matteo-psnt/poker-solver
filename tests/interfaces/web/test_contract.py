"""The contract, checked against the payloads commands actually return.

`contract.py` cannot validate at request time -- FastAPI skips validation for a
handler returning a `Response`, and every endpoint returns `PayloadResponse` so
that `jsonio.dumps` keeps handling the numpy scalars and `Path` objects these
payloads carry. So the enforcement is here.

This is the link that replaced a hand-written one. The chain used to be: handler
returns an untyped dict, a `PAYLOADS` example pins it, a fixture exports the
example, and 682 lines of hand-written Zod declared it all over again. Now
`PAYLOADS` is still the pinned example -- `test_command_renderers.py` drives
every renderer from it, and `test_payloads_cover_every_command` there fails if a
command is missing -- and THIS test says the models describe those examples. The
TypeScript is generated from the models, so nobody writes a shape twice.

What a failure here means
-------------------------
A payload changed shape. That is a change to the contract the console reads, so
the model changes, `types.gen.ts` regenerates, and TypeScript points at whatever
no longer compiles. It is not a reason to loosen a field: `extra="allow"` already
absorbs anything ADDED, so a failure means something the console reads was
removed, renamed, or changed type.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from src.interfaces.web import contract
from tests.interfaces.commands.test_command_renderers import PAYLOADS
from tests.interfaces.web.test_command_coverage import _commands_the_console_invokes

# Only the commands the console READS; `test_command_coverage.py` holds the list
# of what is excluded and why, and this would be the wrong place to re-declare it.
MODELS: dict[str, type[BaseModel]] = {
    "pool-status": contract.Pool,
    "jobs": contract.Jobs,
    "tasks": contract.Tasks,
    "runs": contract.Runs,
    "runinfo": contract.RunInfo,
    "progress": contract.Progress,
    "curve": contract.Curve,
    "ledger": contract.Ledger,
    "logs": contract.LogLines,
    "cost": contract.Cost,
    "report": contract.Report,
    "compare": contract.Comparison,
    "activity": contract.Activity,
    "configs": contract.Configs,
    "autoscale-check": contract.Autoscale,
    # Each dispatch declares its OWN payload now. They shared `Dispatched` while
    # the shapes were restated here, which lost `score`'s rungs and
    # `submit-precompute`'s target every time someone read the schema.
    "submit": contract.SubmitPayload,
    "score": contract.ScorePayload,
    "submit-precompute": contract.PrecomputeDispatchPayload,
    "submit-vector": contract.SubmitVectorPayload,
    "push-code": contract.PushedCode,
    "push-data": contract.PushedData,
    "compact-legs": contract.Compacted,
    "promote": contract.Promoted,
    "cancel": contract.Cancelled,
    "serve-box": contract.Box,
}


class TestTheModelsDescribeRealPayloads:
    @pytest.mark.parametrize("op", sorted(MODELS))
    def test_the_pinned_payload_validates(self, op: str):
        """Every modelled command's example payload satisfies its model."""
        try:
            MODELS[op].model_validate(PAYLOADS[op])
        except ValidationError as error:
            raise AssertionError(
                f"`{op}`'s payload no longer matches {MODELS[op].__name__}. Something "
                f"the console READS was removed, renamed or changed type -- `extra=allow` "
                f"absorbs additions, so this is not a field that merely appeared.\n{error}"
            ) from error

    @pytest.mark.parametrize("op", sorted(MODELS))
    def test_every_modelled_command_still_exists(self, op: str):
        """A model for a command that has been deleted is dead weight that still
        looks like a contract."""
        assert op in PAYLOADS, f"{op} is modelled here but is not a command"

    def test_every_command_the_console_serves_has_a_model(self):
        """The hole the old system had, closed.

        `console/src/api/schemas.test.ts` checked 20 of 24 schemas against the
        fixture, because ITS list was hand-maintained too -- so `cancel` was
        never checked, and its schema disagreed with the command for as long as
        both existed. A coverage list that can silently omit an entry is not
        coverage, so this one is DERIVED, from the same AST read
        `test_command_coverage` already does rather than from a second list here.
        """
        served = _commands_the_console_invokes()
        assert served, "found no `answer(...)` calls — the parser is broken, not the code"
        unmodelled = sorted(served - set(MODELS))
        assert not unmodelled, (
            f"these commands are served but have no model: {unmodelled}. An "
            "unmodelled endpoint is one the generated TypeScript cannot describe, "
            f"so the console would read it as `unknown`. Add it to MODELS in {__file__}."
        )

    def test_a_typed_payload_carries_only_what_it_declares(self):
        """What replaced `extra="allow"`, and why it is no longer needed.

        Leniency existed because `contract.py` RESTATED each shape and had to
        survive the command adding a field it had not heard about. A model the
        command CONSTRUCTS cannot be behind, so there is nothing to tolerate: an
        undeclared key is not part of the contract and does not survive.

        The stronger check is a static one. `ty` rejects
        `PoolPayload(a_field_added_next_week=1)` at pre-commit -- which is the
        drift this whole exercise measured, and the reason this test no longer
        has to be the thing standing between a rename and the console.
        """
        widened = {**PAYLOADS["pool-status"].model_dump(), "a_field_added_next_week": 1}
        assert not hasattr(contract.Pool.model_validate(widened), "a_field_added_next_week")

    def test_a_removed_field_is_caught(self):
        """The guard, verified rather than assumed. Without this, a model whose
        fields were all optional would pass everything and pin nothing."""
        without = {k: v for k, v in PAYLOADS["pool-status"].model_dump().items() if k != "pool_id"}
        with pytest.raises(ValidationError):
            contract.Pool.model_validate(without)

    def test_a_retyped_field_is_caught(self):
        """The likelier drift: a field that stays but changes shape."""
        retyped = {**PAYLOADS["cost"].model_dump(), "task_hours": "seventeen"}
        with pytest.raises(ValidationError):
            contract.Cost.model_validate(retyped)


class TestTheViewEnvelopes:
    """A view's own shape, which no command owns and so nothing else pins."""

    def _part(self, op: str) -> dict:
        """One answered part. Dumped, because that is how `_compose` ships it."""
        payload = PAYLOADS[op]
        dump = getattr(payload, "model_dump", None)
        return {"payload": dump() if callable(dump) else payload, "error": None}

    def test_the_now_view_validates(self):
        view = contract.NowView.model_validate(
            {
                "op": "view-now",
                "at": "2026-08-12T09:00:00+02:00",
                "elapsed_seconds": 2.1,
                "parts": {
                    "pool": self._part("pool-status"),
                    "jobs": self._part("jobs"),
                    "tasks": self._part("tasks"),
                    "autoscale": self._part("autoscale-check"),
                    "cost": self._part("cost"),
                },
            }
        )
        assert view.parts.pool.payload is not None
        assert view.parts.pool.payload.pool_id

    def test_a_failed_part_validates_and_keeps_its_reason(self):
        """The property the fan-out exists for. A view whose model REQUIRED a
        payload would 500 exactly when one panel was unavailable -- turning the
        one failure the design survives into the one that blanks the screen."""
        view = contract.NowView.model_validate(
            {
                "op": "view-now",
                "at": "2026-08-12T09:00:00+02:00",
                "elapsed_seconds": 2.1,
                "parts": {
                    "pool": {"payload": None, "error": "az login has expired"},
                    "jobs": self._part("jobs"),
                    "tasks": self._part("tasks"),
                    "autoscale": self._part("autoscale-check"),
                    "cost": self._part("cost"),
                },
            }
        )
        assert view.parts.pool.payload is None
        assert view.parts.pool.error == "az login has expired"

    def test_the_run_view_carries_its_join(self):
        view = contract.RunView.model_validate(
            {
                "op": "view-run",
                "at": "2026-08-12T09:00:00+02:00",
                "elapsed_seconds": 3.4,
                "parts": {
                    "run": self._part("runinfo"),
                    "progress": self._part("progress"),
                    "curve": self._part("curve"),
                    "evals": self._part("ledger"),
                    # A `TasksSummary`, which is a DIFFERENT model from `Tasks`
                    # and carries no rows -- see `views._summarised`.
                    "tasks": {
                        "payload": {"op": "tasks", "reconciled": 0, "source_rows": 412},
                        "error": None,
                    },
                },
                "run_tasks": [row.model_dump() for row in PAYLOADS["tasks"].rows],
            }
        )
        assert view.parts.tasks.payload is not None
        assert view.parts.tasks.payload.source_rows == 412
        assert not hasattr(view.parts.tasks.payload, "rows"), (
            "a trimmed part must not offer rows — that is the whole point of the type"
        )
        assert view.run_tasks

    def test_the_experiment_view_validates(self):
        view = contract.ExperimentView.model_validate(
            {
                "op": "view-experiment",
                "at": "2026-08-12T09:00:00+02:00",
                "elapsed_seconds": 1.2,
                "parts": {
                    "report": self._part("report"),
                    "runs": self._part("runs"),
                },
                "arm_runs": [row.model_dump() for row in PAYLOADS["runs"].runs],
            }
        )
        assert view.parts.report.payload is not None
        assert view.arm_runs
