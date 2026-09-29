"""Serving several DEPTHS from one reader, the way the chipzen box holds them.

The box already stages every rung for the seat, so the reader answering for all of
them costs the load rather than another copy of the deepest run -- measured 09-29 at
0.77 GB for one 100 bb blueprint under `play_only`, against 1.92 GB before.

Two depths is enough to catch the failure that matters: answering from the WRONG
blueprint while labelling it with the requested stack.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.interfaces.blueprint.app import create_app
from tests.test_helpers import build_trained_test_solver

DEEP = 400
SHALLOW = 200


@pytest.fixture(scope="module")
def ladder():
    return {
        DEEP: build_trained_test_solver(iterations=40, starting_stack=DEEP),
        SHALLOW: build_trained_test_solver(iterations=40, starting_stack=SHALLOW),
    }


@pytest.fixture(scope="module")
def client(ladder):
    return TestClient(
        create_app(
            lambda: ladder,
            run_id="deep-run",
            run_ids={DEEP: "deep-run", SHALLOW: "shallow-run"},
        )
    )


class TestWhatTheServerSaysItHolds:
    def test_run_lists_every_depth_deepest_first(self, client):
        body = client.get("/api/run").json()
        assert body["depths"] == [DEEP, SHALLOW], "deepest first: it is the default"
        assert body["starting_stack"] == DEEP

    def test_each_depth_names_the_run_it_came_from(self, client):
        body = client.get("/api/run").json()
        assert body["runs_by_depth"] == {str(DEEP): "deep-run", str(SHALLOW): "shallow-run"}


class TestSelectingADepth:
    def test_omitting_the_stack_reads_the_deepest(self, client):
        assert client.get("/api/node", params={"path": ""}).status_code == 200

    def test_each_held_stack_answers(self, client):
        for stack in (DEEP, SHALLOW):
            assert client.get("/api/node", params={"stack": stack}).status_code == 200

    def test_the_grids_actually_differ_between_depths(self, client):
        """The bug this pins: `strategy_grid` read the default blueprint.

        Selecting a stack then answering from the deepest would label a 400-chip
        range as a 200-chip one, and nothing downstream could tell. Two depths of
        the same tree trained identically still differ, because the betting tree
        itself differs -- a shallower stack caps the raises.
        """
        deep = client.get("/api/node", params={"stack": DEEP}).json()
        shallow = client.get("/api/node", params={"stack": SHALLOW}).json()
        assert deep["grid"] is not None
        assert shallow["grid"] is not None
        assert deep["grid"] != shallow["grid"], (
            "a depth selector that returns the same grid for every depth is worse "
            "than no selector: it is a wrong label on a real range"
        )

    def test_a_stack_the_server_does_not_hold_is_refused_by_name(self, client):
        response = client.get("/api/node", params={"stack": 12345})
        assert response.status_code == 422
        # The refusal NAMES what is available, because the operator's next move is
        # to pick one and silence would make them guess.
        assert "12345" in response.json()["error"]
        assert str(DEEP) in response.json()["error"]


class TestASingleBlueprintStillWorks:
    """A one-run server must report its one stack, not an empty list."""

    def test_one_blueprint_reports_one_depth(self):
        solver = build_trained_test_solver(iterations=40, starting_stack=DEEP)
        client = TestClient(create_app(lambda: solver, run_id="solo"))
        body = client.get("/api/run").json()
        assert body["depths"] == [DEEP]
        assert body["runs_by_depth"] == {str(DEEP): "solo"}


class TestTheSeatsRungSpellingIsAccepted:
    """`blueprint-serve --rung` must take the SEAT's `run:at:threshold` form.

    The deploy hands both units one list so they cannot drift about what is
    fielded, and every fielded rung carries a threshold (`...:12000:0.20`).
    Splitting on the first colon reads "12000:0.20" as the iteration and refuses
    all seven.
    """

    @pytest.mark.parametrize("spec", ["a-run", "a-run:12000", "a-run:12000:0.20"])
    def test_all_three_field_counts_parse(self, spec, tmp_path):
        from src.interfaces.commands.blueprint_serve import COMMAND, BlueprintServePayload

        (tmp_path / "a-run").mkdir()
        (tmp_path / "deep").mkdir()
        payload = BlueprintServePayload.model_validate(
            COMMAND.invoke(run="deep", runs_dir=str(tmp_path), rung=[spec]).model_dump()
        )
        assert len(payload.rungs) == 1
        assert payload.rungs[0].startswith(str(tmp_path / "a-run"))

    def test_a_non_numeric_iteration_is_still_refused(self, tmp_path):
        from src.interfaces.commands.blueprint_serve import COMMAND
        from src.interfaces.errors import CommandError

        (tmp_path / "a-run").mkdir()
        (tmp_path / "deep").mkdir()
        with pytest.raises(CommandError, match="not an iteration"):
            COMMAND.invoke(run="deep", runs_dir=str(tmp_path), rung=["a-run:latest"])
