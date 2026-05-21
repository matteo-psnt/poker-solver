"""The blueprint server, driven through the same app a node would run.

The factory indirection is what makes this possible: a four-iteration solver
goes through the identical handlers a 30M-iteration one would, so the transport
is tested without a checkpoint, a card abstraction or a node.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.engine.search.range_inference import NUM_COMBOS
from src.interfaces.blueprint.app import create_app, parse_board
from src.pipeline.blueprint.paths import PathError
from tests.test_helpers import build_trained_test_solver


@pytest.fixture(scope="module")
def client():
    solver = build_trained_test_solver(iterations=40)
    return TestClient(create_app(lambda: solver, run_id="test-run"))


class TestBoardParsing:
    @pytest.mark.parametrize("text", ["2c7d9h", "2c 7d 9h", "2c,7d,9h"])
    def test_the_usual_spellings_all_work(self, text):
        assert len(parse_board(text)) == 3

    def test_an_empty_board_is_preflop(self):
        assert parse_board("") == ()

    @pytest.mark.parametrize(
        ("bad", "because"),
        [("2c7", "whole number of cards"), ("2x", "is not a card"), ("2c2c", "repeats a card")],
    )
    def test_a_malformed_board_says_what_is_wrong(self, bad, because):
        with pytest.raises(PathError, match=because):
            parse_board(bad)


class TestTheRunEndpoint:
    def test_it_names_what_is_loaded(self, client):
        body = client.get("/api/run").json()

        assert body["run"] == "test-run"
        assert body["combos"] == NUM_COMBOS
        assert body["starting_stack"] > 0


class TestTheCombosEndpoint:
    def test_it_returns_the_canonical_order_once(self, client):
        combos = client.get("/api/combos").json()["combos"]

        assert len(combos) == NUM_COMBOS
        assert len(set(combos)) == NUM_COMBOS


class TestTheNodeEndpoint:
    def test_the_root_carries_a_grid_and_its_children(self, client):
        body = client.get("/api/node").json()

        assert body["terminal"] is False
        assert body["children"], "someone can act at the root"
        assert len(body["grid"]["combo_buckets"]) == NUM_COMBOS
        assert body["grid"]["actions"]

    def test_every_child_token_is_itself_walkable(self, client):
        """The tree the client is told about must be the tree it can navigate."""
        root = client.get("/api/node").json()

        for child in root["children"]:
            response = client.get("/api/node", params={"path": child["token"], "board": "2c7d9h"})
            assert response.status_code == 200, child["token"]

    def test_a_board_blocks_combos_and_says_how_many(self, client):
        root = client.get("/api/node").json()
        call = next(c for c in root["children"] if c["token"] == "c")
        body = client.get(
            "/api/node", params={"path": f"{call['token']}/x", "board": "2c7d9h"}
        ).json()

        assert body["board"] == ["2c", "7d", "9h"]
        assert body["grid"]["blocked"] > 0

    def test_bucket_keys_are_strings_and_cover_every_unblocked_combo(self, client):
        grid = client.get("/api/node").json()["grid"]

        for bucket in grid["combo_buckets"]:
            if bucket >= 0:
                assert str(bucket) in grid["buckets"]

    def test_an_untrained_bucket_carries_no_strategy(self, client):
        grid = client.get("/api/node").json()["grid"]

        for entry in grid["buckets"].values():
            assert (entry["strategy"] is None) == (not entry["trained"])

    def test_a_terminal_line_is_reported_rather_than_refused(self, client):
        body = client.get("/api/node", params={"path": "f"}).json()

        assert body["terminal"] is True
        assert body["grid"] is None
        assert body["children"] == []


class TestRefusalsSurviveTheWire:
    def test_an_impossible_action_is_a_422_with_a_sentence(self, client):
        response = client.get("/api/node", params={"path": "b999999"})

        assert response.status_code == 422
        assert "On offer" in response.json()["error"]

    def test_a_short_board_is_a_422(self, client):
        response = client.get("/api/node", params={"path": "c/x", "board": ""})

        assert response.status_code == 422
        assert "board cards" in response.json()["error"]

    def test_a_bad_card_is_a_422(self, client):
        response = client.get("/api/node", params={"board": "2x"})

        assert response.status_code == 422
