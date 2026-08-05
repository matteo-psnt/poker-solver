"""Sitting down against the blueprint over HTTP.

Two properties carry real weight. The bot's cards must not cross the wire before
the hand ends -- a client that receives them can show them, and a sit-down where
you see the opponent's hand measures nothing. And an evicted session must be
distinguishable from a finished one, because "your hand is over" is a result and
"your hand is gone" is not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.interfaces.blueprint.app import create_app
from src.interfaces.blueprint.sessions import Sessions, UnknownSessionError
from tests.test_helpers import build_trained_test_solver


@pytest.fixture(scope="module")
def blueprint():
    return build_trained_test_solver(iterations=40)


@pytest.fixture(scope="module")
def client(blueprint):
    return TestClient(create_app(lambda: blueprint, run_id="test-run"))


def start(client, **body) -> dict:
    response = client.post("/api/play", json={"human_seat": 0, "seed": 7, **body})
    assert response.status_code == 200, response.text
    return response.json()


def play_out(client, hand: dict) -> dict:
    """Take the first legal action until the hand ends."""
    while not hand["over"]:
        assert hand["legal"], "not over, but nothing to do"
        response = client.post(
            f"/api/play/{hand['session']}/action", json={"token": hand["legal"][0]["token"]}
        )
        assert response.status_code == 200, response.text
        hand = response.json()
    return hand


class TestDealing:
    def test_a_new_hand_comes_back_playable(self, client):
        hand = start(client)

        assert hand["session"]
        assert len(hand["hole_cards"]) == 2
        assert hand["over"] or hand["legal"]

    def test_the_same_seed_deals_the_same_hand(self, client):
        assert start(client, seed=42)["hole_cards"] == start(client, seed=42)["hole_cards"]

    def test_the_button_alternates_when_not_pinned(self, client):
        buttons = {start(client, seed=None, button=None)["button"] for _ in range(4)}

        assert buttons == {0, 1}, "a fixed button shows only one side of every spot"

    def test_a_bad_seat_is_a_refusal_not_a_crash(self, client):
        response = client.post("/api/play", json={"human_seat": 3})

        assert response.status_code == 422


class TestTheBotsCardsAreHidden:
    def test_they_are_withheld_while_the_hand_is_live(self, client):
        hand = start(client)
        if hand["over"]:
            pytest.skip("this seed ended before the human acted")

        assert hand["bot_hole_cards"] is None

    def test_they_are_shown_once_it_is_over(self, client):
        hand = play_out(client, start(client))

        assert hand["bot_hole_cards"] is not None
        assert len(hand["bot_hole_cards"]) == 2

    def test_the_mix_is_withheld_until_the_end_too(self, client):
        """The bot's mix IS its strategy; mid-hand it would be a peek."""
        hand = start(client)
        if hand["over"]:
            pytest.skip("this seed ended before the human acted")

        assert all(event["mix"] is None for event in hand["log"])

        finished = play_out(client, hand)
        trained = [e for e in finished["log"] if e["actor"] == "bot" and not e["untrained"]]
        assert not trained or any(e["mix"] is not None for e in trained)


class TestPlaying:
    def test_a_hand_reaches_a_settled_terminal(self, client):
        hand = play_out(client, start(client))

        assert hand["over"] is True
        assert hand["payoff"] is not None
        assert hand["to_act"] is None
        assert hand["legal"] == []

    def test_an_illegal_move_is_refused_with_what_was_on_offer(self, client):
        hand = start(client)
        if hand["over"]:
            pytest.skip("this seed ended before the human acted")

        response = client.post(f"/api/play/{hand['session']}/action", json={"token": "b999999"})

        assert response.status_code == 422
        assert "On offer" in response.json()["error"]

    def test_acting_after_the_end_is_refused(self, client):
        hand = play_out(client, start(client))

        response = client.post(f"/api/play/{hand['session']}/action", json={"token": "f"})

        assert response.status_code == 422

    def test_the_untrained_count_is_reported_every_step(self, client):
        hand = play_out(client, start(client))

        assert hand["bot_untrained_decisions"] <= hand["bot_decisions"]
        assert hand["bot_untrained_decisions"] == sum(
            1 for e in hand["log"] if e["actor"] == "bot" and e["untrained"]
        )


class TestSessionLifetime:
    def test_a_hand_can_be_fetched_back(self, client):
        hand = start(client)

        again = client.get(f"/api/play/{hand['session']}").json()

        assert again["hole_cards"] == hand["hole_cards"]

    def test_an_unknown_session_is_a_404_that_says_so(self, client):
        response = client.get("/api/play/not-a-session")

        assert response.status_code == 404
        assert "no longer on the server" in response.json()["error"]

    def test_leaving_is_idempotent(self, client):
        hand = start(client)

        assert client.delete(f"/api/play/{hand['session']}").status_code == 200
        assert client.delete(f"/api/play/{hand['session']}").status_code == 200
        assert client.get(f"/api/play/{hand['session']}").status_code == 404


class TestTheStoreIsBounded:
    def test_the_oldest_is_dropped_rather_than_the_newest_refused(self, blueprint):
        sessions = Sessions(blueprint, limit=3)
        ids = [sessions.start(human_seat=0, seed=i)[0] for i in range(5)]

        assert len(sessions) == 3
        with pytest.raises(UnknownSessionError):
            sessions.get(ids[0])
        assert sessions.get(ids[-1]) is not None

    def test_touching_a_session_keeps_it_alive(self, blueprint):
        """Least-RECENTLY-USED, not oldest-created: an active tab must survive."""
        sessions = Sessions(blueprint, limit=2)
        first, _ = sessions.start(human_seat=0, seed=1)
        sessions.start(human_seat=0, seed=2)

        sessions.get(first)
        sessions.start(human_seat=0, seed=3)

        assert sessions.get(first) is not None
