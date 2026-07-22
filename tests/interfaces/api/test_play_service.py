"""Tests for the play-against-the-blueprint service (sessions + redaction)."""

import pytest

from src.interfaces.api.play_service import PlayService
from tests.test_helpers import build_trained_test_solver


def _service(session_id: str) -> PlayService:
    blueprint = build_trained_test_solver(20, session_id=session_id)
    return PlayService(run_id="test-run", blueprint=blueprint)


def _play_out(service: PlayService, view: dict) -> dict:
    """Drive a hand to completion by always checking/calling."""
    while not view["isOver"]:
        legal = view["legalActions"]
        # Never fold: check/call if offered, else call the jam (a non-fold action).
        passive = next(
            (a for a in legal if a["kind"] in ("check", "call")),
            next((a for a in legal if a["kind"] != "fold"), legal[0]),
        )
        view = service.submit_action(view["sessionId"], passive["id"])
    return view


def test_new_hand_hides_bot_cards():
    service = _service("svc-hide")
    view = service.new_hand(human_seat=0, button=0)

    assert view["botHole"] is None
    assert len(view["yourHole"]) == 2
    assert view["legalActions"]  # human is to act preflop as the button


def test_showdown_reveals_and_fold_hides():
    service = _service("svc-reveal")
    saw_showdown = False
    saw_fold = False
    for _ in range(40):
        final = _play_out(service, service.new_hand())
        result = final["result"]
        if result["terminal"] == "showdown":
            saw_showdown = True
            assert final["botHole"] is not None
            assert len(final["botHole"]) == 2
        else:
            saw_fold = True
            assert final["botHole"] is None
        # Payoff sign matches the reported outcome.
        if result["outcome"] == "win":
            assert result["humanPayoff"] > 0
        elif result["outcome"] == "loss":
            assert result["humanPayoff"] < 0

    assert saw_showdown, "expected at least one showdown over 40 check/call hands"
    assert saw_fold, "expected at least one fold over 40 check/call hands"


def test_legal_action_serialization_is_amount_free_ids():
    service = _service("svc-actions")
    view = service.new_hand(human_seat=0, button=0)

    ids = [a["id"] for a in view["legalActions"]]
    assert ids == list(range(len(ids)))
    for action in view["legalActions"]:
        assert set(action) == {"id", "kind", "label", "chips"}
        assert action["chips"] >= 0


def test_invalid_action_id_rejected():
    service = _service("svc-invalid")
    view = service.new_hand(human_seat=0, button=0)
    with pytest.raises(ValueError):
        service.submit_action(view["sessionId"], 999)


def test_unknown_session_raises_keyerror():
    service = _service("svc-unknown")
    with pytest.raises(KeyError):
        service.get_state("nonexistent")


def test_session_eviction_bounds_registry():
    blueprint = build_trained_test_solver(5, session_id="svc-evict")
    service = PlayService(run_id="test-run", blueprint=blueprint, max_sessions=3)
    ids = [service.new_hand()["sessionId"] for _ in range(5)]
    # Only the last 3 survive; the first two were evicted.
    with pytest.raises(KeyError):
        service.get_state(ids[0])
    assert service.get_state(ids[-1])["sessionId"] == ids[-1]
