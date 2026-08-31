"""The hand loop, and the arithmetic that turns hands into the published metric.

The score is the one number this whole surface exists to produce, and it is
derived from two of their fields by a formula nothing else checks. So the
identity is pinned against a REAL row off their public board rather than
against itself: Bitcrumbs' -138,938.32 AIVAT chips over 44,717 hands at a 100
blind is the -3.11 bb/100 they publish.
"""

from __future__ import annotations

import pytest

from src.interfaces.gtowizard import session
from src.interfaces.gtowizard.agents import AllIn, AlwaysFold, CheckCall
from src.interfaces.gtowizard.protocol import Frame

BB = 100
GAME = {
    "game_id": 1,
    "game_name": "HUNL 200BB",
    "game_format": "heads-up",
    "starting_stack": 20000,
    "blinds": [BB, 50],  # THEIR order: big first
    "stack_reset_per_hand": True,
}


def state(
    *,
    history: list[str],
    over: bool = False,
    legal: list[str],
    winnings: float | None = None,
    aivat: float | None = None,
) -> dict:
    return {
        "street": "preflop",
        "common_pot": 0,
        "total_pot": 150,
        "board_cards": "",
        "is_hand_over": over,
        "players": [
            {"name": "villain", "stack": 1, "position": "?", "hole_cards": None},
            {"name": "hero", "stack": 1, "position": "?", "hole_cards": "AhKd"},
        ],
        "legal_actions": legal,
        "raise_range": {"min": 200, "max": 20000},
        "action_history": history,
        "has_gto_wizard_folded": False,
        "winnings": winnings,
        "aivat_score": aivat,
    }


class FakeServer:
    """Answers one scripted hand: we act once, then it is over.

    Records every request so the loop's wire encoding is inspectable -- the
    thing a live probe checks, checked here for the shape rather than the value.
    """

    def __init__(self, *, winnings: float, aivat: float, fail_after: int | None = None) -> None:
        self.winnings = winnings
        self.aivat = aivat
        self.fail_after = fail_after
        self.hands = 0
        self.sent: list[tuple[str, int | None]] = []

    def new_hand(self, game_name: str = "") -> Frame:
        self.hands += 1
        if self.fail_after is not None and self.hands > self.fail_after:
            raise RuntimeError("engine gave up")
        return Frame.parse(
            {
                "hand_id": self.hands,
                "game": GAME,
                "game_state": state(history=[], legal=["f", "c", "b"]),
            }
        )

    def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
        self.sent.append((action, amount))
        return Frame.parse(
            {
                "hand_id": hand_id,
                "game": GAME,
                "game_state": state(
                    history=[action],
                    over=True,
                    legal=[],
                    winnings=self.winnings,
                    aivat=self.aivat,
                ),
            }
        )


class TestScore:
    def test_the_metric_reproduces_a_real_row_off_their_board(self) -> None:
        # Bitcrumbs, GTO Wizard AI v2, read 2026-08-31: -138,938.31867 AIVAT
        # chips over 44,717 hands at a 100 blind is the -3.11 they publish.
        tally = session.Tally()
        per_hand = -138_938.31867 / 44_717
        for index in range(44_717):
            tally.add(
                session.HandRecord(
                    hand_id=index, big_blind=BB, winnings=0.0, aivat=per_hand, decisions=1
                )
            )
        assert tally.aivat_bb_per_100 == pytest.approx(-3.107, abs=0.001)

    def test_the_raw_result_is_a_different_number_and_is_reported_apart(self) -> None:
        # Roman_SL is +0.51 raw and -9.76 adjusted. Quoting the wrong column is
        # the single easiest way to misreport this benchmark.
        tally = session.Tally()
        for index in range(100):
            tally.add(
                session.HandRecord(
                    hand_id=index, big_blind=BB, winnings=50.0, aivat=-900.0, decisions=1
                )
            )
        assert tally.raw_bb_per_100 == pytest.approx(50.0)
        assert tally.aivat_bb_per_100 == pytest.approx(-900.0)

    def test_one_hand_carries_no_error_bar_rather_than_a_flattering_zero(self) -> None:
        tally = session.Tally()
        tally.add(session.HandRecord(hand_id=1, big_blind=BB, winnings=0.0, aivat=0.0, decisions=1))
        assert tally.aivat_std_bb_per_100 == float("inf")


class TestLoop:
    def test_a_hand_is_played_to_its_end_and_scored(self) -> None:
        server = FakeServer(winnings=-100.0, aivat=-150.0)
        tally = session.run(server, CheckCall(), num_hands=3, concurrency=1)
        assert tally.played == 3
        assert tally.failed == 0
        assert tally.aivat_bb_per_100 == pytest.approx(-150.0)

    def test_check_call_sends_a_call_when_it_cannot_check(self) -> None:
        server = FakeServer(winnings=0.0, aivat=0.0)
        session.run(server, CheckCall(), num_hands=1, concurrency=1)
        assert server.sent == [("c", None)]

    def test_always_fold_folds(self) -> None:
        server = FakeServer(winnings=0.0, aivat=0.0)
        session.run(server, AlwaysFold(), num_hands=1, concurrency=1)
        assert server.sent == [("f", None)]

    def test_a_bet_carries_its_amount_and_nothing_else_does(self) -> None:
        server = FakeServer(winnings=0.0, aivat=0.0)
        session.run(server, AllIn(), num_hands=1, concurrency=1)
        assert server.sent == [("b", 20000)]

    def test_a_failed_hand_is_dropped_from_the_score_not_counted_as_a_loss(self) -> None:
        server = FakeServer(winnings=-100.0, aivat=-150.0, fail_after=2)
        tally = session.run(server, CheckCall(), num_hands=5, concurrency=1)
        assert tally.played == 2
        assert tally.failed == 3
        assert tally.aivat_bb_per_100 == pytest.approx(-150.0)

    def test_hands_run_concurrently_without_losing_any(self) -> None:
        server = FakeServer(winnings=-100.0, aivat=-150.0)
        tally = session.run(server, CheckCall(), num_hands=40, concurrency=8)
        assert tally.played == 40

    @pytest.mark.parametrize("concurrency", [0, session.MAX_CONCURRENT + 1])
    def test_their_concurrency_cap_is_refused_here_not_by_their_server(
        self, concurrency: int
    ) -> None:
        with pytest.raises(ValueError, match="Concurrency"):
            session.run(
                FakeServer(winnings=0.0, aivat=0.0),
                CheckCall(),
                num_hands=1,
                concurrency=concurrency,
            )

    def test_the_log_gets_one_object_per_hand_as_it_lands(self, tmp_path) -> None:
        path = tmp_path / "hands.jsonl"
        server = FakeServer(winnings=-100.0, aivat=-150.0)
        session.run(server, CheckCall(), num_hands=4, concurrency=2, log_path=path)
        assert len(path.read_text().strip().splitlines()) == 4


class TestBadFrames:
    def test_a_live_frame_offering_nothing_is_named_not_answered(self) -> None:
        """The failure `allows`' old permissive-empty default would have caused.

        Not over, and no legal action: every agent falls through to its last
        resort and sends something the server rejects, and a 4xx mid-hand
        abandons a hand that would otherwise have been scored.
        """

        class OffersNothing(FakeServer):
            def new_hand(self, game_name: str = "") -> Frame:
                self.hands += 1
                return Frame.parse(
                    {
                        "hand_id": self.hands,
                        "game": GAME,
                        "game_state": state(history=[], legal=[]),
                    }
                )

        server = OffersNothing(winnings=0.0, aivat=0.0)
        tally = session.run(server, CheckCall(), num_hands=1, concurrency=1)
        assert tally.failed == 1
        assert server.sent == []


class TestRunaway:
    def test_a_hand_that_never_ends_is_cut_off_rather_than_burning_the_run(self) -> None:
        class NeverEnds(FakeServer):
            def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
                super().act(hand_id, action, amount)
                return Frame.parse(
                    {
                        "hand_id": hand_id,
                        "game": GAME,
                        "game_state": state(history=[], legal=["f", "c", "b"]),
                    }
                )

        server = NeverEnds(winnings=0.0, aivat=0.0)
        tally = session.run(server, CheckCall(), num_hands=1, concurrency=1)
        assert tally.played == 0
        assert tally.failed == 1
        # The cap, plus the one fold that hands the slot back on the way out.
        assert len(server.sent) == session.MAX_DECISIONS_PER_HAND + 1
        assert server.sent[-1] == ("f", None)


class DyingServer:
    """Deals a hand, then fails on our first action.

    The shape of the leak: the hand EXISTS server-side, so the slot is taken,
    and the thing that ends the run happens after that.
    """

    def __init__(self, *, legal: list[str]) -> None:
        self.legal = legal
        self.acts: list[tuple[int, str]] = []
        self.dealt = False

    def new_hand(self, game_name: str = "") -> Frame:
        self.dealt = True
        return Frame.parse(
            {"hand_id": 7, "game": GAME, "game_state": state(history=[], legal=self.legal)}
        )

    def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
        self.acts.append((hand_id, action))
        raise RuntimeError("engine died mid-hand")


class TestSlotsComeBack:
    """MEASURED: a 600-hand run played 423 and lost 177 to their 20-hand cap,
    because eleven hands abandoned by earlier runs still held eleven slots. A
    hand that dies mid-play is not scored either way -- but it must not keep
    its slot, or every later run starts poorer than the last.
    """

    def test_a_hand_that_dies_mid_play_is_conceded(self) -> None:
        server = DyingServer(legal=["f", "c", "b"])
        with pytest.raises(RuntimeError):
            session.play_hand(server, CheckCall())
        # Two acts: the agent's call, then the fold that gives the slot back.
        assert server.acts == [(7, "c"), (7, "f")]

    def test_a_check_only_spot_is_conceded_by_checking(self) -> None:
        # Folding is not always legal, and an illegal action leaves the hand
        # exactly as open as doing nothing.
        server = DyingServer(legal=["k", "b"])
        with pytest.raises(RuntimeError):
            session.play_hand(server, CheckCall())
        assert server.acts == [(7, "k"), (7, "k")]

    def test_a_run_does_not_leak_a_slot_per_failed_hand(self) -> None:
        server = DyingServer(legal=["f", "c", "b"])
        tally = session.run(server, CheckCall(), num_hands=3, concurrency=1)
        assert tally.played == 0
        assert tally.failed == 3
        assert [action for _, action in server.acts].count("f") == 3


class FakeLobby:
    """Hands left open by somebody else's run, which is how they are found."""

    def __init__(
        self, *, open_ids: list[int], legal: list[str], refuse: set[int] | None = None
    ) -> None:
        self.open_ids = open_ids
        self.legal = legal
        self.refuse = refuse or set()
        self.folded: list[int] = []

    def in_progress(self, game_name: str = "") -> list[Frame]:
        return [
            Frame.parse(
                {
                    "hand_id": hand_id,
                    "game": GAME,
                    "game_state": state(history=[], legal=self.legal),
                }
            )
            for hand_id in self.open_ids
        ]

    def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
        if hand_id in self.refuse:
            raise RuntimeError("that hand is beyond saving")
        self.folded.append(hand_id)
        return Frame.parse(
            {
                "hand_id": hand_id,
                "game": GAME,
                "game_state": state(history=[action], over=True, legal=[], winnings=0.0, aivat=0.0),
            }
        )


class TestDrain:
    def test_every_open_hand_is_folded_and_reported(self) -> None:
        lobby = FakeLobby(open_ids=[11, 12, 13], legal=["f", "c", "b"])
        assert session.drain(lobby) == [11, 12, 13]
        assert lobby.folded == [11, 12, 13]

    def test_a_hand_that_will_not_close_is_not_reported_as_released(self) -> None:
        # Absence reads as success unless the count is of what ACTUALLY closed.
        lobby = FakeLobby(open_ids=[11, 12], legal=["f", "c", "b"], refuse={11})
        assert session.drain(lobby) == [12]


class CheckOnlyLobby:
    """A hand where folding is not offered, which needs more than one action.

    MEASURED: two of the eleven hands drained live sat in check-only spots. One
    check handed the turn back to their engine, which acted and asked again --
    so the hand was still open, and still holding its slot, after a drain that
    called it released.
    """

    def __init__(self, *, checks_to_end: int) -> None:
        self.checks_to_end = checks_to_end
        self.acts: list[str] = []

    def in_progress(self, game_name: str = "") -> list[Frame]:
        return [
            Frame.parse(
                {"hand_id": 9, "game": GAME, "game_state": state(history=[], legal=["k", "b"])}
            )
        ]

    def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
        self.acts.append(action)
        done = len(self.acts) >= self.checks_to_end
        return Frame.parse(
            {
                "hand_id": hand_id,
                "game": GAME,
                "game_state": state(
                    history=self.acts,
                    over=done,
                    legal=[] if done else ["k", "b"],
                    winnings=0.0 if done else None,
                    aivat=0.0 if done else None,
                ),
            }
        )


class TestConcedingAHandThatWillNotFold:
    def test_it_keeps_checking_until_the_hand_is_actually_over(self) -> None:
        lobby = CheckOnlyLobby(checks_to_end=4)
        assert session.drain(lobby) == [9]
        assert lobby.acts == ["k", "k", "k", "k"]

    def test_a_hand_still_open_at_the_cap_is_not_counted_as_released(self) -> None:
        # The failure this guards is a COUNT that outruns what it counted.
        lobby = CheckOnlyLobby(checks_to_end=10_000)
        assert session.drain(lobby) == []
        assert len(lobby.acts) == session.MAX_DECISIONS_PER_HAND
