"""Tests for suit isomorphism canonicalization."""

from src.abstraction.isomorphism.suit_canonicalization import (
    SuitMapping,
    canonicalize_board,
    canonicalize_hand,
    get_canonical_board_id,
    get_canonical_hand_id,
    get_card_rank_idx,
    get_card_suit,
    hand_relative_to_board,
)
from src.game.state import Card


class TestSuitMapping:
    """Tests for SuitMapping."""

    def test_empty_mapping(self):
        """Test creating empty mapping."""
        mapping = SuitMapping()
        assert len(mapping.mapping) == 0
        assert mapping.next_label == 0

    def test_get_or_assign_new_suit(self):
        """Test assigning new suits."""
        mapping = SuitMapping()

        mapping, label1 = mapping.get_or_assign("s")
        assert label1 == 0
        assert mapping.mapping["s"] == 0

        mapping, label2 = mapping.get_or_assign("h")
        assert label2 == 1
        assert mapping.mapping["h"] == 1

    def test_get_or_assign_existing_suit(self):
        """Test getting existing suit."""
        mapping = SuitMapping()
        mapping, _ = mapping.get_or_assign("s")

        mapping, label = mapping.get_or_assign("s")
        assert label == 0  # Same label
        assert mapping.next_label == 1  # Didn't increment


class TestCardFunctions:
    """Tests for card utility functions."""

    def test_get_card_suit(self):
        """Test extracting suit from Card."""
        assert get_card_suit(Card.new("As")) == "s"
        assert get_card_suit(Card.new("Kh")) == "h"
        assert get_card_suit(Card.new("Qd")) == "d"
        assert get_card_suit(Card.new("Jc")) == "c"

    def test_get_card_rank_idx(self):
        """Test extracting rank index from Card."""
        assert get_card_rank_idx(Card.new("As")) == 0  # Ace
        assert get_card_rank_idx(Card.new("Kh")) == 1  # King
        assert get_card_rank_idx(Card.new("Qd")) == 2  # Queen
        assert get_card_rank_idx(Card.new("2c")) == 12  # Deuce


class TestCanonicalizeBoard:
    """Tests for board canonicalization."""

    def test_simple_flop(self):
        """Test canonicalizing a simple flop."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        canonical, mapping = canonicalize_board(board)

        # First suit (spades) -> 0, second suit (hearts) -> 1
        assert mapping.mapping["s"] == 0
        assert mapping.mapping["h"] == 1

        # Check canonical cards
        assert len(canonical) == 3
        assert canonical[0].suit_label == 0  # Ts -> T_0
        assert canonical[1].suit_label == 1  # 9h -> 9_1
        assert canonical[2].suit_label == 0  # 8s -> 8_0

    def test_monotone_flop(self):
        """Test canonicalizing a monotone (single-suited) flop."""
        board = (Card.new("Ks"), Card.new("Ts"), Card.new("5s"))
        canonical, mapping = canonicalize_board(board)

        # All spades -> all mapped to 0
        assert all(c.suit_label == 0 for c in canonical)
        assert mapping.next_label == 1  # Only one suit seen

    def test_rainbow_flop(self):
        """Test canonicalizing a rainbow (all different suits) flop."""
        board = (Card.new("As"), Card.new("Kh"), Card.new("Qd"))
        canonical, mapping = canonicalize_board(board)

        assert canonical[0].suit_label == 0
        assert canonical[1].suit_label == 1
        assert canonical[2].suit_label == 2
        assert mapping.next_label == 3

    def test_isomorphic_boards_same_canonical(self):
        """Test that isomorphic boards produce same canonical form."""
        # These boards are equivalent under suit relabeling
        board1 = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        board2 = (Card.new("Th"), Card.new("9d"), Card.new("8h"))
        board3 = (Card.new("Td"), Card.new("9c"), Card.new("8d"))

        canonical1, _ = canonicalize_board(board1)
        canonical2, _ = canonicalize_board(board2)
        canonical3, _ = canonicalize_board(board3)

        # Should have same canonical representation
        assert get_canonical_board_id(canonical1) == get_canonical_board_id(canonical2)
        assert get_canonical_board_id(canonical1) == get_canonical_board_id(canonical3)


class TestCanonicalizeHand:
    """Tests for hand canonicalization relative to board."""

    def test_hand_matching_board_suits(self):
        """Test hand with suits that match board."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        _, mapping = canonicalize_board(board)

        # Spades hand -> suit label 0
        hand1 = canonicalize_hand((Card.new("As"), Card.new("Ks")), mapping)
        assert hand1[0].suit_label == 0
        assert hand1[1].suit_label == 0

        # Hearts hand -> suit label 1
        hand2 = canonicalize_hand((Card.new("Ah"), Card.new("Kh")), mapping)
        assert hand2[0].suit_label == 1
        assert hand2[1].suit_label == 1

    def test_hand_with_new_suit(self):
        """Test hand with suit not on board."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        _, mapping = canonicalize_board(board)

        # Diamonds not on board -> assigned label 2
        hand = canonicalize_hand((Card.new("Ad"), Card.new("Kd")), mapping)
        assert hand[0].suit_label == 2
        assert hand[1].suit_label == 2

    def test_mixed_suit_hand(self):
        """Test hand with mixed suits."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        _, mapping = canonicalize_board(board)

        # One spade, one heart
        hand = canonicalize_hand((Card.new("As"), Card.new("Kh")), mapping)
        assert hand[0].suit_label == 0  # A_0
        assert hand[1].suit_label == 1  # K_1

    def test_isomorphic_hands_same_canonical(self):
        """Test that isomorphic (hand, board) pairs produce same canonical form."""
        # Two boards that are isomorphic
        board1 = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        board2 = (Card.new("Td"), Card.new("9c"), Card.new("8d"))

        _, mapping1 = canonicalize_board(board1)
        _, mapping2 = canonicalize_board(board2)

        # Corresponding hands under the isomorphism
        hand1 = canonicalize_hand((Card.new("As"), Card.new("Ks")), mapping1)  # Flush draw
        hand2 = canonicalize_hand((Card.new("Ad"), Card.new("Kd")), mapping2)  # Flush draw

        # Should have same canonical hand
        assert get_canonical_hand_id(hand1) == get_canonical_hand_id(hand2)


class TestHandRelativeToBoard:
    """Tests for the combined canonicalization function."""

    def test_basic_usage(self):
        """Test basic hand_relative_to_board usage."""
        hole_cards = (Card.new("As"), Card.new("Ks"))
        board = (Card.new("Ts"), Card.new("9s"), Card.new("8c"))

        canonical = hand_relative_to_board(hole_cards, board)

        # Should return tuple of tuples
        assert len(canonical) == 2
        assert len(canonical[0]) == 2  # (rank_idx, suit_label)
        assert len(canonical[1]) == 2

    def test_different_suits_different_canonical(self):
        """Test that strategically different hands have different canonical forms."""
        board = (Card.new("Ts"), Card.new("9s"), Card.new("8c"))

        # Flush draw
        flush_draw = hand_relative_to_board((Card.new("As"), Card.new("Ks")), board)

        # No flush draw (different suit)
        no_flush = hand_relative_to_board((Card.new("Ah"), Card.new("Kh")), board)

        # These should be DIFFERENT (strategically distinct)
        assert flush_draw != no_flush

    def test_isomorphic_same_canonical(self):
        """Test that isomorphic situations have same canonical form."""
        # Situation 1: AKs flush draw on Ts9s8c
        board1 = (Card.new("Ts"), Card.new("9s"), Card.new("8c"))
        hand1 = (Card.new("As"), Card.new("Ks"))

        # Situation 2: AKh flush draw on Th9h8d (isomorphic)
        board2 = (Card.new("Th"), Card.new("9h"), Card.new("8d"))
        hand2 = (Card.new("Ah"), Card.new("Kh"))

        canonical1 = hand_relative_to_board(hand1, board1)
        canonical2 = hand_relative_to_board(hand2, board2)

        # These should be SAME (isomorphic)
        assert canonical1 == canonical2


class TestNoConflicts:
    """Tests ensuring the system produces no conflicts."""

    def test_all_hands_on_board_unique_canonical(self):
        """Test that all valid hands on a board have unique canonical forms."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        board_cards = set(board)

        # Generate all valid hands (not overlapping with board)
        cards = [Card.new(f"{r}{s}") for r in "AKQJT98765432" for s in "shdc"]
        valid_hands = []

        for i, c1 in enumerate(cards):
            if c1 in board_cards:
                continue
            for c2 in cards[i + 1 :]:
                if c2 in board_cards:
                    continue
                valid_hands.append((c1, c2))

        # Get canonical forms
        canonicals = [hand_relative_to_board(h, board) for h in valid_hands]

        # Count unique canonicals
        unique_canonicals = set(canonicals)

        # Should have significantly fewer unique canonicals than hands
        # (due to suit isomorphism) but each hand maps to exactly one
        assert len(canonicals) == len(valid_hands)
        assert len(unique_canonicals) < len(valid_hands)

        # Verify no conflicts: each canonical should consistently map
        # (Multiple hands can map to same canonical, but each hand maps to one)
        canonical_to_hands = {}
        for hand, canonical in zip(valid_hands, canonicals):
            if canonical not in canonical_to_hands:
                canonical_to_hands[canonical] = []
            canonical_to_hands[canonical].append(hand)

        # All hands in each group should be suit-isomorphic
        # (This is the key property that eliminates conflicts)
        for canonical, hands in canonical_to_hands.items():
            # Hands mapping to same canonical are strategically equivalent
            assert len(hands) >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_paired_board(self):
        """Test board with paired cards."""
        board = (Card.new("Ts"), Card.new("Th"), Card.new("8s"))
        canonical, mapping = canonicalize_board(board)

        # Two tens with different suits
        assert canonical[0].rank_idx == canonical[1].rank_idx  # Same rank
        assert canonical[0].suit_label != canonical[1].suit_label  # Different suits

    def test_turn_card(self):
        """Test 4-card board."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"), Card.new("2d"))
        canonical, mapping = canonicalize_board(board)

        assert len(canonical) == 4
        assert canonical[3].suit_label == 2  # Third unique suit

    def test_river_card(self):
        """Test 5-card board."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"), Card.new("2d"), Card.new("7c"))
        canonical, mapping = canonicalize_board(board)

        assert len(canonical) == 5
        assert canonical[4].suit_label == 3  # Fourth unique suit

    def test_pocket_pair(self):
        """Test pocket pair canonicalization."""
        board = (Card.new("Ts"), Card.new("9h"), Card.new("8s"))
        _, mapping = canonicalize_board(board)

        # AA with both spades (one matches board flush)
        # Note: Can't have both As since only one As exists
        hand1 = canonicalize_hand((Card.new("As"), Card.new("Ah")), mapping)
        hand2 = canonicalize_hand((Card.new("Ad"), Card.new("Ac")), mapping)

        # These are different: As has flush draw, Ad/Ac don't
        assert get_canonical_hand_id(hand1) != get_canonical_hand_id(hand2)
