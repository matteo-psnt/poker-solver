"""
Exact range-vs-range equity engine.

Computes showdown equity against a uniform random opponent for every valid
hole-card combo on a board in a single pass, sharing all evaluation work:

- On each completed (5-card) board, every combo is evaluated exactly once.
  Win/tie counts against all disjoint opponent combos are then derived from
  sorted-strength counting with card-removal (blocker) corrections, instead
  of pairwise comparisons or Monte Carlo showdowns.
- River and turn equities are exact (all runouts enumerated).
- Flop equities are exact by default (1,176 runouts). ``max_runouts`` caps
  this with a deterministic per-board sample of runouts; each sampled runout
  still contributes an exact river range-vs-range calculation, so accuracy
  degrades far more gracefully than per-showdown sampling.
- Preflop requires ``max_runouts`` (enumerating C(50,5) runouts is
  intentionally unsupported).

Equity semantics match the classic definition: P(win) + 0.5 * P(tie) versus
one uniformly random opponent hand and a uniformly random runout.
"""

from __future__ import annotations

import itertools

import eval7
import numpy as np

from src.core.game.state import Card

# Opponent combos on a completed board once 5 board + 2 hero cards are removed: C(45, 2).
OPPONENTS_PER_FULL_BOARD = 990

_TABLE_CACHE_SIZE = 32


def _combo_key(hole_cards: tuple[Card, Card]) -> tuple[int, int]:
    a, b = hole_cards[0].mask, hole_cards[1].mask
    return (a, b) if a <= b else (b, a)


class BoardEquityTable:
    """Equities for every valid hole-card combo on one board."""

    def __init__(
        self,
        board: tuple[Card, ...],
        combos: list[tuple[Card, Card]],
        equities: np.ndarray,
        histograms: np.ndarray | None,
    ):
        self.board = board
        self.combos = combos
        self.equities = equities
        self.histograms = histograms
        self._index = {_combo_key(combo): i for i, combo in enumerate(combos)}

    def combo_index(self, hole_cards: tuple[Card, Card]) -> int:
        """Index of a combo into ``combos``/``equities``/``histograms``."""
        idx = self._index.get(_combo_key(hole_cards))
        if idx is None:
            board_str = " ".join(str(c) for c in self.board)
            raise KeyError(f"Combo {hole_cards} is not a valid hand on board [{board_str}]")
        return idx

    def equity(self, hole_cards: tuple[Card, Card]) -> float:
        """Equity of a specific combo vs a uniform random opponent."""
        return float(self.equities[self.combo_index(hole_cards)])

    def histogram(self, hole_cards: tuple[Card, Card]) -> np.ndarray:
        """Equity-realization histogram of a specific combo (rows sum to 1)."""
        if self.histograms is None:
            raise ValueError("Table was computed without histograms (pass histogram_bins)")
        return self.histograms[self.combo_index(hole_cards)]

    def __len__(self) -> int:
        return len(self.combos)

    def __str__(self) -> str:
        board_str = " ".join(str(c) for c in self.board)
        return f"BoardEquityTable(board=[{board_str}], combos={len(self.combos)})"


class RangeEquityEngine:
    """
    Batch equity computation for all hole-card combos on a board.

    Args:
        max_runouts: Cap on enumerated runouts per board (None = exact
            enumeration). Only binds on the flop (1,176 runouts) and preflop,
            where it is required; turn (48) and river (1) stay exact unless
            the cap is set below their enumeration size.
        seed: Seed for runout sampling. Sampling is deterministic per board
            (independent of call order).
    """

    def __init__(self, max_runouts: int | None = None, seed: int = 42):
        self.max_runouts = max_runouts
        self.seed = seed
        self._full_deck = Card.get_full_deck()
        self._table_cache: dict[tuple, BoardEquityTable] = {}

    def board_equities(
        self, board: tuple[Card, ...], histogram_bins: int | None = None
    ) -> BoardEquityTable:
        """
        Compute equities for every valid combo on ``board``.

        Args:
            board: 0 (preflop), 3, 4, or 5 community cards.
            histogram_bins: If set, also accumulate a per-combo histogram of
                exact river equities across runouts (the equity-realization
                distribution used for potential-aware bucketing).

        Returns:
            BoardEquityTable with combos in deterministic deck order.
        """
        board = tuple(board)
        if len(board) not in (0, 3, 4, 5):
            raise ValueError(f"Board must have 0, 3, 4, or 5 cards, got {len(board)}")

        board_set = set(board)
        remaining = [c for c in self._full_deck if c not in board_set]
        n_rem = len(remaining)

        combo_pairs = list(itertools.combinations(range(n_rem), 2))
        n_combos = len(combo_pairs)

        # Per-combo eval7 card objects (evaluation hot path works on these).
        rem_e7 = [c.to_eval7() for c in remaining]
        board_e7 = [c.to_eval7() for c in board]
        combo_a_e7 = [rem_e7[a] for a, _ in combo_pairs]
        combo_b_e7 = [rem_e7[b] for _, b in combo_pairs]

        # combos_containing[c] = indices of combos that use remaining-card c.
        containing: list[list[int]] = [[] for _ in range(n_rem)]
        for i, (a, b) in enumerate(combo_pairs):
            containing[a].append(i)
            containing[b].append(i)
        combos_containing = [np.asarray(lst, dtype=np.int64) for lst in containing]

        runouts = self._enumerate_runouts(board, n_rem)

        wins_acc = np.zeros(n_combos, dtype=np.int64)
        ties_acc = np.zeros(n_combos, dtype=np.int64)
        opp_acc = np.zeros(n_combos, dtype=np.int64)
        hist = (
            np.zeros((n_combos, histogram_bins), dtype=np.int64)
            if histogram_bins is not None
            else None
        )

        evaluate = eval7.evaluate
        pos_of = np.empty(n_combos, dtype=np.int64)

        for runout in runouts:
            # Combos sharing a card with the runout do not exist on this river.
            if runout:
                valid_mask = np.ones(n_combos, dtype=bool)
                for r in runout:
                    valid_mask[combos_containing[r]] = False
                valid_idx = np.nonzero(valid_mask)[0]
            else:
                valid_idx = np.arange(n_combos)
            n_valid = valid_idx.size

            # Evaluate every valid combo once on the completed board.
            cards7 = board_e7 + [rem_e7[r] for r in runout] + [None, None]
            strengths = np.empty(n_valid, dtype=np.int64)
            for pos, i in enumerate(valid_idx):
                cards7[5] = combo_a_e7[i]
                cards7[6] = combo_b_e7[i]
                strengths[pos] = evaluate(cards7)

            # Global counts: combos strictly weaker / equal (higher eval7 value wins).
            sorted_strengths = np.sort(strengths, kind="stable")
            below = np.searchsorted(sorted_strengths, strengths, side="left")
            equal = np.searchsorted(sorted_strengths, strengths, side="right") - below

            wins = below.copy()
            ties = equal - 1  # exclude the combo itself
            # n_valid + 1 - deg(c1) - deg(c2), degrees subtracted in the loop below.
            opp = np.full(n_valid, n_valid + 1, dtype=np.int64)

            # Blocker corrections: remove opponent combos sharing a card with
            # the hero combo. A combo shares exactly one card with each
            # conflicting opponent combo (both cards only with itself), so
            # per-card subtraction is exact inclusion-exclusion.
            pos_of[valid_idx] = np.arange(n_valid)
            for r in runout:
                pos_of[combos_containing[r]] = -1
            for members_full in combos_containing:
                members = pos_of[members_full]
                members = members[members >= 0]
                if members.size == 0:
                    continue
                member_strengths = strengths[members]
                sorted_member = np.sort(member_strengths)
                lo = np.searchsorted(sorted_member, member_strengths, side="left")
                hi = np.searchsorted(sorted_member, member_strengths, side="right")
                wins[members] -= lo
                ties[members] -= hi - lo - 1
                opp[members] -= members.size

            wins_acc[valid_idx] += wins
            ties_acc[valid_idx] += ties
            opp_acc[valid_idx] += opp

            if hist is not None:
                n_bins = hist.shape[1]
                river_equity = (wins + 0.5 * ties) / opp
                bin_idx = np.minimum((river_equity * n_bins).astype(np.int64), n_bins - 1)
                hist[valid_idx, bin_idx] += 1

        equities = np.full(n_combos, np.nan)
        covered = opp_acc > 0
        equities[covered] = (wins_acc[covered] + 0.5 * ties_acc[covered]) / opp_acc[covered]

        histograms = None
        if hist is not None:
            runout_counts = hist.sum(axis=1, keepdims=True)
            histograms = np.divide(
                hist,
                runout_counts,
                out=np.zeros_like(hist, dtype=np.float64),
                where=runout_counts > 0,
            )

        combos = [(remaining[a], remaining[b]) for a, b in combo_pairs]
        return BoardEquityTable(board, combos, equities, histograms)

    def hand_equity(self, hole_cards: tuple[Card, Card], board: tuple[Card, ...]) -> float:
        """
        Equity of a single combo (convenience wrapper).

        Computes (and caches) the full board table, so repeated lookups on the
        same board are free after the first call.
        """
        board = tuple(board)
        cache_key = tuple(sorted(c.mask for c in board))
        table = self._table_cache.get(cache_key)
        if table is None:
            table = self.board_equities(board)
            if len(self._table_cache) >= _TABLE_CACHE_SIZE:
                self._table_cache.pop(next(iter(self._table_cache)))
            self._table_cache[cache_key] = table
        return table.equity(hole_cards)

    def _enumerate_runouts(self, board: tuple[Card, ...], n_rem: int) -> list[tuple[int, ...]]:
        """Enumerate (or deterministically sample) runouts as remaining-card indices."""
        cards_needed = 5 - len(board)

        if cards_needed == 0:
            return [()]

        if cards_needed > 2:  # preflop: C(50,5) is too large to enumerate
            if self.max_runouts is None:
                raise ValueError(
                    "Preflop equity requires max_runouts (exact enumeration of "
                    "5-card runouts is not supported)"
                )
            rng = self._board_rng(board)
            return [
                tuple(rng.choice(n_rem, size=cards_needed, replace=False))
                for _ in range(self.max_runouts)
            ]

        all_runouts = list(itertools.combinations(range(n_rem), cards_needed))
        if self.max_runouts is None or self.max_runouts >= len(all_runouts):
            return all_runouts

        rng = self._board_rng(board)
        chosen = rng.choice(len(all_runouts), size=self.max_runouts, replace=False)
        return [all_runouts[i] for i in chosen]

    def _board_rng(self, board: tuple[Card, ...]) -> np.random.Generator:
        """Per-board RNG so sampled runouts don't depend on call order."""
        return np.random.default_rng([self.seed, *sorted(c.mask for c in board)])

    def __str__(self) -> str:
        runouts = "exact" if self.max_runouts is None else str(self.max_runouts)
        return f"RangeEquityEngine(max_runouts={runouts})"
