"""What an LBR evaluation is configured WITH, separable from what runs it.

Its own module for a reason that is not tidiness. ``LBRConfig`` is the settings
object every caller of the evaluator has to construct, and it also identifies a
KNOB TIER -- `ledger.tiers` derives the comparability key from the same config
the evaluation consumed, which is what stops two runs scored under different
settings being compared. So a module that only wants to know "which tier is
this" had to import the evaluator, and the evaluator imports scipy, numpy, tqdm
and the whole engine.

That single edge put ~1.7s of scientific-stack import in front of EVERY
`poker-solver` invocation, `jobs` and `--help` included: the ledger is reached
from `commands._base`, which every command imports. Nothing in this file needs
more than the standard library and one frozen model.

Deliberately not re-exported from :mod:`hunl_local_best_response`. An alias
there would let the expensive path come back silently, which is the whole thing
being prevented -- and `tests/interfaces/test_import_weight.py` fails if it does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.shared.config import ResolverConfig

# Off-tree bet sizes the LBR player may bet when first to put money in on a
# street. Deliberately NOT the blueprint's trained sizes; overbets in
# particular probe the action abstraction.
DEFAULT_OFF_TREE_POT_FRACTIONS: tuple[float, ...] = (0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0)


@dataclass
class LBRConfig:
    """Settings for the HUNL LBR evaluator."""

    num_hands: int = 1000
    equity_runouts: int = 12
    off_tree_pot_fractions: tuple[float, ...] = DEFAULT_OFF_TREE_POT_FRACTIONS
    # Add off-tree bet/raise sizes to the exploiter's menu. Rigorous: opponent
    # lookups go through a persistent on-tree shadow state (see module docs), so
    # off-tree amounts never leak into infoset keys. Off by default only for
    # comparability with recorded baselines — it changes the measured completion,
    # so never mix on/off numbers in one comparison.
    include_off_tree: bool = False
    seed: int | None = None
    # Parallel workers for hand evaluation. LBR is embarrassingly parallel over hands;
    # each hand is seeded independently so the result is identical for any worker count.
    num_workers: int = 1
    # Board runouts averaged at all-in showdown terminals (board incomplete). All-in
    # pots are the largest payoffs in the game, so valuing them on a single sampled
    # runout was the dominant remaining variance source; averaging is a pure
    # Rao-Blackwellization (same expectation, lower variance). When exactly one card
    # is missing the runout is enumerated instead.
    allin_runouts: int = 50
    # WHICH strategy the LBR player exploits on the realized path. "blueprint" is
    # the raw table (historical numbers); "deployed" routes the opponent's actual
    # decisions through the runtime subgame resolver — the system that really
    # plays. The exploiter's myopic scorer stays blueprint-backed either way
    # (selection-only: approximations there loosen the bound, never invalidate it).
    opponent: str = "blueprint"
    # Resolver settings for opponent="deployed". Must set max_iterations (wall-
    # clock budgets make the measured strategy machine-dependent). None with
    # opponent="deployed" raises at engine construction.
    resolver: ResolverConfig | None = None
    # HOW the exploiter selects its actions. "myopic" is the classic one-step
    # check/call-to-showdown arithmetic; "lookahead" scores candidates by a
    # depth-limited best-response walk against the blueprint policy (see
    # :mod:`lookahead_scorer`). Selection-only: any scorer keeps the bound valid.
    scorer: str = "myopic"
    # Opponent-response levels the lookahead expands (depth - 1 exploiter
    # re-decisions; depth 1 ~= branch-resolved myopic). Ignored under "myopic".
    lookahead_depth: int = 2
    # Myopic prefilter width: lookahead-rescore only the top-k myopic candidates
    # (the myopic argmax is always included). <= 0 rescores the whole menu.
    lookahead_top_k: int = 3
