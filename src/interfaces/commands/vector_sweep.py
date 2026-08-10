"""The `vector-sweep` subcommand: score a vector-CFR kernel against iteration.

One arm of the board-free-vs-hand-space comparison — one abstraction, one
kernel, one derivation size — producing an exploitability-vs-iteration curve.
Both kernels are scored the same way, by the hand-space kernel's *exact* best
response over a held-out set of boards, so the two are directly comparable and
neither marks its own homework.

This exists so the comparison runs where every other long operation in this
project runs. It is the sort of work a node is for: the hand-space arm is tens
of minutes of pinned CPU, and a laptop measurably throttles under it — a
1,000-iteration run drifted 4.1 s to 9.7 s per iteration purely from heat, which
corrupts every wall-clock figure in the result while leaving the (deterministic)
exploitability numbers intact.

The derivation universe is streamed rather than held: a board's context carries
an ``(H, H)`` blocking matrix, so twenty thousand of them is 23 GB in a list.
Populating a 600-bucket river needs that many boards, so streaming is what makes
the fine-abstraction arm possible at all.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import FULL_DECK, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.vector import (
    BOARD_FREE,
    HAND_SPACE,
    KERNELS,
    SCALAR,
    bucket_game,
    compile_tree,
)
from src.engine.solver.vector.bucket_kernel import BucketVectorCFR
from src.engine.solver.vector.fixed_board_scalar import FixedBoardStaticSolver
from src.engine.solver.vector.mixture import BoardMixtureCFR
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.pipeline.abstraction.postflop.bucketer import DenseBucketer
from src.pipeline.abstraction.vector_universe import (
    build_hand_context,
    iter_universe,
    sample_boards,
)
from src.shared.config.loader import load_config

# Per kernel, because they differ by ~70x in cost per iteration: board-free is
# ~0.14 s and hand-space ~10 s on a 32-board scoring set. One shared list is how
# a sweep ends up asking the slow kernel for 18 hours of work under an 8 h
# timeout -- which it then loses entirely, having published nothing.
DEFAULT_CHECKPOINTS = {
    BOARD_FREE: "10,25,50,100,200,400,800,1600,6400",
    HAND_SPACE: "10,25,50,100,200,400",
    # The shipped scalar kernel touches ~90 infoset rows per iteration against
    # the vector kernels' whole table, so its axis is millions, not hundreds.
    # Comparing them by ITERATION would be meaningless; the comparison is by
    # wall-clock, which every arm reports.
    SCALAR: "20000,100000,400000,1600000,6400000",
}
# The scoring universe is materialised (the scorer walks it every time it is
# asked), so this is the one board count that costs resident memory.
DEFAULT_SCORE_BOARDS = 32


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run vector-sweep`."""
    parser.add_argument(
        "--abstraction",
        required=True,
        help="Abstraction directory name, e.g. buckets-F100T300R600-rexact-a1542e88.",
    )
    parser.add_argument(
        "--abstractions-dir",
        default="data/combo_abstraction",
        help="Where abstraction directories live.",
    )
    parser.add_argument(
        "--kernel",
        required=True,
        choices=list(KERNELS),
        help="board-free carries a range over buckets and covers every board at "
        "once; hand-space carries a range over hands and takes one pass per "
        "board; scalar is the shipped ES-MCCFR kernel, pinned to the same boards "
        "so the three solve one identical game.",
    )
    parser.add_argument(
        "--derive-boards",
        type=int,
        default=6000,
        help="Boards the board-free transition and terminal matrices average over. "
        "Ignored by the hand-space kernel, which needs no derivation.",
    )
    parser.add_argument(
        "--train-boards",
        type=int,
        default=0,
        help="Boards the hand-space and scalar kernels TRAIN on. 0 means train on "
        "the scoring boards, which grades those two kernels on their own training "
        "set -- in-sample, and not what 'how exploitable is this' asks. Board-free "
        "is unaffected: it always derives from --derive-boards.",
    )
    parser.add_argument(
        "--score-boards",
        type=int,
        default=DEFAULT_SCORE_BOARDS,
        help="Held-out boards the exact best response scores against.",
    )
    parser.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated iteration counts to score at. Default depends on "
        "the kernel, which is the point: they differ ~70x in cost per iteration.",
    )
    parser.add_argument(
        "--progress-file",
        default="",
        help="Write the result after EVERY checkpoint, not just at the end. A "
        "task killed by its timeout keeps whatever it had reached.",
    )
    parser.add_argument("--config", default="", help="Config stem supplying the action model.")
    parser.add_argument(
        "--stack",
        type=int,
        default=20,
        help="Starting stack in big blinds' chips. Smaller trees make the "
        "hand-space arm affordable; the comparison is between kernels on ONE tree.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Derivation universe seed.")
    parser.add_argument(
        "--score-seed",
        type=int,
        default=999,
        help="Held-out scoring universe seed. Must differ from --seed.",
    )


def _bucket_counts(abstraction: DenseBucketer) -> dict[Street, int]:
    return {
        street: abstraction.num_buckets(street)
        for street in (Street.FLOP, Street.TURN, Street.RIVER)
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Train one kernel, scoring it against the exact best response as it goes."""
    if args.seed == args.score_seed:
        raise CommandError("--seed and --score-seed must differ, or scoring is not held out.")

    path = Path(args.abstractions_dir) / args.abstraction
    if not path.is_dir():
        raise CommandError(f"No such abstraction: {path}")
    abstraction = DenseBucketer.load(path)
    counts = _bucket_counts(abstraction)

    config = load_config(f"config/training/{args.config}.yaml") if args.config else load_config()
    if config.game.starting_stack != args.stack:
        # The tree below is enumerated for --stack. Anything that deals from the
        # config instead lands on states the tree has no node for, which showed
        # up as `Illegal action ALL_IN` deep in a traversal rather than here.
        config = config.model_copy(
            update={"game": config.game.model_copy(update={"starting_stack": args.stack})}
        )
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = BettingTree(
        rules, ActionModel(config), starting_stack=args.stack, buckets_per_street=counts
    )
    compiled = compile_tree(tree, rules)

    boards = sample_boards(np.random.default_rng(args.score_seed), args.score_boards)
    held_out = [build_hand_context(board, abstraction) for board in boards]
    pairs = float(np.mean([(~c.blocks).sum() for c in held_out]))
    initial = np.ones(held_out[0].num_hands, dtype=np.float32)
    scorer = BoardMixtureCFR(compiled, held_out, cfr_plus=True, boards=boards)

    # Where hand-space and scalar TRAIN. Drawn from --seed, the same stream
    # board-free derives from, so no kernel sees a scoring board while training.
    # The strategy table is indexed by (node, bucket) and carries no board axis,
    # which is what lets one trained here be scored over there at all.
    if args.train_boards:
        train_boards = sample_boards(np.random.default_rng(args.seed), args.train_boards)
        train_contexts = [build_hand_context(board, abstraction) for board in train_boards]
        train_initial = np.ones(train_contexts[0].num_hands, dtype=np.float32)
    else:
        train_boards, train_contexts, train_initial = boards, held_out, initial
    baseline = (
        float(scorer.exploitability(initial, pairs)),
        float(scorer.exploitability(initial, pairs, unconstrained=True)),
    )

    requested = args.checkpoints or DEFAULT_CHECKPOINTS[args.kernel]
    checkpoints = [int(part) for part in requested.split(",") if part.strip()]
    points: list[dict[str, Any]] = []
    started = time.perf_counter()

    def checkpoint_reached(derive_seconds: float | None) -> None:
        """Persist what is known so far, so a kill is not a total loss."""
        if not args.progress_file:
            return
        Path(args.progress_file).write_text(
            json.dumps(
                _payload(
                    args, counts, compiled, baseline, points, derive_seconds, len(checkpoints)
                ),
                indent=2,
            )
        )

    if args.kernel == SCALAR:
        # The shipped kernel, restricted to the same runouts. Its storage is the
        # same flat (node, bucket, action) table the vector kernels write, so the
        # very same exact best response scores it.
        storage = StaticArrayStorage(compiled.tree)
        try:
            solver = FixedBoardStaticSolver(
                ActionModel(config),
                abstraction,
                storage,
                config,
                tree=compiled.tree,
                runouts=[tuple(FULL_DECK[int(c)] for c in board) for board in train_boards],
            )
            done = 0
            for target in checkpoints:
                while done < target:
                    solver.train_iteration()
                    done += 1
                scorer.strategy_sum[:] = storage.strategy_sum
                points.append(
                    {
                        "iterations": done,
                        "train_seconds": round(time.perf_counter() - started, 1),
                        "exploitability": round(float(scorer.exploitability(initial, pairs)), 6),
                        "unconstrained": round(
                            float(scorer.exploitability(initial, pairs, unconstrained=True)), 6
                        ),
                    }
                )
                checkpoint_reached(None)
        finally:
            storage.close()
    elif args.kernel == HAND_SPACE:
        solver = BoardMixtureCFR(compiled, train_contexts, cfr_plus=True, boards=train_boards)
        done = 0
        for target in checkpoints:
            while done < target:
                solver.iterate(train_initial)
                done += 1
            scorer.strategy_sum[:] = solver.strategy_sum
            points.append(
                {
                    "iterations": done,
                    "train_seconds": round(time.perf_counter() - started, 1),
                    # Scored through `scorer`, never `solver`: one carries the
                    # scoring boards and the other the training boards, and with
                    # --train-boards those are disjoint. Scoring on `solver`
                    # grades the kernel on its own training set.
                    "exploitability": round(float(scorer.exploitability(initial, pairs)), 6),
                    # Against an opponent who sees its own cards rather than only
                    # its bucket -- always the larger of the two.
                    "unconstrained": round(
                        float(scorer.exploitability(initial, pairs, unconstrained=True)), 6
                    ),
                }
            )
            checkpoint_reached(None)
    else:
        mass = np.zeros(compiled.tree.num_buckets(Street.PREFLOP))

        def universe():
            nonlocal mass
            for context in iter_universe(
                abstraction, args.derive_boards, rng=np.random.default_rng(args.seed)
            ):
                mass = mass + np.bincount(
                    context.buckets_for(Street.PREFLOP), minlength=mass.shape[0]
                )
                yield context

        game = bucket_game.derive(universe(), {Street.PREFLOP: mass.shape[0], **counts})
        mass = mass / args.derive_boards
        derive_seconds = round(time.perf_counter() - started, 1)

        solver = BucketVectorCFR(compiled, game, cfr_plus=True)
        trained, spent = 0, 0.0
        for target in checkpoints:
            block = time.perf_counter()
            while trained < target:
                solver.iterate(mass)
                trained += 1
            spent += time.perf_counter() - block
            scorer.strategy_sum[:] = solver.strategy_sum
            points.append(
                {
                    "iterations": trained,
                    "train_seconds": round(spent, 1),
                    "exploitability": round(float(scorer.exploitability(initial, pairs)), 6),
                    "unconstrained": round(
                        float(scorer.exploitability(initial, pairs, unconstrained=True)), 6
                    ),
                }
            )
            checkpoint_reached(derive_seconds)
        return _payload(args, counts, compiled, baseline, points, derive_seconds, len(checkpoints))

    return _payload(args, counts, compiled, baseline, points, None, len(checkpoints))


def _payload(
    args, counts, compiled, baseline, points, derive_seconds, total_checkpoints=0
) -> dict[str, Any]:
    best = min(points, key=lambda p: p["exploitability"]) if points else None
    return {
        "op": "vector-sweep",
        # Read back by the node's progress sampler: a curve's honest unit is
        # checkpoints scored, and it cannot know the denominator otherwise.
        "done": len(points),
        "total": total_checkpoints,
        "abstraction": args.abstraction,
        "buckets": {street.name.lower(): count for street, count in counts.items()},
        "kernel": args.kernel,
        "derive_boards": args.derive_boards if args.kernel == BOARD_FREE else 0,
        "train_boards": args.train_boards or args.score_boards,
        "score_boards": args.score_boards,
        "in_sample": not args.train_boards,
        "stack": args.stack,
        "nodes": compiled.num_nodes,
        "infoset_rows": compiled.tree.num_rows,
        "derive_seconds": derive_seconds,
        "uniform_baseline": round(baseline[0], 6),
        "uniform_baseline_unconstrained": round(baseline[1], 6),
        "points": points,
        "best_exploitability": best["exploitability"] if best else None,
        "best_at_iterations": best["iterations"] if best else None,
    }


def render(payload: dict[str, Any]) -> None:
    buckets = payload["buckets"]
    print(
        f"vector-sweep {payload['kernel']} on {payload['abstraction']} "
        f"(F{buckets['flop']}/T{buckets['turn']}/R{buckets['river']}, "
        f"{payload['nodes']:,} nodes, {payload['infoset_rows']:,} rows)"
    )
    if payload["derive_seconds"] is not None:
        print(f"  derived from {payload['derive_boards']:,} boards in {payload['derive_seconds']}s")
    print(
        f"  uniform baseline {payload['uniform_baseline']:.4f} in-abstraction, "
        f"{payload['uniform_baseline_unconstrained']:.4f} per-hand"
    )
    print(f"  {'iters':>8} {'train s':>9} {'in-abs':>12} {'per-hand':>12}")
    for point in payload["points"]:
        print(
            f"  {point['iterations']:>8,} {point['train_seconds']:>9.1f} "
            f"{point['exploitability']:>12.5f} {point['unconstrained']:>12.5f}"
        )
    if payload["best_exploitability"] is not None:
        print(
            f"  best {payload['best_exploitability']:.5f} at "
            f"{payload['best_at_iterations']:,} iterations"
        )


COMMAND = Command(
    name="vector-sweep",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Exploitability vs iteration for one vector-CFR kernel on one abstraction.",
)
