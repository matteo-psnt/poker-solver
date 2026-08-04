"""Verify a published static ladder and mark the rungs that actually load.

Usage:  verify_ladder.py <training-config> <published-run-dir-on-the-share>

WHY. Completion markers guarded ``checkpoint-*|keys-*`` but not ``static-*``, so
every static rung was published unmarked -- and an unmarked rung is
indistinguishable from one interrupted mid-copy. The fetch therefore has to
refuse them all, which would strand an entire 30M-iteration ladder. Blanket
marking would reinstate exactly the bug markers exist to prevent, so each rung is
PROVEN instead: loaded end to end, and marked only if every chunk decompresses.

Reads the share IN PLACE rather than copying to the node. Verification has to
read every byte regardless, so a copy-then-read is pure duplicate I/O -- the
first version of this did that AND started a fresh interpreter per rung,
rebuilding the 57,604-node tree and reloading the 385 MB abstraction thirty
times. The tree, abstraction and storage are built once here and reused.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint
from src.pipeline.training.abstraction_resolver import ComboAbstractionResolver
from src.shared.config_loader import load_training_config

_RUNG = re.compile(r"static-(\d+)\.zarr")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    config_name, share = argv[1], Path(argv[2])
    config = load_training_config(config_name)

    action_model = ActionModel(config)
    abstraction = ComboAbstractionResolver().load(abstraction_config=config.card_abstraction.config)
    tree = build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        action_model,
        abstraction,
        starting_stack=config.game.starting_stack,
    )
    storage = StaticArrayStorage(tree)
    print(f"[repair] tree {len(tree):,} nodes, {tree.num_rows:,} rows", flush=True)

    rungs = sorted(
        int(match.group(1))
        for path in share.glob("static-*.zarr")
        if (match := _RUNG.fullmatch(path.name))
    )
    if not rungs:
        print("[repair] no static rungs found", flush=True)
        return 1

    good = bad = 0
    for iteration in rungs:
        try:
            # Loading reads every chunk, which IS the verification -- there is no
            # cheaper check that would still catch a truncated or torn copy.
            load_checkpoint(storage, share, at_iteration=iteration)
        except Exception as exc:  # noqa: BLE001 -- any load failure IS the corruption being reported
            print(
                f"[repair] {iteration:>12,}: CORRUPT -- {type(exc).__name__}: {exc}"[:180],
                flush=True,
            )
            bad += 1
            continue
        (share / f".complete-static-{iteration}.zarr").write_text("")
        print(f"[repair] {iteration:>12,}: OK, marked", flush=True)
        good += 1

    print(f"[repair] complete: {good} verified and marked, {bad} unusable", flush=True)
    # Non-zero only if NOTHING is usable: a ladder with some bad rungs is still a
    # ladder, and the log names exactly which points are missing.
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
