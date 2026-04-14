"""Exact evaluation for SMALL games -- the harness that validates the method.

Nothing in here knows about HUNL. :mod:`game_tree` defines an engine-agnostic
extensive-form protocol over opaque hashable states, and the rest computes
exact answers over it by full traversal: a best response, exploitability, and a
vanilla CFR solver to drive a toy game to a known equilibrium.

**These are not production evaluators, and that is the distinction this
directory exists to make.** Full-HUNL exact BR is intractable, so the modules
one level up approximate it -- exactly on a board-restricted game
(``public_tree_br``) or from below (``lbr``). Flat alongside those, a file named
``best_response.py`` reads like a third way to score a blueprint; it is
tractable only at Kuhn/Leduc scale. What it is FOR is proving the expensive
evaluators right: ``tests/pipeline/evaluation/restricted_hunl.py`` validates the
vectorised public-tree engine against this one to 1e-9, which is the only
check that the fast path computes what it claims.
"""
