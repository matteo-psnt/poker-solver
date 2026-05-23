"""Vector-form CFR: one traversal updates every infoset at every node.

The scalar kernel in ``mccfr/`` samples one deal per iteration and reaches a
measured 42.7 infoset visits out of ~32M. This package propagates a whole
*range* through the public betting tree instead, so a single pass touches every
infoset at every node it reaches. See ``compiled_tree`` for the structure the
traversal needs and ``kernel`` for the traversal itself.
"""

from src.engine.solver.vector.compiled_tree import CompiledTree, TerminalKind, compile_tree

# The kernels a sweep can measure. Defined here, in the layer they live in,
# because both the CLI command and the cloud leg validator need the list and a
# second copy is how one of them comes to reject a kernel the other accepts.
BOARD_FREE = "board-free"
HAND_SPACE = "hand-space"
SCALAR = "scalar"
KERNELS: tuple[str, ...] = (BOARD_FREE, HAND_SPACE, SCALAR)

__all__ = (
    "BOARD_FREE",
    "HAND_SPACE",
    "KERNELS",
    "SCALAR",
    "CompiledTree",
    "TerminalKind",
    "compile_tree",
)
