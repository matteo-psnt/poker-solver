"""The one cast in the solver hierarchy, and what keeps it honest.

``StaticTreeSolver.__init__`` passes its ``StaticArrayStorage`` to the base as
``cast("Storage", storage)``. That cast is a lie the type checker cannot see
through: ``StaticArrayStorage`` deliberately does not implement the key-addressed
``Storage`` ABC, because reintroducing ``InfoSetKey`` would restore the string
hashing the static design exists to remove.

It is safe for exactly one reason -- ``MCCFRSolver`` touches ``self.storage`` in
only two methods, and ``StaticTreeSolver`` overrides both. That reason was a
comment. Delete an override and the cast becomes a live ``AttributeError`` deep
in a training run rather than a type error at the seam, so it is pinned here.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from src.engine.solver.mccfr.solver import MCCFRSolver
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage


def _methods_touching_storage() -> set[str]:
    """Base-class methods whose body reads ``self.storage``."""
    tree = ast.parse(Path(inspect.getfile(MCCFRSolver)).read_text())
    found: set[str] = set()
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef) or class_node.name != "MCCFRSolver":
            continue
        for method in class_node.body:
            if not isinstance(method, ast.FunctionDef) or method.name == "__init__":
                continue
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "storage"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    found.add(method.name)
    return found


def test_every_base_method_reaching_storage_is_overridden():
    """The cast's whole safety argument, as an assertion."""
    reaching = _methods_touching_storage()
    assert reaching, "expected to find base methods using self.storage"

    not_overridden = {name for name in reaching if name not in StaticTreeSolver.__dict__}
    assert not not_overridden, (
        f"MCCFRSolver.{sorted(not_overridden)} read self.storage but StaticTreeSolver "
        "does not override them. StaticArrayStorage does not implement the Storage "
        "ABC, so the cast in StaticTreeSolver.__init__ would reach a missing method."
    )


def test_static_storage_really_does_not_satisfy_the_abc():
    """If this ever fails, the cast is no longer needed -- delete it."""
    missing = [
        name
        for name in ("get_or_create_infoset", "get_infoset", "checkpoint")
        if not hasattr(StaticArrayStorage, name)
    ]
    assert missing, (
        "StaticArrayStorage now implements the whole Storage surface; the "
        "cast in StaticTreeSolver.__init__ can go."
    )
