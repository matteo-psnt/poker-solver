"""What this package is allowed to depend on, asserted rather than intended.

The node loads every module in ``src/shared/cloudtask`` before ``uv sync``, so
one import of something that pulls pydantic or PyYAML kills every Batch task at
bootstrap -- invisibly, because the task dies before it can write the record
that would explain it. ``src.shared.config`` is exactly that shape, sits one
directory up, and is the most natural thing in the world to reach for.

Derived and fail-CLOSED, which is the whole reason this is a test rather than an
``.importlinter`` forbidden contract: a list of what may not be imported is
correct only until the next module is added to ``src/shared``, and the failure it
would then miss is the one that cannot be diagnosed from the outside.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pytest

from src.shared import repo

if TYPE_CHECKING:
    import pathlib

PACKAGE = repo.SRC / "shared" / "cloudtask"

# The only first-party modules the node may reach outside this package. Both are
# stdlib-only themselves -- that is the property being borrowed, so anything
# added here has to be checked for it by hand.
ALLOWED = {"src.shared.records", "src.shared.jsonio", "src.shared.cache"}


def _sources() -> list[pathlib.Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _imports(source: pathlib.Path) -> set[str]:
    """Every ``src.*`` name this file imports, however it spells the import."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.startswith("src.")}


def test_the_package_was_found():
    """A walk that finds nothing passes every check below, on nothing."""
    assert len(_sources()) >= 8


@pytest.mark.parametrize("source", _sources(), ids=lambda p: p.name)
def test_nothing_reaches_outside_the_allowed_set(source):
    outside = {
        name
        for name in _imports(source)
        if not name.startswith("src.shared.cloudtask")
        # `from src.shared import records` also offers `src.shared`, which is an
        # empty package init, not a dependency.
        and name != "src.shared"
        and not any(name == allowed or name.startswith(f"{allowed}.") for allowed in ALLOWED)
    }
    assert not outside, (
        f"{source.name} imports {sorted(outside)}, which the node may not have. "
        f"Allowed outside this package: {sorted(ALLOWED)}."
    )
