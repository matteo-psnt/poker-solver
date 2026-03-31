"""The rule the console is subordinate to, checked rather than trusted.

The previous browser UI was deleted in `fbcf9a8` (−4,776 lines) because it
carried its own data layer: `api/chart_service.py`, `api/play_service.py` and
`interfaces/chart/data.py` were a second way to ask questions the CLI already
answered. They drifted from it, then rotted, and nothing failed until someone
looked.

Nothing here prevents that by good intentions. Every endpoint body must be a
`Command.invoke`, and the web package must not be able to reach past the command
layer at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

WEB = Path("src/interfaces/web")

FORBIDDEN_ROOTS = ("src.pipeline", "src.engine", "src.core")


def _modules() -> list[Path]:
    return sorted(WEB.rglob("*.py"))


def test_the_package_exists_and_is_scanned():
    """A glob that matches nothing would pass every assertion below."""
    assert _modules(), f"no modules under {WEB} — this guard is checking nothing"


@pytest.mark.parametrize("module", _modules(), ids=lambda p: p.name)
def test_it_cannot_reach_past_the_command_layer(module: Path):
    """No direct import of pipeline/engine/core.

    Those are where a second read path would come from: the moment this layer
    can call `eval_ledger` or `services` itself, it can answer a question
    differently from the command that owns it.
    """
    tree = ast.parse(module.read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    reaching = [name for name in imported if name.startswith(FORBIDDEN_ROOTS)]
    assert not reaching, (
        f"{module} imports {reaching}. The console reads through "
        "`Command.invoke` or not at all — that is what `fbcf9a8` deleted 4,776 "
        "lines for."
    )


def test_every_endpoint_answers_through_a_command():
    """Each route body must reach `answer(...)`, which is the only fetch site.

    Catches the plausible-looking regression: an endpoint that assembles a
    response itself because the shape it wanted was 'almost' what a command
    returns.
    """
    tree = ast.parse((WEB / "app.py").read_text())
    routes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr in {"get", "post"}
            for d in node.decorator_list
        )
    ]
    assert routes, "found no decorated routes — the parser is broken, not the code"

    offenders = []
    for route in routes:
        calls = {
            node.func.id
            for node in ast.walk(route)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        # `_unbuilt` and `_spa` serve the page, not data; they answer nothing.
        if "answer" not in calls and route.name not in {"_unbuilt", "_spa"}:
            offenders.append(route.name)

    assert not offenders, f"these endpoints do not go through `answer()`: {offenders}"


def test_endpoints_are_sync_so_they_do_not_block_the_event_loop():
    """`def`, never `async def`.

    Every Azure client in `src.interfaces.cloud` is synchronous. FastAPI runs a
    sync handler in a threadpool; an `async def` handler would hold the event
    loop for the whole 2-4s of a cloud read and serialise every other request
    behind it. It reads like a tidy-up, which is why it is pinned.
    """
    tree = ast.parse((WEB / "app.py").read_text())
    coroutines = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and any(isinstance(d, ast.Call) for d in node.decorator_list)
    ]
    assert not coroutines, (
        f"{coroutines} are `async def`. Sync handlers run in a threadpool; a "
        "coroutine blocks the loop for the whole cloud read."
    )
