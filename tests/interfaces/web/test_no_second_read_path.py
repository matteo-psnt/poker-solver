"""The rule the console is subordinate to, checked rather than trusted.

The previous browser UI was deleted in `fbcf9a8` (−4,776 lines) because it
carried its own data layer: `api/chart_service.py`, `api/play_service.py` and
`interfaces/chart/data.py` were a second way to ask questions the CLI already
answered. They drifted from it, then rotted, and nothing failed until someone
looked.

Nothing here prevents that by good intentions. Every endpoint body must be a
`Command.invoke`.

The other half of the rule -- that the package cannot reach past the command
layer at all -- is the `web_reads_through_the_command_layer` contract in
`.importlinter`, not an AST walk here. It is the same property, declared once in
the tool the repo already runs over every module, which sees import forms a walk
of this package would not.
"""

from __future__ import annotations

import ast
from pathlib import Path

WEB = Path(__file__).resolve().parents[3] / "src" / "interfaces" / "web"


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
