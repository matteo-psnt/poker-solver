"""The rule the console is subordinate to, checked rather than trusted.

The previous browser UI was deleted in `fbcf9a8` (−4,776 lines) because it
carried its own data layer: `api/chart_service.py`, `api/play_service.py` and
`interfaces/chart/data.py` were a second way to ask questions the CLI already
answered. They drifted from it, then rotted, and nothing failed until someone
looked.

Nothing here prevents that by good intentions.

The rule used to be *every endpoint body is one `Command.invoke`*, which stopped
the disease by forbidding composition -- and so pushed the composing onto the
browser, where it became four requests per screen and a client that downloaded
the whole task log to find one run's rows. Composition was never the disease.
Deriving answers was. So the rule is now:

    **The web layer may COMPOSE command payloads. It may not COMPUTE one.**

which is checked as two things here: an endpoint gets its data from `answer` or
`view` and nowhere else, and `views.py` reaches the outside world only through
the command registry. A join in `views.py` may filter, group and
cross-reference; the moment it needs a quantity no command can answer, it needs
a command first.

The other half -- that the package cannot reach past the command layer at all --
is the `web_reads_through_the_command_layer` contract in `.importlinter`, not an
AST walk here. It is the same property, declared once in the tool the repo
already runs over every module, which sees import forms a walk of this package
would not.
"""

from __future__ import annotations

import ast

from src.shared import repo

WEB = repo.SRC / "interfaces" / "web"

# The command registry, the fan-out, and the stdlib. Not `cloud`, not `pipeline`,
# not `shared.records`: a view that imports a reader can answer a question
# itself, which is the failure being guarded against. `_compose` is the fan-out.
VIEW_IMPORTS = {
    "src.interfaces.commands",
    "src.interfaces.commands._compose",
}


def test_every_endpoint_answers_through_a_command():
    """Each route body must reach `answer(...)` or `view(...)`, and nothing else.

    Catches the plausible-looking regression: an endpoint that assembles a
    response itself because the shape it wanted was 'almost' what a command
    returns. `view` is admitted alongside `answer` because it is composition --
    several `Command.invoke`s and a join -- not a second way to ask.
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
        if not (calls & {"answer", "view"}) and route.name not in {"_unbuilt", "_spa"}:
            offenders.append(route.name)

    assert not offenders, f"these endpoints go through neither `answer()` nor `view()`: {offenders}"


def test_views_reach_the_world_only_through_commands():
    """A view composes commands. It must not be able to read anything itself.

    The import list is the whole check, and it is the strong one: a view that
    cannot import `cloud`, `pipeline` or `shared.records` cannot grow a second
    read path no matter what its joins do, because there is nothing for a join
    to read FROM except a payload a command already returned.
    """
    tree = ast.parse((WEB / "views.py").read_text())
    reached = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
            reached.add(node.module or "")
        if isinstance(node, ast.Import):
            reached.update(alias.name for alias in node.names if alias.name.startswith("src."))

    assert reached, "found no `src.` imports in views.py — the parser is broken, not the code"
    stowaways = reached - VIEW_IMPORTS
    assert not stowaways, (
        f"views.py imports {sorted(stowaways)}. A view may only COMPOSE command "
        f"payloads; anything outside {sorted(VIEW_IMPORTS)} lets it compute one."
    )


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
