"""What an endpoint passes a command, checked against the command's own parser.

`test_command_coverage` asks whether a command is REACHABLE. Both failures here
were one level below that: `submit` had an endpoint the whole time, and the
console still could not queue a board-free or a warm-started run, because
`SubmitBody` was missing the seven flags those two features arrived as. The gap
looked like a feature nobody had built rather than a field nobody had added.

The other direction is the same mistake mirrored. `/api/box` handed `serve-box`
its own `--resource-group`, `--vm` and `--subscription` defaults straight back,
so the blueprint host's name was declared twice and renaming it in `serve_box.py`
would have left this file pointing at a VM that no longer exists.

Read from the AST for the reason the coverage guard is: a mapping kept here is
another thing that can be right about a file it does not read.
"""

from __future__ import annotations

import ast
import json
from typing import Any

import pytest

from src.interfaces import telemetry
from src.interfaces.commands import _compose, activity, configs, load
from src.interfaces.commands._compose import Part
from src.interfaces.web import app
from src.interfaces.web.cache import TtlCache
from src.shared import repo

APP = repo.SRC / "interfaces" / "web" / "app.py"

TREE = ast.parse(APP.read_text())

# Given to every subcommand by `headless.build_parser` as parents, so they are
# not the command's own flags and no endpoint is expected to offer them.
COMMON = {"json", "log_level", "help"}


def _answer_calls() -> list[ast.Call]:
    """Every ``answer(cache, <module>.COMMAND, ...)`` call in `app.py`."""
    return [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "answer"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Attribute)
    ]


def _command_of(call: ast.Call) -> str:
    module = call.args[1].value  # type: ignore[attr-defined]
    return module.id.replace("_", "-")


def _flags(name: str) -> tuple[dict[str, Any], set[str]]:
    """One command's declared flag defaults, minus the ones every command gets."""
    defaults, _ = load(name).declared()
    return {key: value for key, value in defaults.items() if key not in COMMON}, set()


MODELS: dict[str, ast.ClassDef] = {
    node.name: node for node in ast.walk(TREE) if isinstance(node, ast.ClassDef)
}


def _bodies() -> dict[str, str]:
    """``request-model name -> command name``, for each POST that splats a body.

    The shape is always ``answer(TtlCache(0.0), <module>.COMMAND, **given(body))``
    inside a handler whose parameters include one annotated with the model.

    Only annotations naming a class declared in `app.py` count. A handler is free
    to take a second parameter -- a path segment, a query flag -- and reading
    `dry: bool` as a request model would fail this file with a true statement
    about the wrong thing ("bool is declared nowhere").
    """
    found: dict[str, str] = {}
    for handler in ast.walk(TREE):
        if not isinstance(handler, ast.FunctionDef):
            continue
        annotations = {
            argument.annotation.id
            for argument in handler.args.args
            if isinstance(argument.annotation, ast.Name) and argument.annotation.id in MODELS
        }
        for call in _answer_calls():
            if call not in ast.walk(handler):
                continue
            splatted = {
                keyword.value.func.id  # type: ignore[attr-defined]
                for keyword in call.keywords
                if keyword.arg is None
                and isinstance(keyword.value, ast.Call)
                and isinstance(keyword.value.func, ast.Name)
            }
            if "given" not in splatted:
                continue
            for model in annotations:
                found[model] = _command_of(call)
    return found


def _fields(model: str) -> set[str]:
    """The field names of a request model, read where it is declared."""
    return {
        statement.target.id
        for statement in MODELS[model].body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    }


BODIES = _bodies()


def test_the_bodies_were_found_at_all():
    """A parser that matches nothing would make every check below vacuous."""
    assert BODIES, f"found no `**given(body)` dispatch in {APP} — the parser is broken"


@pytest.mark.parametrize(("model", "command"), sorted(BODIES.items()))
def test_a_request_model_offers_every_flag_its_command_accepts(model, command):
    """A flag with no field is a capability the console silently does not have.

    Not the reverse of the coverage guard but the finer grain of it: the command
    is reachable and the feature still is not. Adding a flag to a dispatching
    command means adding it here, or the CLI grows a door the console has not.
    """
    defaults, _ = _flags(command)
    unreachable = sorted(set(defaults) - _fields(model))
    assert not unreachable, (
        f"`{command}` accepts {unreachable}, which `{model}` does not offer — so the "
        f"console cannot ask for them. Add the field(s) to `{model}` in {APP} as "
        "`| None = None`, which `given()` reads as omitted."
    )


@pytest.mark.parametrize(("model", "command"), sorted(BODIES.items()))
def test_a_request_model_invents_no_flag(model, command):
    """A field with no flag reaches `Command.arguments` and is refused there.

    As a 422 naming an argument the caller never sent, which is a true message
    about the wrong file.
    """
    defaults, _ = _flags(command)
    invented = sorted(_fields(model) - set(defaults))
    assert not invented, f"`{model}` offers {invented}, which `{command}` does not accept"


def _resolvable(node: ast.expr) -> tuple[bool, Any]:
    """The value of an endpoint's keyword argument, when it is a constant.

    Literals and ``<module>.NAME`` module constants -- the only two forms an
    endpoint uses for a fixed argument, and the only two that can restate a
    parser default. Anything else is a request parameter and is not a default.
    """
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module = __import__(f"src.interfaces.commands.{node.value.id}", fromlist=[node.attr])
        return hasattr(module, node.attr), getattr(module, node.attr, None)
    return False, None


def test_no_endpoint_restates_a_parser_default():
    """`Command.invoke` fills every default from `add_arguments`, so re-passing
    one is a second declaration that can drift from the first.

    `action="status"` on `/api/box` is the one declared exception: those three
    endpoints differ in exactly one argument, and leaving it implicit on the one
    whose value happens to be the default makes the set read as though it were
    doing something else.
    """
    allowed = {("serve-box", "action")}
    restated: list[str] = []
    for call in _answer_calls():
        command = _command_of(call)
        try:
            defaults, _ = _flags(command)
        except KeyError:  # pragma: no cover — a name that is not a command
            continue
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg not in defaults:
                continue
            known, value = _resolvable(keyword.value)
            if known and value == defaults[keyword.arg] and (command, keyword.arg) not in allowed:
                restated.append(f"{command}({keyword.arg}={value!r})")
    assert not restated, (
        f"{sorted(restated)} pass a value equal to the command's own parser default. "
        "Drop the argument: `Command.invoke` supplies it, and passing it here makes "
        "this file a second place the value has to be kept true."
    )


def test_every_command_named_by_an_endpoint_still_exists():
    """An endpoint naming a deleted command imports fine and 500s on the first
    request, because the attribute is resolved at call time."""
    for call in _answer_calls():
        name = _command_of(call)
        try:
            load(name)
        except KeyError:
            pytest.fail(f"{APP} has an endpoint for `{name}`, which is not a command")


def test_a_composed_view_files_its_parts_under_the_console(tmp_path, monkeypatch):
    """A view's parts cross a thread boundary before they are recorded.

    `_served` opens the surface, `_compose._bound` copies the calling thread's
    context per submit, and the part runs on a pool thread -- so the attribution
    survives only because the copy is taken on the caller. It did not once, and
    every panel's cost was filed under the default. Nothing asserted it until the
    `with` moved off the four endpoints and into `_served`.
    """
    monkeypatch.setenv("POKER_SOLVER_TELEMETRY", "1")
    monkeypatch.setenv("POKER_SOLVER_CACHE", str(tmp_path))

    def probe() -> dict[str, Any]:
        # Two LOCAL commands, so this fans out for real and touches no cloud.
        return _compose.compose(
            "view-probe",
            [
                Part("configs", configs.COMMAND),
                Part("activity", activity.COMMAND, {"days": 1.0, "limit": 1}),
            ],
        )

    assert app._served(TtlCache(0.0), ("probe", ()), probe).status_code == 200

    rows = [
        json.loads(line)
        for path in telemetry.logs()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert rows, "the fan-out recorded nothing, so this asserts nothing"
    misfiled = sorted({row["command"] for row in rows if row["surface"] != "console"})
    assert not misfiled, f"{misfiled} were filed under another surface than the console"
