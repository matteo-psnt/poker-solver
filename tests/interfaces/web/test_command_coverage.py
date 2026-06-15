"""Every command is reachable from the console, or its absence is DECLARED.

The console is expected to grow toward what the commands can do -- that is the
stated relationship between the two surfaces, and it is the one property of it
that nothing checked. Coverage was 12 of 29 when this was written, and no single
commit had made it so: each new command simply arrived without a matching
endpoint, and nothing said which of the seventeen gaps were decisions.

So the gap becomes a declaration, in the `records.REGISTRY` idiom the layout
guards already use. A command with no endpoint fails until someone writes down
which of exactly two reasons applies -- and the reasons are different, which is
the point:

- ``NO_PAYLOAD`` -- it blocks forever and never returns one. `Command.invoke`
  has nothing to hand a surface, and a button that starts a server inside the
  server process is not a coherent thing to build.
- ``NODE_ONLY`` -- it is compute, invoked BY the node wrapper on the box
  executing a task. A console button here would train on the laptop, which is
  the one thing this project does not do. The scriptable path already covers it:
  `submit` queues the pool task that runs `train-static`.

Adding to either list is fine. Doing it without deciding is what this argues
against, so the reason is stored beside the name and read back in the failure.
"""

from __future__ import annotations

import ast

from src.interfaces.commands import COMMANDS
from src.shared import repo

APP = repo.SRC / "interfaces" / "web" / "app.py"


NO_PAYLOAD: dict[str, str] = {
    "serve": "it IS this server; invoking it from inside would bind the port again",
    "blueprint-serve": "blocks serving one run; the console reaches it through "
    "`blueprint_proxy` and controls its host through `serve-box`",
}

NODE_ONLY: dict[str, str] = {
    "train-static": "compute, run BY the node wrapper — `submit` is the console's door",
    "precompute": "compute, run BY the node wrapper — `submit-precompute` is the door",
    "evaluate": "compute, run BY the node wrapper — `score` is the door",
    "vector-sweep": "compute, run BY the node wrapper — `submit-vector` is the door",
    "train-vector": "compute, run BY the node wrapper — `submit --kernel board-free` is the door",
    "abstraction-coupling": "compute, run BY the node wrapper — the fine abstraction is on the share",
}

EXCLUDED = NO_PAYLOAD | NODE_ONLY

# NOT an exclusion: `status` composes three commands the console already renders
# as panels, so every question it answers is on the Overview page. An
# `/api/status` would be the second read path `test_no_second_read_path` forbids.
COMPOSED: dict[str, tuple[str, ...]] = {"status": ("pool-status", "jobs", "tasks")}


def _commands_the_console_invokes() -> set[str]:
    """Which command each `answer(...)` call in `app.py` names.

    Read from the AST rather than from a list kept here, for the same reason
    `test_no_second_read_path` walks it: a hand-maintained list of endpoints is
    another thing that can be right about a file it does not read.

    The second argument is always ``<module>.COMMAND``, and the module name is
    the command's own with hyphens as underscores -- the convention the registry
    itself relies on -- so the attribute access identifies it without importing
    anything.
    """
    tree = ast.parse(APP.read_text())
    named: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "answer" or len(node.args) < 2:
            continue
        command = node.args[1]
        if isinstance(command, ast.Attribute) and isinstance(command.value, ast.Name):
            named.add(command.value.id.replace("_", "-"))
    return named


def test_every_command_is_reachable_from_the_console():
    """The console covers each command, or the gap is one of the declared ones."""
    covered = _commands_the_console_invokes()
    assert covered, "found no `answer(...)` calls — the parser is broken, not the code"

    reachable = covered | set(COMPOSED)
    missing = sorted({ref.name for ref in COMMANDS} - reachable - set(EXCLUDED))
    assert not missing, (
        f"{missing} have no console endpoint and no declared reason. The console is "
        "expected to grow toward what the commands can do: add an endpoint in "
        f"`web/app.py`, or add the name to NO_PAYLOAD/NODE_ONLY in {__file__} with "
        "the reason it does not belong on a screen."
    )


def test_the_exclusions_still_name_real_commands():
    """A declared gap for a command that no longer exists is a stale decision.

    It reads as a considered omission forever, which is worse than no list: the
    next person sees three node-only commands and does not notice one of them
    was deleted two refactors ago.
    """
    names = {ref.name for ref in COMMANDS}
    stale = sorted((set(EXCLUDED) | set(COMPOSED)) - names)
    assert not stale, f"{stale} are declared here but are not commands any more"


def test_nothing_is_both_excluded_and_covered():
    """An exclusion that gained an endpoint is a decision that was reversed.

    Silently, and in the other file -- so the reason recorded here goes on
    arguing for something that is no longer true.
    """
    covered = _commands_the_console_invokes()
    contradicted = sorted(covered & set(EXCLUDED))
    assert not contradicted, (
        f"{contradicted} have an endpoint but are still listed as excluded here. "
        "Delete the entry: the reason it carries is now false."
    )


def test_composed_commands_are_covered_by_their_parts():
    """`status` counts as reachable only while all three panels are."""
    covered = _commands_the_console_invokes()
    for name, parts in COMPOSED.items():
        absent = sorted(set(parts) - covered)
        assert not absent, (
            f"{name} is declared as covered by {list(parts)}, but {absent} "
            "no longer have endpoints — so it is not covered at all."
        )
