"""The node runs the wrapper before `uv sync`, so it runs it on stdlib alone.

`infra/run_task.py` is executed by the interpreter the pool's START TASK
installs -- not the OS's, which is 3.10 on the pinned 22.04 image -- with no
third-party package present. A non-stdlib import anywhere it REACHES does not
fail visibly: the task dies before it can say why, and `tasks` reports no
record, indistinguishable from "nothing ran".

**What carries the floor is derived, not listed.** It used to be a literal of
four paths, and a literal is only correct on the day it is written: the entry
point's real closure is eleven files, so `jsonio` and `records` were reached on
the node and checked by nothing. `_closure` below walks first-party imports
from the entry point, so the guarded set is whatever the node actually loads.

Two checks, deliberately at different costs: a substring scan for third-party
imports, and a real import of the whole closure on the node's interpreter --
the one that actually proves the contract.
"""

from __future__ import annotations

import ast
import pathlib
import shutil
import subprocess

import pytest

from src.shared import repo

REPO_ROOT = repo.ROOT
ENTRY_POINT = REPO_ROOT / "infra" / "run_task.py"


def _first_party_targets(source: pathlib.Path) -> set[str]:
    """Every ``src.*`` name this file imports, however it spells the import.

    ``from src.shared.cloudtask import kinds, task_log`` names two modules that the dotted
    prefix alone does not, so each imported name is also offered as a candidate
    module; the ones that are attributes rather than modules simply resolve to
    nothing. Relative imports do not appear -- the node package has none, and a
    file executed by path could not use them anyway.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.startswith("src.")}


def _resolve(module: str) -> pathlib.Path | None:
    """A dotted name to the file that defines it, if this repo defines one."""
    base = REPO_ROOT / pathlib.Path(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    package_init = base / "__init__.py"
    return package_init if package_init.is_file() else None


def _closure(entry: pathlib.Path) -> list[pathlib.Path]:
    """Every first-party file reachable from ``entry``, the entry point included.

    ``TYPE_CHECKING``-only imports are deliberately NOT excluded. Including one
    that the node never executes costs a redundant check; excluding one that a
    future edit promotes to runtime costs a task that dies without a record.
    """
    seen: set[pathlib.Path] = set()
    pending = [entry]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for module in _first_party_targets(current):
            resolved = _resolve(module)
            if resolved is not None and resolved not in seen:
                pending.append(resolved)
    return sorted(seen)


GUARDED_SOURCES = _closure(ENTRY_POINT)

NODE_PYTHON = "3.13"

THIRD_PARTY = (
    "numpy",
    "pydantic",
    "zarr",
    "yaml",
    "xxhash",
    "tqdm",
    "rich",
    "azure",
    # Installed BESIDE the node's interpreter by the pool's start task, unlike
    # the rest of these -- so importing it is legal, but only from inside a
    # function that catches the failure. See `legmirror`.
    "psycopg",
)


def _code(source: pathlib.Path) -> str:
    """Prose in a docstring may name a hazard; only executable lines may not."""
    text = source.read_text()
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(("#", "*"))]
    return "\n".join(lines).split('"""', 2)[-1]


def test_the_guarded_set_is_discovered_and_not_empty():
    """A derived list can fail OPEN; a literal one cannot.

    If `_resolve` stopped finding files -- a layout change, a rename -- the
    closure would quietly collapse to the entry point alone and every check
    below would still pass, on nothing. So assert the walk actually reached
    past the first hop, and name the two files a hand-written list forgot: they
    are reached only transitively (`lifecycle` -> `task_log` -> `records` ->
    `jsonio`), which is precisely the depth a person stops tracing at.
    """
    found = {path.relative_to(REPO_ROOT).as_posix() for path in GUARDED_SOURCES}
    assert ENTRY_POINT in GUARDED_SOURCES
    assert {
        "src/shared/cloudtask/node/lifecycle.py",
        "src/shared/cloudtask/node/handlers.py",
        "src/shared/cloudtask/node/progress.py",
        "src/shared/cloudtask/node/process.py",
        "src/shared/cloudtask/node/paths.py",
        "src/shared/cloudtask/node/archive.py",
        "src/shared/cloudtask/node/plan.py",
        "src/shared/cloudtask/task_log.py",
        "src/shared/cloudtask/kinds.py",
        "src/shared/cache.py",
        "src/shared/records.py",
        "src/shared/jsonio.py",
    } <= found, f"the node closure lost members; found {sorted(found)}"


def _module_level_imports(source: pathlib.Path) -> set[str]:
    """Top-level package names imported when this module is LOADED.

    Module scope only, which is the rule the node actually needs: an import that
    runs at load time and is missing kills the task at bootstrap, before it can
    write the record that would explain it. One inside a function runs later, on
    a path that can catch it.

    Parsed rather than scanned for substrings. The scan this replaces looked for
    `import numpy` and would have missed `from numpy import array` entirely.
    """
    names: set[str] = set()
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("source", GUARDED_SOURCES, ids=lambda p: p.name)
def test_no_third_party_import_at_module_level(source):
    """A task dying during dependency install must still leave a record."""
    offending = _module_level_imports(source) & set(THIRD_PARTY)
    assert not offending, f"{source.name} imports {sorted(offending)} at module level"


def test_a_deferred_third_party_import_is_caught_where_it_happens():
    """`psycopg` is the one the start task installs, so the node may use it --
    but only where a node that somehow lacks it mirrors nothing instead of dying
    at bootstrap. The guard above permits the import; this is what makes
    permitting it safe."""
    source = REPO_ROOT / "src" / "shared" / "cloudtask" / "node" / "legmirror.py"
    text = source.read_text()
    assert "psycopg" not in _module_level_imports(source)
    assert "import psycopg" in text, "the module this protects no longer imports it"
    assert "except Exception" in text, "the deferred import must be caught by its caller"


def test_the_interpreter_is_installed_by_the_pool_not_the_image():
    """The floor is no longer the OS's.

    It was 3.10, because the wrapper ran the system python3 on a pinned 22.04
    image, and that cost a scan for 3.11+ constructs plus per-file ruff ignores
    -- which still bit twice: `datetime.UTC` inside a call whose errors are
    swallowed, and again when renaming a module moved it out from under its
    ignore. The start task installs the interpreter now, so what has to stay
    true is that Terraform installs the version this asserts against.
    """
    main_tf = (REPO_ROOT / "infra" / "main.tf").read_text()
    assert f"uv python install {NODE_PYTHON}" in main_tf


def test_the_pool_installs_the_wrappers_one_dependency_where_it_will_be_found():
    """The wrapper mirrors a task's records into the database itself, and it
    runs before `uv sync` -- so the driver arrives with the INTERPRETER, not the
    project. Two halves that must agree: Terraform installs it, `spec.py` puts
    that directory on the wrapper's path, and neither can see the other.

    `--target` and not `--system`: uv REFUSES to install into the interpreter it
    manages ("externally managed ... should not be modified"), which failed the
    start task and left a node START_TASK_FAILED.
    """
    from src.interfaces.cloud.tasks.spec import NODE_DEPS_DIR, TASK_COMMAND_TEMPLATE

    main_tf = (REPO_ROOT / "infra" / "main.tf").read_text()
    install = next(line for line in main_tf.splitlines() if "psycopg[binary]" in line)
    assert f"--target {NODE_DEPS_DIR}" in install, "installed somewhere the wrapper does not look"
    assert "--system" not in install, "uv refuses to modify the interpreter it manages"
    assert f"PYTHONPATH={NODE_DEPS_DIR}" in TASK_COMMAND_TEMPLATE


def test_the_dependency_install_cannot_brick_a_node():
    """A start task that FAILS bricks the node, and this file has lost nodes to
    that -- including one to this very line. The one thing here whose absence
    costs nothing that matters must not be the thing that takes a node down:
    `legmirror` catches the missing driver and the task trains exactly as
    before, which is what an unset DSN already does."""
    main_tf = (REPO_ROOT / "infra" / "main.tf").read_text()
    install = main_tf[main_tf.index("psycopg[binary]") :][:200]
    assert "|| echo" in install, "an unguarded install in a `set -e` start task bricks the node"


def test_the_entry_point_adds_the_repo_to_the_path_before_importing():
    """It is executed as a file inside the extracted tarball, not as a module,
    so nothing puts the repo root on sys.path for it."""
    source = (REPO_ROOT / "infra" / "run_task.py").read_text()
    marker = "from src.shared.cloudtask.node.lifecycle import"
    assert source.index("sys.path.insert") < source.index(marker)


@pytest.mark.timeout(300)
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to provide the node interpreter")
def test_the_whole_package_imports_on_the_node_interpreter():
    """The check the substring scan cannot make.

    Imports the entry point's whole chain -- lifecycle, handlers, progress,
    archive, plan, task_log, records -- on the node's real interpreter, with `--no-project` so not one project
    dependency is installed. That is the node, before `uv sync`.
    """
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from src.shared.cloudtask.node.lifecycle import main;"
        "from src.shared.cloudtask.node.plan import parse_environment;"
        "plan = parse_environment({'RUN_OP': 'train', 'RUN_CONFIG': 'c', 'RUN_TO': '5'});"
        "assert plan.commands[0][0] == 'train-static';"
        "print('ok')"
    )
    result = subprocess.run(
        ["uv", "run", "--python", NODE_PYTHON, "--no-project", "python", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if "no interpreter found" in result.stderr.lower():
        pytest.skip(f"python {NODE_PYTHON} unavailable on this machine")
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.timeout(300)
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to provide the node interpreter")
def test_a_task_record_can_be_written_on_the_node_interpreter(tmp_path):
    """The one thing that must work even when everything else has failed."""
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from src.shared.cloudtask.task_log import write_node_record;"
        f"write_node_record({str(tmp_path)!r}, task_id='t', event='started');"
        "print('ok')"
    )
    result = subprocess.run(
        ["uv", "run", "--python", NODE_PYTHON, "--no-project", "python", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )
    if "no interpreter found" in result.stderr.lower():
        pytest.skip(f"python {NODE_PYTHON} unavailable on this machine")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "legs" / "t.1.start.json").exists()
