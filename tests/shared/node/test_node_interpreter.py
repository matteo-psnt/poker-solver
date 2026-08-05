"""The node runs OLDER Python than this project, and runs it before `uv sync`.

`infra/main.tf` pins `batch.node.ubuntu 22.04`, whose system `python3` is 3.10,
not the 3.12+ developed against here -- and `infra/run_task.py` is executed by
that interpreter, with no third-party package installed. A 3.11+ construct or a
non-stdlib import anywhere it REACHES does not fail visibly: the task dies before
it can say why, and `tasks` reports "no task records", indistinguishable from
"no tasks ran".

**What carries the floor is derived, not listed.** It used to be a literal of
four paths, and a literal is only correct on the day it is written: the entry
point's real closure is eleven files, so `jsonio` and `records` were reached on
the node and checked by nothing. Neither happens to violate the floor today,
which is exactly why a list drifts quietly -- the cost of an import added to
`runner` is paid on the one machine that has no way to report it. `_closure`
below walks first-party imports from `infra/run_task.py`, so the guarded set is
whatever the node actually loads, and a new import extends it with no edit here.

Two checks, deliberately at different costs:

* a substring scan, in the fast gate, that catches the constructs already known
  to bite. It runs under THIS interpreter, so it can only look for names.
* a real 3.10 import of the whole package, via `uv run --python 3.10
  --no-project`. This is the one that actually proves the contract --
  `datetime.UTC` was added, passed every test, and silently disabled task
  records on the only machine that runs them. It is in the FAST gate despite
  spawning an interpreter: measured at ~85ms warm, which is less than the
  substring scan costs to justify. The generous timeout covers the one cold
  run that downloads the interpreter.

Ruff's `target-version` is py312, so `UP017` and friends would happily rewrite
these modules into something the node cannot import. `pyproject.toml` disables
those rules for exactly these files; if that per-file ignore is dropped, the
3.10 run below is what notices.
"""

from __future__ import annotations

import ast
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ENTRY_POINT = REPO_ROOT / "infra" / "run_task.py"


def _first_party_targets(source: pathlib.Path) -> set[str]:
    """Every ``src.*`` name this file imports, however it spells the import.

    ``from src.shared import cache, task_log`` names two modules that the dotted
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

NODE_PYTHON = "3.10"

THIRD_PARTY = ("numpy", "pydantic", "zarr", "yaml", "xxhash", "tqdm", "rich", "azure")

# Names that do not exist on 3.10. Extend when the floor moves.
FORBIDDEN = (
    ("datetime import UTC", "datetime.UTC is 3.11+; use timezone.utc"),
    ("from typing import Self", "typing.Self is 3.11+"),
    ("ExceptionGroup", "ExceptionGroup is 3.11+"),
    ("tomllib", "tomllib is 3.11+"),
    ("itertools.batched", "itertools.batched is 3.12+"),
    ("@override", "typing.override is 3.12+"),
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
    are reached only transitively (`runner` -> `task_log` -> `records` ->
    `jsonio`), which is precisely the depth a person stops tracing at.
    """
    found = {path.relative_to(REPO_ROOT).as_posix() for path in GUARDED_SOURCES}
    assert ENTRY_POINT in GUARDED_SOURCES
    assert {
        "src/shared/node/runner.py",
        "src/shared/node/archive.py",
        "src/shared/node/plan.py",
        "src/shared/task_log.py",
        "src/shared/cache.py",
        "src/shared/describe.py",
        "src/shared/records.py",
        "src/shared/jsonio.py",
    } <= found, f"the node closure lost members; found {sorted(found)}"


@pytest.mark.parametrize("source", GUARDED_SOURCES, ids=lambda p: p.name)
def test_no_third_party_import(source):
    """A task dying during dependency install must still leave a record."""
    text = source.read_text()
    for name in THIRD_PARTY:
        assert f"import {name}" not in text, f"{source.name} must not import {name}"


@pytest.mark.parametrize("source", GUARDED_SOURCES, ids=lambda p: p.name)
def test_no_construct_newer_than_the_node_interpreter(source):
    body = _code(source)
    for needle, why in FORBIDDEN:
        assert needle not in body, f"{needle} in {source.name}: {why}"


def test_the_pinned_node_image_is_still_what_this_assumes():
    """If the image moves, the floor above moves with it."""
    main_tf = (REPO_ROOT / "infra" / "main.tf").read_text()
    assert "batch.node.ubuntu 22.04" in main_tf, (
        "the node image changed; re-check the system python3 version this package must import under"
    )


def test_the_entry_point_adds_the_repo_to_the_path_before_importing():
    """It is executed as a file inside the extracted tarball, not as a module,
    so nothing puts the repo root on sys.path for it."""
    source = (REPO_ROOT / "infra" / "run_task.py").read_text()
    assert source.index("sys.path.insert") < source.index("from src.shared.node.runner import")


@pytest.mark.timeout(300)
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to provide a 3.10 interpreter")
def test_the_whole_package_imports_on_the_node_interpreter():
    """The check the substring scan cannot make.

    Imports the entry point's whole chain -- runner, archive, plan, task_log,
    records -- on a real 3.10, with `--no-project` so not one project
    dependency is installed. That is the node, before `uv sync`.
    """
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from src.shared.node.runner import main;"
        "from src.shared.node.plan import parse_environment;"
        "plan = parse_environment({'RUN_OP': 'train', 'RUN_CONFIG': 'c', 'RUN_TO': '5'});"
        "assert plan.train_argv()[0] == 'train-static';"
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
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to provide a 3.10 interpreter")
def test_a_task_record_can_be_written_on_the_node_interpreter(tmp_path):
    """The one thing that must work even when everything else has failed."""
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from src.shared.task_log import write_node_record;"
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
