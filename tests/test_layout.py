"""Module paths spelled inside STRINGS, resolved against the tree.

`.importlinter` enforces the direction of every import. This covers the one
staleness it structurally cannot see: a dotted path that is not an import.

A path in an `import` statement fails loudly the moment the module moves. The
same path inside a string -- a monkeypatch target, a subprocess program, a
logger name, a docstring citation -- fails silently or late. After the cloud
split, seven files still cited pre-move paths and nothing complained.

This file used to also pin filing CONVENTION: which modules may sit loose beside
a sub-package, which basenames may repeat, whether `tests/` mirrors `src/`.
Those were registries of taste with no violations in them, and keeping a
declaration in step with the tree cost more than the drift it caught. They are
gone; the layer contracts and this check are what remain.
"""

from __future__ import annotations

import ast
import collections
import importlib
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent


# Strings that look like module paths but are not, and why each is allowed.
NOT_MODULE_PATHS: dict[str, str] = {
    "src.pipeline.demo": "a fake logger name in test_log.py, module-shaped on purpose",
}

MODULE_PATH_IN_STRING = re.compile(r"(?<![\w.])(?:src|tests)(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _is_module(dotted: str) -> bool:
    path = ROOT / dotted.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def _resolves(dotted: str) -> bool:
    """True if `dotted` names a module, or an attribute chain on one."""
    if _is_module(dotted):
        return True
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:cut])
        if _is_module(prefix):
            try:
                obj = importlib.import_module(prefix)
                for attr in parts[cut:]:
                    obj = getattr(obj, attr)
                return True
            except (ImportError, AttributeError):
                return False
    return False


class TestModulePathsInStringsResolve:
    @pytest.mark.timeout(20)
    def test_every_dotted_path_in_a_string_names_something_real(self):
        stale = collections.defaultdict(set)
        for base in (SRC, TESTS, ROOT / "infra"):
            for path in sorted(base.rglob("*.py")):
                if "__pycache__" in str(path):
                    continue
                for node in ast.walk(ast.parse(path.read_text())):
                    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                        continue
                    for match in MODULE_PATH_IN_STRING.findall(node.value):
                        if match in NOT_MODULE_PATHS or _resolves(match):
                            continue
                        stale[str(path.relative_to(ROOT))].add(match)
        stale = {file: sorted(paths) for file, paths in stale.items()}
        assert not stale, (
            f"dotted module path(s) in strings that resolve to nothing: {stale}\n"
            "The module moved and the string did not follow -- an import would have "
            "failed loudly; the string just went stale. Update it, or if it is "
            "deliberately not a module path, declare it in NOT_MODULE_PATHS."
        )
