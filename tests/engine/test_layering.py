"""Layering guardrails for package dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path


def test_engine_does_not_import_pipeline() -> None:
    """Engine modules must not depend on pipeline modules."""
    engine_root = Path("src/engine")
    violations: list[str] = []

    for py_file in engine_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src.pipeline"):
                        violations.append(f"{py_file}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("src.pipeline"):
                    violations.append(f"{py_file}:{node.lineno} imports from {module}")

    assert not violations, "Engine->pipeline import violations found:\n" + "\n".join(violations)
