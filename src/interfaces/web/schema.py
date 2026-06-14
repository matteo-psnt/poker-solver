"""The OpenAPI document the console generates its TypeScript from.

Lives here rather than in the test that used to own it because two callers
need it now: `test_openapi_export` and the pre-commit hook. A generator that
exists only inside a test is a generator nothing can run.

`sort_keys` because FastAPI builds the document by walking routes and models,
and that order is not stable across versions or a reorder of `create_app`.
Filtered to `/api/` because `_mount_console` registers a DIFFERENT catch-all
depending on whether `console/dist/index.html` exists, so an unfiltered
document changes shape with the build state of an unrelated directory.
"""

from __future__ import annotations

import json

from src.interfaces.web.app import create_app
from src.shared import repo

SCHEMA = repo.ROOT / "console" / "src" / "api" / "openapi.json"


def export_schema() -> str:
    """The schema as it should be on disk: the API, and nothing else."""
    document = create_app().openapi()
    document["paths"] = {
        path: operations
        for path, operations in document["paths"].items()
        if path.startswith("/api/")
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_if_stale() -> bool:
    """Bring the file up to date. True if it MOVED, which is a contract change."""
    expected = export_schema()
    if SCHEMA.exists() and SCHEMA.read_text() == expected:
        return False
    SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA.write_text(expected)
    return True


if __name__ == "__main__":
    import sys

    if write_if_stale():
        sys.stderr.write(
            f"{SCHEMA.relative_to(repo.ROOT)} was stale and has been regenerated.\n"
            "Review the diff, stage it, and commit again -- staging it is what makes the\n"
            "console hook regenerate types.gen.ts from it.\n"
        )
        sys.exit(1)
