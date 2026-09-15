"""The OpenAPI schema, exported for the console to generate its types from.

This is the file that replaced `payloads.fixture.json`, and the difference is
the whole point of the exercise. Both are generated -- but the fixture was
generated so that a HAND-WRITTEN `schemas.ts` could be checked against it, and
this is generated so that `types.gen.ts` can be built from it. The count of
hand-written declarations of a payload shape goes from two to one.

Committed rather than produced on demand, for the same reason the fixture was:
the console's build must not need a running server, or a Python environment, to
know what an endpoint returns.

The suite only CHECKS. The pre-commit hook is the writer, and it fails the
commit it rewrote the file in so the diff is staged on purpose. A test that
also wrote was a test that changed the working tree under `-n auto`, and in
practice its failure was answered by running pytest again rather than by
reading anything.
"""

from __future__ import annotations

import json

from src.interfaces.web import schema


def test_the_exported_schema_is_current() -> None:
    """Check only: the message names the writer, this never is one."""
    on_disk = schema.SCHEMA.read_text() if schema.SCHEMA.exists() else ""
    assert on_disk == schema.export_schema(), (
        f"{schema.SCHEMA.name} is stale. Regenerate it with\n"
        "    uv run python -m src.interfaces.web.schema\n"
        "    npm --prefix console run gen:types\n"
        "-- the console's TypeScript is generated from this file, so a change here "
        "is a change to what the UI believes every endpoint returns."
    )


def test_the_schema_only_describes_the_api() -> None:
    """`_mount_console` serves the page from a catch-all whose shape depends on
    whether `console/dist` exists, which would make this file's contents a
    function of an unrelated directory's build state."""

    document = json.loads(schema.export_schema())
    assert document["paths"]
    assert all(path.startswith("/api/") for path in document["paths"])
