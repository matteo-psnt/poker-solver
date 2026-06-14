"""The OpenAPI schema, exported for the console to generate its types from.

This is the file that replaced `payloads.fixture.json`, and the difference is
the whole point of the exercise. Both are generated -- but the fixture was
generated so that a HAND-WRITTEN `schemas.ts` could be checked against it, and
this is generated so that `types.gen.ts` can be built from it. The count of
hand-written declarations of a payload shape goes from two to one.

Committed rather than produced on demand, for the same reason the fixture was:
the console's build must not need a running server, or a Python environment, to
know what an endpoint returns.

Regenerate-and-fail rather than regenerate-silently. A schema change is a change
to the contract the UI reads; writing it quietly would let a payload change
reach the console without anyone seeing that the contract moved. The same
writer runs as a pre-commit hook, so the suite is the second net here rather
than the only one.
"""

from __future__ import annotations

import json

from src.interfaces.web import schema


def test_the_exported_schema_is_current() -> None:
    """Regenerate on mismatch, and fail -- so the diff is reviewed, not absorbed."""
    assert not schema.write_if_stale(), (
        f"{schema.SCHEMA.name} was stale and has been regenerated. Review the diff, then "
        "run `npm --prefix console run gen:types` -- the console's TypeScript is "
        "generated from this file, so a change here is a change to what the UI "
        "believes every endpoint returns."
    )


def test_the_schema_only_describes_the_api() -> None:
    """`_mount_console` serves the page from a catch-all whose shape depends on
    whether `console/dist` exists, which would make this file's contents a
    function of an unrelated directory's build state."""

    document = json.loads(schema.export_schema())
    assert document["paths"]
    assert all(path.startswith("/api/") for path in document["paths"])
