"""The OpenAPI schema, exported for the console to generate its types from.

This is the file that replaced `payloads.fixture.json`, and the difference is
the whole point of the exercise. Both are generated -- but the fixture was
generated so that a HAND-WRITTEN `schemas.ts` could be checked against it, and
this is generated so that `types.gen.ts` can be built from it. The count of
hand-written declarations of a payload shape goes from two to one.

Committed rather than produced on demand, for the same reason the fixture was:
the console's build must not need a running server, or a Python environment, to
know what an endpoint returns.

Regenerate-and-fail rather than regenerate-silently, also for the same reason.
A schema change is a change to the contract the UI reads; writing it quietly
would let a payload change reach the console without anyone seeing that the
contract moved.
"""

from __future__ import annotations

import json

from src.interfaces.web.app import create_app
from src.shared import repo

SCHEMA = repo.ROOT / "console" / "src" / "api" / "openapi.json"


def _exported() -> str:
    """The schema as it should be on disk: the API, and nothing else.

    `sort_keys` because FastAPI builds the document by walking routes and models,
    and that order is not guaranteed stable across versions or across a reorder
    of `create_app`. An unstable export would produce a diff on every unrelated
    change, which is how a generated file stops being read.

    **Filtered to `/api/`, and that is a correctness fix rather than tidiness.**
    `_mount_console` registers a DIFFERENT catch-all depending on whether
    `console/dist/index.html` exists -- `_spa` when the console is built,
    `_unbuilt` when it is not -- so an unfiltered document changes shape with the
    build state of an unrelated directory. This test then failed or passed
    according to whether someone had run `npm run build`, which is the kind of
    flake that gets a guard deleted rather than fixed. The catch-all serves the
    page, not data, so it has no place in a client's schema either way.
    """
    document = create_app().openapi()
    document["paths"] = {
        path: operations
        for path, operations in document["paths"].items()
        if path.startswith("/api/")
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def test_the_exported_schema_is_current() -> None:
    """Regenerate on mismatch, and fail -- so the diff is reviewed, not absorbed."""
    expected = _exported()
    if not SCHEMA.exists() or SCHEMA.read_text() != expected:
        SCHEMA.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA.write_text(expected)
        raise AssertionError(
            f"{SCHEMA.name} was stale and has been regenerated. Review the diff, then "
            "run `npm --prefix console run gen:types` — the console's TypeScript is "
            "generated from this file, so a change here is a change to what the UI "
            "believes every endpoint returns."
        )


def test_every_endpoint_declares_what_it_returns() -> None:
    """A route with no `response_model` is a hole in the generated client.

    It does not fail the build -- it types as `unknown`, and the page that reads
    it compiles anyway and breaks at runtime. That is precisely the failure this
    whole seam exists to convert into a compile error, so an undeclared endpoint
    has to be loud here instead.
    """
    document = create_app().openapi()
    undeclared = []
    for path, operations in document["paths"].items():
        if not path.startswith("/api/"):
            continue
        for method, operation in operations.items():
            body = operation["responses"].get("200", {}).get("content", {})
            schema = body.get("application/json", {}).get("schema", {})
            # FastAPI emits a bare `{}` for a handler that declares nothing.
            if not schema:
                undeclared.append(f"{method.upper()} {path}")
    assert not undeclared, (
        f"these endpoints declare no response model: {sorted(undeclared)}. Add one "
        "from `src/interfaces/web/contract.py`, or the console reads them as `unknown`."
    )
