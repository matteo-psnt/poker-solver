"""Shared fixtures for the interfaces tests.

Every reading command now answers against the published record: there is no
`--source local` to point at a directory. That is right for the product and
awkward for a test, which wants a tree it controls and no network.

`records_root` imports `share_records` lazily, inside the function, so there is
exactly one place to intercept — patch it and the whole reader stack works
against a local tree without knowing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for the share: readers materialise THIS tree.

    Returns the directory, so a test seeds it exactly as it used to seed a
    `--runs-dir`.
    """
    import src.interfaces.cloud.store.workspace as workspace

    @contextmanager
    def _materialise(run: str | None = None) -> Iterator[Path]:
        yield tmp_path

    monkeypatch.setattr(workspace, "share_records", _materialise)
    return tmp_path
