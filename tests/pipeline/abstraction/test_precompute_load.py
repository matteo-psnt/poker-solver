"""Loading a stale abstraction pickle surfaces a clear, actionable error."""

from __future__ import annotations

import pytest

from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer


def test_stale_abstraction_pickle_raises_friendly_error(tmp_path):
    """A pickle referencing a module that no longer exists must not leak a raw
    ModuleNotFoundError; it should tell the user to regenerate the abstraction.
    """
    # Minimal pickle: a GLOBAL opcode referencing a nonexistent module, which
    # raises ModuleNotFoundError on load exactly like a pre-refactor artifact.
    (tmp_path / "combo_abstraction.pkl").write_bytes(b"c__nonexistent_pkg__\nThing\n.")

    with pytest.raises(RuntimeError, match=r"[Ss]tale abstraction"):
        PostflopPrecomputer.load(tmp_path)
