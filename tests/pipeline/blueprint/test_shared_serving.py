"""Sharing one blueprint between the SERVING processes on the reader box.

The seat holds every depth of the chipzen ladder and the reader holds the deepest
run, so on 09-29 the same 100 bb blueprint was resident TWICE on a 15 GB box:
measured 2.82 GB private in the seat and 1.84 GB private in the reader, with only
0.08 GB (libraries) shared. `StaticArrayStorage` already has the create/attach
split that training workers use; these are the guards on wiring it to serving.
"""

from __future__ import annotations

import pytest

from src.pipeline.blueprint import construction
from src.shared.config import Config


class TestAttachRefusesWhatItCannotDo:
    """Both refusals happen BEFORE `build_card_abstraction`, which loads ~773 MB."""

    def test_attach_without_a_session_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="needs share_as"):
            construction.build_static_evaluation_solver(
                Config.default(),
                checkpoint_dir=tmp_path,
                attach=True,
                at_iteration=8000,
            )

    def test_attach_without_a_rung_is_refused(self, tmp_path):
        """The rung is what the creator named its segments after.

        An attacher reads no checkpoint, so it cannot discover the iteration the
        way a loader does -- and `assemble_policy` needs it.
        """
        with pytest.raises(ValueError, match="needs at_iteration"):
            construction.build_static_evaluation_solver(
                Config.default(),
                checkpoint_dir=tmp_path,
                share_as="serve-ladder",
                attach=True,
            )
