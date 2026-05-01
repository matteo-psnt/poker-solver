"""Suite-wide fixtures.

The one thing here is the card-abstraction guard. A test that trains for real
needs a precomputed combo abstraction on this machine, and that artifact is
**gitignored, unversioned, and ~194 MB** -- so whether it is present is a
property of the laptop, not of the code. A test whose result depends on that
must say "not run", never "broken".
"""

from __future__ import annotations

import pytest


@pytest.fixture
def requires_card_abstraction():
    """Skip unless the `quick_test` combo abstraction is precomputed locally.

    WHY A SKIP AND NOT A FAILURE. `data/` holds no run data and no training
    artifact by design -- runs live on the share and nowhere else. The only
    thing that ever put 194 MB back on a laptop was this one test, and when the
    artifact was absent the FULL suite went red with a `FileNotFoundError`
    pointing at `precompute`. That is an environment report dressed as a
    regression: it tells a fresh checkout its code is broken when it is not,
    and it is the loudest possible signal for the least actionable cause.

    The coverage this protects is real (parallel exact-BR must agree with
    serial EXACTLY, or the zero-variance instrument stops being zero-variance),
    so the skip names how to get it back rather than quietly passing.
    """
    from src.pipeline import blueprint
    from src.shared.config_loader import load_training_config

    # `load_training_config`, NOT `load_config` -- the latter takes a PATH and
    # raises FileNotFoundError on a bare stem, which this guard would have
    # caught and reported as a missing abstraction. That skips forever, with a
    # plausible message, on a machine where the artifact is present. Verified
    # by running with the artifact restored and watching the test actually run.
    config = load_training_config("quick_test")
    try:
        blueprint.resolve_card_abstraction_hash(config)
    except FileNotFoundError:
        pytest.skip(
            "no local `quick_test` combo abstraction (~194 MB, gitignored). "
            "This test trains for real and needs one. To run it: "
            "`uv run poker-solver submit-precompute --config quick_test` "
            "and copy the result into data/combo_abstraction/, or pull it from "
            "the share. Everything else in the suite runs without it."
        )
