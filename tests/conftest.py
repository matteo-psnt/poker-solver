"""Suite-wide fixtures.

The card-abstraction guard, the telemetry kill switch, and the hypothesis
profile.

The abstraction guard is the older of the two. A test that trains for real
needs a precomputed combo abstraction on this machine, and that artifact is
**gitignored, unversioned, and ~194 MB** -- so whether it is present is a
property of the laptop, not of the code. A test whose result depends on that
must say "not run", never "broken".
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings

from src.interfaces import telemetry

# `derandomize=True` is the whole reason property tests are allowed in here.
# `pytest-randomly` was rejected for this suite because random order x xdist x a
# 5s wall-clock budget is a flake generator, and a hypothesis default profile
# reproduces exactly that failure mode -- a new input set every run, under a
# timeout, on 12 workers. Derandomized, a property test is a fixed set of
# examples derived from the source: it fails for everyone or for nobody.
# `deadline=None` because the numba kernels JIT-compile on the first example and
# nothing else in the file is slow; the per-test `@pytest.mark.timeout` is the
# real budget.
settings.register_profile(
    "suite",
    derandomize=True,
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("suite")


@pytest.fixture(autouse=True, scope="session")
def _no_telemetry_from_the_suite():
    """The suite records nothing, and this must be autouse.

    `Command.execute` writes one row per invocation into the developer's real
    cache -- `cache_root()` reads the environment, and nothing here redirects
    it. A full run calls commands hundreds of times, so without this every
    `pytest` would append hundreds of rows describing commands that ran against
    fakes and measured nothing. `activity` would then report a p95 for `tasks`
    derived mostly from a stub returning a dict.

    Session-scoped and set in the environment rather than monkeypatched,
    because `-n auto` runs 12 worker PROCESSES and a patched module attribute
    would only silence the one that applied it.
    """
    import os

    previous = os.environ.get(telemetry.ENV_VAR)
    os.environ[telemetry.ENV_VAR] = "0"
    yield
    if previous is None:
        del os.environ[telemetry.ENV_VAR]
    else:
        os.environ[telemetry.ENV_VAR] = previous


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
    from src.shared.config.loader import load_training_config

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
