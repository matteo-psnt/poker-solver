"""What a `poker-solver` invocation may NOT import, expressed as a closure.

The sibling of `tests/interfaces/cloud/test_read_cost.py`. That one pins network
round trips because latency is invisible in a test; this pins IMPORT weight, for
exactly the same reason -- a `from sklearn.cluster import KMeans` at module scope
costs half a second on every command the tool has, and no test would ever notice.

Measured, all of it. `poker-solver --help` took **3.15s** before any of this, on
a warm interpreter and with no network at all, and every one of the edges below
was an accident rather than a need:

* `ledger.tiers` derives a knob tier from an ``LBRConfig`` -- a stdlib dataclass
  that happened to live inside the 850-line LBR evaluator, so deriving a
  comparability key imported scipy, numpy, tqdm and the whole engine. The ledger
  is reached from ``commands._base``, which EVERY command imports. Moving the
  dataclass to ``lbr.config`` took that module from 0.94s to 0.13s.
* `abstraction.postflop.precompute` and `abstraction.preflop.opponent_clusters`
  imported ``KMeans`` at module scope for three call sites. Both are reachable
  from the ``src.pipeline.services`` facade, so every reader command paid.

Result: 3.15s -> 1.93s.

These run in SUBPROCESSES on purpose. Under ``-n auto`` some other test has
almost certainly already imported numpy's world into the worker, so asking
``sys.modules`` in-process would answer about the worker's history rather than
about the closure under test.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from src.shared import repo

HEAVY = ("scipy", "sklearn", "numba")

SRC = repo.SRC
TESTS = repo.ROOT / "tests"


def _closure(module: str) -> set[str]:
    """The heavy top-level packages importing ``module`` drags in."""
    code = (
        "import sys, importlib;"
        f"importlib.import_module({module!r});"
        f"print(' '.join(sorted({{m.split('.')[0] for m in sys.modules}} & set({HEAVY!r}))))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, f"importing {module} failed:\n{done.stderr}"
    return set(done.stdout.split())


@pytest.mark.timeout(180)
class TestTheCommandSurfaceStaysLight:
    def test_the_command_registry_does_not_import_sklearn(self):
        """It is reached only through the abstraction facade, and only three
        functions in the tree actually cluster anything."""
        assert "sklearn" not in _closure("src.interfaces.commands")

    def test_the_ledger_imports_none_of_them(self):
        """`commands._base` imports `rebuild_ledger`, so whatever the ledger
        pulls, EVERY command pays for -- `jobs` and `--help` included."""
        assert _closure("src.pipeline.evaluation.ledger") == set()

    def test_the_tier_key_does_not_reach_the_evaluator(self):
        """The specific edge: a knob tier is derived from `LBRConfig`, which is
        a stdlib dataclass. It must not live behind numpy and scipy again."""
        assert _closure("src.pipeline.evaluation.ledger.tiers") == set()


class TestNothingReachesLBRConfigThroughTheEvaluator:
    """The old import path still RESOLVES, and cannot be made not to.

    ``hunl_local_best_response`` imports ``LBRConfig`` for its own use, so
    ``from ...hunl_local_best_response import LBRConfig`` keeps working -- and
    any module that does it pays scipy, numpy, tqdm and the engine to get a
    stdlib dataclass. Python offers no way to forbid that at import time.

    So the guard is on the SOURCE, the same shape as
    ``test_records.py::test_no_module_writes_a_json_artifact_outside_the_substrate``.
    An earlier version of this test asserted ``"LBRConfig" not in __all__``,
    which passed while the path it was meant to close stayed wide open -- a
    guard that checks nothing is worse than no guard, because it is believed.
    """

    def test_the_config_is_importable_from_its_own_module(self):
        from src.pipeline.evaluation.estimators.lbr.config import LBRConfig

        assert LBRConfig().num_hands > 0

    def test_no_module_imports_it_from_the_evaluator(self):
        offenders = []
        pattern = re.compile(
            r"from src\.pipeline\.evaluation\.lbr\.hunl_local_best_response import [^\n]*LBRConfig"
        )
        for path in (*SRC.rglob("*.py"), *TESTS.rglob("*.py")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path}:{number}")
        assert not offenders, (
            "these import LBRConfig from the evaluator, paying ~1.7s of scientific "
            "stack for a stdlib dataclass — import it from "
            "src.pipeline.evaluation.estimators.lbr.config instead: " + ", ".join(offenders)
        )
