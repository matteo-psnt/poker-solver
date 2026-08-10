"""The eval index is derived on every read, and never written to a default path.

Recording an evaluation used to also append a row to `data/eval_ledger.jsonl` --
a module-level default that the node wrapper never overrode, so every cloud eval
wrote a stored index the architecture says does not exist. A stored index beside
a derived one is a second answer waiting to disagree, and deriving per read is
what makes concurrent evaluation from several boxes safe.

The obvious guard -- "no module names a `data/` path" -- would be WRONG here:
`DEFAULT_RUNS_DIR = "data/runs"` is deliberate, because `$CODE/data` is a symlink
to the node's work disk. What actually must not exist is an *implicit* place to
put the index, so that is what this pins.
"""

from __future__ import annotations

import inspect

import pytest

from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.ledger import queries as ledger_queries
from src.pipeline.evaluation.ledger import records as ledger_records
from src.pipeline.services import experiments
from src.pipeline.services import scoring as scoring_service

MODULES = (eval_ledger, ledger_queries, ledger_records, scoring_service, experiments)

PARAM = "ledger_path"


def _functions():
    for module in MODULES:
        for name, obj in vars(module).items():
            if inspect.isfunction(obj) and obj.__module__.startswith("src."):
                yield module.__name__, name, obj


def test_there_are_functions_to_check():
    """A generator that silently yielded nothing would pass the rule below."""
    found = [f"{m}.{n}" for m, n, _ in _functions()]
    assert len(found) > 10, f"introspection found almost nothing: {found}"


def test_something_still_takes_a_ledger_path():
    """If `ledger_path` were renamed, the rule below would enforce nothing."""
    takers = [f"{m}.{n}" for m, n, fn in _functions() if PARAM in inspect.signature(fn).parameters]
    assert takers, f"nothing takes `{PARAM}` any more -- this guard needs rewriting"


CASES = list(_functions())


@pytest.mark.parametrize(
    ("module_name", "name", "fn"),
    CASES,
    ids=[f"{module_name}.{name}" for module_name, name, _ in CASES],
)
def test_no_function_defaults_the_ledger_path(module_name, name, fn):
    """No function may default `ledger_path`.

    WHY THE IDS ARE SPELLED OUT. They used to be `str(x)[:40]` applied to each
    tuple element, and for the function element that is `<function f at 0x...>`
    -- a heap address, different in every interpreter. A test whose ID changes
    per process cannot be named: `-k`, `--last-failed` and any distributed run
    address a case that no longer exists under that name. Under `pytest -n`
    the workers collected mutually different suites and the whole run errored
    out before executing a single test. Module-qualified name is stable and
    is what the failure message quotes anyway.
    """
    parameter = inspect.signature(fn).parameters.get(PARAM)
    if parameter is None:
        return
    assert parameter.default is inspect.Parameter.empty, (
        f"{module_name}.{name} defaults `{PARAM}`, so a caller that forgets it writes a "
        f"stored index somewhere implicit. The index is derived per read -- pass the "
        f"path explicitly, from a tree the caller owns."
    )


def test_recording_an_evaluation_takes_no_ledger_path():
    """The write path itself must have no way to name an index at all."""
    assert PARAM not in inspect.signature(eval_ledger.record_evaluation).parameters


def test_there_is_no_append_helper():
    """`append_record` was the mechanism; its absence is the fix."""
    assert not hasattr(eval_ledger, "append_record")
    assert not hasattr(eval_ledger, "DEFAULT_LEDGER_PATH")
