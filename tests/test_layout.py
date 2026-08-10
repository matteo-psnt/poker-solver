"""Where files sit, asserted rather than intended.

`.importlinter` enforces the DIRECTION of every import and does it well -- ten
contracts, and the layer graph is clean. It cannot see either of the two things
below, because neither is an edge:

- whether a set of PEERS is filed consistently, and
- whether two files share a name.

Both drifted. `lbr/` was a package while `public_tree_br.py` -- the primary
estimator, and the largest module in that package -- sat loose beside the
ledger and the reference oracles, so "which of these scores a blueprint" was
not answerable from the directory. And `services.evaluation` sat above
`pipeline.evaluation`, so one word named both a service and the subsystem it
wrapped.

Both are declared here, in the `records.REGISTRY` idiom this project already
uses: the exceptions are a list, the test asserts reality matches it, and
adding to the list is a decision someone makes on purpose in a diff.

HONESTLY: the basename registry ships with eleven entries and zero violations,
so it is documentation with a tripwire attached rather than enforcement. It
earns its place by making the TWELFTH duplicate a decision instead of an
accident -- which is exactly how the third `paths.py` would otherwise arrive.
"""

from __future__ import annotations

import collections
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


"""Loose modules that sit beside sub-packages, and why each set is allowed.

A package that holds sub-packages AND loose modules is not automatically wrong
-- `shared/` is a flat utility package with one subsystem in it, and that is the
right shape. It is wrong when a loose module is a PEER of something filed inside
a sub-package, because then the same kind of thing is filed two ways and the
directory stops answering what it looks like it answers.

That distinction needs a person, so this asks for one: the set is pinned, and a
new loose module fails until someone writes down which case it is.
"""
LOOSE_BESIDE_PACKAGES: dict[str, tuple[str, ...]] = {
    # Each is a single thing with no siblings to be filed away from.
    "engine.solver": ("betting_tree.py", "numba_ops.py", "protocols.py"),
    # Leaf helpers with no owner among the surfaces; `errors` is imported by all
    # of them and belongs above each.
    "interfaces": ("errors.py", "run_names.py"),
    # `config` is the credentials and resource ids all three sub-packages need,
    # so it sits above them; `serve_box` provisions the play-server VM and is
    # its own job, sharing nothing with dispatching work or reading the bill.
    "interfaces.cloud": ("config.py", "serve_box.py"),
    # Shared across postflop/ and preflop/, which is why they are not in either.
    "pipeline.abstraction": ("base.py", "config.py", "paths.py", "resolver.py"),
    # Measurement helpers used BY the estimators, not estimators themselves.
    "pipeline.evaluation": ("statistics.py", "units.py"),
    # Peers of `lbr/`; each is one module and `lbr/` is six, so the difference
    # is size, not kind. If one of these grows a second module it becomes a
    # package -- it does not gain a `_helpers.py` sibling out here.
    "pipeline.evaluation.estimators": (
        "blueprint_match.py",
        "public_tree_br.py",
        "resolver_match.py",
    ),
    # Same rule: `scoring/` is five modules, these are one each.
    "pipeline.services": ("bucketing.py", "experiments.py", "runs.py", "static_training.py"),
    "pipeline.training": ("static_parallel.py",),
    # `shared/` IS the flat one -- a module per concern, deliberately small, so
    # that anything may import exactly what it needs. `cloudtask/` is the one
    # subsystem, and it is a subsystem because the node imports it as a closure.
    "shared": (
        "action_tokens.py",
        "cache.py",
        "config.py",
        "config_loader.py",
        "dicts.py",
        "gitinfo.py",
        "jsonio.py",
        "log.py",
        "numeric.py",
        "records.py",
        "run_events.py",
    ),
    # Read from BOTH ends of the wire, so above `node/` rather than inside it.
    "shared.cloudtask": ("kinds.py", "task_log.py"),
}


"""Module basenames used more than once, and why each is not a collision.

Most are the command-and-its-implementation pattern (`commands/runs.py` invokes
`services/runs.py`), which is informative rather than confusing. The rest are
per-package `config.py`/`base.py`, which is ordinary Python.

The one this exists to catch is the kind that already happened: a name that
means DIFFERENT things in the two places, where a sentence naming the file is
ambiguous and a grep returns both.
"""
DUPLICATE_BASENAMES: dict[str, tuple[str, ...]] = {
    "app.py": ("interfaces/blueprint/app.py", "interfaces/web/app.py"),
    "base.py": ("engine/solver/storage/base.py", "pipeline/abstraction/base.py"),
    "cache.py": ("interfaces/web/cache.py", "shared/cache.py"),
    "config.py": (
        "interfaces/cloud/config.py",
        "pipeline/abstraction/config.py",
        "pipeline/evaluation/estimators/lbr/config.py",
        "shared/config.py",
    ),
    "paths.py": (
        "pipeline/abstraction/paths.py",
        "pipeline/blueprint/paths.py",
        "shared/cloudtask/node/paths.py",
    ),
    # command -> implementation, below.
    "precompute.py": (
        "interfaces/commands/precompute.py",
        "pipeline/abstraction/postflop/precompute.py",
    ),
    "progress.py": ("interfaces/commands/progress.py", "shared/cloudtask/node/progress.py"),
    "records.py": ("pipeline/evaluation/ledger/records.py", "shared/records.py"),
    "resolver.py": ("engine/search/resolver.py", "pipeline/abstraction/resolver.py"),
    "runs.py": ("interfaces/commands/runs.py", "pipeline/services/runs.py"),
    "serve_box.py": ("interfaces/cloud/serve_box.py", "interfaces/commands/serve_box.py"),
}


def _packages_with_subpackages() -> dict[str, tuple[str, ...]]:
    """Every package that holds sub-packages, mapped to its loose modules."""
    found = {}
    for directory in sorted(SRC.rglob("*")):
        if not directory.is_dir() or "__pycache__" in str(directory):
            continue
        has_subpackages = any(
            child.is_dir() and (child / "__init__.py").is_file() for child in directory.iterdir()
        )
        loose = tuple(
            sorted(
                child.name
                for child in directory.iterdir()
                if child.suffix == ".py" and child.name != "__init__.py"
            )
        )
        if has_subpackages and loose:
            found[str(directory.relative_to(SRC)).replace("/", ".")] = loose
    return found


def _modules_by_basename() -> dict[str, tuple[str, ...]]:
    by_name = collections.defaultdict(list)
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path) or path.name == "__init__.py":
            continue
        by_name[path.name].append(str(path.relative_to(SRC.parent).relative_to("src")))
    return {name: tuple(sorted(paths)) for name, paths in by_name.items() if len(paths) > 1}


class TestPeersAreFiledTheSameWay:
    def test_no_undeclared_loose_module_beside_a_subpackage(self):
        actual = _packages_with_subpackages()
        surprises = {
            package: sorted(set(loose) - set(LOOSE_BESIDE_PACKAGES.get(package, ())))
            for package, loose in actual.items()
        }
        surprises = {package: names for package, names in surprises.items() if names}
        assert not surprises, (
            f"loose module(s) beside sub-packages, undeclared: {surprises}\n"
            "If one is a PEER of something inside a sub-package, move it there -- that is "
            "the drift this catches (`public_tree_br.py` loose while `lbr/` was a package).\n"
            "If it genuinely belongs at this level, add it to LOOSE_BESIDE_PACKAGES with "
            "the reason. Declaring it is a decision, not a formality."
        )

    def test_the_declaration_names_no_module_that_moved_away(self):
        """The other direction. A declaration outliving its module is how the
        list stops describing the tree and starts being folklore."""
        actual = _packages_with_subpackages()
        stale = {
            package: sorted(set(declared) - set(actual.get(package, ())))
            for package, declared in LOOSE_BESIDE_PACKAGES.items()
        }
        stale = {package: names for package, names in stale.items() if names}
        assert not stale, (
            f"LOOSE_BESIDE_PACKAGES names module(s) that are no longer there: {stale}\n"
            "They were moved or deleted -- drop them from the list. Nothing is wrong "
            "with the tree; the list is behind it."
        )


class TestANameMeansOneThing:
    @pytest.mark.parametrize("basename", sorted(DUPLICATE_BASENAMES), ids=str)
    def test_a_declared_duplicate_still_lives_where_it_says(self, basename):
        actual = _modules_by_basename()
        assert actual.get(basename) == DUPLICATE_BASENAMES[basename], (
            f"'{basename}' is declared at {DUPLICATE_BASENAMES[basename]} "
            f"but is actually at {actual.get(basename)}"
        )

    def test_no_new_duplicate_basename(self):
        undeclared = {
            name: paths
            for name, paths in _modules_by_basename().items()
            if name not in DUPLICATE_BASENAMES
        }
        assert not undeclared, (
            f"module basename(s) now used more than once: {undeclared}\n"
            "Ask whether the two mean the SAME thing. A command and the service it "
            "invokes sharing a name is informative; two unrelated files sharing one "
            "means every sentence about either has to name the directory too.\n"
            "If it is deliberate, add it to DUPLICATE_BASENAMES with the reason."
        )
