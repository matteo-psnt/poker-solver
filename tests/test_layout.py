"""Where files sit, asserted rather than intended.

`.importlinter` enforces the DIRECTION of every import and does it well -- nine
contracts, and the layer graph is clean. It cannot see any of the things
below, because none of them is an edge:

- whether a set of PEERS is filed consistently,
- whether two files share a name,
- whether `tests/` mirrors the src layout it claims to, and
- whether a module path spelled inside a STRING still names a real module.

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

import ast
import collections
import importlib
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = SRC.parent


"""Loose modules that sit beside sub-packages, and why each set is allowed.

A package that holds sub-packages AND loose modules is not automatically wrong
-- `shared/` is a flat utility package with two subsystems in it, and that is
the right shape. It is wrong when a loose module is a PEER of something filed inside
a sub-package, because then the same kind of thing is filed two ways and the
directory stops answering what it looks like it answers.

That distinction needs a person, so this asks for one: the set is pinned, and a
new loose module fails until someone writes down which case it is.
"""
LOOSE_BESIDE_PACKAGES: dict[str, tuple[str, ...]] = {
    # Each is a single thing with no siblings to be filed away from.
    "engine.solver": ("betting_tree.py", "numba_ops.py", "protocols.py"),
    # Leaf helpers with no owner among the surfaces; `errors` is imported by all
    # of them and belongs above each, and `telemetry` observes the command seam
    # that `cli/` and `web/` both go through -- so it is above both for the same
    # reason, not a peer of either.
    "interfaces": ("errors.py", "run_names.py", "telemetry.py"),
    # `config` is the credentials and resource ids all three sub-packages need,
    # so it sits above them; `serve_box` provisions the play-server VM and is
    # its own job, sharing nothing with dispatching work or reading the bill.
    "interfaces.cloud": ("config.py", "serve_box.py"),
    # Shared across postflop/ and preflop/, which is why they are not in either.
    # `vector_universe` is the odd one: it CONSUMES a finished abstraction
    # rather than building one, bridging the artifact to the vector kernels'
    # `HandContext`. It is in `pipeline/` only because import-linter forbids
    # `engine/` from reaching an abstraction directly, so it is a peer of
    # `resolver.py` — also a consumer — and of nothing inside postflop/ or
    # preflop/.
    "pipeline.abstraction": (
        "base.py",
        "config.py",
        "paths.py",
        "resolver.py",
        "vector_universe.py",
    ),
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
    # that anything may import exactly what it needs, and each of these has 3-6
    # importers spanning two or more LAYERS. That fan-out is the test: a module
    # only two things use, both of them in one package, is that package's detail
    # (`dicts.py` was, and moved into `config/`). Two subsystems: `config/`,
    # and `cloudtask/` because the node imports it as a closure.
    "shared": (
        "action_tokens.py",
        "cache.py",
        "gitinfo.py",
        "jsonio.py",
        "log.py",
        "numeric.py",
        "records.py",
        "repo.py",
        "run_events.py",
        # The READING half of the task record, out here rather than beside the
        # writer in `cloudtask/` precisely because it is not node code: a
        # fail-closed guard walks every file in that package and would hold a
        # laptop-only join to the node's stdlib-only floor. Seven importers
        # across `interfaces` and `pipeline`, which is the fan-out test above.
        "task_history.py",
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


"""Test directories that deliberately mirror no src directory."""
TEST_ONLY_DIRS: dict[str, str] = {
    "integration": "cross-layer smoke tests; their subject is the seams, not one package",
}


"""Loose test modules whose subject imports all land in ONE sub-package of the
mirrored src package, but which stay loose anyway. Empty on purpose: when the
src regroup happened, twenty-seven test files were left filed the old way, and
nothing said so until a manual sweep. A test that trips the rule either moves
into the sub-package it tests, or gets declared here with the reason it is
genuinely about the parent."""
CROSS_CUTTING_TESTS: dict[str, str] = {}


"""Strings that look like module paths but are not, and why each is allowed.

A dotted path in an import statement fails loudly the moment the module moves.
The same path inside a STRING -- a monkeypatch target, a subprocess program, a
logger name -- fails silently or late: after the cloud split, seven files still
cited pre-move paths and nothing complained. The test below resolves every
`src.`/`tests.` dotted string against the tree, so a move breaks in CI instead
of in a docstring nobody rereads."""
NOT_MODULE_PATHS: dict[str, str] = {
    "src.pipeline.demo": "a fake logger name in test_log.py, module-shaped on purpose",
}


def _test_dirs() -> list[pathlib.Path]:
    return [d for d in sorted(TESTS.rglob("*")) if d.is_dir() and "__pycache__" not in str(d)]


def _src_imports(path: pathlib.Path) -> set[str]:
    """Dotted module names a test imports from src, with `from pkg import mod`
    resolved to the module when the name IS a module rather than an attribute."""
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names if a.name.startswith("src."))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            for alias in node.names:
                full = f"{node.module}.{alias.name}"
                mods.add(full if _is_module(full) else node.module)
    return mods


def _is_module(dotted: str) -> bool:
    path = ROOT / dotted.replace(".", "/")
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


class TestTestsMirrorSrc:
    def test_every_test_directory_mirrors_a_src_directory(self):
        strays = [
            str(d.relative_to(TESTS))
            for d in _test_dirs()
            if not (SRC / d.relative_to(TESTS)).is_dir()
            and str(d.relative_to(TESTS)) not in TEST_ONLY_DIRS
        ]
        assert not strays, (
            f"test director(ies) mirroring no src directory: {strays}\n"
            "Either the src package moved and the tests did not follow, or the "
            "directory is deliberately test-only -- declare it in TEST_ONLY_DIRS."
        )

    def test_a_test_of_subpackage_code_is_filed_in_that_subpackage(self):
        misfiled = {}
        for directory in _test_dirs():
            rel = directory.relative_to(TESTS)
            srcdir = SRC / rel
            if not srcdir.is_dir():
                continue
            subpackages = {
                child.name
                for child in srcdir.iterdir()
                if child.is_dir() and (child / "__init__.py").is_file()
            }
            if not subpackages:
                continue
            mirrored = "src." + str(rel).replace("/", ".")
            for test in sorted(directory.glob("test_*.py")):
                key = str(test.relative_to(TESTS))
                if key in CROSS_CUTTING_TESTS:
                    continue
                subjects = {m for m in _src_imports(test) if m.startswith(mirrored + ".")}
                if not subjects:
                    continue
                hits = {m[len(mirrored) + 1 :].split(".")[0] for m in subjects}
                inside = hits & subpackages
                if hits == inside and len(inside) == 1:
                    misfiled[key] = f"{rel}/{inside.pop()}/"
        assert not misfiled, (
            f"test(s) filed loose but importing from exactly one sub-package: {misfiled}\n"
            "Move each into the sub-package it tests -- that is how the src regroup "
            "left twenty-seven tests behind. If one genuinely tests the parent "
            "package, declare it in CROSS_CUTTING_TESTS with the reason."
        )


MODULE_PATH_IN_STRING = re.compile(r"(?<![\w.])(?:src|tests)(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def _resolves(dotted: str) -> bool:
    """True if `dotted` names a module, or an attribute chain on one."""
    if _is_module(dotted):
        return True
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:cut])
        if _is_module(prefix):
            try:
                obj = importlib.import_module(prefix)
                for attr in parts[cut:]:
                    obj = getattr(obj, attr)
                return True
            except (ImportError, AttributeError):
                return False
    return False


class TestModulePathsInStringsResolve:
    @pytest.mark.timeout(20)
    def test_every_dotted_path_in_a_string_names_something_real(self):
        stale = collections.defaultdict(set)
        for base in (SRC, TESTS, ROOT / "infra"):
            for path in sorted(base.rglob("*.py")):
                if "__pycache__" in str(path):
                    continue
                for node in ast.walk(ast.parse(path.read_text())):
                    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                        continue
                    for match in MODULE_PATH_IN_STRING.findall(node.value):
                        if match in NOT_MODULE_PATHS or _resolves(match):
                            continue
                        stale[str(path.relative_to(ROOT))].add(match)
        stale = {file: sorted(paths) for file, paths in stale.items()}
        assert not stale, (
            f"dotted module path(s) in strings that resolve to nothing: {stale}\n"
            "The module moved and the string did not follow -- an import would have "
            "failed loudly; the string just went stale. Update it, or if it is "
            "deliberately not a module path, declare it in NOT_MODULE_PATHS."
        )
