"""A README that names a module which no longer exists is worse than no README.

Doc rot here is not cosmetic. `src/pipeline/training/README.md` spent months
documenting `TrainingSession`, `ARRAY_SPECS`, an ownership map keyed on
`xxhash(infoset_key)` and a `trainer/` package -- the entire deleted dynamic
backend -- while the code had moved to static enumeration. A reader following
it would have built the one thing the project no longer does. Nothing detected
that, because a stale path is still valid prose.

What is checked: a backticked token that ends in `.py` or `.md` and contains a
`/` must resolve, either from the repo root or relative to the document that
names it. Both spellings appear in these docs and both are unambiguous.

What is deliberately NOT checked, so the failure above stays honest about its
own reach:

* a bare filename with no directory (``metrics.py``). It cannot be resolved
  without guessing which package is meant, and guessing produces false alarms
  on test fixtures named in prose.
* anything with a brace or a space -- ``src/interfaces/cli/{app,flows,ui}``
  names three deleted paths ON PURPOSE, and a checker that flagged it would
  punish the docs for recording history.
* tokens without a code extension. ``data/cache`` is named precisely because it
  must NOT exist, and `tests/shared/test_cache.py` is what enforces that.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from src.shared import repo

REPO_ROOT = repo.ROOT
REFERENCE = re.compile(r"`([^`\n]+?\.(?:py|md))`")


def _tracked_markdown() -> list[pathlib.Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.split()
    return [REPO_ROOT / name for name in listed]


DOCS = _tracked_markdown()


def _unresolved(document: pathlib.Path) -> list[str]:
    dead = []
    for token in REFERENCE.findall(document.read_text()):
        if "{" in token or " " in token or "/" not in token:
            continue
        if not (REPO_ROOT / token).exists() and not (document.parent / token).exists():
            dead.append(token)
    return dead


def test_there_are_documents_to_check():
    """Guards the guard: `git ls-files` returning nothing must not read as pass."""
    assert len(DOCS) >= 5, f"only found {[d.name for d in DOCS]}"
    assert any(d.name == "AGENTS.md" for d in DOCS)


@pytest.mark.parametrize("document", DOCS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_every_path_a_document_names_exists(document):
    dead = _unresolved(document)
    assert not dead, (
        f"{document.relative_to(REPO_ROOT)} names paths that do not exist: {dead}. "
        "Update the document -- or delete the reference if the thing is gone."
    )
