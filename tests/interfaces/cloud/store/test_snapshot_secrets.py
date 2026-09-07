"""A credential in the working tree must not ride a dispatch up to the share.

A code snapshot seals the WORKING TREE, not the index, so `.gitignore` protects
nothing here: an untracked `.env` or `chipzen.toml` in the root is tarballed by
`submit`/`score` like any other file and lands in `code/` on the share, readable
by every node that ever extracts it.

Measured 2026-08-24 while wiring the Chipzen seat, which needs a `cz_extbot_`
token: before this, both names sailed straight through `_snapshot_filter`. The
test builds a real tarball rather than asserting against the exclusion set,
because the set is the mechanism and the tarball is the thing that leaves.
"""

from __future__ import annotations

import tarfile

import pytest

from src.interfaces.cloud.store.blob import build_code_snapshot

SECRETS = (".env", ".env.local", "chipzen.toml")


@pytest.fixture
def sealed(tmp_path):
    """A tree carrying one of every credential file, sealed into a tarball."""
    root = tmp_path / "tree"
    (root / "src").mkdir(parents=True)
    (root / "src" / "solver.py").write_text("# real code\n")
    for name in SECRETS:
        (root / name).write_text("token = 'cz_extbot_deadbeef'\n")
    (root / ".chipzen").mkdir()
    (root / ".chipzen" / "chipzen.toml").write_text("token = 'cz_extbot_deadbeef'\n")

    destination = tmp_path / "snapshot.tar.gz"
    build_code_snapshot(root, destination)
    with tarfile.open(destination) as archive:
        return set(archive.getnames())


@pytest.mark.parametrize("name", SECRETS)
def test_a_credential_file_never_enters_the_tarball(name, sealed):
    assert name not in sealed


def test_a_credential_nested_in_a_dot_chipzen_dir_is_dropped_too(sealed):
    assert not any(entry.startswith(".chipzen") for entry in sealed)


def test_no_entry_anywhere_carries_a_token(tmp_path):
    """The blunt version of the check above: nothing in the archive holds one."""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "keep.py").write_text("x = 1\n")
    (root / ".env").write_text("CHIPZEN_EXTBOT_TOKEN=cz_extbot_deadbeef\n")
    destination = tmp_path / "snapshot.tar.gz"
    build_code_snapshot(root, destination)

    with tarfile.open(destination) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            assert extracted is not None
            assert b"cz_extbot_" not in extracted.read()


def test_ordinary_source_still_ships(sealed):
    """The exclusion must not have eaten the tree it is protecting."""
    assert "src/solver.py" in sealed
