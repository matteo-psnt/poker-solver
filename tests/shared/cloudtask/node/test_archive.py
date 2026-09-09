"""The publish/fetch rules, each of which cost a run to learn.

This logic lived in ~250 lines of `cp -ru`/`find` inside run_task.sh and had no
test at all: it ran only on a Batch node, and every regression in it was
discovered as a corrupt checkpoint hours later. The cases below are the
production failures its comments describe, reproduced on a tmp_path.
"""

from __future__ import annotations

import json
import os

import pytest

from src.shared import records
from src.shared.cloudtask.node import archive


def _snapshot(run_dir, name: str, *files: str) -> None:
    """A rung as the trainer writes one: ONE file, not a tree.

    `name` keeps whatever spelling the caller uses -- a manifest written before
    the format changed still says `static-N.zarr` and is never repointed, so
    both spellings reach these paths.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text("".join(files) or "chunk")


def _manifest(run_dir, current: str, retained=(), iteration: int = 0) -> None:
    (run_dir / records.STATIC_CHECKPOINT).write_text(
        json.dumps(
            {
                "zarr": current,
                "iteration": iteration,
                "retained": [{"iteration": r, "zarr": f"static-{r}.zarr"} for r in retained],
            }
        )
    )


def _run(tmp_path, name: str = "run-a"):
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / ".run.json").write_text('{"run_id": "run-a"}')
    return run_dir


class TestCopySemantics:
    def test_no_timestamp_is_preserved(self, tmp_path):
        """`cp --preserve=timestamps` fails on this mount AFTER copying the
        data, which suppressed the manifest and made a good publish look
        broken. shutil.copy2/copytree would reintroduce exactly that."""
        source, destination = tmp_path / "s", tmp_path / "d"
        source.mkdir()
        (source / "f").write_text("x")
        os.utime(source / "f", (1_000_000_000, 1_000_000_000))

        archive.copy_tree(source, destination)

        assert (destination / "f").stat().st_mtime != pytest.approx(1_000_000_000)

    def test_the_update_rule_skips_an_older_source(self, tmp_path):
        source, destination = tmp_path / "s.txt", tmp_path / "d.txt"
        source.write_text("old")
        destination.write_text("newer")
        os.utime(source, (1_000_000_000, 1_000_000_000))
        assert not archive.needs_copy(source, destination)

    def test_the_update_rule_copies_a_newer_source(self, tmp_path):
        source, destination = tmp_path / "s.txt", tmp_path / "d.txt"
        destination.write_text("old")
        os.utime(destination, (1_000_000_000, 1_000_000_000))
        source.write_text("new")
        assert archive.needs_copy(source, destination)

    def test_update_false_copies_regardless(self, tmp_path):
        """The fetch direction: a file already on the node is evidence of a
        cancelled task, not of a complete copy."""
        source, destination = tmp_path / "s", tmp_path / "d"
        source.mkdir()
        (source / "f").write_text("real")
        destination.mkdir()
        (destination / "f").write_text("truncated")
        os.utime(source / "f", (1_000_000_000, 1_000_000_000))

        archive.copy_tree(source, destination, update=False)
        assert (destination / "f").read_text() == "real"


class TestFetchCurrentRung:
    def _published(self, tmp_path, *, marked: bool = True, current: str = "static-2000.zarr"):
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        (share / ".run.json").write_text("{}")
        for name in ("static-1000.zarr", "static-2000.zarr"):
            (share / name / "regrets").mkdir(parents=True)
            (share / name / "regrets" / "0").write_text(name)
            if marked:
                (share / archive.marker_for(name)).write_text("")
        (share / records.STATIC_CHECKPOINT).write_text(
            json.dumps({"zarr": current, "iteration": 2000, "retained": []})
        )
        return share

    def test_only_the_current_rung_comes_down(self, tmp_path):
        """The ladder stays on the share: taking all 31 rungs was ~25 GB and
        ~40 minutes to load the 809 MB the trainer actually reads."""
        share = self._published(tmp_path)
        node = tmp_path / "runs" / "run-a"

        archive.fetch_current_rung(share, node)

        assert (node / "static-2000.zarr" / "regrets" / "0").exists()
        assert not (node / "static-1000.zarr").exists()
        assert (node / records.STATIC_CHECKPOINT).exists()
        assert (node / ".run.json").exists()

    def test_a_manifest_naming_an_absent_rung_is_refused(self, tmp_path):
        share = self._published(tmp_path, current="static-9999.zarr")
        with pytest.raises(archive.FetchRefusedError, match="no store holds it"):
            archive.fetch_current_rung(share, tmp_path / "runs" / "run-a")

    def test_a_manifest_naming_nothing_is_refused(self, tmp_path):
        share = self._published(tmp_path, current="")
        with pytest.raises(archive.FetchRefusedError, match="no current snapshot"):
            archive.fetch_current_rung(share, tmp_path / "runs" / "run-a")

    def test_a_dynamic_backend_run_is_refused_by_name(self, tmp_path):
        """Its checkpoints are unreadable at HEAD by design. Fetching them
        would buy a confusing failure several minutes deeper."""
        share = tmp_path / "archive" / "old-run"
        (share / "checkpoint-500.zarr").mkdir(parents=True)
        (share / archive.LEGACY_MANIFEST).write_text('{"zarr": "checkpoint-500.zarr"}')

        with pytest.raises(archive.FetchRefusedError, match="dynamic backend"):
            archive.fetch_current_rung(share, tmp_path / "runs" / "old-run")

    def test_a_run_with_no_manifest_starts_the_ladder(self, tmp_path):
        """A task that died before its first checkpoint published .run.json and
        nothing else. Refusing that would strand the run id forever."""
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        (share / ".run.json").write_text("{}")
        node = tmp_path / "runs" / "run-a"

        archive.fetch_current_rung(share, node)
        assert (node / ".run.json").exists()

    def test_markers_are_not_copied_onto_the_node(self, tmp_path):
        """They describe the SHARE's copy. Carrying them down would let a later
        publish skip a rung it never actually uploaded."""
        share = self._published(tmp_path)
        node = tmp_path / "runs" / "run-a"
        archive.fetch_current_rung(share, node)
        assert not list(node.glob(archive.MARKER_PREFIX + "*"))


class TestFetchForEvaluation:
    def _published(self, tmp_path, rungs=(1000, 2000, 3000), unmarked=()):
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        for rung in rungs:
            name = f"static-{rung}.zarr"
            (share / name).mkdir()
            (share / name / "chunk").write_text(name)
            if rung not in unmarked:
                (share / archive.marker_for(name)).write_text("")
        # The manifest is what NAMES each rung; without one there is nothing to
        # resolve an iteration to a file, which is the point of the lookup.
        _manifest(share, f"static-{rungs[-1]}.zarr", retained=rungs, iteration=rungs[-1])
        return share

    def test_only_the_named_rungs_come_down(self, tmp_path):
        share = self._published(tmp_path)
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_for_evaluation(share, node, ["1000", "3000"]) == ["1000", "3000"]
        assert (node / "static-1000.zarr" / "chunk").exists()
        assert (node / "static-3000.zarr" / "chunk").exists()
        assert not (node / "static-2000.zarr").exists()

    def test_a_missing_rung_is_skipped_not_fatal(self, tmp_path):
        share = self._published(tmp_path)
        node = tmp_path / "runs" / "run-a"
        assert archive.fetch_for_evaluation(share, node, ["9999"]) == []

    def test_a_partial_node_copy_is_replaced_not_merged(self, tmp_path):
        """Rung 10000000: "fetched" in one second, then a read error. `cp -u`
        had treated a cancelled task's leftovers as already present."""
        share = self._published(tmp_path)
        node = tmp_path / "runs" / "run-a"
        (node / "static-1000.zarr").mkdir(parents=True)
        (node / "static-1000.zarr" / "chunk").write_text("truncated")
        (node / "static-1000.zarr" / "orphan").write_text("from a dead attempt")

        archive.fetch_for_evaluation(share, node, ["1000"])

        assert (node / "static-1000.zarr" / "chunk").read_text() == "static-1000.zarr"
        assert not (node / "static-1000.zarr" / "orphan").exists()


class TestLadderState:
    def test_it_changes_when_the_current_snapshot_advances(self, tmp_path):
        """checkpoint_every below the retain interval advances `iteration`
        while the ladder stands still; watching only the ladder would sit idle
        through exactly those chunks."""
        run_dir = tmp_path / "runs" / "run-a"
        run_dir.mkdir(parents=True)
        _manifest(run_dir, "static-1000.zarr", iteration=1000)
        before = archive.ladder_state(run_dir)
        _manifest(run_dir, "static-2000.zarr", iteration=2000)
        assert archive.ladder_state(run_dir) != before

    def test_a_run_that_does_not_exist_yet_reads_as_empty_then_appears(self, tmp_path):
        """The id is fixed before the trainer starts; the directory is not."""
        run_dir = tmp_path / "runs" / "run-new"
        assert archive.ladder_state(run_dir) == ""
        run_dir.mkdir(parents=True)
        _manifest(run_dir, "static-10.zarr", iteration=10)
        assert "run-new" in archive.ladder_state(run_dir)

    def test_it_never_reads_a_neighbouring_run(self, tmp_path):
        """The reused-node bug: a watcher that read the runs directory pushed
        every run an earlier evaluate task had fetched there."""
        runs = tmp_path / "runs"
        mine, theirs = runs / "run-mine", runs / "run-theirs"
        mine.mkdir(parents=True)
        theirs.mkdir()
        _manifest(theirs, "static-5.zarr", iteration=5)
        assert archive.ladder_state(mine) == ""

    def test_a_torn_manifest_reads_as_absent(self, tmp_path):
        """Half-written JSON is the expected residue of a kill mid-checkpoint
        and must not take down the watcher that would publish the rest."""
        run_dir = tmp_path / "runs" / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / records.STATIC_CHECKPOINT).write_text('{"zarr": ')
        assert archive.ladder_state(run_dir) == ""

    def test_a_missing_run_directory_reads_as_empty(self, tmp_path):
        assert archive.ladder_state(tmp_path / "nothing") == ""


class TestTheManifestNamesTheRung:
    """`fetch_for_evaluation` built `static-<rung>.zarr` by hand.

    That is a second opinion about a name the manifest already holds, and it
    holds only while every snapshot is a zarr directory. A run repointed to the
    new format would have reported every rung missing while its bytes sat on
    the share untouched.
    """

    def _published(self, tmp_path, name: str):
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        (share / name).mkdir()
        (share / name / "chunk").write_text(name)
        (share / archive.marker_for(name)).write_text("")
        _manifest(share, name, iteration=1000)
        return share

    def test_a_repointed_manifest_still_resolves_its_rung(self, tmp_path):
        share = self._published(tmp_path, f"static-1000{records.SNAPSHOT_SUFFIX}")
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_for_evaluation(share, node, ["1000"]) == ["1000"]
        assert (node / f"static-1000{records.SNAPSHOT_SUFFIX}" / "chunk").exists()

    def test_a_rung_the_manifest_does_not_name_is_skipped_and_said_so(self, tmp_path):
        """Not guessed at. A rung outside the manifest has no published bytes
        under any spelling, so inventing one only moves the failure later."""
        share = self._published(tmp_path, "static-1000.zarr")
        node = tmp_path / "runs" / "run-a"
        lines: list[str] = []

        assert archive.fetch_for_evaluation(share, node, ["9999"], lines.append) == []
        assert any("9999" in line and "manifest names no snapshot" in line for line in lines)


class TestTheContainerIsActuallyReached:
    """The manifest names `static-N.zarr`; the container holds
    `static-N.ckpt.zst`. Every lookup passed the manifest's spelling straight
    through, so all 1,081 migrated objects 404ed and the fetch fell back to the
    share -- which worked, right up until the share was deleted.
    """

    def _share(self, tmp_path):
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        _manifest(share, "static-1000.zarr", iteration=1000)
        return share

    def test_the_object_name_is_what_is_asked_for(self, tmp_path, monkeypatch):
        asked: list[str] = []
        monkeypatch.setattr(
            archive.blobstore,
            "exists",
            lambda _s, name: (asked.append(name.split("/", 1)[1]), True)[1],
        )
        archive.require_complete(self._share(tmp_path), "static-1000.zarr", "sas")
        assert asked == ["static-1000.ckpt.zst"]

    def test_a_rung_in_the_container_needs_no_share_copy(self, tmp_path, monkeypatch):
        """The share holds no directory at all here; existence in the container
        IS completeness, which is the whole point of the flip."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: True)
        archive.require_complete(self._share(tmp_path), "static-1000.zarr", "sas")

    def test_the_fetch_pulls_the_object_and_lands_it_under_that_name(self, tmp_path, monkeypatch):
        share, node = self._share(tmp_path), tmp_path / "runs" / "run-a"

        def _get(_s, name, destination):
            name = name.split("/", 1)[1]
            (destination / name).write_text("the object")
            return True

        monkeypatch.setattr(archive.blobstore, "get_object", _get)
        archive.fetch_snapshot(share, node, "static-1000.zarr", "sas")

        assert (node / "static-1000.ckpt.zst").read_text() == "the object"

    def test_a_stale_copy_of_the_other_spelling_is_cleared(self, tmp_path, monkeypatch):
        """A cancelled task leaves a partial rung under whichever name it was
        fetching. Removing only the name asked for leaves the other beside the
        one that just landed, and the loader's fallback picks it up."""
        share, node = self._share(tmp_path), tmp_path / "runs" / "run-a"
        (node / "static-1000.zarr").mkdir(parents=True)
        (node / "static-1000.zarr" / "chunk").write_text("from a dead attempt")

        monkeypatch.setattr(
            archive.blobstore,
            "get_object",
            lambda _s, name, destination: (
                (destination / name.split("/", 1)[1]).write_text("fresh"),
                True,
            )[1],
        )
        archive.fetch_snapshot(share, node, "static-1000.zarr", "sas")

        assert not (node / "static-1000.zarr").exists()
        assert (node / "static-1000.ckpt.zst").read_text() == "fresh"

    def test_a_file_snapshot_on_the_share_is_fetched_as_a_file(self, tmp_path):
        """Published without a SAS, a rung lands on the share as one FILE.
        `is_dir()` refused it and `copy_tree` could not have copied it."""
        share, node = self._share(tmp_path), tmp_path / "runs" / "run-a"
        name = f"static-1000{records.SNAPSHOT_SUFFIX}"
        (share / name).write_text("one object")
        (share / archive.marker_for(name)).write_text("")

        archive.require_complete(share, name)
        archive.fetch_snapshot(share, node, name)

        assert (node / name).read_text() == "one object"
