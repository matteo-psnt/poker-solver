"""The publish/fetch rules, each of which cost a run to learn.

This logic lived in ~250 lines of `cp -ru`/`find` inside run_task.sh and had no
test at all: it ran only on a Batch node, and every regression in it was
discovered as a corrupt checkpoint hours later. The cases below are the
production failures its comments describe, reproduced on a tmp_path.
"""

from __future__ import annotations

import json

import pytest

from src.shared import records
from src.shared.cloudtask.node import archive
from tests.shared.cloudtask.node.conftest import SAS


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


class TestFetchCurrentRung:
    """The container is the only store, so a rung is one OBJECT, not a tree."""

    def _published(self, container, *, current: str = "static-2000.zarr"):
        for name in ("static-1000.ckpt.zst", "static-2000.ckpt.zst"):
            container[f"run-a/{name}"] = name.encode()
        container["run-a/.run.json"] = b"{}"
        container["run-a/" + records.STATIC_CHECKPOINT] = json.dumps(
            {"zarr": current, "iteration": 2000, "retained": []}
        ).encode()

    def test_only_the_current_rung_comes_down(self, tmp_path, container):
        """The ladder stays in the container: taking all 31 rungs was ~25 GB and
        ~40 minutes to load the 809 MB the trainer actually reads."""
        self._published(container)
        node = tmp_path / "runs" / "run-a"

        archive.fetch_current_rung("run-a", node, SAS)

        assert (node / "static-2000.ckpt.zst").exists()
        assert not (node / "static-1000.ckpt.zst").exists()
        assert (node / records.STATIC_CHECKPOINT).exists()
        assert (node / ".run.json").exists(), "loose metadata comes with it"

    def test_a_manifest_naming_an_absent_rung_is_refused(self, tmp_path, container):
        self._published(container, current="static-9999.zarr")
        with pytest.raises(archive.FetchRefusedError, match="does not hold it"):
            archive.fetch_current_rung("run-a", tmp_path / "runs" / "run-a", SAS)

    def test_a_manifest_naming_nothing_is_refused(self, tmp_path, container):
        self._published(container, current="")
        with pytest.raises(archive.FetchRefusedError, match="no current snapshot"):
            archive.fetch_current_rung("run-a", tmp_path / "runs" / "run-a", SAS)

    def test_a_dynamic_backend_run_is_refused_by_name(self, tmp_path, container):
        """Its checkpoints are unreadable at HEAD by design. Fetching them
        would buy a confusing failure several minutes deeper.

        A LEGACY manifest with no static one beside it is the whole signal, and
        it is only looked for in that case -- the common path pays nothing."""
        container["old-run/" + archive.LEGACY_MANIFEST] = b'{"zarr": "checkpoint-500.zarr"}'

        with pytest.raises(archive.FetchRefusedError, match="dynamic backend"):
            archive.fetch_current_rung("old-run", tmp_path / "runs" / "old-run", SAS)

    def test_a_run_with_no_manifest_starts_the_ladder(self, tmp_path, container):
        """A task that died before its first checkpoint published .run.json and
        nothing else. Refusing that would strand the run id forever."""
        container["run-a/.run.json"] = b"{}"
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_current_rung("run-a", node, SAS) == ""
        assert (node / ".run.json").exists()

    def test_a_marker_object_is_not_carried_down(self, tmp_path, container):
        """Presence is completeness now, so nothing writes one -- but a marker
        left over from the share era must not land on the node, where a later
        publish could read it as "already uploaded"."""
        self._published(container)
        container["run-a/" + archive.marker_for("static-2000.ckpt.zst")] = b""
        node = tmp_path / "runs" / "run-a"

        archive.fetch_current_rung("run-a", node, SAS)

        assert not list(node.glob(archive.MARKER_PREFIX + "*"))


class TestFetchForEvaluation:
    def _published(self, container, rungs=(1000, 2000, 3000)):
        for rung in rungs:
            container[f"run-a/static-{rung}.ckpt.zst"] = f"static-{rung}".encode()
        # The manifest is what NAMES each rung; without one there is nothing to
        # resolve an iteration to an object, which is the point of the lookup.
        container["run-a/" + records.STATIC_CHECKPOINT] = json.dumps(
            {
                "zarr": f"static-{rungs[-1]}.zarr",
                "iteration": rungs[-1],
                "retained": [{"iteration": r, "zarr": f"static-{r}.zarr"} for r in rungs],
            }
        ).encode()

    def test_only_the_named_rungs_come_down(self, tmp_path, container):
        self._published(container)
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_for_evaluation("run-a", node, ["1000", "3000"], SAS) == [
            "1000",
            "3000",
        ]
        assert (node / "static-1000.ckpt.zst").exists()
        assert (node / "static-3000.ckpt.zst").exists()
        assert not (node / "static-2000.ckpt.zst").exists()

    def test_a_missing_rung_is_skipped_not_fatal(self, tmp_path, container):
        """A partial curve beats none."""
        self._published(container)
        node = tmp_path / "runs" / "run-a"
        assert archive.fetch_for_evaluation("run-a", node, ["9999"], SAS) == []

    def test_a_partial_node_copy_is_replaced_not_merged(self, tmp_path, container):
        """Rung 10000000: "fetched" in one second, then a read error. `cp -u`
        had treated a cancelled task's leftovers as already present."""
        self._published(container)
        node = tmp_path / "runs" / "run-a"
        node.mkdir(parents=True)
        (node / "static-1000.ckpt.zst").write_text("truncated")

        archive.fetch_for_evaluation("run-a", node, ["1000"], SAS)

        assert (node / "static-1000.ckpt.zst").read_text() == "static-1000"


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
    new format reported every rung missing while its bytes sat untouched.
    """

    def _published(self, container, name: str):
        container[f"run-a/{records.object_name(name)}"] = b"the rung"
        container["run-a/" + records.STATIC_CHECKPOINT] = json.dumps(
            {"zarr": name, "iteration": 1000, "retained": []}
        ).encode()

    def test_a_repointed_manifest_still_resolves_its_rung(self, tmp_path, container):
        self._published(container, f"static-1000{records.SNAPSHOT_SUFFIX}")
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_for_evaluation("run-a", node, ["1000"], SAS) == ["1000"]
        assert (node / f"static-1000{records.SNAPSHOT_SUFFIX}").read_text() == "the rung"

    def test_a_manifest_still_spelling_zarr_resolves_the_object(self, tmp_path, container):
        """The other direction, and the one 1,081 migrated rungs depend on."""
        self._published(container, "static-1000.zarr")
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_for_evaluation("run-a", node, ["1000"], SAS) == ["1000"]
        assert (node / "static-1000.ckpt.zst").read_text() == "the rung"

    def test_a_rung_the_manifest_does_not_name_is_skipped_and_said_so(self, tmp_path, container):
        """Not guessed at. A rung outside the manifest has no published bytes
        under any spelling, so inventing one only moves the failure later."""
        self._published(container, "static-1000.zarr")
        node = tmp_path / "runs" / "run-a"
        lines: list[str] = []

        assert archive.fetch_for_evaluation("run-a", node, ["9999"], SAS, lines.append) == []
        assert any("9999" in line and "manifest names no snapshot" in line for line in lines)


class TestTheObjectNameIsWhatIsAskedFor:
    """The manifest names `static-N.zarr`; the container holds
    `static-N.ckpt.zst`. Every lookup passed the manifest's spelling straight
    through, so all 1,081 migrated objects 404ed.
    """

    def test_completeness_is_asked_of_the_object(self, tmp_path, monkeypatch):
        asked: list[str] = []
        monkeypatch.setattr(
            archive.blobstore,
            "exists",
            lambda _s, name: (asked.append(name), True)[1],
        )
        archive.require_complete("run-a", "static-1000.zarr", SAS)
        assert asked == ["run-a/static-1000.ckpt.zst"]

    def test_a_rung_the_container_lacks_is_refused(self, tmp_path, container):
        """Presence IS completeness: there is no marker and no second store."""
        with pytest.raises(archive.FetchRefusedError, match="does not hold it"):
            archive.require_complete("run-a", "static-1000.zarr", SAS)

    def test_the_fetch_lands_the_object_under_the_object_name(self, tmp_path, container):
        container["run-a/static-1000.ckpt.zst"] = b"the object"
        node = tmp_path / "runs" / "run-a"

        archive.fetch_snapshot("run-a", node, "static-1000.zarr", SAS)

        assert (node / "static-1000.ckpt.zst").read_text() == "the object"

    def test_a_stale_copy_of_the_other_spelling_is_cleared(self, tmp_path, container):
        """A cancelled task leaves a partial rung under whichever name it was
        fetching. Removing only the name asked for leaves the other beside the
        one that just landed, and the loader's fallback picks it up."""
        container["run-a/static-1000.ckpt.zst"] = b"fresh"
        node = tmp_path / "runs" / "run-a"
        (node / "static-1000.zarr").mkdir(parents=True)
        (node / "static-1000.zarr" / "chunk").write_text("from a dead attempt")

        archive.fetch_snapshot("run-a", node, "static-1000.zarr", SAS)

        assert not (node / "static-1000.zarr").exists()
        assert (node / "static-1000.ckpt.zst").read_text() == "fresh"

    def test_a_rung_that_is_not_there_refuses_rather_than_landing_nothing(
        self, tmp_path, container
    ):
        """There is no share to fall through to, so a silent no-op would leave
        the loader to discover the absence minutes later."""
        with pytest.raises(archive.FetchRefusedError, match="does not hold"):
            archive.fetch_snapshot("run-a", tmp_path / "runs" / "run-a", "static-1.zarr", SAS)
