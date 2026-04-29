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

from src.shared.node import archive


def _snapshot(run_dir, name: str, *files: str) -> None:
    """A zarr-shaped directory: nested chunk files, not one blob."""
    directory = run_dir / name
    (directory / "regrets" / "0").mkdir(parents=True, exist_ok=True)
    for index, content in enumerate(files or ("chunk",)):
        (directory / "regrets" / "0" / f"{index}").write_text(content)


def _manifest(run_dir, current: str, retained=(), iteration: int = 0) -> None:
    (run_dir / archive.MANIFEST).write_text(
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


class TestPublish:
    def test_a_snapshot_is_copied_whole_and_marked(self, tmp_path):
        run_dir = _run(tmp_path)
        _snapshot(run_dir, "static-1000.zarr", "a", "b")
        destination = tmp_path / "archive" / "run-a"

        assert archive.publish_run(run_dir, destination)

        assert (destination / "static-1000.zarr" / "regrets" / "0" / "0").read_text() == "a"
        assert (destination / "static-1000.zarr" / "regrets" / "0" / "1").read_text() == "b"
        assert (destination / archive.marker_for("static-1000.zarr")).exists()

    def test_the_manifest_lands_after_the_snapshot_it_names(self, tmp_path):
        """Publish it first and an interrupted copy leaves the share naming a
        rung that is only half there -- which a later fetch reads as whole."""
        run_dir = _run(tmp_path)
        _snapshot(run_dir, "static-1000.zarr")
        _manifest(run_dir, "static-1000.zarr")
        destination = tmp_path / "archive" / "run-a"

        order: list[str] = []
        real = archive.copy_file

        def record(source, target):
            order.append(target.name)
            real(source, target)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(archive, "copy_file", record)
            assert archive.publish_run(run_dir, destination)

        assert order[-1] == archive.MANIFEST, order

    def test_a_failed_snapshot_suppresses_the_manifest(self, tmp_path):
        """The whole point of the ordering: the share keeps describing the last
        checkpoint that fully copied, rather than one that did not."""
        run_dir = _run(tmp_path)
        _snapshot(run_dir, "static-1000.zarr")
        _manifest(run_dir, "static-1000.zarr")
        destination = tmp_path / "archive" / "run-a"

        def explode(source, target, **kwargs):
            raise OSError("share went away")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(archive, "copy_tree", explode)
            assert not archive.publish_run(run_dir, destination)

        assert not (destination / archive.MANIFEST).exists()
        assert not (destination / archive.marker_for("static-1000.zarr")).exists()

    def test_a_failed_snapshot_also_suppresses_the_static_manifest(self, tmp_path):
        """STATIC_CHECKPOINT.json once fell to the unguarded loose-file copy and
        was published even when a rung had failed."""
        run_dir = _run(tmp_path)
        _snapshot(run_dir, "static-1000.zarr")
        _manifest(run_dir, "static-1000.zarr")
        (run_dir / "metrics.jsonl").write_text("{}\n")
        destination = tmp_path / "archive" / "run-a"

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(archive, "copy_tree", _raise)
            archive.publish_run(run_dir, destination)

        assert (destination / "metrics.jsonl").exists(), "loose files still publish"
        assert not (destination / archive.MANIFEST).exists()

    def test_an_already_marked_snapshot_is_not_recopied(self, tmp_path):
        """Measured at 6.6 minutes re-uploading 809 MB already on the share: a
        resumed task's starting rung has a newer node mtime than the share copy,
        so the update rule alone would copy it every time."""
        run_dir = _run(tmp_path)
        _snapshot(run_dir, "static-1000.zarr")
        destination = tmp_path / "archive" / "run-a"
        archive.publish_run(run_dir, destination)

        published = destination / "static-1000.zarr" / "regrets" / "0" / "0"
        published.write_text("SHARE COPY -- must not be overwritten")
        os.utime(run_dir / "static-1000.zarr" / "regrets" / "0" / "0", (2e9, 2e9))

        assert archive.publish_run(run_dir, destination)
        assert published.read_text() == "SHARE COPY -- must not be overwritten"

    def test_a_growing_directory_keeps_publishing(self, tmp_path):
        """`evals/` is NOT write-once, so it must never take a marker -- doing
        so would freeze it at whatever it held on the first publish."""
        run_dir = _run(tmp_path)
        (run_dir / "evals").mkdir()
        (run_dir / "evals" / "first.json").write_text("{}")
        destination = tmp_path / "archive" / "run-a"
        archive.publish_run(run_dir, destination)

        (run_dir / "evals" / "second.json").write_text("{}")
        assert archive.publish_run(run_dir, destination)

        assert (destination / "evals" / "second.json").exists()
        assert not (destination / archive.marker_for("evals")).exists()

    def test_an_interrupted_snapshot_republishes_from_scratch(self, tmp_path):
        """No marker means the previous attempt was cut short, so the rung is
        copied again rather than trusted."""
        run_dir = _run(tmp_path)
        _snapshot(run_dir, "static-1000.zarr", "a", "b")
        destination = tmp_path / "archive" / "run-a"
        # What a kill mid-copy leaves: some files, no marker.
        (destination / "static-1000.zarr" / "regrets" / "0").mkdir(parents=True)
        (destination / "static-1000.zarr" / "regrets" / "0" / "0").write_text("truncated")

        assert archive.publish_run(run_dir, destination)

        assert (destination / "static-1000.zarr" / "regrets" / "0" / "0").read_text() == "a"
        assert (destination / "static-1000.zarr" / "regrets" / "0" / "1").read_text() == "b"
        assert (destination / archive.marker_for("static-1000.zarr")).exists()

    def test_publish_all_covers_every_run_on_the_disk(self, tmp_path):
        for name in ("run-a", "run-b"):
            _snapshot(_run(tmp_path, name), "static-10.zarr")
        assert archive.publish_all(tmp_path / "runs", tmp_path / "archive")
        assert (tmp_path / "archive" / "run-a" / "static-10.zarr").is_dir()
        assert (tmp_path / "archive" / "run-b" / "static-10.zarr").is_dir()

    def test_publish_all_tolerates_a_missing_runs_directory(self, tmp_path):
        assert archive.publish_all(tmp_path / "nothing", tmp_path / "archive")


def _raise(*args, **kwargs):
    raise OSError("share went away")


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
        (share / archive.MANIFEST).write_text(
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
        assert (node / archive.MANIFEST).exists()
        assert (node / ".run.json").exists()

    def test_an_unmarked_current_rung_is_refused(self, tmp_path):
        """Resuming from a truncated snapshot trains on garbage rather than
        failing; the marker is the only thing that can tell them apart."""
        share = self._published(tmp_path, marked=False)
        with pytest.raises(archive.FetchRefusedError, match="completion marker"):
            archive.fetch_current_rung(share, tmp_path / "runs" / "run-a")

    def test_a_manifest_naming_an_absent_rung_is_refused(self, tmp_path):
        share = self._published(tmp_path, current="static-9999.zarr")
        with pytest.raises(archive.FetchRefusedError, match="not on the share"):
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
        return share

    def test_only_the_named_rungs_come_down(self, tmp_path):
        share = self._published(tmp_path)
        node = tmp_path / "runs" / "run-a"

        assert archive.fetch_for_evaluation(share, node, ["1000", "3000"]) == ["1000", "3000"]
        assert (node / "static-1000.zarr" / "chunk").exists()
        assert (node / "static-3000.zarr" / "chunk").exists()
        assert not (node / "static-2000.zarr").exists()

    def test_an_unmarked_rung_is_skipped_not_fatal(self, tmp_path):
        """A partial curve beats none; the gap is visible in the task log and
        absent from the ledger."""
        share = self._published(tmp_path, unmarked=(2000,))
        node = tmp_path / "runs" / "run-a"
        lines: list[str] = []

        assert archive.fetch_for_evaluation(share, node, ["1000", "2000"], lines.append) == ["1000"]
        assert any("2000" in line and "completion marker" in line for line in lines)

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
        runs = tmp_path / "runs"
        run_dir = runs / "run-a"
        run_dir.mkdir(parents=True)
        _manifest(run_dir, "static-1000.zarr", iteration=1000)
        before = archive.ladder_state(runs)
        _manifest(run_dir, "static-2000.zarr", iteration=2000)
        assert archive.ladder_state(runs) != before

    def test_it_sees_a_run_that_did_not_exist_yet(self, tmp_path):
        """A fresh train's id appears only once the trainer creates it, so the
        watcher must poll the whole runs directory."""
        runs = tmp_path / "runs"
        runs.mkdir()
        assert archive.ladder_state(runs) == ""
        run_dir = runs / "run-new"
        run_dir.mkdir()
        _manifest(run_dir, "static-10.zarr", iteration=10)
        assert "run-new" in archive.ladder_state(runs)

    def test_a_torn_manifest_reads_as_absent(self, tmp_path):
        """Half-written JSON is the expected residue of a kill mid-checkpoint
        and must not take down the watcher that would publish the rest."""
        runs = tmp_path / "runs"
        run_dir = runs / "run-a"
        run_dir.mkdir(parents=True)
        (run_dir / archive.MANIFEST).write_text('{"zarr": ')
        assert archive.ladder_state(runs) == ""

    def test_a_missing_runs_directory_reads_as_empty(self, tmp_path):
        assert archive.ladder_state(tmp_path / "nothing") == ""
