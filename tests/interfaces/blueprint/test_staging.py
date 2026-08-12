"""Putting one run on local disk, copying as little as will do.

The layout these fixtures imitate is the real one, because the bug they exist
for was a misreading of it: a published run holds its whole LADDER of
checkpoints, and the server loads exactly one of them.
"""

from __future__ import annotations

import json

import pytest

from src.interfaces.blueprint.staging import MANIFEST, StagingError, stage_run

RUNG_FILES = 4


@pytest.fixture
def share(tmp_path):
    """A published run with three rungs, shaped like the ones on the share."""
    published = tmp_path / "share" / "archive" / "a-run"
    rungs = [1_000_000, 2_000_000, 3_000_000]
    for rung in rungs:
        zarr = published / f"static-{rung}.zarr"
        zarr.mkdir(parents=True)
        for index in range(RUNG_FILES):
            (zarr / f"chunk-{index}").write_text(f"{rung}-{index}")
        (published / f".complete-static-{rung}.zarr").write_text("")
    (published / "run.jsonl").write_text('{"event": "started"}\n')
    (published / MANIFEST).write_text(
        json.dumps(
            {
                "iteration": 3_000_000,
                "zarr": "static-3000000.zarr",
                "retained": [{"iteration": r, "zarr": f"static-{r}.zarr"} for r in rungs[:-1]],
            }
        )
    )
    return tmp_path / "share"


class TestItStagesOneRung:
    def test_the_head_checkpoint_and_nothing_else(self, tmp_path, share):
        """The whole point: the ladder stays on the share.

        Copying the run directory moved ~127 GB in 400,000 files to load one
        850 MB checkpoint — about six hours at the box's measured throughput.
        """
        run_dir = stage_run("a-run", runs_dir=tmp_path / "runs", share=share)

        staged = {path.name for path in run_dir.glob("static-*.zarr")}
        assert staged == {"static-3000000.zarr"}
        assert (run_dir / MANIFEST).is_file()
        assert (run_dir / "run.jsonl").is_file()

    def test_a_named_rung_instead_of_the_head(self, tmp_path, share):
        run_dir = stage_run(
            "a-run", runs_dir=tmp_path / "runs", share=share, at_iteration=1_000_000
        )
        staged = {path.name for path in run_dir.glob("static-*.zarr")}
        assert staged == {"static-1000000.zarr"}

    def test_a_rung_that_does_not_exist_says_which_do(self, tmp_path, share):
        with pytest.raises(StagingError, match="1000000, 2000000, 3000000"):
            stage_run("a-run", runs_dir=tmp_path / "runs", share=share, at_iteration=9_999)


class TestItDoesNotRecopy:
    def test_a_second_stage_reads_nothing_from_the_share(self, tmp_path, share):
        """The repeat switch, which has to be fast or the button is a lie."""
        runs_dir = tmp_path / "runs"
        stage_run("a-run", runs_dir=runs_dir, share=share)

        # Removing the share entirely is the strongest possible assertion that
        # the second call did not touch it.
        for path in sorted(share.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()

        run_dir = stage_run("a-run", runs_dir=runs_dir, share=share)
        assert (run_dir / "static-3000000.zarr").is_dir()

    def test_an_interrupted_stage_is_not_mistaken_for_a_finished_one(self, tmp_path, share):
        """A manifest beside a half-copied checkpoint must not read as staged.

        This is what a box that deallocated mid-copy leaves behind, and treating
        it as done would serve a truncated checkpoint as a run.
        """
        runs_dir = tmp_path / "runs"
        run_dir = stage_run("a-run", runs_dir=runs_dir, share=share)
        (run_dir / ".complete-static-3000000.zarr").unlink()

        # It re-stages rather than returning early, and lands complete.
        again = stage_run("a-run", runs_dir=runs_dir, share=share)
        assert (again / ".complete-static-3000000.zarr").is_file()


class TestRefusals:
    def test_an_unknown_run_says_so(self, tmp_path, share):
        with pytest.raises(StagingError, match="No published run"):
            stage_run("not-a-run", runs_dir=tmp_path / "runs", share=share)

    def test_no_share_and_no_local_copy_is_its_own_message(self, tmp_path):
        """Different problem, different fix — a laptop has no share at all."""
        with pytest.raises(StagingError, match="share is not mounted"):
            stage_run("a-run", runs_dir=tmp_path / "runs", share=tmp_path / "nope")

    def test_a_run_with_no_manifest_is_refused(self, tmp_path, share):
        published = share / "archive" / "legacy"
        published.mkdir(parents=True)
        (published / "run.jsonl").write_text("{}\n")
        with pytest.raises(StagingError, match="retired dynamic backend"):
            stage_run("legacy", runs_dir=tmp_path / "runs", share=share)
