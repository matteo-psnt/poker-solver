"""Answering "is this run loadable from local disk?", and saying so when not.

This module used to COPY one rung off the SMB share. The share holds nothing
since the move to blob, and its tests passed anyway because they built a share
in `tmp_path` -- a fixture proving copy logic that production could never reach.
What is pinned now is the contract that remains: local disk answers, and every
refusal names what a person should do next.
"""

from __future__ import annotations

import json

import pytest

from src.interfaces.blueprint.staging import MANIFEST, StagingError, stage_run

RUNG_FILES = 4
RUNGS = [1_000_000, 2_000_000, 3_000_000]


def _stage(root, *, head=3_000_000, complete=True):
    """A run on local disk, shaped like one `serve-deploy` leaves behind."""
    run = root / "a-run"
    zarr = run / f"static-{head}.zarr"
    zarr.mkdir(parents=True)
    for index in range(RUNG_FILES):
        (zarr / f"chunk-{index}").write_text(f"{head}-{index}")
    if complete:
        (run / f".complete-static-{head}.zarr").write_text("")
    (run / "run.jsonl").write_text('{"event": "started"}\n')
    (run / MANIFEST).write_text(
        json.dumps(
            {
                "iteration": 3_000_000,
                "zarr": "static-3000000.zarr",
                "retained": [{"iteration": r, "zarr": f"static-{r}.zarr"} for r in RUNGS[:-1]],
            }
        )
    )
    return run


class TestWhatIsOnDisk:
    def test_a_staged_run_resolves_to_its_directory(self, tmp_path):
        runs = tmp_path / "runs"
        _stage(runs)
        assert stage_run("a-run", runs_dir=runs) == runs / "a-run"

    def test_a_named_rung_resolves_when_it_is_present(self, tmp_path):
        runs = tmp_path / "runs"
        _stage(runs, head=1_000_000)
        assert stage_run("a-run", runs_dir=runs, at_iteration=1_000_000) == runs / "a-run"


class TestEveryRefusalNamesTheNextStep:
    def test_a_run_the_box_does_not_hold(self, tmp_path):
        """The message the share used to get wrong: it answered "not published"
        about runs that ARE published, because it read an empty mount."""
        with pytest.raises(StagingError, match=r"serve-deploy"):
            stage_run("a-run", runs_dir=tmp_path / "runs")

    def test_a_rung_the_manifest_does_not_list(self, tmp_path):
        runs = tmp_path / "runs"
        _stage(runs)
        with pytest.raises(StagingError, match=r"has: 1000000, 2000000, 3000000"):
            stage_run("a-run", runs_dir=runs, at_iteration=9_999)

    def test_a_rung_listed_but_half_copied(self, tmp_path):
        """A truncated snapshot fails deep in the loader minutes later, so it is
        refused here where the cause is still legible."""
        runs = tmp_path / "runs"
        _stage(runs, complete=False)
        with pytest.raises(StagingError, match=r"not complete on disk"):
            stage_run("a-run", runs_dir=runs)
