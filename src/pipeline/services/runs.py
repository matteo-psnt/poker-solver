"""Reading the run directory: what runs exist, and what state each is in.

Every other service module starts here — training resumes a run, evaluation
scores one — so this holds the readers and nothing that mutates.
"""

from dataclasses import dataclass
from pathlib import Path

from src.engine.solver.storage.helpers import (
    CHECKPOINT_MANIFEST_FILE,
    read_checkpoint_manifest,
)
from src.pipeline.training.run_tracker import RunMetadata, RunTracker
from src.pipeline.training.versioning import REPRESENTATION_VERSION
from src.shared.gitinfo import commits_ahead_of


def checkpoint_iteration_of(run_dir: Path, at_iteration: int | None = None) -> int | None:
    """Iteration of the checkpoint an evaluator will actually score.

    Read from the manifest rather than run metadata: the manifest is committed in
    the same atomic replace as the arrays it names, so it describes exactly what an
    evaluator loading this directory will see. None for pre-manifest runs.

    With ``at_iteration`` the evaluator loads that ladder rung instead, so the
    reported iteration must be the rung -- reporting the published one would
    relabel every point of a convergence curve with the run's final iteration,
    which is the exact mislabelling this field exists to prevent.
    """
    if at_iteration is not None:
        return at_iteration
    manifest = read_checkpoint_manifest(run_dir)
    return int(manifest["iteration"]) if manifest is not None else None


def list_runs(runs_dir: Path) -> list[str]:
    """List available training runs in the provided base directory."""
    return RunTracker.list_runs(runs_dir)


@dataclass(frozen=True)
class RunSummary:
    """A run annotated with how old it is and whether it can still be loaded.

    ``commits_ago`` is the number of commits HEAD is ahead of the run's train
    commit (0 == trained on the current HEAD, None == commit unknown to this
    checkout). ``loadable`` is False when the run cannot be opened at HEAD --
    either it never checkpointed or its on-disk format predates the current
    ``REPRESENTATION_VERSION`` -- with ``blocker`` naming the reason for the UI.
    """

    name: str
    commits_ago: int | None
    git_dirty: bool | None
    representation_version: int | None
    current_version: int
    has_checkpoint: bool
    loadable: bool
    blocker: str | None
    # Descriptive metadata for the picker (None when metadata is unreadable).
    iterations: int | None
    num_infosets: int | None
    config_name: str | None
    status: str | None


def _has_checkpoint(run_dir: Path) -> bool:
    """Whether ``run_dir`` holds a checkpoint the loader can open.

    Covers both the versioned manifest and the legacy fixed-name/iteration-suffixed
    zarr layouts (``checkpoint.zarr`` / ``checkpoint-N.zarr``).
    """
    return (run_dir / CHECKPOINT_MANIFEST_FILE).exists() or any(run_dir.glob("checkpoint*.zarr"))


def _summarize_run(runs_dir: Path, name: str) -> RunSummary:
    run_dir = runs_dir / name
    try:
        metadata = load_run_metadata(run_dir)
    except (OSError, ValueError, KeyError):
        return RunSummary(
            name=name,
            commits_ago=None,
            git_dirty=None,
            representation_version=None,
            current_version=REPRESENTATION_VERSION,
            has_checkpoint=_has_checkpoint(run_dir),
            loadable=False,
            blocker="unreadable metadata",
            iterations=None,
            num_infosets=None,
            config_name=None,
            status=None,
        )

    has_checkpoint = _has_checkpoint(run_dir)
    version = metadata.representation_version
    if not has_checkpoint:
        blocker: str | None = "no checkpoint"
    elif version != REPRESENTATION_VERSION:
        blocker = f"format v{version} ≠ v{REPRESENTATION_VERSION}"
    else:
        blocker = None

    return RunSummary(
        name=name,
        commits_ago=commits_ahead_of(metadata.git_commit),
        git_dirty=metadata.git_dirty,
        representation_version=version,
        current_version=REPRESENTATION_VERSION,
        has_checkpoint=has_checkpoint,
        loadable=blocker is None,
        blocker=blocker,
        iterations=metadata.iterations,
        num_infosets=metadata.num_infosets,
        config_name=metadata.config_name,
        status=metadata.status,
    )


def describe_runs(runs_dir: Path) -> list[RunSummary]:
    """Summarize every run, newest first, with age and loadability annotations."""
    names = list_runs(runs_dir)
    return [_summarize_run(runs_dir, name) for name in reversed(names)]


def load_run_metadata(run_dir: Path) -> RunMetadata:
    """Load run metadata from an existing run directory."""
    tracker = RunTracker.load(run_dir)
    return tracker.metadata
