"""Preflop chart generation for CLI."""

import subprocess
import webbrowser
from pathlib import Path

from src.actions.betting_actions import BettingActions
from src.bucketing.base import BucketingStrategy
from src.cli.flows.chart.server import ChartServer
from src.cli.ui import prompts, ui
from src.cli.ui.context import CliContext
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.in_memory import InMemoryStorage
from src.training.run_tracker import RunTracker
from src.utils.config import Config


class _DummyCardAbstraction(BucketingStrategy):
    """Placeholder abstraction for viewing charts (not used)."""

    def get_bucket(self, hole_cards, board, street):
        return 0

    def num_buckets(self, street):
        return 1


def view_preflop_chart(ctx: CliContext) -> None:
    """Handle viewing preflop strategy chart."""
    runs = RunTracker.list_runs(ctx.runs_dir)

    if not runs:
        ui.error(f"No trained runs found in {ctx.runs_dir}")
        ui.pause()
        return

    selected_run = prompts.select(
        ctx,
        "Select run to visualize:",
        choices=runs + ["Cancel"],
    )

    if selected_run == "Cancel" or selected_run is None:
        return

    if not _ensure_ui_build(ctx):
        ui.pause()
        return

    print(f"\nLoading solver from {selected_run}...")
    run_dir = ctx.runs_dir / selected_run
    config = _load_run_config(run_dir)
    # Use read-only InMemoryStorage for viewing charts
    storage = InMemoryStorage(checkpoint_dir=run_dir)

    print(f"  Loaded {storage.num_infosets():,} infosets")

    if storage.num_infosets() == 0:
        ui.error("No strategy data found in this run")
        ui.pause()
        return

    # For viewing charts, we don't need the full solver - just the storage
    # But MCCFRSolver requires action/card abstraction, so provide minimal instances
    action_abs = BettingActions(config.action_abstraction, big_blind=config.game.big_blind)

    solver = MCCFRSolver(
        action_abstraction=action_abs,
        card_abstraction=_DummyCardAbstraction(),
        storage=storage,
        config=config.merge({"system": {"seed": 42}}),
    )

    print("\nStarting chart viewer...")

    server = ChartServer(
        solver=solver,
        run_id=selected_run,
        base_dir=ctx.base_dir,
    )

    try:
        server.start()
    except OSError as exc:
        ui.error(f"Unable to start chart server: {exc}")
        ui.pause()
        return

    print(f"  Viewer running at {server.base_url}")
    print("  Opening in browser...")

    webbrowser.open(server.base_url)

    try:
        ui.pause("\nPress Enter to stop the chart viewer and return to the menu...")
    finally:
        server.stop()


def _ensure_ui_build(ctx: CliContext) -> bool:
    ui_dist = ctx.base_dir / "ui" / "dist" / "index.html"
    if ui_dist.exists():
        return True

    should_build = prompts.confirm(
        ctx,
        "UI build not found. Build it now?",
        default=True,
    )

    if not should_build:
        ui.info("UI build is required to open the viewer.")
        print("       Run: cd ui && npm install && npm run build")
        return False

    print("\nBuilding UI...")
    ui_dir = ctx.base_dir / "ui"

    try:
        subprocess.run(
            ["npm", "install"],
            cwd=ui_dir,
            check=True,
        )
        subprocess.run(
            ["npm", "run", "build"],
            cwd=ui_dir,
            check=True,
        )
        return True
    except FileNotFoundError:
        ui.error("npm was not found. Install Node.js and try again.")
    except subprocess.CalledProcessError as exc:
        ui.error(f"UI build failed (exit code {exc.returncode}).")

    return False


def _load_run_config(run_dir: Path) -> Config:
    tracker = RunTracker.load(run_dir)
    return tracker.metadata.config
