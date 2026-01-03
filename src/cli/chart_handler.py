"""Preflop chart generation for CLI."""

import json
import subprocess
import webbrowser
from pathlib import Path

import questionary

from src.actions.betting_actions import BettingActions
from src.bucketing.base import BucketingStrategy
from src.cli.chart_server import ChartServer
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.in_memory import InMemoryStorage
from src.training.run_tracker import RunTracker
from src.utils.config import Config


def handle_view_preflop_chart(
    runs_dir: Path,
    base_dir: Path,
    custom_style,
):
    """
    Handle viewing preflop strategy chart.

    Args:
        runs_dir: Directory containing training runs
        base_dir: Base project directory
        custom_style: Questionary style
    """

    runs = RunTracker.list_runs(runs_dir)

    if not runs:
        print("\n[ERROR] No trained runs found in data/runs/")
        input("Press Enter to continue...")
        return

    selected_run = questionary.select(
        "Select run to visualize:",
        choices=runs + ["Cancel"],
        style=custom_style,
    ).ask()

    if selected_run == "Cancel" or selected_run is None:
        return

    if not _ensure_ui_build(base_dir, custom_style):
        input("\nPress Enter to continue...")
        return

    print(f"\nLoading solver from {selected_run}...")
    run_dir = runs_dir / selected_run
    run_config = _load_run_config(run_dir)
    config = Config.from_dict(run_config) if run_config else Config.default()
    # Use read-only InMemoryStorage for viewing charts
    storage = InMemoryStorage(checkpoint_dir=run_dir)

    print(f"  Loaded {storage.num_infosets():,} infosets")

    if storage.num_infosets() == 0:
        print("\n[ERROR] No strategy data found in this run")
        input("Press Enter to continue...")
        return

    # For viewing charts, we don't need the full solver - just the storage
    # But MCCFRSolver requires action/card abstraction, so provide minimal instances
    action_abs = BettingActions(
        run_config.get("action_abstraction") if isinstance(run_config, dict) else None,
        big_blind=config.game.big_blind,
    )

    # Card abstraction not actually used for viewing stored strategies
    # Create a dummy implementation since MCCFRSolver requires it
    class DummyCardAbstraction(BucketingStrategy):
        """Placeholder abstraction for viewing charts (not used)."""

        def get_bucket(self, hole_cards, board, street):
            return 0

        def num_buckets(self, street):
            return 1

    solver = MCCFRSolver(
        action_abstraction=action_abs,
        card_abstraction=DummyCardAbstraction(),
        storage=storage,
        config=config.merge({"system": {"seed": 42}}),
    )

    print("\nStarting chart viewer...")

    server = ChartServer(
        solver=solver,
        run_id=selected_run,
        base_dir=base_dir,
    )

    try:
        server.start()
    except OSError as exc:
        print(f"\n[ERROR] Unable to start chart server: {exc}")
        input("Press Enter to continue...")
        return

    print(f"  Viewer running at {server.base_url}")
    print("  Opening in browser...")

    webbrowser.open(server.base_url)

    try:
        input("\nPress Enter to stop the chart viewer and return to the menu...")
    finally:
        server.stop()


def _ensure_ui_build(base_dir: Path, custom_style) -> bool:
    ui_dist = base_dir / "ui" / "dist" / "index.html"
    if ui_dist.exists():
        return True

    should_build = questionary.confirm(
        "UI build not found. Build it now?",
        default=True,
        style=custom_style,
    ).ask()

    if not should_build:
        print("\n[INFO] UI build is required to open the viewer.")
        print("       Run: cd ui && npm install && npm run build")
        return False

    print("\nBuilding UI...")
    ui_dir = base_dir / "ui"

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
        print("\n[ERROR] npm was not found. Install Node.js and try again.")
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] UI build failed (exit code {exc.returncode}).")

    return False


def _load_run_config(run_dir: Path) -> dict:
    metadata_path = run_dir / ".run.json"
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path) as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    config = metadata.get("config")
    return config if isinstance(config, dict) else {}
