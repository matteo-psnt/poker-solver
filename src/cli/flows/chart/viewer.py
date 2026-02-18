"""Preflop chart generation for CLI."""

import os
import subprocess
import webbrowser

from src.api.app import FastAPIChartServer
from src.api.chart_service import ChartService
from src.cli.flows.chart.server import ChartServer
from src.cli.ui import prompts, ui
from src.cli.ui.context import CliContext
from src.training import services


def view_preflop_chart(ctx: CliContext) -> None:
    """Handle viewing preflop strategy chart."""
    runs = services.list_runs(ctx.runs_dir)

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
    chart_service = ChartService.from_run_dir(run_dir, run_id=selected_run)
    storage = chart_service.runtime.storage

    print(f"  Loaded {storage.num_infosets():,} infosets")

    if storage.num_infosets() == 0:
        ui.error("No strategy data found in this run")
        ui.pause()
        return

    print("\nStarting chart viewer...")

    use_legacy = os.getenv("POKER_SOLVER_USE_LEGACY_CHART_SERVER") == "1"
    if use_legacy:
        server = ChartServer(
            chart_service=chart_service,
            base_dir=ctx.base_dir,
        )
    else:
        server = FastAPIChartServer(
            service=chart_service,
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
