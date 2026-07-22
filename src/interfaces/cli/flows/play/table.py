"""Launch the browser table to play heads-up against a trained blueprint."""

import webbrowser

from src.interfaces.api.app import FastAPIChartServer
from src.interfaces.api.chart_service import ChartService
from src.interfaces.api.play_service import PlayService
from src.interfaces.cli.flows.chart.viewer import _ensure_ui_build
from src.interfaces.cli.flows.run_picker import select_run
from src.interfaces.cli.ui import ui
from src.interfaces.cli.ui.context import CliContext
from src.pipeline import services
from src.pipeline.training.components import build_evaluation_solver


def play_against_model(ctx: CliContext) -> None:
    """Load a run and serve the interactive poker table (with charts) in the browser."""
    selected_run = select_run(ctx, "Select run to play against:")
    if selected_run is None:
        return

    if not _ensure_ui_build(ctx):
        ui.pause()
        return

    print(f"\nLoading solver from {selected_run}...")
    run_dir = ctx.runs_dir / selected_run
    metadata = services.load_run_metadata(run_dir)
    # Load the blueprint once and share it between the two services (the load is
    # the expensive step; both are read-only consumers of the same table).
    blueprint, _ = build_evaluation_solver(metadata.config, checkpoint_dir=run_dir)
    play_service = PlayService(run_id=selected_run, blueprint=blueprint)
    chart_service = ChartService(run_id=selected_run, blueprint=blueprint)

    infosets = play_service.num_infosets()
    print(f"  Loaded {infosets:,} infosets")
    if infosets == 0:
        ui.error("No strategy data found in this run")
        ui.pause()
        return

    print("\nStarting table server...")
    server = FastAPIChartServer(
        service=chart_service,
        base_dir=ctx.base_dir,
        play_service=play_service,
    )
    try:
        server.start()
    except OSError as exc:
        ui.error(f"Unable to start server: {exc}")
        ui.pause()
        return

    print(f"  Table running at {server.base_url}")
    print("  Opening in browser...")
    webbrowser.open(server.base_url)

    try:
        ui.pause("\nPress Enter to stop the server and return to the menu...")
    finally:
        server.stop()
