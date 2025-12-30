"""CLI entrypoint and top-level menus."""

from src.cli.flows import training
from src.cli.flows.chart.viewer import view_preflop_chart
from src.cli.flows.combo_menu import combo_menu
from src.cli.ui import ui
from src.cli.ui.context import CliContext
from src.cli.ui.menu import MenuItem, run_menu


def main() -> None:
    ctx = CliContext.from_project_root()

    ui.header("POKER SOLVER CLI")

    items = [
        MenuItem("Train Solver", training.train_solver),
        MenuItem("Resume Training", training.resume_training),
        MenuItem("View Past Runs", training.view_runs),
        MenuItem("Evaluate Solver", training.evaluate_solver),
        MenuItem("View Preflop Chart", view_preflop_chart),
        MenuItem("Combo Abstraction Tools", combo_menu),
    ]

    run_menu(ctx, "What would you like to do?", items, exit_label="Exit")
    print("\nGoodbye!")
