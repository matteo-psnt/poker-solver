"""CLI entrypoint and top-level menus."""

from src.interfaces.cli.flows import training
from src.interfaces.cli.flows.combo_menu import combo_menu
from src.interfaces.cli.ui import ui
from src.interfaces.cli.ui.context import CliContext
from src.interfaces.cli.ui.menu import MenuItem, run_menu
from src.shared.log import configure_logging


def main() -> None:
    configure_logging()
    ctx = CliContext.from_project_root()

    ui.header("POKER SOLVER CLI")

    items = [
        MenuItem("Submit Training Leg", training.submit_training_leg),
        MenuItem("Score a Run", training.score_run),
        MenuItem("Cloud Status", training.cloud_status),
        MenuItem("Combo Abstraction Tools", combo_menu),
    ]

    run_menu(ctx, "What would you like to do?", items, exit_label="Exit")
    print("\nGoodbye!")
