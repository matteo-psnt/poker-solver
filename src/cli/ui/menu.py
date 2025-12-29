"""Menu wiring for the CLI."""

import traceback
from collections.abc import Callable
from dataclasses import dataclass

from questionary import Choice

from src.cli.ui import prompts, ui
from src.cli.ui.context import CliContext


@dataclass(frozen=True)
class MenuItem:
    label: str
    handler: Callable[[CliContext], None]


_EXIT = object()


def run_action(ctx: CliContext, handler: Callable[[CliContext], None]) -> None:
    try:
        handler(ctx)
    except KeyboardInterrupt:
        ui.warn("Operation cancelled by user")
    except Exception as exc:
        ui.error(str(exc))
        traceback.print_exc()
        ui.pause()


def run_menu(
    ctx: CliContext,
    title: str,
    items: list[MenuItem],
    exit_label: str = "Back",
) -> None:
    choices = [Choice(title=item.label, value=item.handler) for item in items]
    choices.append(Choice(title=exit_label, value=_EXIT))

    while True:
        handler = prompts.select(ctx, title, choices)
        if handler is None or handler is _EXIT:
            return
        run_action(ctx, handler)
