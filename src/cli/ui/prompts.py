"""Questionary prompt helpers."""

from collections.abc import Sequence
from typing import Any

import questionary
from questionary import Choice

from src.cli.ui.context import CliContext


def select(
    ctx: CliContext,
    message: str,
    choices: Sequence[str | Choice],
    default: str | None = None,
) -> Any:
    return questionary.select(
        message,
        choices=list(choices),
        default=default,
        style=ctx.style,
    ).ask()


def checkbox(
    ctx: CliContext,
    message: str,
    choices: Sequence[str | Choice],
) -> list[Any] | None:
    return questionary.checkbox(
        message,
        choices=list(choices),
        style=ctx.style,
    ).ask()


def confirm(ctx: CliContext, message: str, default: bool = False) -> bool | None:
    return questionary.confirm(
        message,
        default=default,
        style=ctx.style,
    ).ask()


def text(
    ctx: CliContext,
    message: str,
    default: str = "",
    validate=None,
) -> str | None:
    return questionary.text(
        message,
        default=default,
        validate=validate,
        style=ctx.style,
    ).ask()


def prompt_int(
    ctx: CliContext,
    message: str,
    default: int | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
    allow_blank: bool = False,
) -> int | None:
    default_text = "" if default is None else str(default)

    def _validate(value: str) -> bool | str:
        if allow_blank and not value.strip():
            return True
        try:
            parsed = int(value)
        except ValueError:
            return "Enter a whole number"
        if min_value is not None and parsed < min_value:
            return f"Must be >= {min_value}"
        if max_value is not None and parsed > max_value:
            return f"Must be <= {max_value}"
        return True

    answer = text(ctx, message, default=default_text, validate=_validate)
    if answer is None:
        return None
    if allow_blank and not answer.strip():
        return None
    return int(answer)


def prompt_float(
    ctx: CliContext,
    message: str,
    default: float | None = None,
    min_value: float | None = None,
    allow_blank: bool = False,
) -> float | None:
    default_text = "" if default is None else str(default)

    def _validate(value: str) -> bool | str:
        if allow_blank and not value.strip():
            return True
        try:
            parsed = float(value)
        except ValueError:
            return "Enter a number"
        if min_value is not None and parsed < min_value:
            return f"Must be >= {min_value}"
        return True

    answer = text(ctx, message, default=default_text, validate=_validate)
    if answer is None:
        return None
    if allow_blank and not answer.strip():
        return None
    return float(answer)
