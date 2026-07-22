"""Shared run selector that annotates runs by age and hides unloadable ones.

Old runs are the main papercut in every "pick a run" flow: a run trained on a
stale format or one that never checkpointed cannot be opened at HEAD, but the
plain name list gives no hint until the load fails. This picker annotates each
run with how many commits back it was trained (``HEAD`` / ``N commits ago``) and,
for flows that will *load* the run, hides the unloadable ones behind a toggle and
greys them out (with the reason) when expanded. Informational flows pass
``allow_unloadable=True`` to keep every run selectable for inspection.
"""

from __future__ import annotations

from collections.abc import Sequence

import questionary
from questionary import Choice

from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.pipeline import services
from src.pipeline.services import RunSummary

# Sentinel choice values; the NUL prefix cannot collide with a run directory name.
_SHOW_ALL = "\x00::show-all"
_HIDE = "\x00::hide"


def _format_age(summary: RunSummary) -> str:
    if summary.commits_ago is None:
        age = "commit unknown"
    elif summary.commits_ago == 0:
        age = "HEAD"
    elif summary.commits_ago == 1:
        age = "1 commit ago"
    else:
        age = f"{summary.commits_ago} commits ago"
    if summary.git_dirty:
        age += " · dirty"
    return age


def _human_count(value: int | None) -> str | None:
    """Compact magnitude label, e.g. 6000 -> '6k', 10_600_000 -> '10.6M'."""
    if value is None:
        return None
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def _format_stats(summary: RunSummary) -> str:
    """Descriptive facts that distinguish one run from another in the list."""
    parts: list[str] = []
    iters = _human_count(summary.iterations)
    if iters is not None:
        parts.append(f"{iters} it")
    infosets = _human_count(summary.num_infosets)
    if infosets is not None:
        parts.append(f"{infosets} infosets")
    if summary.config_name:
        parts.append(summary.config_name)
    # Surface a non-terminal status (a completed run is the unremarkable default).
    if summary.status and summary.status != "completed":
        parts.append(f"[{summary.status}]")
    return " · ".join(parts)


def run_title(summary: RunSummary, *, note_blocker: bool) -> str:
    title = f"{summary.name}   ·   {_format_age(summary)}"
    stats = _format_stats(summary)
    if stats:
        title += f"   ·   {stats}"
    if note_blocker and summary.blocker is not None:
        title += f"   ·   ⚠ {summary.blocker}"
    return title


def build_choices(
    summaries: Sequence[RunSummary],
    *,
    show_all: bool,
    cancel_label: str,
    allow_unloadable: bool,
) -> list:
    """Build the questionary choice list (pure; unit-tested without prompting)."""
    if allow_unloadable:
        # Everything is selectable (for inspection); flag the broken ones inline.
        choices: list = [
            Choice(title=run_title(s, note_blocker=True), value=s.name) for s in summaries
        ]
        choices.append(questionary.Separator())
        choices.append(Choice(title=cancel_label, value=cancel_label))
        return choices

    loadable = [s for s in summaries if s.loadable]
    blocked = [s for s in summaries if not s.loadable]

    choices = [Choice(title=run_title(s, note_blocker=False), value=s.name) for s in loadable]
    if show_all:
        choices += [
            Choice(title=run_title(s, note_blocker=False), value=s.name, disabled=s.blocker)
            for s in blocked
        ]
    choices.append(questionary.Separator())
    if blocked:
        if show_all:
            choices.append(Choice(title="↑ Hide incompatible runs", value=_HIDE))
        else:
            plural = "s" if len(blocked) != 1 else ""
            choices.append(
                Choice(title=f"⋯ Show {len(blocked)} incompatible run{plural}", value=_SHOW_ALL)
            )
    choices.append(Choice(title=cancel_label, value=cancel_label))
    return choices


def select_run(
    ctx: CliContext,
    message: str,
    *,
    cancel_label: str = "Cancel",
    allow_unloadable: bool = False,
) -> str | None:
    """Prompt for a run; return its name, or None on cancel / when none exist.

    Handles the empty-runs message itself. With ``allow_unloadable=False`` (the
    default), runs that cannot be opened at HEAD are hidden behind a toggle and
    greyed out when shown; with True, all runs stay selectable for inspection.
    """
    summaries = services.describe_runs(ctx.runs_dir)
    if not summaries:
        ui.error(f"No trained runs found in {ctx.runs_dir}")
        ui.pause()
        return None

    # Expand by default only when nothing is loadable, so the reasons are visible
    # instead of an empty list behind a toggle.
    show_all = allow_unloadable or not any(s.loadable for s in summaries)
    while True:
        choices = build_choices(
            summaries,
            show_all=show_all,
            cancel_label=cancel_label,
            allow_unloadable=allow_unloadable,
        )
        answer = prompts.select(ctx, message, choices=choices)
        if answer == _SHOW_ALL:
            show_all = True
            continue
        if answer == _HIDE:
            show_all = False
            continue
        if answer is None or answer == cancel_label:
            return None
        return answer
