"""The subcommand registry -- the layer BOTH surfaces read through.

One module per subcommand, each owning its flags, its handler and its renderer.
Adding a command means adding a module and one line here; there is no second
place that can be forgotten.

This sits beside ``cli`` and ``web`` rather than inside either, because it
belongs to neither: ``cli.headless`` renders a command to a terminal, and
``web.app`` calls the same command through :meth:`Command.invoke` and serves
the payload. It lived under ``cli`` while the terminal was the only caller, and
the path went on naming one owner after the console became a second -- an
import-linter contract exists purely to keep the console reading through here,
so the seam is load-bearing enough that its path should not misattribute it.

``render()`` is deliberately NOT abstracted away: it is the terminal's
renderer, and for any other surface the payload is the interface. Parser,
handler and renderer stay together on the one dataclass because when they lived
apart a command borrowed another's renderer and died on a missing key.

Naming a command is not running it
----------------------------------
The registry is :class:`CommandRef` -- a name and a help line -- and importing
it imports NO handler. That distinction is the whole point: listing what this
tool can do, and doing one of them, are different questions, and only the second
needs `evaluate`'s evaluator or `precompute`'s clusterer.

Eagerly importing every one of these modules cost 1.2s on every invocation, `--help`
included, because the union of what they need is the union of everything: the
engine, the abstraction pipeline, numba and scipy. Nothing about listing
subcommand names requires any of it. `cli.headless` builds flags only for the
subcommand argv actually names.

The module is the name with hyphens as underscores. A convention the registry
RELIES on rather than a mapping it stores -- one fewer field to get wrong, and
``load_all()`` in the test suite fails loudly for any command that breaks it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache

from src.interfaces.commands._base import Command


@dataclass(frozen=True)
class CommandRef:
    """What a subcommand is CALLED, without importing what it does.

    ``help`` is repeated from the module's own :class:`Command` because reading
    it there would mean importing there, which is the cost being avoided. That
    makes it the one thing here that can drift, so it is the one thing pinned:
    ``tests/interfaces/commands/test_registry.py`` fails if a ref and its module
    disagree about either field.
    """

    name: str
    help: str

    @property
    def module(self) -> str:
        return self.name.replace("-", "_")

    def load(self) -> Command:
        """Import this subcommand's module and hand back its ``Command``."""
        module = importlib.import_module(f"{__name__}.{self.module}")
        return module.COMMAND


"""Order is the order they appear in `--help`, in four groups: open a surface
(one screen, or a server on localhost), dispatch work to the pool and account
for it, run it here (these are what a node invokes), then read the record."""
COMMANDS: tuple[CommandRef, ...] = (
    CommandRef("status", "Pool, Batch and task history on one screen (--watch to follow)."),
    CommandRef("serve", "Serve the console on localhost."),
    CommandRef("blueprint-serve", "Serve one trained run for reading, on localhost."),
    CommandRef("serve-box", "Report, wake, or stop the blueprint host."),
    CommandRef("submit", "Queue a training task on the pool (--run continues an existing run)."),
    CommandRef("score", "Evaluate a published run on the pool, one task per ladder rung."),
    CommandRef(
        "submit-precompute", "Build a card abstraction on a node and publish it to the share."
    ),
    CommandRef("jobs", "Every queued/running task on the pool (--all includes finished jobs)."),
    CommandRef("logs", "Read a task's log from the share (default) or live from its node."),
    CommandRef("tasks", "Per-task outcomes from the share, reconciled against Batch."),
    CommandRef("cancel", "Terminate a running task; its partial progress is published first."),
    CommandRef(
        "pool-status", "Pool node counts, and the real cause behind any allocation failure."
    ),
    CommandRef(
        "autoscale-check",
        "Evaluate the deployed autoscale formula on the live pool, errors included.",
    ),
    CommandRef("push-code", "Publish an immutable snapshot of the working tree; echoes its id."),
    CommandRef(
        "push-data", "Publish card abstractions to the share (copied, never recomputed on a node)."
    ),
    CommandRef(
        "compact-legs",
        "Bundle sealed task records into one file, so reading legs/ is one round trip.",
    ),
    CommandRef(
        "train-static", "Train over the statically-enumerated tree (fixed memory, no key maps)."
    ),
    CommandRef("precompute", "Precompute a combo abstraction into data/combo_abstraction/."),
    CommandRef("evaluate", "Evaluate a run's exploitability (Local Best Response by default)."),
    CommandRef("ledger", "List recorded evaluations from the eval ledger."),
    CommandRef(
        "curve", "Within-run exploitability vs iteration, from the retained checkpoint ladder."
    ),
    CommandRef("cost", "Billed spend from Azure, and node time derived from the task log."),
    CommandRef("progress", "Per-checkpoint coverage, visits and throughput for a run."),
    CommandRef("runs", "Every published run, newest first."),
    CommandRef(
        "runinfo", "Everything recorded about a run: provenance, curve, scores, tasks, gaps."
    ),
    CommandRef("report", "Score every arm of an experiment, each attributed against its control."),
    CommandRef(
        "compare", "Paired (CRN) comparison of two runs' latest evals; refuses mismatched tiers."
    ),
    CommandRef("promote", "Make a run the new baseline (closes one turn of the base-fork loop)."),
)

BY_NAME: dict[str, CommandRef] = {ref.name: ref for ref in COMMANDS}


def load(name: str) -> Command:
    """The one subcommand called ``name``. Raises ``KeyError`` if there is none."""
    return BY_NAME[name].load()


@cache
def load_all() -> tuple[Command, ...]:
    """Every subcommand, imported.

    For callers that genuinely need all of them: the test suite, and
    ``build_parser()`` with no argv. Named so that paying for it is a decision
    rather than a side effect of touching the registry.
    """
    return tuple(ref.load() for ref in COMMANDS)


__all__ = ("BY_NAME", "COMMANDS", "Command", "CommandRef", "load", "load_all")
