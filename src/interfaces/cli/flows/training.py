"""Training and evaluation operations for CLI.

**Every compute operation here is a cloud submission.** The menu builds the
same :class:`~src.interfaces.cloud.spec.LegSpec` the headless ``submit`` and
``score`` commands build, so it inherits their contract rather than keeping a
second one: absolute iteration targets, experiment tagging, and a snapshot
pinned per submission.

That is a deliberate reversal. The menu used to fork workers on the laptop and
prompt for *additional* iterations, which made it the one surface in the
project speaking a relative-target dialect -- and runs started from it could
never appear in ``report --experiment``, because it had nowhere to put an
experiment id.

Nothing here reads a card abstraction, either. Submitting used to call
``build_card_abstraction`` first, which is not an existence check: it loads the
artifact (~773 MB) into the laptop. Worse, it answered the question about the
wrong machine -- the abstraction a leg needs is the one on the share, which the
node mounts. A laptop that has never precomputed anything can submit perfectly
valid work, and the node refuses the leg if the abstraction is genuinely
missing.
"""

from src.interfaces.cli.commands import status
from src.interfaces.cli.commands.evaluate import EVAL_METHODS
from src.interfaces.cli.flows.config_menu import select_config
from src.interfaces.cli.flows.queueing import queue_legs
from src.interfaces.cli.flows.run_picker import select_run
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.interfaces.cloud import spec

MENU_EVAL_METHODS = ("exact_br", *(m for m in EVAL_METHODS if m != "exact_br"))


def submit_training_leg(ctx: CliContext) -> None:
    """Queue a training leg on the pool -- fresh, or continuing an existing run.

    One flow for both, matching ``submit``/``train-static``: the target is
    absolute, so continuing is just submitting the same run with a larger
    number, and a retry converges instead of training twice.
    """
    ui.header("Submit Training Leg")

    continuing = prompts.confirm(ctx, "Continue an existing run?", default=False)
    if continuing is None:
        return

    run_id = ""
    if continuing:
        selected = select_run(ctx, "Select run to continue:", allow_unloadable=True)
        if selected is None:
            return
        run_id = selected

    # BOTH branches. The config builds the tree and the solver, and the
    # checkpoint stores neither, so a continuing leg needs it just as much as a
    # fresh one. This used to skip the prompt when continuing and send an empty
    # config to the node, which died on
    # `Config file not found: config/training/.yaml` after the snapshot upload,
    # the pool spin-up and every Batch retry.
    config = select_config(ctx)
    if config is None:
        return
    ctx.set_runs_dir(config.training.runs_dir)
    config_name = config.system.config_name

    target = prompts.prompt_int(
        ctx,
        "Absolute iteration target (NOT an increment):",
        default=1_000_000,
        min_value=1,
    )
    if target is None:
        return

    tags = _prompt_experiment_tags(ctx)
    if tags is None:
        return
    experiment, arm, parent = tags

    queue_legs(
        lambda snapshot: [
            spec.LegSpec(
                code_snapshot=snapshot,
                op=spec.TRAIN,
                config=config_name,
                run_id=run_id,
                to=target,
                experiment=experiment,
                arm=arm,
                parent=parent,
            )
        ]
    )
    ui.pause()


def _prompt_experiment_tags(ctx: CliContext) -> tuple[str, str, str] | None:
    """Ask for the bookkeeping that makes a run comparable later.

    Optional, but offered every time: a run submitted without an experiment id
    can never be paired against a control by ``report``, and that cannot be
    fixed after the fact.
    """
    experiment = prompts.text(ctx, "Experiment id (blank for none):", default="")
    if experiment is None:
        return None
    if not experiment.strip():
        return "", "", ""

    arm = prompts.text(ctx, "Arm label (e.g. control, variant:pruning):", default="")
    if arm is None:
        return None
    parent = prompts.text(ctx, "Parent run id (blank for none):", default="")
    if parent is None:
        return None
    return experiment.strip(), arm.strip(), parent.strip()


def cloud_status(ctx: CliContext) -> None:  # noqa: ARG001
    """Show what the pool is doing right now.

    This used to read Batch itself and print its own tables, which made it a
    second renderer for payloads that already had one -- free to disagree with
    `jobs` about what a task looks like, and blind to the leg log entirely, so
    the menu could not answer why a leg died. It now shows the same three
    panels `poker-solver-run status` shows, from the same call.

    No error clause of its own, either: a panel that fails renders as
    unavailable beside the ones that did not, which is the behaviour a status
    screen wants and the reason `gather` isolates per panel.
    """
    ui.header("Cloud Status")
    status.render(status.gather())
    ui.pause()


def score_run(ctx: CliContext) -> None:
    """Queue an evaluation of a published run on the pool.

    Scoring belongs on a node: the share is a local mount there, and one
    checkpoint is ~540 MB of small zarr chunks -- roughly twenty minutes to
    pull over SMB, which is what made scoring a ladder from a laptop
    impractical in the first place.
    """
    ui.header("Score a Run")

    selected_run = select_run(ctx, "Select run to score:", allow_unloadable=True)
    if selected_run is None:
        return
    run_id: str = selected_run

    method = prompts.select(
        ctx, "Estimator:", list(MENU_EVAL_METHODS), default=MENU_EVAL_METHODS[0]
    )
    if method is None:
        return

    rungs_text = prompts.text(
        ctx,
        "Ladder rungs, comma-separated (blank = latest checkpoint):",
        default="",
    )
    if rungs_text is None:
        return
    rungs = [part.strip() for part in rungs_text.split(",") if part.strip()] or [""]

    queue_legs(
        lambda snapshot: [
            spec.LegSpec(
                code_snapshot=snapshot,
                op=spec.EVALUATE,
                run_id=run_id,
                eval_method=method,
                eval_at=rung,
            )
            for rung in rungs
        ]
    )
    ui.pause()
