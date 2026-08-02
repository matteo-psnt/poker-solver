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
"""

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from src.interfaces.cli.commands.evaluate import EVAL_METHODS
from src.interfaces.cli.flows.combo_precompute import handle_combo_precompute
from src.interfaces.cli.flows.config_menu import select_config
from src.interfaces.cli.flows.run_picker import select_run
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.interfaces.cloud import batch, dispatch, spec
from src.interfaces.cloud.config import CloudConfig, CloudConfigError
from src.pipeline import services
from src.pipeline.training.abstraction_resolver import AbstractionHashMismatchError
from src.pipeline.training.components import (
    build_card_abstraction,
)
from src.shared.config import Config

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
    config_name = ""
    if continuing:
        selected = select_run(ctx, "Select run to continue:", allow_unloadable=True)
        if selected is None:
            return
        run_id = selected
    else:
        config = select_config(ctx)
        if config is None:
            return
        ctx.set_runs_dir(config.training.runs_dir)
        if not _ensure_combo_abstraction(ctx, config):
            ui.pause()
            return
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

    _queue(
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


def _queue(make_legs) -> None:
    """Stage and queue, turning infrastructure problems into readable messages.

    The Azure exceptions are caught by name alongside our own. An expired
    ``az login`` raises ``ClientAuthenticationError`` and an unreachable
    endpoint raises ``HttpResponseError``; uncaught, either tracebacks out of
    the menu. ``SystemExit`` is caught for the same reason -- the shared
    dispatch path raises it for a caller that is a command-line process, and
    letting it through would close the whole interactive session.
    """
    try:
        payload = dispatch.stage_and_queue(make_legs)
    except (CloudConfigError, ValueError) as error:
        ui.error(str(error))
        return
    except (ClientAuthenticationError, HttpResponseError) as error:
        ui.error(f"Azure rejected the request: {error}")
        print("  If this is an auth failure, `az login` and try again.")
        return
    except SystemExit as error:
        ui.error(str(error))
        return
    print()
    dispatch.render_queued(payload)


def cloud_status(ctx: CliContext) -> None:  # noqa: ARG001
    """Show what the pool is doing right now."""
    ui.header("Cloud Status")
    try:
        config = CloudConfig.load()
        client = batch.client(config)
        status = batch.pool_status(client, config.pool_id)
        jobs = batch.list_jobs_with_tasks(client)
    except (CloudConfigError, ClientAuthenticationError, HttpResponseError) as error:
        ui.error(str(error))
        ui.pause()
        return

    print(f"Pool {status['pool_id']} ({status['vm_size']})")
    print(f"  state: {status['allocation_state']}")
    print(f"  nodes: {status['current_dedicated_nodes']} / {status['target_dedicated_nodes']}")
    print(f"  cost:  {config.hourly_cost} (0 nodes at rest)")

    live = [job for job in jobs if str(job["state"]).rsplit(".", 1)[-1].lower() == "active"]
    print("\nActive jobs:" if live else "\nNothing running.")
    for job in live:
        print(f"  == {job['job']}")
        for task in job["tasks"]:
            state = str(task["state"]).rsplit(".", 1)[-1].lower()
            print(f"     {state:<10} {task['task']}")
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

    _queue(
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


def view_runs(ctx: CliContext) -> None:
    """View past training runs."""
    ui.header("Past Training Runs")

    selected = select_run(
        ctx, "Select run to view details:", cancel_label="Back", allow_unloadable=True
    )
    if selected is None:
        return

    run_dir = ctx.runs_dir / selected
    meta = services.load_run_metadata(run_dir)

    print(f"\nRun: {selected}")
    print("-" * 60)
    print(f"Status: {meta.status or 'unknown'}")
    print(f"Started: {meta.started_at or 'N/A'}")
    if meta.completed_at:
        print(f"Completed: {meta.completed_at}")

    print("\nStatistics:")
    print(f"  Iterations: {meta.iterations}")
    runtime = meta.runtime_seconds
    print(f"  Runtime: {runtime:.2f}s ({runtime / 60:.1f}m)")
    if runtime > 0 and meta.iterations > 0:
        print(f"  Speed: {meta.iterations / runtime:.2f} it/s")
    print(f"  Infosets: {meta.num_infosets:,}")

    print(f"\nConfig: {meta.config_name or 'unknown'}")

    ui.pause()


def _run_precompute_and_verify(ctx: CliContext, config: Config) -> bool:
    """Run precomputation and verify it completed successfully."""
    print("\n" + "=" * 60)
    print("RUNNING PRECOMPUTATION")
    print("=" * 60)
    handle_combo_precompute(ctx)

    try:
        build_card_abstraction(config)
        print("\n✓ Precomputation completed successfully!")
        print("Continuing with training setup...\n")
        return True
    except (FileNotFoundError, ValueError):
        ui.error("\nPrecomputation did not complete successfully.")
        print("Training cancelled.")
        return False


def _ensure_combo_abstraction(ctx: CliContext, config: Config) -> bool:
    """
    Ensure combo abstraction exists for the given config.

    If the abstraction is missing, prompts the user to run precomputation.

    Returns:
        True if abstraction exists or was successfully created, False otherwise
    """
    try:
        build_card_abstraction(config)
        return True
    except FileNotFoundError as e:
        ui.error(str(e))
        print()

        run_precompute = prompts.confirm(
            ctx,
            "Would you like to run precomputation now?",
            default=True,
        )

        if not run_precompute:
            print("\nTraining cancelled. Please run precomputation from the main menu first.")
            return False

        return _run_precompute_and_verify(ctx, config)

    except AbstractionHashMismatchError as e:
        ui.error(str(e))
        print()

        run_precompute = prompts.confirm(
            ctx,
            "Would you like to recompute the abstraction now?",
            default=True,
        )

        if not run_precompute:
            print("\nTraining cancelled.")
            return False

        return _run_precompute_and_verify(ctx, config)

    except ValueError as e:
        ui.error(str(e))
        print()
        return False
