"""Upgrade an old-format run to the current representation version so it loads.

Old runs on disk are frequently a format version behind the loader (the picker
greys them out as ``format vN ≠ vM``). ``migrate_run`` rewrites a copy into the
current format -- functional, so the original and its recorded numbers stay
intact. Only runs that actually have a local checkpoint can be migrated here; a
run whose checkpoint still lives on the Modal Volume must be pulled down first
(``modal volume get poker-data <run_id> data/runs/``) or migrated on Modal via
the ``migrate`` entrypoint.
"""

from questionary import Choice

from src.interfaces.cli.flows import run_picker
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.pipeline import services
from src.pipeline.training.migrations import MigrationBarrierError, migrate_run
from src.pipeline.training.versioning import REPRESENTATION_VERSION


def migrate_run_flow(ctx: CliContext) -> None:
    """Pick an old-format local run and migrate a copy to the current format."""
    ui.header("Migrate Run To Current Format")

    summaries = services.describe_runs(ctx.runs_dir)
    migratable = [
        s
        for s in summaries
        if not s.loadable and s.has_checkpoint and s.blocker and s.blocker.startswith("format")
    ]
    if not migratable:
        ui.info("No local runs need migrating.")
        print(
            "  Runs blocked by 'no checkpoint' live only on the Modal Volume — pull them\n"
            "  first: modal volume get poker-data <run_id> data/runs/"
        )
        ui.pause()
        return

    choices = [
        Choice(title=run_picker.run_title(s, note_blocker=True), value=s.name) for s in migratable
    ]
    choices.append(Choice(title="Cancel", value="Cancel"))
    selected = prompts.select(
        ctx, "Select a run to migrate to the current format:", choices=choices
    )
    if selected is None or selected == "Cancel":
        return

    src = ctx.runs_dir / selected
    dest_name = f"{selected}-v{REPRESENTATION_VERSION}"
    dst = ctx.runs_dir / dest_name
    if dst.exists():
        ui.error(f"Destination already exists: {dest_name}")
        ui.pause()
        return

    print(f"\nMigrating {selected} → {dest_name} (original untouched)...")
    try:
        migrate_run(src, dst)
    except MigrationBarrierError as exc:
        ui.error(str(exc))
        ui.pause()
        return
    except (OSError, ValueError) as exc:
        ui.error(f"Migration failed: {exc}")
        ui.pause()
        return

    ui.info(
        f"Done — {dest_name} is now at v{REPRESENTATION_VERSION} and ready to play or evaluate."
    )
    ui.pause()
