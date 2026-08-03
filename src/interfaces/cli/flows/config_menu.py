"""Configuration selection for the interactive CLI.

Selection only -- there is deliberately no editor here. One used to live in
``flows/config_editors/``: eight modules that walked the user through every
knob and returned a modified :class:`Config`. The caller read one field off it
(``system.config_name``) and dropped the rest on the floor, and nothing wrote
to disk, so every edit was discarded the moment the function returned.

It could not have worked as written either. A leg carries a config NAME plus
``LegSpec.sets``; the node loads the YAML out of the code snapshot. An in-memory
``Config`` has no way to reach a node. Overrides go through ``--set k=v``, which
is one dialect for the thing the editor was a second, silent dialect of.
"""

from src.interfaces.cli.flows.config_helpers import list_config_names
from src.interfaces.cli.ui import prompts, ui
from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config
from src.shared.config_loader import load_config


def select_config(ctx: CliContext) -> Config | None:
    """
    Select a config file from ``config/training/``.

    Returns:
        Loaded Config object or None if cancelled.
    """
    training_config_dir = ctx.config_dir / "training"
    config_names = list_config_names(training_config_dir)

    if not config_names:
        ui.error(f"No config files found in {training_config_dir}/")
        return None

    choices = [*config_names, "Cancel"]
    selected = prompts.select(ctx, "Select configuration:", choices=choices)

    if selected is None or selected == "Cancel":
        return None

    return load_config(training_config_dir / f"{selected}.yaml")
