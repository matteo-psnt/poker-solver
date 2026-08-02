"""The headless subcommand registry.

One module per subcommand, each owning its flags, its handler and its renderer.
Adding a command means adding a module and one line here; there is no second
place that can be forgotten.
"""

from src.interfaces.cli.commands._base import Command
from src.interfaces.cli.commands.ab import COMMAND as AB
from src.interfaces.cli.commands.checkpoint_profile import COMMAND as CHECKPOINT_PROFILE
from src.interfaces.cli.commands.compare import COMMAND as COMPARE
from src.interfaces.cli.commands.curve import COMMAND as CURVE
from src.interfaces.cli.commands.evaluate import COMMAND as EVALUATE
from src.interfaces.cli.commands.ledger import COMMAND as LEDGER
from src.interfaces.cli.commands.precompute import COMMAND as PRECOMPUTE
from src.interfaces.cli.commands.promote import COMMAND as PROMOTE
from src.interfaces.cli.commands.report import COMMAND as REPORT
from src.interfaces.cli.commands.train_static import COMMAND as TRAIN_STATIC

# Order is the order they appear in `--help`: produce a run, then measure it,
# then read the record.
COMMANDS: tuple[Command, ...] = (
    TRAIN_STATIC,
    PRECOMPUTE,
    EVALUATE,
    LEDGER,
    CURVE,
    REPORT,
    PROMOTE,
    CHECKPOINT_PROFILE,
    COMPARE,
    AB,
)

__all__ = ("COMMANDS", "Command")
