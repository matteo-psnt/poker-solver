"""The headless subcommand registry.

One module per subcommand, each owning its flags, its handler and its renderer.
Adding a command means adding a module and one line here; there is no second
place that can be forgotten.
"""

from src.interfaces.cli.commands._base import Command
from src.interfaces.cli.commands.autoscale_check import COMMAND as AUTOSCALE_CHECK
from src.interfaces.cli.commands.cancel import COMMAND as CANCEL
from src.interfaces.cli.commands.compare import COMMAND as COMPARE
from src.interfaces.cli.commands.curve import COMMAND as CURVE
from src.interfaces.cli.commands.evaluate import COMMAND as EVALUATE
from src.interfaces.cli.commands.fetch import COMMAND as FETCH
from src.interfaces.cli.commands.jobs import COMMAND as JOBS
from src.interfaces.cli.commands.ledger import COMMAND as LEDGER
from src.interfaces.cli.commands.legs import COMMAND as LEGS
from src.interfaces.cli.commands.logs import COMMAND as LOGS
from src.interfaces.cli.commands.pool_status import COMMAND as POOL_STATUS
from src.interfaces.cli.commands.precompute import COMMAND as PRECOMPUTE
from src.interfaces.cli.commands.progress import COMMAND as PROGRESS
from src.interfaces.cli.commands.promote import COMMAND as PROMOTE
from src.interfaces.cli.commands.push_code import COMMAND as PUSH_CODE
from src.interfaces.cli.commands.push_data import COMMAND as PUSH_DATA
from src.interfaces.cli.commands.repair_ladder import COMMAND as REPAIR_LADDER
from src.interfaces.cli.commands.report import COMMAND as REPORT
from src.interfaces.cli.commands.runinfo import COMMAND as RUNINFO
from src.interfaces.cli.commands.score import COMMAND as SCORE
from src.interfaces.cli.commands.submit import COMMAND as SUBMIT
from src.interfaces.cli.commands.submit_precompute import COMMAND as SUBMIT_PRECOMPUTE
from src.interfaces.cli.commands.train_static import COMMAND as TRAIN_STATIC

# Order is the order they appear in `--help`, in three groups: dispatch work to
# the pool, run it here (these are what a node invokes), then read the record.
COMMANDS: tuple[Command, ...] = (
    SUBMIT,
    SCORE,
    SUBMIT_PRECOMPUTE,
    JOBS,
    LOGS,
    LEGS,
    CANCEL,
    POOL_STATUS,
    AUTOSCALE_CHECK,
    REPAIR_LADDER,
    FETCH,
    PUSH_CODE,
    PUSH_DATA,
    TRAIN_STATIC,
    PRECOMPUTE,
    EVALUATE,
    LEDGER,
    CURVE,
    PROGRESS,
    RUNINFO,
    REPORT,
    COMPARE,
    PROMOTE,
)

__all__ = ("COMMANDS", "Command")
