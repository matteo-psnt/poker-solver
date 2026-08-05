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
"""

from src.interfaces.commands._base import Command
from src.interfaces.commands.autoscale_check import COMMAND as AUTOSCALE_CHECK
from src.interfaces.commands.cancel import COMMAND as CANCEL
from src.interfaces.commands.compare import COMMAND as COMPARE
from src.interfaces.commands.cost import COMMAND as COST
from src.interfaces.commands.curve import COMMAND as CURVE
from src.interfaces.commands.evaluate import COMMAND as EVALUATE
from src.interfaces.commands.jobs import COMMAND as JOBS
from src.interfaces.commands.ledger import COMMAND as LEDGER
from src.interfaces.commands.logs import COMMAND as LOGS
from src.interfaces.commands.pool_status import COMMAND as POOL_STATUS
from src.interfaces.commands.precompute import COMMAND as PRECOMPUTE
from src.interfaces.commands.progress import COMMAND as PROGRESS
from src.interfaces.commands.promote import COMMAND as PROMOTE
from src.interfaces.commands.push_code import COMMAND as PUSH_CODE
from src.interfaces.commands.push_data import COMMAND as PUSH_DATA
from src.interfaces.commands.repair_ladder import COMMAND as REPAIR_LADDER
from src.interfaces.commands.report import COMMAND as REPORT
from src.interfaces.commands.runinfo import COMMAND as RUNINFO
from src.interfaces.commands.runs import COMMAND as RUNS
from src.interfaces.commands.score import COMMAND as SCORE
from src.interfaces.commands.serve import COMMAND as SERVE
from src.interfaces.commands.status import COMMAND as STATUS
from src.interfaces.commands.submit import COMMAND as SUBMIT
from src.interfaces.commands.submit_precompute import COMMAND as SUBMIT_PRECOMPUTE
from src.interfaces.commands.tasks import COMMAND as TASKS
from src.interfaces.commands.train_static import COMMAND as TRAIN_STATIC

# Order is the order they appear in `--help`, in three groups: dispatch work to
# the pool, run it here (these are what a node invokes), then read the record.
COMMANDS: tuple[Command, ...] = (
    STATUS,
    SERVE,
    SUBMIT,
    SCORE,
    SUBMIT_PRECOMPUTE,
    JOBS,
    LOGS,
    TASKS,
    CANCEL,
    POOL_STATUS,
    AUTOSCALE_CHECK,
    REPAIR_LADDER,
    PUSH_CODE,
    PUSH_DATA,
    TRAIN_STATIC,
    PRECOMPUTE,
    EVALUATE,
    LEDGER,
    CURVE,
    COST,
    PROGRESS,
    RUNS,
    RUNINFO,
    REPORT,
    COMPARE,
    PROMOTE,
)

__all__ = ("COMMANDS", "Command")
