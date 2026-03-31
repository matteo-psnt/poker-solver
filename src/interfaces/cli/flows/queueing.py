"""Queueing legs from the interactive menu.

Every compute operation the menu offers ends here, which is the point: the menu
is a cloud client, so there is exactly one place where it hands work to the
pool and exactly one place that turns an infrastructure failure into a line the
user can read instead of a traceback that closes the session.
"""

from collections.abc import Callable

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

from src.interfaces.cli.ui import ui
from src.interfaces.cloud import dispatch, spec
from src.interfaces.cloud.config import CloudConfigError


def queue_legs(make_legs: Callable[[str], list[spec.LegSpec]]) -> bool:
    """Stage the tree and queue the legs it builds; report problems readably.

    The Azure exceptions are caught by name alongside our own. An expired
    ``az login`` raises ``ClientAuthenticationError`` and an unreachable
    endpoint raises ``HttpResponseError``; uncaught, either tracebacks out of
    the menu. ``SystemExit`` is caught for the same reason -- the shared
    dispatch path raises it for a caller that is a command-line process, and
    letting it through would close the whole interactive session.

    Returns:
        True if the legs were queued.
    """
    try:
        payload = dispatch.stage_and_queue(make_legs)
    except (CloudConfigError, ValueError) as error:
        ui.error(str(error))
        return False
    except (ClientAuthenticationError, HttpResponseError) as error:
        ui.error(f"Azure rejected the request: {error}")
        print("  If this is an auth failure, `az login` and try again.")
        return False
    except SystemExit as error:
        ui.error(str(error))
        return False
    print()
    dispatch.render_queued(payload)
    return True
