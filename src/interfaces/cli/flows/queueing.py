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
from src.interfaces.errors import CommandError


def queue_legs(make_legs: Callable[[str], list[spec.LegSpec]]) -> bool:
    """Stage the tree and queue the legs it builds; report problems readably.

    ``CommandError`` covers everything the dispatch path can refuse -- an
    unreadable Terraform state (``CloudConfigError`` is one) and an empty
    submission alike. This clause used to include ``SystemExit`` as well,
    because dispatch signalled a refusal the way a command-line process does
    and letting it through would have closed the whole interactive session.
    That is now :class:`CommandError`, so the menu no longer pays for the CLI's
    convention. ``ValueError`` stays: it is what ``LegSpec.validate`` raises.

    The Azure exceptions are still caught by name. An expired ``az login``
    raises ``ClientAuthenticationError`` and an unreachable endpoint raises
    ``HttpResponseError``; neither is ours to reclassify at this distance from
    the call.

    Returns:
        True if the legs were queued.
    """
    try:
        payload = dispatch.stage_and_queue(make_legs)
    except (CommandError, ValueError) as error:
        ui.error(str(error))
        return False
    except (ClientAuthenticationError, HttpResponseError) as error:
        ui.error(f"Azure rejected the request: {error}")
        print("  If this is an auth failure, `az login` and try again.")
        return False
    print()
    dispatch.render_queued(payload)
    return True
