"""The failure a surface can show, as distinct from the failure that is a bug.

Raising ``SystemExit`` to signal a bad request is a correct thing for a
command-line process to do exactly once, and it is the assumption no other
surface can make.

``CommandError`` is that same signal carried as a value. :mod:`headless` turns
it back into a message and an exit code, so the command line behaves exactly as
it did. Anything else -- the console's ``/api`` handlers above all -- catches it
and renders ONE panel unavailable instead of dying: a status screen fetches
several commands concurrently, so a single unpublished run must grey out its own
panel and nothing else. ``raise SystemExit`` at 16 call sites made that
impossible. The core reports; the surface decides what a report means.

The Azure SDK's exceptions, and why they are here now
-----------------------------------------------------
``ClientAuthenticationError`` and ``HttpResponseError`` are raised from a dozen
call sites in :mod:`src.interfaces.cloud.tasks.batch` with no single chokepoint
to wrap, so they are caught HERE and nowhere else. Two surfaces talk to Batch --
``status`` composing three panels and the console's ``answer`` -- and each
otherwise carries the same three-arm ladder, differing only in whether the
result becomes an HTTP status or a dict field. The cost of one of them
forgetting an arm is the failure the ladder exists to prevent: an expired
``az login`` blanking a whole screen rather than the two panels that actually
talk to Batch.

:func:`attempt` is that ladder, once. It classifies rather than translates,
because the two callers need different renderings of the same distinction: a
refusal is "understood, and the answer is no" (422), unavailable is "the service
could not be reached" (503). Anything that is neither still propagates -- a bug
in a panel must look like a bug, not like an unavailable panel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Literal


class CommandError(Exception):
    """A command could not answer, and the reason is for the user to read.

    Raise this for anything the caller could have got right: a run that is not
    published, a ``--set`` pair with no ``=``, two checkpoints that cannot be
    compared. A bug stays an ordinary exception, because a traceback is the
    correct output for one.
    """


@dataclass(frozen=True, slots=True)
class Failure:
    """Why a surface has nothing to show, in the terms a surface acts on.

    ``kind`` is the distinction every caller has had to make for itself:
    ``refusal`` is the command answering "no" to a question it understood, and
    is not a fault; ``unavailable`` is Azure not answering at all, which is
    transient and worth retrying. ``message`` is already user-facing.
    """

    kind: Literal["refusal", "unavailable"]
    message: str


@cache
def _azure_failures() -> tuple[type[BaseException], type[BaseException]]:
    """The SDK exception types, imported on first use rather than at module scope.

    Measured at 76ms to import ``azure.core.exceptions`` -- against a 0.18s
    ``poker-solver --help``, which is the budget the lazy command registry
    exists to protect. This module is imported by ``_base`` and therefore by
    every command, so a module-scope import would put that 76ms on every
    invocation, ``--help`` included, to serve the two that talk to Batch. By the
    time :func:`attempt` runs, the caller has already imported the SDK.
    """
    from azure.core.exceptions import ClientAuthenticationError, HttpResponseError

    return (ClientAuthenticationError, HttpResponseError)


def attempt[T](call: Callable[[], T]) -> tuple[T | None, Failure | None]:
    """Run ``call``, returning either its result or a classified failure.

    Exactly one of the two is ever not None. The Azure types are resolved before
    the ``try`` because an ``except`` clause needs them as values, which is also
    what keeps the import out of the exception path.
    """
    auth, http = _azure_failures()
    try:
        return call(), None
    except CommandError as error:
        return None, Failure("refusal", str(error))
    except auth:
        return None, Failure("unavailable", "Azure rejected the credential — try `az login`.")
    except http as error:
        return None, Failure("unavailable", f"Azure did not answer: {error}")
