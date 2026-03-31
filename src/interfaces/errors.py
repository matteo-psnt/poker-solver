"""The failure a surface can show, as distinct from the failure that is a bug.

Every headless command used to signal a bad request by raising ``SystemExit``.
That is a correct thing for a command-line process to do exactly once, and it is
the assumption no other surface can make. The interactive menu already had to
catch ``SystemExit`` in :mod:`src.interfaces.cli.flows.queueing` to stop a
missing config from closing the whole session -- one surface paying, in a
``except SystemExit`` clause, for the other surface's convention.

``CommandError`` is that same signal carried as a value. :mod:`headless` turns
it back into a message and an exit code, so the command line behaves exactly as
it did; anything else -- a menu, a live status view, an HTTP handler -- catches
it and renders one panel unavailable instead of dying. That is the property
that makes a second surface cheap to add: the core reports, and the surface
decides what a report means.

Deliberately NOT covered here: the Azure SDK's ``ClientAuthenticationError`` and
``HttpResponseError``. They are raised from a dozen call sites in
:mod:`src.interfaces.cloud.batch` with no single chokepoint to wrap, so a
surface that talks to Batch still catches them by name. Worth closing, but it
is a separate change from this one.
"""

from __future__ import annotations


class CommandError(Exception):
    """A command could not answer, and the reason is for the user to read.

    Raise this for anything the caller could have got right: a run that is not
    published, a ``--set`` pair with no ``=``, two checkpoints that cannot be
    compared. A bug stays an ordinary exception, because a traceback is the
    correct output for one.
    """
