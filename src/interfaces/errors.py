"""The failure a surface can show, as distinct from the failure that is a bug.

Every headless command used to signal a bad request by raising ``SystemExit``.
That is a correct thing for a command-line process to do exactly once, and it is
the assumption no other surface can make. It was first paid for by a menu that
had to wrap its own calls in ``except SystemExit`` so a missing config would not
close the whole session -- one surface paying, in an exception clause, for
another surface's convention. That menu is gone; the constraint outlived it,
because the console is a worse case of the same thing.

``CommandError`` is that same signal carried as a value. :mod:`headless` turns
it back into a message and an exit code, so the command line behaves exactly as
it did. Anything else -- the console's ``/api`` handlers above all -- catches it
and renders ONE panel unavailable instead of dying: a status screen fetches
several commands concurrently, so a single unpublished run must grey out its own
panel and nothing else. ``raise SystemExit`` at 16 call sites made that
impossible. The core reports; the surface decides what a report means.

Deliberately NOT covered here: the Azure SDK's ``ClientAuthenticationError`` and
``HttpResponseError``. They are raised from a dozen call sites in
:mod:`src.interfaces.cloud.tasks.batch` with no single chokepoint to wrap, so a
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
