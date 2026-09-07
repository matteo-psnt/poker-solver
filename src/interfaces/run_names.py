"""What a run FRAGMENT identifies.

Run ids are long, share a prefix and differ only at the end
(``run-production-025433-1095``), so the piece a person remembers is the tail.
Every reader accepts a fragment for that reason.

This sits at the top of ``interfaces`` because two surfaces UNDER it resolve one:
``cloud.workspace`` decides which run to MATERIALISE from the share, and
``cli.commands`` then resolves a directory inside the tree it got. If those two
disagreed about what ``1095`` means, a reader would pull one run and answer
about another -- so there is one definition and both import it. Nothing below
``interfaces`` resolves a fragment: a run id reaches ``pipeline`` already whole.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def matching(fragment: str, names: Iterable[str]) -> list[str]:
    """Every run a fragment identifies, exact match winning outright.

    An exact name is never ambiguous, even when it is also a substring of a
    longer one: with ``run-a`` and ``run-a-2`` both published, asking for
    ``run-a`` has an unambiguous answer and reporting a conflict would make the
    full id unusable.
    """
    candidates = list(names)
    if fragment in candidates:
        return [fragment]
    return sorted(name for name in candidates if fragment in name)


def ambiguous_message(fragment: str, matches: list[str], *, limit: int = 6) -> str:
    """The refusal text, shared so both surfaces read the same way."""
    shown = ", ".join(matches[:limit])
    more = f", +{len(matches) - limit} more" if len(matches) > limit else ""
    return f"'{fragment}' matches {len(matches)} runs ({shown}{more}). Be more specific."


def unknown_message(fragment: str, published: list[str], *, limit: int = 6) -> str:
    """The refusal for a fragment nothing matches -- the SIMILAR ids, not all of them.

    The ambiguous path above was given a limit and this one was not, so a typo
    printed every published run: measured 09-01, 303 ids in one paragraph, which
    is a wall to scroll past rather than an answer. Ranked by similarity, because
    the useful reply to a mistyped id is the id that was meant and alphabetical
    order puts that nowhere in particular.
    """
    if not published:
        return f"'{fragment}' is not published, and neither is anything else."
    close = difflib.get_close_matches(fragment, published, n=limit, cutoff=0.4)
    # Nothing similar -- a fragment of a run that was never created, or one still
    # training and so not yet published. The newest ids are where that is.
    shown = close or sorted(published)[-limit:]
    return (
        f"'{fragment}' is not published. {'Closest' if close else 'Newest'}: "
        f"{', '.join(shown)} ({len(published)} published; `poker-solver runs` lists them)."
    )
