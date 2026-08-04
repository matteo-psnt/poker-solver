"""What a run FRAGMENT identifies.

Run ids are long, share a prefix and differ only at the end
(``run-production-025433-1095``), so the piece a person remembers is the tail.
Every reader accepts a fragment for that reason.

This lives in ``shared`` because two layers resolve one: the cloud side decides
which run to MATERIALISE from the share, and the command side then resolves a
directory inside the tree it got. If those two disagreed about what ``1095``
means, a reader would pull one run and answer about another -- so there is one
definition and both import it.
"""

from __future__ import annotations

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
