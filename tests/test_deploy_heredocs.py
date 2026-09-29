"""An unquoted heredoc in `infra/serve/deploy.sh` may not carry a backtick.

Both env files interpolate ($RUN_ID, $SEAT_RUNGS, $RECORD_DSN), so neither
heredoc can be quoted -- which makes a backtick inside one command substitution.
MEASURED 09-29: five comment lines documenting the reader's ladder shipped inside
the `/etc/blueprint.env` heredoc, and a real deploy answered

    bash: line 409: play_only: command not found
    bash: line 409: /api/run: No such file or directory

then wrote the comment with those words deleted. Explanations go ABOVE the
heredoc; only assignments go inside it.
"""

from __future__ import annotations

import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "infra" / "serve" / "deploy.sh"
# An unquoted delimiter: `<<EOF`, not `<<'EOF'` or `<<"EOF"`.
UNQUOTED = re.compile(r"<<-?([A-Za-z_][A-Za-z0-9_]*)\s*$")


def _unquoted_heredoc_bodies() -> list[tuple[int, str]]:
    """Every line inside an unquoted heredoc, as ``(line number, text)``."""
    lines = DEPLOY.read_text().splitlines()
    inside: str | None = None
    body: list[tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        if inside is None:
            match = UNQUOTED.search(line)
            if match:
                inside = match.group(1)
            continue
        if line.strip() == inside:
            inside = None
            continue
        body.append((number, line))
    return body


def test_the_unquoted_heredocs_are_found_at_all() -> None:
    """Guards the guard: a regex that matches nothing passes the test below."""
    assert _unquoted_heredoc_bodies(), "no unquoted heredoc found; the pattern drifted"


def test_no_backtick_inside_an_unquoted_heredoc() -> None:
    offenders = [(number, line) for number, line in _unquoted_heredoc_bodies() if "`" in line]

    assert not offenders, "backtick inside an unquoted heredoc runs as a command: " + "; ".join(
        f"line {number}: {line.strip()}" for number, line in offenders
    )
