"""`deploy.sh keep` must field exactly what the seat fields, or refuse.

Every deploy rewrites the seat's env from its own arguments, so a deploy that
mis-reads the settings it is meant to keep silently re-fields something else.
These run the real `keep_seat` function, cut out of deploy.sh by its markers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1] / "infra" / "serve" / "deploy.sh"

# The seat env as the box held it on 10-16, less the DSN.
LIVE = """RUN=run-pcs-production_opens-to12k-d100-145201-6584
RUNS_DIR=/mnt/work/data/runs
CHIPZEN_ENV=prod
POLICY_THRESHOLD=0.35
SEAT_EXTRA=--at 60200 --no-resolver --rung run-a-d50:34600:0.35 --rung run-b-d35:50000:0.20
"""


def _keep(env: str) -> subprocess.CompletedProcess[str]:
    text = DEPLOY.read_text()
    body = text[text.index("# >>> keep_seat") : text.index("# <<< keep_seat")]
    script = (
        body
        + 'keep_seat "$ENV_TEXT" || exit 1\n'
        + 'printf "%s\\n" "$RUN" "$AT" "$RUNGS" "${RESOLVER:-off}" "$POLICY_THRESHOLD" '
        + '"$CHIPZEN_ENV" "${BUDGET_MS:-none}" "${RESOLVER_OFF_TREE:-no}"\n'
    )
    return subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script],
        env={"ENV_TEXT": env, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_it_carries_every_setting_the_seat_runs_with():
    out = _keep(LIVE)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [
        "run-pcs-production_opens-to12k-d100-145201-6584",
        "60200",
        "run-a-d50:34600:0.35,run-b-d35:50000:0.20",
        "off",
        "0.35",
        "prod",
        "none",
        "no",
    ]


def test_a_single_tree_seat_stays_single_rather_than_taking_the_default_ladder():
    out = _keep(LIVE.replace(" --rung run-a-d50:34600:0.35 --rung run-b-d35:50000:0.20", ""))
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines()[2] == "none"


def test_a_flag_it_cannot_carry_over_is_a_refusal():
    out = _keep(LIVE.replace("--no-resolver", "--no-resolver --some-new-flag"))
    assert out.returncode != 0
    assert "--some-new-flag" in out.stderr
