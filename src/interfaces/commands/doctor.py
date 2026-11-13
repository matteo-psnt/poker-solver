"""The `doctor` subcommand: is this laptop set up to operate the project?

Each check names the step that fixes it, in `just setup` order, because setting
up a second laptop (11-11) meant finding every step by reading the code. Any
failure is a refusal carrying the whole report, so the exit status is honest.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.interfaces.cloud import config as cloud_config
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

NVMRC = Path("console/.nvmrc")
TOOLS = ("git", "uv", "just", "terraform", "az", "node", "npm")


class Check(BaseModel):
    name: str
    ok: bool
    detail: str
    fix: str = ""  # the command that makes it pass; empty when it passes


class DoctorPayload(BaseModel):
    """Every check, in the order `just setup` performs them."""

    checks: list[Check]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """`doctor` takes no flags."""


def _output(*argv: str, timeout: float = 30) -> str:
    """stdout of `argv`, or a CommandError carrying its stderr."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CommandError(str(error)) from error
    if done.returncode != 0:
        lines = (done.stderr or done.stdout).strip().splitlines()
        raise CommandError(lines[-1] if lines else f"exit {done.returncode}")
    return done.stdout.strip()


def _tools() -> str:
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        raise CommandError("missing: " + ", ".join(missing))
    return "all present"


def _node() -> str:
    wanted = NVMRC.read_text().strip()
    have = _output("node", "--version").lstrip("v")
    if have.split(".")[0] != wanted:
        raise CommandError(f"node {have}, the console wants {wanted} (console/.nvmrc)")
    return f"node {have}"


def _azure() -> str:
    return _output("az", "account", "show", "--query", "name", "-o", "tsv")


def _terraform() -> str:
    roots = (cloud_config.INFRA_DIR, cloud_config.STORE_DIR, cloud_config.SERVE_DIR)
    missing = [str(root) for root in roots if not (root / ".terraform").is_dir()]
    if missing:
        raise CommandError("not initialised: " + ", ".join(missing))
    return "3 roots initialised"


def _record() -> str:
    from src.adapters.postgres.connect import engine_for  # noqa: PLC0415 -- sqlalchemy is heavy

    try:
        with engine_for(cloud_config.record_dsn()).connect() as connection:
            connection.exec_driver_sql("select 1")
    except Exception as error:
        raise CommandError(str(error).strip().splitlines()[0]) from error
    return "answers"


def _console() -> str:
    if not Path("console/node_modules/.package-lock.json").is_file():
        raise CommandError("console/node_modules is not installed")
    return "installed"


def _hook() -> str:
    hook = Path(_output("git", "rev-parse", "--git-path", "hooks/pre-commit"))
    if not hook.is_file():
        raise CommandError("no pre-commit hook")
    return "installed"


CHECKS: tuple[tuple[str, Callable[[], str], str], ...] = (
    ("tools", _tools, "brew install just azure-cli node hashicorp/tap/terraform"),
    ("node version", _node, "brew install node (or `nvm use` in console/)"),
    ("azure sign-in", _azure, "az login"),
    ("terraform", _terraform, "just setup (runs terraform init in all three roots)"),
    ("record database", _record, "uv run poker-solver record-admit"),
    ("console packages", _console, "npm --prefix console ci"),
    ("pre-commit hook", _hook, "uv run pre-commit install"),
)


def run(_args: argparse.Namespace) -> DoctorPayload:
    """Run every check; refuse with the full report if any failed."""
    checks = []
    for name, probe, fix in CHECKS:
        try:
            checks.append(Check(name=name, ok=True, detail=probe()))
        except (CommandError, OSError) as failure:
            checks.append(Check(name=name, ok=False, detail=str(failure), fix=fix))
    payload = DoctorPayload(checks=checks)
    if not all(check.ok for check in checks):
        raise CommandError("this laptop is not set up:\n" + "\n".join(_lines(payload)))
    return payload


def _lines(payload: DoctorPayload) -> list[str]:
    lines = []
    for check in payload.checks:
        lines.append(f"  {'✓' if check.ok else '✗'} {check.name:<17} {check.detail}")
        if not check.ok:
            lines.append(f"      fix: {check.fix}")
    return lines


def render(payload: DoctorPayload) -> None:
    print("\n".join(_lines(payload)))


COMMAND = Command(
    name="doctor",
    help="Check this laptop is set up: tools, Azure sign-in, Terraform, record, console, hook.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
