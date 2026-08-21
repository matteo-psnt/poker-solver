"""Where the deployed infrastructure lives, asked of the thing that built it.

Terraform owns what exists, so Terraform is the authority on its coordinates.
This module is the single place that asks. Two states are read, and the split
is deliberate: ``infra/`` holds the Batch account and pool, ``infra/store/``
holds the durable share in its own resource group, so ``just destroy`` can tear
down compute without being able to reach the experiment record.

``terraform output -json`` is shelled rather than reading a generated outputs
file, and the answer is kept under the cache root for an hour. The shell-out
is 3 s against a remote state: terraform starts, authenticates to the backend
and downloads the state, all to read a dozen strings that change only at an
apply -- and it was paid twice by any command that reconciles with Batch. The
cached file holds the store's ``access_key`` and the record DSN, so it is
written owner-only; the `just` apply recipes delete it, since an apply is the
one event that changes the answer.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.interfaces.errors import CommandError
from src.shared import cache

INFRA_DIR = Path("infra")
STORE_DIR = Path("infra/store")

# The one environment variable a task carries: the node has no Terraform, so
# dispatch seals the resolved DSN into the task under this name, and a reader
# on the laptop may set it to point elsewhere (a restored server, a test).
RECORD_DSN_ENV = "POKER_SOLVER_RECORD_DSN"

# Guards the cold-cache computation in `_outputs`, not the cache itself.
_OUTPUTS_LOCK = threading.Lock()

# The on-disk copy of each state's outputs, keyed by the root's relative path
# so every worktree reads the same file: the states are remote and one per
# root, so the answer is the same from any checkout. An hour bounds how long a
# direct `terraform apply` (one that bypassed `just`) can leave a stale
# coordinate in play; a stale one fails loudly, as a refused connection.
OUTPUTS_CACHE = "terraform-outputs"
OUTPUTS_TTL_SECONDS = 3600.0

_TERRAFORM_MISSING = (
    "terraform is not on PATH. The cloud commands read the deployed pool and "
    "share coordinates from Terraform state; install terraform, or run the "
    "operation from a checkout where `just create` has been applied."
)


class CloudConfigError(CommandError):
    """Infrastructure coordinates could not be read.

    Raised in preference to letting a ``FileNotFoundError`` on ``terraform``
    surface: the cloud commands carry an undeclared dependency on terraform and
    the az CLI credential, and a bare OSError gives no hint which is missing.

    A :class:`CommandError` because it is one: a checkout with no applied
    Terraform state is something the caller can fix, and every surface should be
    able to say so in the same clause it uses for every other readable failure.
    """


def _outputs(chdir: str) -> dict[str, Any]:
    """Read one Terraform state's outputs, once per process.

    ``functools.cache`` alone does not give "once": it makes the LOOKUP atomic,
    not the computation, so callers arriving together on a cold cache each miss
    and each shell out. Measured on the first concurrent reader (`status`, whose
    panels run on three threads): ``terraform output`` ran twice against
    ``infra`` and twice against ``infra/store``. The lock costs a dict-lookup's
    worth of contention once warm, and a future poller with more panels would
    have multiplied the cold cost further.
    """
    with _OUTPUTS_LOCK:
        return _read_outputs(chdir)


@functools.cache
def _read_outputs(chdir: str) -> dict[str, Any]:
    """The per-process read: the on-disk copy if it is fresh, else terraform.
    Keyed by ``str`` rather than ``Path`` so the cache is hashable and stable."""
    stored = cache.cached_json(OUTPUTS_CACHE, chdir, OUTPUTS_TTL_SECONDS)
    if stored is not None:
        return stored
    parsed = _terraform_output(chdir)
    cache.store_json(OUTPUTS_CACHE, chdir, parsed)
    return parsed


def _terraform_output(chdir: str) -> dict[str, Any]:
    """The shell-out itself."""
    if shutil.which("terraform") is None:
        raise CloudConfigError(_TERRAFORM_MISSING)
    try:
        completed = subprocess.run(
            ["terraform", f"-chdir={chdir}", "output", "-json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CloudConfigError(
            f"`terraform -chdir={chdir} output -json` failed. Has `just create` been "
            f"applied against this checkout?\n{exc.stderr.strip()}"
        ) from exc
    parsed = json.loads(completed.stdout)
    if not parsed:
        raise CloudConfigError(
            f"Terraform state at {chdir} has no outputs -- nothing has been applied yet. "
            "Run `just store-create` then `just create`."
        )
    return parsed


def _value(chdir: str, name: str) -> str:
    """Pull one output's value, failing with the name that was missing."""
    entry = cast("dict[str, Any] | None", _outputs(chdir).get(name))
    if entry is None or "value" not in entry:
        raise CloudConfigError(
            f"Terraform state at {chdir} has no output named '{name}'. "
            "The infra and the code are out of step -- re-apply, or check infra/outputs.tf."
        )
    return str(entry["value"])


def _optional_value(chdir: str, name: str) -> str:
    """An output that may predate the current infra -- absent reads as "".

    `pool_big_id` exists only after the `train-big` pool has been applied;
    every command that does not name that pool must keep working against the
    older state.
    """
    try:
        return _value(chdir, name)
    except CloudConfigError:
        return ""


def store_value(name: str) -> str:
    """One output of the store state, for the commands that act on the record
    server itself and need none of the compute coordinates."""
    return _value(str(STORE_DIR), name)


def record_dsn() -> str:
    """The record database's DSN: the environment if set, else the store state.

    Terraform is the authority, the same way it is for every other coordinate
    here. The environment wins only so a sealed task and a deliberate override
    can point elsewhere; nothing falls back to "no record" -- a reader with no
    database answers nothing, and says so.
    """
    override = os.environ.get(RECORD_DSN_ENV, "").strip()
    if override:
        return override
    return _value(str(STORE_DIR), "postgres_dsn")


def export_record_dsn() -> None:
    """Put the resolved DSN into this process's environment, once, at startup.

    Best effort on purpose: on a node there is no Terraform and the variable is
    already sealed; on a laptop with no applied store state the readers raise
    their own, specific refusal when they find nothing.
    """
    if os.environ.get(RECORD_DSN_ENV, "").strip():
        return
    try:
        os.environ[RECORD_DSN_ENV] = record_dsn()
    except CloudConfigError:
        return


@dataclass(frozen=True)
class CloudConfig:
    """Everything the control plane needs to talk to Azure.

    ``batch_endpoint`` carries a scheme even though Terraform's
    ``account_endpoint`` does not. ``azurerm`` reports the bare host
    (``acct.region.batch.azure.com``) and ``BatchClient`` wants a URL, so the
    normalisation happens here rather than at each of the call sites that would
    otherwise each have to remember.
    """

    batch_endpoint: str
    pool_id: str
    pool_big_id: str
    pool_huge_id: str
    pool_mem_id: str
    storage_account: str
    share_name: str
    share_key: str
    code_container: str
    hourly_cost: str
    pool_big_hourly_cost: str
    pool_huge_hourly_cost: str
    pool_mem_hourly_cost: str
    subscription_id: str

    @classmethod
    def load(cls) -> CloudConfig:
        """Read both Terraform states and assemble the coordinates."""
        endpoint = _value(str(INFRA_DIR), "batch_account_endpoint")
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        return cls(
            batch_endpoint=endpoint,
            pool_id=_value(str(INFRA_DIR), "pool_id"),
            pool_big_id=_optional_value(str(INFRA_DIR), "pool_big_id"),
            pool_huge_id=_optional_value(str(INFRA_DIR), "pool_huge_id"),
            pool_mem_id=_optional_value(str(INFRA_DIR), "pool_mem_id"),
            storage_account=_value(str(STORE_DIR), "storage_account"),
            share_name=_value(str(STORE_DIR), "share_name"),
            share_key=_value(str(STORE_DIR), "access_key"),
            code_container=_value(str(STORE_DIR), "code_container_name"),
            hourly_cost=_value(str(INFRA_DIR), "hourly_cost"),
            pool_big_hourly_cost=_optional_value(str(INFRA_DIR), "pool_big_hourly_cost"),
            pool_huge_hourly_cost=_optional_value(str(INFRA_DIR), "pool_huge_hourly_cost"),
            pool_mem_hourly_cost=_optional_value(str(INFRA_DIR), "pool_mem_hourly_cost"),
            subscription_id=_value(str(INFRA_DIR), "subscription_id"),
        )
