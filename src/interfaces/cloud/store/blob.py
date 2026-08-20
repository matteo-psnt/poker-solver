"""Minting the checkpoint container's SAS, at dispatch.

THE LAPTOP SIDE of `shared.cloudtask.node.blobstore`. The SDK lives here and
cannot live there: `archive` is imported by the wrapper before `uv sync`, on
the interpreter the start task installs, so the node speaks REST through
`urllib` and needs a URL that already carries its own authorisation.

Minted per dispatch rather than stored anywhere. A SAS expires by design, so
there is nothing to rotate and nothing to leak from a config file -- and the
task it is sealed into outlives it only if the task outlives `SAS_LIFETIME`,
which is why that is measured against the longest job this pool runs rather
than the longest task.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.shared.cloudtask.kinds import TaskName

CONTAINER = "checkpoints"

# The ops that PUT a rung. Everything else fetches one, and a fetch has no
# business holding a credential that can overwrite what it read. Kept beside the
# minting rather than at the call site so there is one list of who may write.
#
# MIGRATE_CHECKPOINTS is here because moving the history IS writing rungs -- it
# was omitted, handed a read-only SAS, and every upload came back 403
# `AuthorizationPermissionMismatch`. The scoping worked exactly as designed;
# the list was wrong. A task that writes checkpoints must be named here, and
# `test_every_kind_is_classified` now fails if a new one is not.
WRITES_CHECKPOINTS = frozenset({TaskName.TRAIN, TaskName.TRAIN_PCS, TaskName.MIGRATE_CHECKPOINTS})

# Longer than any job, not any task. A ladder score fans out 30 rungs behind one
# dispatch and the last of them can start hours after the first; a SAS that
# expired between them would fail a task that had done nothing wrong, and the
# retry would fail identically.
SAS_LIFETIME = timedelta(days=7)

# Backdated, because the node's clock is not this machine's. A SAS whose window
# opens "now" is refused by a node running a few seconds behind, and the failure
# reads as a 403 that looks exactly like a bad signature.
CLOCK_SKEW = timedelta(minutes=15)


def container_sas(account: str, key: str, *, write: bool) -> str:
    """A URL for the checkpoint container, carrying its own authorisation.

    `write` is the discriminator between a training task and a reader: a score
    or an evaluate FETCHES rungs and must not be able to overwrite one. Both
    need `list`, because a fetch of "the current rung" resolves a name first.
    """
    from azure.storage.blob import (  # noqa: PLC0415 -- Azure only when dispatching
        AccountSasPermissions,
        ResourceTypes,
        generate_account_sas,
    )

    now = datetime.now(UTC)
    token = generate_account_sas(
        account_name=account,
        account_key=key,
        resource_types=ResourceTypes(container=True, object=True),
        permission=AccountSasPermissions(read=True, list=True, write=write, create=write),
        start=now - CLOCK_SKEW,
        expiry=now + SAS_LIFETIME,
    )
    return f"https://{account}.blob.core.windows.net/{CONTAINER}?{token}"
