"""A token must cost one `az` subprocess, not one per Azure client.

The sibling of `test_config_concurrency.py`, for the other subprocess on the
read path. `AzureCliCredential.get_token` shells out every call and caches
nothing -- measured 0.38s on a fresh AND on a reused credential -- and a client
is built per command, so a Batch read paid it every time: 0.70s against 0.30s
once the token is shared. A five-part fan-out spawned five.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from src.interfaces.cloud.credential import REFRESH_MARGIN_SECONDS, CachedCliCredential

SCOPE = "https://batch.core.windows.net/.default"
OTHER_SCOPE = "https://management.azure.com/.default"


class FakeToken:
    def __init__(self, token: str, expires_on: float):
        self.token = token
        self.expires_on = expires_on


class CountingCli:
    """Stands in for `az account get-access-token`, counting invocations.

    It sleeps for the same reason the config test's stub does: without it the
    first caller can finish before the others start, and the test would pass
    against an unlocked implementation too.
    """

    def __init__(self, lifetime: float = 3600.0):
        self.calls: list[tuple[str, ...]] = []
        self._lock = threading.Lock()
        self._lifetime = lifetime
        self._issued = 0

    def get_token(self, *scopes: str, **_kwargs: Any) -> FakeToken:
        with self._lock:
            self.calls.append(scopes)
            self._issued += 1
            issued = self._issued
        threading.Event().wait(0.05)
        return FakeToken(f"token-{issued}", time.time() + self._lifetime)


def test_concurrent_clients_shell_out_once():
    cli = CountingCli()
    credential = CachedCliCredential(inner=cli)
    seen: list[str] = []
    guard = threading.Lock()

    def _read():
        token = credential.get_token(SCOPE)
        with guard:
            seen.append(token.token)

    threads = [threading.Thread(target=_read) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cli.calls) == 1, f"shelled out {len(cli.calls)} times"
    assert len(seen) == 6
    assert all(token == seen[0] for token in seen)


def test_different_scopes_are_fetched_separately():
    """The cache must not answer a Cost Management reader with a Batch token."""
    cli = CountingCli()
    credential = CachedCliCredential(inner=cli)

    first = credential.get_token(SCOPE)
    second = credential.get_token(OTHER_SCOPE)

    assert cli.calls == [(SCOPE,), (OTHER_SCOPE,)]
    assert first.token != second.token


def test_a_token_about_to_expire_is_replaced():
    """A read must not start with a token that expires mid-flight."""
    cli = CountingCli(lifetime=REFRESH_MARGIN_SECONDS / 2)
    credential = CachedCliCredential(inner=cli)

    credential.get_token(SCOPE)
    credential.get_token(SCOPE)

    assert len(cli.calls) == 2


def test_a_challenge_bypasses_the_cache():
    """`enable_cae` demands a DIFFERENT token.

    Serving the cached one would answer an ARM challenge with the token that
    just failed it, and keying on scopes alone cannot tell the two apart.
    """
    cli = CountingCli()
    credential = CachedCliCredential(inner=cli)

    credential.get_token(SCOPE)
    credential.get_token(SCOPE)
    credential.get_token(SCOPE, enable_cae=True)

    assert len(cli.calls) == 2, "the cached call and the challenge must not collapse"


def test_get_token_info_is_absent():
    """`BearerTokenCredentialPolicy` prefers `get_token_info` wherever it
    exists, and that path would bypass this cache entirely."""
    assert not hasattr(CachedCliCredential(inner=CountingCli()), "get_token_info")


@pytest.mark.parametrize("module", ["tasks.batch", "cost.billing", "serve_box"])
def test_no_module_builds_its_own_cli_credential(module: str):
    """The defect returns the moment one construction site is missed."""
    import importlib

    imported = importlib.import_module(f"src.interfaces.cloud.{module}")
    assert not hasattr(imported, "AzureCliCredential"), (
        f"{module} imports AzureCliCredential directly; it should take credential.shared()"
    )
