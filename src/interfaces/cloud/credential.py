"""One `az` subprocess per token, instead of one per Azure client.

``AzureCliCredential.get_token`` shells out to ``az account get-access-token``
on every call and caches nothing -- 0.38s, measured identical on a fresh and a
reused credential. The token is cached on the *client's* pipeline, so a fresh
client per command paid it on every read: 0.70s against 0.30s once shared.

Cached behind the credential rather than by sharing a client, because a shared
client is what :func:`src.interfaces.commands._compose.fan_out` rules out, and
Azure's sync clients are not documented as safe across threads.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from azure.identity import AzureCliCredential

if TYPE_CHECKING:
    from azure.core.credentials import AccessToken

REFRESH_MARGIN_SECONDS = 300.0
"""Reuse a token only while it has this long left, so a read cannot start with
a token that expires mid-flight."""


class CachedCliCredential:
    """An ``AzureCliCredential`` that shells out once per token, not per call.

    Exposes ``get_token`` and deliberately NOT ``get_token_info``:
    ``BearerTokenCredentialPolicy`` prefers the latter wherever it exists, and
    that path would bypass this cache entirely.
    """

    def __init__(self, inner: Any = None) -> None:
        self._inner = AzureCliCredential() if inner is None else inner
        self._tokens: dict[tuple[str, ...], AccessToken] = {}
        self._locks: dict[tuple[str, ...], threading.Lock] = {}
        self._guard = threading.Lock()

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        """A cached token for ``scopes``, fetching at most once per expiry.

        Any keyword argument bypasses the cache. The one that matters is
        ``enable_cae``, which an ARM challenge sets to demand a *different*
        token: serving the cached one would answer the challenge with the token
        that just failed it, and keying on scopes alone cannot tell them apart.
        """
        if kwargs:
            return self._inner.get_token(*scopes, **kwargs)

        cached = self._tokens.get(scopes)
        if cached is not None and self._usable(cached):
            return cached

        # Per-scope, so a Batch reader and a Cost Management reader arriving
        # together still overlap -- one lock would serialise them behind a
        # subprocess neither of them needs.
        with self._lock_for(scopes):
            cached = self._tokens.get(scopes)
            if cached is not None and self._usable(cached):
                return cached
            fresh: AccessToken = self._inner.get_token(*scopes)
            self._tokens[scopes] = fresh
            return fresh

    @staticmethod
    def _usable(token: AccessToken) -> bool:
        return token.expires_on - time.time() > REFRESH_MARGIN_SECONDS

    def _lock_for(self, scopes: tuple[str, ...]) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(scopes, threading.Lock())


_SHARED: CachedCliCredential | None = None
_SHARED_GUARD = threading.Lock()


def shared() -> CachedCliCredential:
    """The process-wide credential. Every Azure client here takes this one."""
    global _SHARED
    with _SHARED_GUARD:
        if _SHARED is None:
            _SHARED = CachedCliCredential()
        return _SHARED
