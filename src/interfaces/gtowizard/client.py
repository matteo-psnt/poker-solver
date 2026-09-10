"""HTTP against the GTO Wizard research API.

Their engine answers 502/503/504 when it is busy rather than queueing, and the
reference client treats exactly those three as retryable -- so a run of tens of
thousands of hands has to as well, or it dies hours in on a transient.

Their 409 on `POST /hands` is the one 4xx that is not an answer: it is the
concurrency cap, nothing has been dealt, and a slot frees on its own. See
:data:`CAPPED`.

``/leaderboard`` and ``/winnings`` need no key. That is what makes the field and
the two reference floors (Always Fold, Check Call) readable before we hold one.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import httpx

from src.interfaces.errors import CommandError
from src.interfaces.gtowizard.protocol import GAME_NAME, Frame

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

BASE_URL = "https://researcher.gtowizard.com"
KEY_ENV = "GTOWIZARD_API_KEY"

# Their AI is versioned and the versions are DIFFERENT OPPONENTS: Claude Opus
# 4.6 reads -13.62 bb/100 on v1 and -19.74 on v2. Every read is pinned so a
# score cannot silently span two of them, and the pin is recorded beside it.
DEFAULT_VERSION = 2

BUSY = frozenset(
    {HTTPStatus.BAD_GATEWAY, HTTPStatus.SERVICE_UNAVAILABLE, HTTPStatus.GATEWAY_TIMEOUT}
)

# Their 20-hand concurrency cap answers `POST /hands` with a 409. That is BACK
# PRESSURE, not an answer: no card has been dealt, and a slot frees the moment
# one of our own hands ends. Read as a failure it cost 177 of 600 hands in one
# run. Retryable only on `new_hand` -- a 409 from `act` says the hand is
# already over, and repeating that changes nothing.
CAPPED = HTTPStatus.CONFLICT


@dataclass(frozen=True)
class Retry:
    """Backoff for a busy engine. Bounded, so a wedged server ends the run.

    JITTERED, because the failure this fixes was not one slow request. Their
    engine answers 503 intermittently and recovers within seconds, but every
    concurrent worker backs off on the same schedule, so a deterministic pause
    marches them into the server together and they exhaust together: 30 hands
    at concurrency 5 failed 30/30 while a lone request beside them succeeded.
    Full jitter spreads the herd over the window instead of stacking it on the
    edge of one.
    """

    attempts: int = 8
    base_seconds: float = 0.5
    max_seconds: float = 30.0

    def pause(self, attempt: int) -> float:
        """Uniform over [0, window]: full jitter, not the window itself."""
        return random.uniform(0.0, min(self.base_seconds * 2**attempt, self.max_seconds))


def key_from_environment() -> str:
    """The key, or a refusal that says where one comes from.

    Read HERE rather than in each command, because the refusal is the same
    refusal and the second copy of it drifted from the first.
    """
    key = os.environ.get(KEY_ENV)
    if not key:
        raise CommandError(
            f"No API key. Set ${KEY_ENV} to the key GTO Wizard approved "
            "(request one at https://benchmark.gtowizard.com/)."
        )
    return key


class BenchmarkClient:
    """One authenticated session against the research API.

    The key is read from the environment by the caller and never logged: it is
    the whole credential, it is chosen once at registration and cannot be
    rotated from here, and a run's log is published to the share.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 60.0,
        retry: Retry | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = client or httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, follow_redirects=True
        )
        self._retry = retry or Retry()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BenchmarkClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _call(self, method: str, path: str, *, wait_for_slot: bool = False, **kwargs: Any) -> Any:
        """One request, retried while the engine says it is busy.

        A 4xx is otherwise never retried: a bad key, a rejected action or a hand
        that is already over are all answers, and repeating them just burns the
        clock. ``wait_for_slot`` adds the one exception -- see :data:`CAPPED`.
        """
        retryable = BUSY | ({CAPPED} if wait_for_slot else frozenset())
        last: httpx.HTTPStatusError | None = None
        for attempt in range(self._retry.attempts):
            response = self._client.request(method, path, **kwargs)
            if response.status_code in retryable:
                last = httpx.HTTPStatusError(
                    f"engine busy ({response.status_code})",
                    request=response.request,
                    response=response,
                )
                pause = self._retry.pause(attempt)
                waiting = (
                    "All hand slots taken" if response.status_code == CAPPED else "Engine busy"
                )
                logger.warning("%s (%s); retrying in %.1fs", waiting, response.status_code, pause)
                time.sleep(pause)
                continue
            if response.status_code == HTTPStatus.UNAUTHORIZED:
                raise CommandError(
                    f"The API rejected the key. Set ${KEY_ENV} to the key GTO Wizard "
                    "approved, exactly as registered."
                )
            if response.is_error:
                raise CommandError(f"{method} {path} -> {response.status_code}: {response.text}")
            return response.json()
        if last is not None and last.response.status_code == CAPPED:
            raise CommandError(
                f"{method} {path}: every one of their hand slots stayed taken for "
                f"{self._retry.attempts} attempts. Hands left open by an earlier run hold "
                "slots indefinitely; `poker-solver benchmark-drain` folds them and gives "
                f"them back. Their answer was: {last.response.text}"
            ) from last
        raise CommandError(
            f"{method} {path}: the engine stayed busy for {self._retry.attempts} attempts."
        ) from last

    # ---- playing ---------------------------------------------------------

    def new_hand(self, game_name: str = GAME_NAME) -> Frame:
        return Frame.parse(
            self._call("POST", "/hands", json={"game_name": game_name}, wait_for_slot=True)
        )

    def act(self, hand_id: int, action: str, amount: int | None = None) -> Frame:
        body: dict[str, Any] = {"action": action}
        if amount is not None:
            body["amount"] = int(amount)
        return Frame.parse(self._call("POST", f"/hands/{hand_id}/act", json=body))

    def in_progress(self, game_name: str = GAME_NAME) -> Sequence[Frame]:
        payload = self._call("GET", "/hands/in-progress", params={"game_name": game_name})
        return [Frame.parse(entry) for entry in payload]

    # ---- reading ---------------------------------------------------------

    def results(
        self, game_name: str = GAME_NAME, *, version: int | None = DEFAULT_VERSION
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"game_name": game_name}
        if version is not None:
            params["gto_wizard_version"] = version
        return self._call("GET", "/results", params=params)

    def leaderboard(
        self,
        game_name: str = GAME_NAME,
        *,
        limit: int = 100,
        min_hands: int = 1000,
        version: int | None = DEFAULT_VERSION,
    ) -> list[dict[str, Any]]:
        """The public board. Needs no key, so it works before one is approved."""
        params: dict[str, Any] = {
            "game_name": game_name,
            "limit": limit,
            "min_hands": min_hands,
        }
        if version is not None:
            params["gto_wizard_version"] = version
        return list(self._call("GET", "/leaderboard", params=params))
