"""Layout helpers for shared-memory naming and partition ranges."""

import hashlib


def get_shm_name(base: str, session_id: str) -> str:
    """Get session-namespaced shared memory name.

    The session component is a 16-hex-char blake2b digest of ``session_id``,
    not the id itself: macOS caps POSIX shm names at 31 chars, and the old
    approach (truncating the id to 8 chars upstream) collapsed every same-day
    run id into one namespace ("run-2026..."), so concurrent runs would
    collide on each other's segments. Hashing keeps names short and unique
    while letting callers hold the full, readable session id.
    """
    token = hashlib.blake2b(session_id.encode(), digest_size=8).hexdigest()
    return f"{base}_{token}"
