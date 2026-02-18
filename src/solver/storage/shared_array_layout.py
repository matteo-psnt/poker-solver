"""Layout helpers for shared-memory naming and partition ranges."""


def get_shm_name(base: str, session_id: str) -> str:
    """Get session-namespaced shared memory name."""
    return f"{base}_{session_id}"
