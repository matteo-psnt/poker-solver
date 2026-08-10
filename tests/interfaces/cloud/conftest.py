"""Nothing in this package may touch the real caches.

`billing` gained a DISK cache -- under `$POKER_SOLVER_CACHE`, shared by every
worktree and every process on the machine -- so a test that exercises it writes
where a real `poker-solver cost` will later read.

That is not hypothetical. It happened while this was being written: the stub in
`test_read_cost.py` (`total: 1.0`, one Virtual Machines row) landed in
`~/.cache/poker-solver/billing/` during a test run and sat there as a real cache
entry. It was harmless only because its window key differed from the one the
command asks for. Had the dates lined up, `poker-solver cost` would have
reported a fabricated $1.00 as the authority on spend -- which is precisely the
failure this whole change set exists to remove.

So the isolation is autouse and lives here rather than in one module: the next
test to call into `billing` should not have to remember. Scoped to this package
rather than the whole suite on purpose -- the abstraction caches elsewhere are
legitimately shared, and redirecting those would silently recompute the river's
2.6M boards on every run.
"""

from __future__ import annotations

import pytest

from src.interfaces.cloud import billing


@pytest.fixture(autouse=True)
def isolated_caches(tmp_path, monkeypatch):
    """Redirect the disk cache and clear the in-process memo around each test.

    Redirected rather than disabled, so the caching itself stays under test --
    and cleared on the way IN as well as out, because a memo entry surviving
    from a previous test would make a stubbed call pass without being made.
    """
    billing._MEMO.clear()
    monkeypatch.setattr(billing.cache, "cache_root", lambda: tmp_path)
    yield
    billing._MEMO.clear()
