"""The schema's version, asked of the server and of the migration scripts.

A writer that opens against a server one migration behind fails on its first
INSERT, hours into a run, with a column error nothing upstream can explain.
Comparing the two revisions costs one SELECT and is done once per process.
"""

from __future__ import annotations

from typing import Any

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from src.shared import repo


class SchemaBehindError(RuntimeError):
    """The server's schema is not the one this code was written against."""


def _config(url: str | None = None) -> Config:
    config = Config(str(repo.ROOT / "alembic.ini"))
    if url is not None:
        config.set_main_option(
            "sqlalchemy.url", url.replace("postgresql://", "postgresql+psycopg://", 1)
        )
    return config


def head_revision() -> str:
    """The revision the code expects: the single head of the scripts directory."""
    return ScriptDirectory.from_config(_config()).get_current_head() or ""


def current_revision(engine: Any) -> str:
    """What the server holds; empty when the version table does not exist."""
    with engine.connect() as connection:
        try:
            row = connection.execute(text("SELECT version_num FROM alembic_version")).first()
        except Exception:  # noqa: BLE001 -- a missing table is the answer, not a fault
            connection.rollback()
            return ""
    return str(row[0]) if row else ""


def assert_current(engine: Any) -> None:
    """Refuse a writer whose server is behind (or ahead of) its code."""
    have, want = current_revision(engine), head_revision()
    if have != want:
        raise SchemaBehindError(
            f"record schema is {have or 'unversioned'} and this code expects {want}: "
            "run `poker-solver record-migrate` before writing."
        )


def upgrade(dsn: str) -> tuple[str, str]:
    """Apply every pending migration; returns the revision before and after."""
    from src.adapters.postgres import connect  # noqa: PLC0415 -- one engine, made here

    before = current_revision(connect.engine_for(dsn))
    command.upgrade(_config(dsn), "head")
    return before, head_revision()
