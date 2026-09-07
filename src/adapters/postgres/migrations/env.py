"""Alembic's entry point, pointed at the models rather than at a config file.

The URL comes from the environment, never from `alembic.ini`: the DSN carries a
password, and a file in the repo is the wrong place for one.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.adapters.postgres.models import Base

config = context.config
target_metadata = Base.metadata

DSN_ENV = "POKER_SOLVER_RECORD_DSN"


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url") or os.environ.get(DSN_ENV)
    if not url:
        raise RuntimeError(
            f"{DSN_ENV} is unset. `poker-solver record-migrate` resolves it from the store "
            "state; the bare `alembic` CLI needs it exported."
        )
    # Alembic writes the URL into a config that logs it; psycopg is the driver.
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
