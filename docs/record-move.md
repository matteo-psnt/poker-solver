# Moving the record to Canada Central

The record database sat in Sweden Central because `infra/store` has one
`location` and the share must sit with the boxes. Every reader of the database
is the laptop, and from Toronto the round trip to Sweden is ~110 ms; a cold
command spent seven or more of those before its first query. The database now
has its own region, `postgres_location`, and the share stays where it was.

The Sweden server is not moved, it is REPLACED: a location change recreates a
Flexible Server, and this one is the only copy. So the old server stays in
state as the retired `azurerm_postgresql_flexible_server.record`, the new one
is `record_ca`, and the data crosses by dump and restore. This is that runbook.

## Before

1. `git status --short` clean on `main`, this branch merged.
2. **Drain the queue.** A task seals its DSN at dispatch, so anything running
   through the switch writes to Sweden until it ends. `uv run poker-solver
   tasks --limit 20` shows nothing running. If something must run through,
   see *Delta* below.
3. **A version-16 `pg_dump`.** Homebrew's `postgresql@14` refuses a 16 server.
   `brew install libpq` puts one at `/opt/homebrew/opt/libpq/bin`.

## Apply

1. `just plan` — expect one change, the allowed-locations policy gaining
   `canadacentral`. `just create`. The policy is subscription-wide and denies
   the store apply until this lands.
2. `terraform -chdir=infra/store plan` — expect **5 to add, 0 to change, 0 to
   destroy**: the server, its database, the TLS setting, `allow-azure-services`
   and `operator_ca[0]`. A plan that destroys `operator[0]` means
   `infra/store/terraform.tfvars` with `operator_ip` is missing from this
   checkout. `just store-create`. The server is the slow part, ~10 minutes;
   the recipe ends by forgetting the cached coordinates.
3. `uv run poker-solver record-admit` — the rule Terraform made carries the
   tfvars address, which is whatever the ISP handed out on 09-02.

## Copy

```sh
PGBIN=/opt/homebrew/opt/libpq/bin
NEW=$(terraform -chdir=infra/store output -raw postgres_dsn)
OLD=${NEW/poker-solver-record-ca./poker-solver-record.}    # same password

$PGBIN/pg_dump --format=custom --no-owner --no-privileges "$OLD" -f record.dump
$PGBIN/pg_restore --no-owner --no-privileges --dbname "$NEW" record.dump
```

68 MB; a minute or two each way. Then the same counts on both:

```sh
for dsn in "$OLD" "$NEW"; do
  $PGBIN/psql "$dsn" -Atc "SELECT (SELECT count(*) FROM runs), (SELECT count(*) FROM run_events),
    (SELECT count(*) FROM checkpoints), (SELECT count(*) FROM evals), (SELECT count(*) FROM legs),
    (SELECT count(*) FROM progress), (SELECT count(*) FROM abstractions),
    (SELECT version_num FROM alembic_version)"
done
```

And the readers, which now answer from Canada without being told:
`uv run poker-solver runs --limit 3`, `ledger --limit 3`, `tasks --limit 3`.
Delete `record.dump` afterwards; it holds the whole record.

## Delta

If a task ran through the window, its rows are on the Sweden server and not
in the dump. Rather than merge, copy again from scratch:

```sh
ADMIN=${NEW/\/record?/\/postgres?}
$PGBIN/psql "$ADMIN" -c 'DROP DATABASE record' -c 'CREATE DATABASE record'
$PGBIN/pg_dump --format=custom --no-owner --no-privileges "$OLD" -f record.dump
$PGBIN/pg_restore --no-owner --no-privileges --dbname "$NEW" record.dump
```

Terraform's database resource is by name and is unaffected.

## After

- The nodes reach Canada the way they reached Sweden: the
  `allow-azure-services` rule, which admits any Azure address in any region.
  Their writes go from ~5 ms to ~110 ms each, a few per rung, in a background
  thread.
- `docs/record-recovery.md` names the new server. Point-in-time history on it
  starts at the restore; the Sweden server keeps the 35 days before.
- **Leave the Sweden server for a week**, then delete it: remove the retired
  section of `infra/store/postgres.tf`, drop its `prevent_destroy`, apply, and
  add `moved { from = azurerm_postgresql_flexible_server.record_ca, to =
  ...record }` (and the same for its four dependents) so the live server gets
  the plain name back.
