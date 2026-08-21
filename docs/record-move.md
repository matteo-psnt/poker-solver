# Moving the record to Canada Central

The record database sat in Sweden Central because `infra/store` has one
`location` and the share must sit with the boxes. Every reader of the database
is the laptop, and from Toronto the round trip to Sweden is ~110 ms; a cold
command spent seven or more of those before its first query. The database now
has its own region, `postgres_location`, and the share stays where it was.

The Sweden server is not moved, it is REPLACED: a location change recreates a
Flexible Server, and this one is the only copy. So the old server stays in
state as the retired `azurerm_postgresql_flexible_server.record`, the new one
is `record_ca`, the data crosses by dump and restore, and `record_live` says
which one the readers see. Three stages: build and validate, switch, delete.

## Stage 1 -- build and validate, readers still on Sweden

`record_live` defaults to `sweden`, so this stage changes nothing anyone reads.

1. `just plan` — expect one change, the allowed-locations policy gaining
   `canadacentral`. `just create`. The policy is subscription-wide and denies
   the store apply until this lands.
2. `terraform -chdir=infra/store plan` — expect **5 to add, 0 to change, 0 to
   destroy**: the server, its database, the TLS setting, `allow-azure-services`
   and `operator_ca[0]`. A plan that destroys `operator[0]` means
   `infra/store/terraform.tfvars` with `operator_ip` is missing from this
   checkout. `just store-create`. The server is the slow part, ~10 minutes.
3. Admit the laptop to the new server -- `record-admit` addresses the LIVE one:

   ```sh
   az postgres flexible-server firewall-rule create \
     --server-name poker-solver-record-ca --resource-group poker-solver-store-rg \
     --name allow-operator \
     --start-ip-address "$(curl -s https://api.ipify.org)" \
     --end-ip-address   "$(curl -s https://api.ipify.org)"
   ```
4. A version-16 `pg_dump`: Homebrew's `postgresql@14` refuses a 16 server.
   `brew install libpq` puts one at `/opt/homebrew/opt/libpq/bin`.
5. Copy and compare (below), then read the copy through the override:
   `POKER_SOLVER_RECORD_DSN=$NEW uv run poker-solver runs --limit 3`, and the
   same for `ledger` and `tasks --skip-reconcile`, against the same commands
   without it. Delete `record.dump` afterwards; it holds the whole record.

## The copy

```sh
PGBIN=/opt/homebrew/opt/libpq/bin
OLD=$(terraform -chdir=infra/store output -raw postgres_dsn)              # while record_live=sweden
NEW=${OLD/poker-solver-record./poker-solver-record-ca.}                   # same password

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

A second copy into a database that already holds one is a conflict, not a
merge. Empty it first:

```sh
ADMIN=${NEW/\/record?/\/postgres?}
$PGBIN/psql "$ADMIN" -c 'DROP DATABASE record' -c 'CREATE DATABASE record'
```

Terraform's database resource is by name and is unaffected.

## Stage 2 -- switch, at a quiet moment

1. **Drain the queue.** A task seals its DSN at dispatch, so anything running
   through the switch writes to Sweden until it ends. `uv run poker-solver
   tasks --limit 20` shows nothing running.
2. **Copy again**, from scratch: the stage-1 copy is stale by now. Empty,
   dump, restore, compare counts.
3. `record_live = "canada"` in `infra/store/terraform.tfvars`, then
   `just store-create` -- the outputs flip and the recipe forgets the cached
   coordinates. `uv run poker-solver record-admit` now addresses Canada.
4. `runs`, `ledger`, `tasks` answer from Canada with no override.

## Stage 3 -- delete Sweden, right after

The data is not worth a second server: once stage 2 reads correctly, delete.
In `infra/store/postgres.tf` remove the retired section and `record_live`
with its `locals`; point the outputs at `record_ca`; then

```sh
terraform -chdir=infra/store plan      # expect: 5 to destroy, nothing else
```

`prevent_destroy` on the Sweden server and its database refuses this until
those two lifecycle lines go too -- that is the guard doing its job; remove
them in the same change. Apply, then add `moved { from =
azurerm_postgresql_flexible_server.record_ca, to = ...record }` (and the same
for its four dependents) so the live server gets the plain name back.

## After

- The nodes reach Canada the way they reached Sweden: the
  `allow-azure-services` rule, which admits any Azure address in any region.
  Their writes go from ~5 ms to ~110 ms each, a few per rung, in a background
  thread.
- `docs/record-recovery.md` names the new server. Point-in-time history on it
  starts at the restore; what was before is gone with Sweden.
