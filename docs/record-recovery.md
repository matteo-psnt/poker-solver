# Recovering the record database

**Point-in-time restore is the whole story.** Publishing stopped on 2026-09-03
and `backfill-record` is deleted; the share holds snapshots and leg documents,
and nothing that can rebuild a run's events, its checkpoints or its evals. This
is what a restore actually involves — measured, not assumed.

## Exercised 2026-09-02

A restore to an 8-minute-old point reproduced the record exactly: 318 runs,
3,738 events, 1,122 checkpoints, 2,238 evals, 13,938 legs, `alembic_version`
intact, JSONB payloads readable. The throwaway server was deleted afterwards.

## The step that is easy to miss

**A restored server inherits NO firewall rules.** It comes up unreachable, and
nothing says so beyond a connection timeout — which looks exactly like the
operator's own IP having rotated. Add one before concluding anything is wrong:

```sh
az postgres flexible-server firewall-rule create \
  --server-name <restored> --resource-group poker-solver-store-rg \
  --name allow-operator \
  --start-ip-address "$(curl -s https://api.ipify.org)" \
  --end-ip-address   "$(curl -s https://api.ipify.org)"
```

Note the CLI takes `--server-name` here and `--name` for the RULE, while
`firewall-rule list` spells the server `--server-name` and `flexible-server
restore` spells it `--name`. Two of those got typed wrong on the first attempt
and one failed silently enough to look like a network problem.

## The restore itself

```sh
az postgres flexible-server restore \
  --name <restored> --resource-group poker-solver-store-rg \
  --source-server poker-solver-record-ca \
  --restore-time 2026-09-02T23:51:48Z          # UTC, >= a few minutes ago
```

Then point the DSN at it by host substitution — admin credentials are preserved:

```sh
export POKER_SOLVER_RECORD_DSN=$(terraform -chdir=infra/store output -raw postgres_dsn \
  | sed 's/poker-solver-record-ca\./<restored>./')
```

## The window

`backup_retention_days = 35`, the Flexible Server maximum, set before publishing
stopped. `geo_redundant_backup_enabled = false`, matching the share's
`Standard_LRS` — there has never been region protection on either, and turning
it on FORCES REPLACEMENT of the server.

## Promoting a restore

The restored server is a separate resource, not in Terraform state. To adopt it,
rename or re-point `postgres_server_name` and `terraform import` it — do NOT
`terraform apply` over the old name while `prevent_destroy` is set on the
original, which is the guard doing its job rather than an obstacle.
