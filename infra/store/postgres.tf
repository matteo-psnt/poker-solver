# The run record, as a database.
#
# In THIS state rather than the compute one for the same reason the share is:
# `just destroy` tears down the boxes, and the record of what was learned on
# them must not go with it.
#
# PUBLIC ENDPOINT, NOT VNET-INTEGRATED, and that is a measured decision rather
# than a shortcut. VNet integration requires the pools to sit in a subnet, a
# pool's network configuration is fixed at CREATION, and there is no subnet in
# `infra/main.tf` today -- so adopting one would recreate all four pools and
# kill every running task, including other sessions'. A probe on a live node
# (`net-probe-192720-14202`) measured outbound TCP to 5432 as OPEN, so the
# public path is proven to work from exactly where it needs to.

resource "random_password" "postgres" {
  length  = 32
  special = true
  # Azure rejects these in the admin password rather than escaping them, and the
  # failure arrives as a generic provisioning error several minutes in.
  override_special = "-_=+"
}

resource "azurerm_postgresql_flexible_server" "record" {
  name                = var.postgres_server_name
  resource_group_name = azurerm_resource_group.store.name
  location            = azurerm_resource_group.store.location
  version             = "16"

  administrator_login    = var.postgres_admin_user
  administrator_password = random_password.postgres.result

  # Burstable. The workload is a few hundred rows a second at its peak and the
  # read side is single-digit milliseconds on 10x the current record (measured
  # locally at 3,000 runs / 2M events / 759 MB). General Purpose would be paying
  # for headroom nothing has asked for; this SKU is one `terraform apply` away
  # from B2ms or a GP tier if that stops being true.
  sku_name   = "B_Standard_B2s"
  storage_mb = 32768
  # Grows rather than failing writes at the ceiling. The record is 26 MB today,
  # so this is not a number anyone should have to watch.
  auto_grow_enabled = true

  backup_retention_days        = 14
  geo_redundant_backup_enabled = false

  public_network_access_enabled = true

  # NOT set: `high_availability`. It doubles the bill to protect against a zone
  # loss, and the thing being protected is a record that is reconstructible from
  # the share for as long as the share exists.

  tags = local.tags

  lifecycle {
    # DELIBERATELY ABSENT while this is being brought up: `prevent_destroy`.
    # Recreating the server is a normal move until it holds the only copy of
    # anything. Turn it on in the same commit that stops publishing JSON to the
    # share -- that is the moment this becomes irreplaceable.
    ignore_changes = [zone]
  }
}

# TLS is enforced by default on Flexible Server and this makes it explicit, so a
# future `terraform plan` shows an attempt to weaken it rather than silently
# accepting one.
resource "azurerm_postgresql_flexible_server_configuration" "require_tls" {
  name      = "require_secure_transport"
  server_id = azurerm_postgresql_flexible_server.record.id
  value     = "ON"
}

# A node opens one connection per task and holds it for the task's lifetime, so
# the ceiling that matters is concurrent TASKS, not workers. B2s defaults to a
# low max_connections; this lifts it well clear of the 40-node pool cap plus the
# console plus headroom for a fan-out.
resource "azurerm_postgresql_flexible_server_configuration" "max_connections" {
  name      = "max_connections"
  server_id = azurerm_postgresql_flexible_server.record.id
  value     = "200"
}

# Batch nodes leave through a SHARED SNAT address and have NO public IP of their
# own (measured by the same probe), so there is no node address to allow. This
# is the only rule that can admit them.
#
# It is broader than an IP allowlist and worth being honest about: it admits
# connections from any Azure resource, not only ours. What stands between that
# and the data is TLS plus a 32-character generated password. The tighter
# version is VNet integration, which costs a full pool recreation -- revisit it
# the next time the pools are being replaced for another reason anyway.
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.record.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# The laptop, so the console and the CLI can read. A home ISP rotates this;
# when `psql` starts timing out rather than refusing, this variable is why.
resource "azurerm_postgresql_flexible_server_firewall_rule" "operator" {
  count            = var.operator_ip == null ? 0 : 1
  name             = "allow-operator"
  server_id        = azurerm_postgresql_flexible_server.record.id
  start_ip_address = var.operator_ip
  end_ip_address   = var.operator_ip
}

resource "azurerm_postgresql_flexible_server_database" "record" {
  name      = var.postgres_database
  server_id = azurerm_postgresql_flexible_server.record.id
  charset   = "UTF8"
  collation = "en_US.utf8"

  lifecycle {
    prevent_destroy = true
  }
}
