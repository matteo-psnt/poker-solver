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
#
# AND IT IS A ONE-WAY DOOR, which the first draft of this file got wrong by
# suggesting VNet could be revisited later. Azure: "We currently don't support
# moving in and out of a virtual network." Changing our mind means a NEW server
# and a dump/restore, not a reconfiguration. Choosing public access here is
# therefore a decision about the life of this server.

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
  #
  # Its connection ceiling is the SKU's own: B2s defaults to max_connections=429,
  # of which 414 are user connections. That is ten times the 40-node pool cap, so
  # nothing here sets the parameter -- an override could only LOWER it, and an
  # earlier draft of this file did exactly that by assuming the default was small.
  #
  # Note also that BURSTABLE SKUs have no built-in PgBouncer. If pooling is ever
  # needed it means moving to General Purpose, not enabling a feature.
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

# Batch nodes leave through a SHARED SNAT address and have NO public IP of their
# own (measured by the same probe), so there is no node address to allow. This
# is the only rule that can admit them.
#
# It is broader than an IP allowlist and worth being honest about: it admits
# connections from any Azure resource, not only ours. What stands between that
# and the data is TLS plus a 32-character generated password. The tighter
# version is VNet integration, which costs a full pool recreation AND a new
# server (see the one-way door above) -- so it is a migration, not a tweak.
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
