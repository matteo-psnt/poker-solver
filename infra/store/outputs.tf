output "storage_account" {
  value = azurerm_storage_account.store.name
}

output "share_name" {
  value = azurerm_storage_share.data.name
}

output "resource_group" {
  value = azurerm_resource_group.store.name
}

output "subscription_id" {
  value = var.subscription_id
}

# The mount needs the account key. Marked sensitive so it never lands in a log or
# a plan diff; `just store-mount` reads it with `terraform output -raw`, which is
# the only path that prints it, and pipes it straight into the mount credentials.
output "access_key" {
  value     = azurerm_storage_account.store.primary_access_key
  sensitive = true
}

output "smb_path" {
  description = "UNC path the boxes mount."
  value       = "//${azurerm_storage_account.store.name}.file.core.windows.net/${azurerm_storage_share.data.name}"
}

output "code_container_name" {
  value = azurerm_storage_container.code.name
}

output "postgres_host" {
  value = azurerm_postgresql_flexible_server.record_ca.fqdn
}

output "postgres_database" {
  value = azurerm_postgresql_flexible_server_database.record_ca.name
}

# Sensitive so it never reaches a log or a plan diff. `terraform output -raw
# postgres_dsn` is the only path that prints it -- the same shape as the share's
# access key above.
output "postgres_dsn" {
  value = format(
    "postgresql://%s:%s@%s:5432/%s?sslmode=require",
    var.postgres_admin_user,
    urlencode(random_password.postgres.result),
    azurerm_postgresql_flexible_server.record_ca.fqdn,
    azurerm_postgresql_flexible_server_database.record_ca.name,
  )
  sensitive = true
}
