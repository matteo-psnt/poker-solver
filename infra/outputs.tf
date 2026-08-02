# Consumed by submit.py and the justfile, which read `terraform output -json` once
# rather than duplicating any of these values in a second config file.

output "batch_account" {
  value = azurerm_batch_account.main.name
}

output "batch_account_endpoint" {
  description = "Data-plane endpoint the Batch SDK talks to."
  value       = azurerm_batch_account.main.account_endpoint
}

output "pool_id" {
  value = azurerm_batch_pool.train.name
}

output "pool_vm_size" {
  value = azurerm_batch_pool.train.vm_size
}

output "resource_group" {
  value = azurerm_resource_group.main.name
}

output "subscription_id" {
  value = var.subscription_id
}



output "hourly_cost" {
  description = "USD/hr PER NODE while running. The pool is 0 nodes at rest."
  # Both sides lowercased. Batch echoes the SKU back in its own casing --
  # `STANDARD_D16als_v6`, not the `Standard_D16als_v6` written in variables.tf --
  # so a literal-keyed lookup silently fell through to the default and this
  # output read "see the Azure price list" for the SKU actually deployed.
  value = lookup({
    "standard_d8als_v6"  = "$0.40/hr/node"
    "standard_d16als_v6" = "$0.80/hr/node"
    "standard_d32als_v6" = "$1.60/hr/node"
  }, lower(azurerm_batch_pool.train.vm_size), "see the Azure price list")
}

output "autoscale_formula" {
  description = "The live formula, so `just autoscale-check` evaluates exactly what is deployed."
  value       = azurerm_batch_pool.train.auto_scale[0].formula
}
