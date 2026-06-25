# Consumed by submit.py and the justfile, which read `terraform output -json` once
# rather than duplicating any of these values in a second config file.

locals {
  # MEASURED, not quoted from a price page -- see the `hourly_cost` output.
  node_rates = {
    "standard_d8als_v6"  = "$0.344/hr/node"
    "standard_d16als_v6" = "$0.688/hr/node"
    "standard_d32als_v6" = "$1.376/hr/node"
    # 64 x the $0.043/vCPU-hr line the three measured SKUs all land on;
    # re-measure with `poker-solver cost` once the pool has billing history.
    "standard_d64als_v6" = "$2.752/hr/node"
  }
}

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

output "pool_big_id" {
  value = azurerm_batch_pool.train_big.name
}

output "pool_huge_id" {
  value = azurerm_batch_pool.train_huge.name
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
  #
  # MEASURED, not quoted from a price page. Cost Management, subscription total
  # 2026-07-26..2026-08-09: D16als_v6 $214.12 over 311.218 node-hours = $0.6880,
  # D8als_v6 $13.80 over 40.117 = $0.3440. Both land on the rate exactly, so the
  # per-vCPU line is $0.043/hr and D32 follows. The previous $0.80/$0.40/$1.60
  # were round numbers nothing had ever checked, and they overstated the compute
  # bill by 16%.
  #
  # RE-MEASURE rather than edit by hand if a SKU is added: `poker-solver cost`
  # prints billed dollars beside billed node-hours for exactly this purpose.
  value = lookup(local.node_rates, lower(azurerm_batch_pool.train.vm_size), "see the Azure price list")
}

output "pool_big_hourly_cost" {
  description = "USD/hr PER NODE on the train-big pool. Same table, other SKU."
  value       = lookup(local.node_rates, lower(azurerm_batch_pool.train_big.vm_size), "see the Azure price list")
}

output "pool_huge_hourly_cost" {
  description = "USD/hr PER NODE on the train-huge pool. Same table, other SKU."
  value       = lookup(local.node_rates, lower(azurerm_batch_pool.train_huge.vm_size), "see the Azure price list")
}
