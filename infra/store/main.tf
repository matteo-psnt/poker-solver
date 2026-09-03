# The durable experiment store: outlives every box, by construction.
#
# SEPARATE STATE, ON PURPOSE. This is its own Terraform root module in its own
# resource group, so `just destroy` -- which tears down the compute -- cannot
# reach it. Boxes are disposable; the record of what was learned on them is not.
# The two are only coupled by the share name, which the boxes mount at runtime.
#
# What lives here: card abstractions (distributed once instead of pushed per box),
# per-run eval records, the baseline pointer, and sealed archived runs.
#
# What does NOT live here: active run directories. A production checkpoint is
# ~2,000 small files and the read path mmaps them; SMB turns every page fault into
# a network round-trip and offers no atomic replace. Runs are born and stay on a
# box's local NVMe and are published here only once sealed. Pointing `runs_dir` at
# this share would be slow AND unsafe -- see infra/README.md.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  subscription_id = var.subscription_id
  features {}
}

locals {
  tags = {
    project    = "poker-solver"
    managed_by = "terraform"
    lifecycle  = "durable" # never destroyed with the compute
  }
}

resource "azurerm_resource_group" "store" {
  name     = var.resource_group
  location = var.location
  tags     = local.tags

  # The whole point of this module. Removing the store means losing every
  # experiment record, so it may not go away as a side effect of anything.
  lifecycle {
    prevent_destroy = true
  }
}

resource "azurerm_storage_account" "store" {
  name                = var.storage_account_name
  resource_group_name = azurerm_resource_group.store.name
  location            = azurerm_resource_group.store.location
  account_tier        = "Standard"
  # LRS: this is a convenience copy plus an archive, and the authoritative ledger
  # is still the local one. Geo-redundancy would double the cost to protect
  # against a regional loss that would not lose unique data.
  account_replication_type = "LRS"
  # Without this a standard share is capped at ~60 MiB/s TOTAL, split across every
  # node publishing at once: measured 2026-08-23 at 60 nodes, a 9-minute 30M train
  # spent 26 minutes publishing its ~8 GB ladder. One-way (cannot be disabled),
  # no cost change on LRS, lifts the cap to the 10 GiB/s large-share limit.
  large_file_share_enabled = true
  # SMB from the boxes needs no public blob access, and the mount uses the account
  # key over TLS.
  https_traffic_only_enabled      = true
  allow_nested_items_to_be_public = false
  tags                            = local.tags

  lifecycle {
    prevent_destroy = true
  }
}

resource "azurerm_storage_share" "data" {
  name               = var.share_name
  storage_account_id = azurerm_storage_account.store.id
  quota              = var.share_quota_gb

  # HOT, not TransactionOptimized. The premium tier buys cheap transactions at
  # roughly twice the per-GiB price, and it was the right trade when the record
  # was written to this share a document at a time: 241 MILLION transactions on
  # 2026-08-25, where Hot's per-operation rate would have dominated everything.
  #
  # The record moved to Postgres on 09-03 and the share stopped being written
  # per-record. Measured the same day: 1.3 million transactions, a ~200x drop,
  # against ~830 GiB of mostly-cold checkpoints. The premium now buys nothing
  # and is charged on every GiB.
  #
  # Hot rather than Cool because a resume still FETCHES a rung: Cool halves
  # storage again but multiplies the read, and this share exists to be read from
  # by nodes. Checkpoints are moving to Blob, where lifecycle tiering can do
  # what Files cannot -- this is the right tier for what REMAINS.
  access_tier = "Hot"

  lifecycle {
    prevent_destroy = true
  }
}
