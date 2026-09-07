variable "subscription_id" {
  description = "Same sponsorship subscription the boxes use; only the lifetime differs."
  type        = string
  default     = "f9c31345-15ac-413f-8841-5d0151baca66"
}

variable "location" {
  description = "Must match the boxes' region — a cross-region SMB mount pays WAN latency per file."
  type        = string
  default     = "swedencentral"
}

variable "resource_group" {
  description = "Deliberately NOT the compute resource group, so `just destroy` cannot reach it."
  type        = string
  default     = "poker-solver-store-rg"
}

variable "storage_account_name" {
  description = <<-EOT
    Globally unique across all of Azure, 3-24 chars, lowercase alphanumeric only.
    Change this if `just store-create` fails with a name-taken error.
  EOT
  type        = string
  default     = "pokersolverstore"
}

variable "share_name" {
  type    = string
  default = "poker-data"
}

variable "share_quota_gb" {
  description = <<-EOT
    Provisioned size. Standard shares bill on data actually stored, not on quota,
    so this is a ceiling rather than a cost, and headroom is free.

    512 was NOT enough: 56 archived runs filled it exactly, and a full share
    fails a submit mid-upload with ShareSizeLimitReached rather than at
    validation, so the dispatch path breaks before the work starts. The account
    has large_file_share_enabled (for throughput, see main.tf), so the 5 TiB
    standard ceiling no longer binds; raise this freely.
  EOT
  type        = number
  default     = 4096
}

variable "postgres_location" {
  description = <<-EOT
    The record server's region, and deliberately NOT `location`. The share must
    sit with the boxes -- SMB pays WAN latency per file -- but the database
    must sit with its READERS, and every reader is the laptop: the CLI and the
    console. The nodes write a few rows a rung, each through a fresh
    connection, and cannot tell 5 ms from 120.

    Measured from the laptop to Sweden Central: 116-175 ms per round trip, and
    a cold command spent seven or more of them before its first query. Canada
    Central is Toronto, the same city the laptop usually is.
  EOT
  type        = string
  default     = "canadacentral"
}

variable "postgres_server_name" {
  description = <<-EOT
    Globally unique across Azure; change it if creation fails with a name-taken
    error. `-ca` because the Sweden server still holds `poker-solver-record`
    until it is deleted (postgres.tf, the retired `record`).
  EOT
  type        = string
  default     = "poker-solver-record-ca"
}

variable "postgres_admin_user" {
  type    = string
  default = "solver"
}

variable "postgres_database" {
  type    = string
  default = "record"
}

variable "operator_ip" {
  description = <<-EOT
    The laptop's public address, allowed through the server firewall so the
    console and the CLI can read.

    NO DEFAULT, on purpose. A home ISP rotates this, and a committed default
    would keep opening the firewall to whoever holds the address next -- a
    stale allow-rule that nothing would ever report. Set it per-apply:

        terraform apply -var "operator_ip=$(curl -s https://api.ipify.org)"

    or put it in `infra/store/terraform.tfvars`, which `.gitignore` covers.
    Left unset, the rule is simply not created and the laptop cannot connect --
    which is the safe direction to fail.
  EOT
  type        = string
  default     = null
}

variable "checkpoints_container_name" {
  description = "Blob container holding one tar object per retained rung."
  type        = string
  default     = "checkpoints"
}

variable "code_container_name" {
  description = "Blob container holding one sealed code tarball per dispatch; expired by the lifecycle policy."
  type        = string
  default     = "code"
}

variable "record_live" {
  description = <<-EOT
    Which server the readers and every dispatch are pointed at: `sweden` or
    `canada`. The Canada server is created, copied into and validated while
    this still says `sweden`; the switch is this value and one apply, at a
    moment nothing is running. Deleted with the Sweden server.
  EOT
  type        = string
  default     = "sweden"

  validation {
    condition     = contains(["sweden", "canada"], var.record_live)
    error_message = "record_live is `sweden` or `canada`."
  }
}
