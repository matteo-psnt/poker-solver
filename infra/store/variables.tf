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
    so this is a ceiling rather than a cost. 773 MB of abstractions plus archived
    runs (a sealed production run is multi-GB) fits comfortably in 512.
  EOT
  type        = number
  default     = 512
}
