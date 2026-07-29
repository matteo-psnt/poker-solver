variable "subscription_id" {
  description = <<-EOT
    Target subscription. Defaults to the SPONSORSHIP one, not "Azure for Students":
    the student subscription caps at 6 total vCPUs (4 per family), which cannot run
    a production train. When the credit grant moves to a different subscription,
    this is the only value that needs to change.
  EOT
  type        = string
  default     = "f9c31345-15ac-413f-8841-5d0151baca66"
}

variable "location" {
  description = <<-EOT
    Region. Established empirically, not chosen: westeurope refuses new
    deployments outright ("not accepting new customers") and northeurope reports
    capacity restrictions on every D/F SKU tried. swedencentral works.
  EOT
  type        = string
  default     = "swedencentral"
}

variable "resource_group" {
  type    = string
  default = "poker-batch-rg"
}

variable "pool_vm_size" {
  description = <<-EOT
    Node size for the training pool. MEASURED CONSTRAINT: Batch does NOT support
    the Dalds_v6 family in swedencentral -- only Dals_v6, which is the same AMD
    Genoa silicon but WITHOUT a local temp disk (hence data_disk_gb below).

      D8als_v6    8 vCPU / 16 GB  ~$0.40/hr  validating the path
      D16als_v6  16 vCPU / 32 GB  ~$0.80/hr  likely sweet spot (16w ~= 32w past 10M)
      D32als_v6  32 vCPU / 64 GB  ~$1.60/hr  matches modal_app.py's run_train shape

    als_v6 is Gen2-only; the image in main.tf must stay a *-gen2 SKU.
  EOT
  type        = string
  default     = "Standard_D8als_v6"
}

variable "max_nodes" {
  description = <<-EOT
    Ceiling for the autoscale formula, and the STRONGEST cost control here: no
    failure can burn faster than max_nodes x the per-node rate. Everything else
    bounds duration; this bounds rate.

      2 x D8als_v6  ~$0.80/hr  ~$19/day
      4 x D8als_v6  ~$1.60/hr  ~$38/day   (blows a $250 budget in ~6 days)
      4 x D16als_v6 ~$3.20/hr  ~$77/day   (~3 days)

    Deliberately 2 while the Batch path is unproven -- it is a one-line change
    once a real leg has completed end to end, and until then the extra
    parallelism cannot be used anyway. Regional quota allows 8 x D8 or 4 x D16.
  EOT
  type        = number
  default     = 2
}

variable "stall_cpu_floor" {
  description = <<-EOT
    Average CPU (as a FRACTION, 0-1) below which a node is treated as stalled and
    force-deallocated. A deadlocked or stuck-I/O trainer sits near zero; a healthy
    one pegs its cores, so this discriminates well.

    Chosen low on purpose. If Azure ever reports $CPUPercent as 0-100 rather than
    0-1, this threshold simply never fires -- the backstop goes inert rather than
    killing healthy work. That is the safe direction for an unvalidated heuristic.
    Confirm against a live pool with `just autoscale-check` before trusting it.
  EOT
  type        = number
  default     = 0.05
}

variable "stall_window_minutes" {
  description = <<-EOT
    How long CPU must stay below the floor before a node is considered stalled.
    Generous because staging is legitimately low-CPU: copying ~2,000 checkpoint
    files over SMB is I/O-bound, and killing a leg mid-stage would be a false
    positive of the worst kind.
  EOT
  type        = number
  default     = 60
}

variable "data_disk_gb" {
  description = <<-EOT
    Per-node working disk, mounted at /mnt/work. Required because als_v6 has no
    local temp disk: this holds the repo, the 773 MB card abstraction, and the
    run's checkpoints (~1.9 GB per snapshot, times the retained ladder).
  EOT
  type        = number
  default     = 256
}

variable "store_account_name" {
  description = "Storage account of the durable share (infra/store outputs it)."
  type        = string
  default     = "pokersolverstore"
}

variable "store_resource_group" {
  type    = string
  default = "poker-solver-store-rg"
}

variable "store_share_name" {
  type    = string
  default = "poker-data"
}

variable "batch_account_name" {
  description = "Lowercase alphanumeric, 3-24 chars, unique within the region."
  type        = string
  default     = "pokerbatchus31321"
}

variable "key_vault_name" {
  description = <<-EOT
    Globally unique. Soft-delete reserves a name for 90 days after deletion, so
    this defaults to the vault already created and wired to the Batch account --
    import it rather than minting a new name.
  EOT
  type        = string
  default     = "pokerbatchkv31321"
}

variable "batch_service_principal_object_id" {
  description = <<-EOT
    Object id of the first-party "Microsoft Azure Batch" service principal in this
    tenant (app id ddbf3205-c6bd-46ae-8127-60eb93363864). It needs vault access to
    store pool certificates, and -- separately and NOT via Terraform -- Contributor
    at subscription scope so UserSubscription pools can create VMs.
  EOT
  type        = string
  default     = "2736183d-125f-4bd0-8cc8-4f1189c65986"
}

variable "alert_email" {
  type    = string
  default = "matteopesenti229@gmail.com"
}

variable "budget_amount" {
  description = "Monthly budget in USD. An ALERT threshold — Azure budgets never cap spending."
  type        = number
  default     = 250
}

variable "budget_start_date" {
  description = "Must be the first of a month, RFC3339. Azure rejects a start date far in the past."
  type        = string
  default     = "2026-07-01T00:00:00Z"
}

variable "allowed_vm_skus" {
  description = "Policy whitelist — nothing outside this list can be created, by you or by a mistaken command."
  type        = list(string)
  default = [
    "Standard_D8als_v6",
    "Standard_D16als_v6",
    "Standard_D32als_v6",
    "Standard_D16_v5",
    "Standard_E16-4ads_v5",
  ]
}

variable "allowed_locations" {
  description = <<-EOT
    Deliberately broader than the one region in use. Regional capacity
    restrictions appear without notice, and a whitelist of exactly one region
    plus a capacity blip leaves you unable to deploy anywhere.
  EOT
  type        = list(string)
  default = [
    "swedencentral",
    "northeurope",
    "uksouth",
    "germanywestcentral",
    "francecentral",
    "eastus",
    "eastus2",
    "westus2",
  ]
}
