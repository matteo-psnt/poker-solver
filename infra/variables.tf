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

    CHANGING THIS REPLACES THE POOL. vm_size forces replacement, so an apply
    that disagrees with the deployed pool destroys and rebuilds it -- which is
    how this default came to be a trap: the live pool had been moved to D16
    out of band while this still read D8, leaving every `just create` one
    confirmation away from silently halving the node.
  EOT
  type        = string
  default     = "Standard_D16als_v6"
}

variable "pool_big_vm_size" {
  description = <<-EOT
    SKU of the `train-big` pool. D32als_v6 = 16 physical cores x SMT at
    $1.376/hr (the measured $0.043/vCPU-hr line). The worker curve on a D16
    is linear per PHYSICAL core (08-23), so this box should ~2x a D16;
    that claim is what the pool exists to test. D48/D64 additionally need
    `allowed_vm_skus` extended.
  EOT
  type        = string
  default     = "Standard_D32als_v6"
}

variable "pool_big_max_nodes" {
  description = <<-EOT
    Autoscale ceiling for `train-big`, and its burn bound: 2 x D32als_v6 is
    ~$2.75/hr. Sized for A/B probes, not campaigns -- raise it deliberately
    if the big box wins and work moves here.
  EOT
  type        = number
  default     = 2
}

variable "pool_huge_vm_size" {
  description = <<-EOT
    SKU of the `train-huge` pool. D64als_v6 = 32 physical cores x SMT at the
    $0.043/vCPU-hr line (~$2.75/hr). The kernel is DRAM-latency-bound and the
    D32/D16 steady-state ratio was 1.76x and rising with run length, so this
    box projects ~3x a D16; that projection is what the pool exists to test.
  EOT
  type        = string
  default     = "Standard_D64als_v6"
}

variable "pool_huge_max_nodes" {
  description = <<-EOT
    Autoscale ceiling for `train-huge`: 4 x D64als_v6 is ~$11.01/hr. The big
    box won and the work moved here, so this is no longer probe-sized.

    PAID FOR OUT OF `max_nodes`, not out of new quota -- the three caps share
    one 1024 vCPU allowance and must clear the +8 offset described there.
    40/2/4 = 960 is deliberate; raise this only by lowering another.
  EOT
  type        = number
  default     = 4
}

variable "max_nodes" {
  description = <<-EOT
    Ceiling for the autoscale formula, and the STRONGEST cost control here: no
    failure can burn faster than max_nodes x the per-node rate. Everything else
    bounds duration; this bounds rate.

      4 x D16als_v6  ~$3.20/hr   ~$77/day
     16 x D16als_v6 ~$12.80/hr  ~$307/day
     30 x D16als_v6 ~$24.00/hr  ~$576/day
     40 x D16als_v6 ~$32.00/hr  ~$768/day
     60 x D16als_v6 ~$48.00/hr ~$1152/day

    Quota went 256 -> 512 and then 512 -> 1024 on 2026-08-22. One `az quota update` on
    `standardDalv6Family` moved `cores` (Total Regional vCPUs) with it, so the
    pair does not need requesting separately, and Batch is in UserSubscription
    mode -- `dedicatedCoreQuota` is null, so there is no second Batch-side
    ceiling behind these.

    NOT 32, and the missing 2 are the point. The previous 16 was justified as
    "16 x 16 vCPU = 256 uses it exactly", and it was never reachable: the usage
    counter runs a constant +8 above the pool's own cores -- 232 against 14
    nodes, 136 against 8 -- and the only 8-wide VM in the subscription is the
    DEALLOCATED `blueprint-server` D8als_v6. The offset is measured; the
    accounting behind it is not established. Either way the pool capped at 14
    carrying a persisted `AllocationFailed: core limit has reached` that read
    like a live fault. Do not size this to consume the quota exactly.

    THE THREE POOL CAPS SHARE ONE QUOTA, so this number is not free to raise.
    40 x 16 + 2 x 32 + 4 x 64 = 960 vCPU of 1024, which the +8 offset above
    leaves reachable. The previous 60/2/2 summed to 1152 -- the whole quota,
    exactly the mistake the paragraph above records. `train` carries scoring,
    which is throwaway; the D64s carry the critical path. Re-do this
    arithmetic before changing ANY of the three.

    `infra/credit_watch.py --daily-burn` is the other half and is NOT derived
    from here -- 1152.00 accompanies this, and still bounds it: 40 D16 + 2 D32
    + 4 D64 is ~$45.76/hr, ~$1098/day worst-case. Against a ~$9.0k remaining
    balance that is ~8 days of runway, which puts `--warn-days 30` permanently
    in alarm. The pool scales to zero at rest, so the figure bounds a runaway,
    not a day. Drop this back to 20-30 when the programme ends.
  EOT
  type        = number
  default     = 40
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
  # Raised from 250 to match what was already deployed: the budget had been
  # raised in the portal and this default never followed, so every plan carried
  # a stray 1000 -> 250 revert waiting to be applied by an unrelated change.
  #
  # 250 is also simply too low to be an alert now. The first full month billed
  # ~$290, so a 250 budget fires all four thresholds every month on ordinary
  # work -- and an alert that always fires is not one. Money is not the
  # constraint on this project; the alert exists to catch a RUNAWAY, and 1000 is
  # roughly two weeks of the pool pinned at max_nodes.
  default = 1000
}

variable "budget_start_date" {
  description = "Must be the first of a month, RFC3339. Azure rejects a start date far in the past."
  type        = string
  default     = "2026-07-01T00:00:00Z"
}

variable "allowed_vm_skus" {
  description = <<-EOT
    Policy whitelist — nothing outside this list can be created, by you or by a
    mistaken command. A SKU that is not here is refused at CREATE time with a
    403 naming the policy rather than the size, which reads as a capacity
    problem and is trap #4 in infra/README.md.

    The single GPU size is here for the JAX kernel arm and is NOT free to use:
    `StandardNCADSA100v4Family` quota is separate from the Dalv6 family's,
    starts at 0, and is not self-service adjustable. Listing it only means the
    policy will stop refusing it once that quota is granted.

    One GPU SKU, not the family. NC24ads_A100_v4 is a single A100 80GB at
    ~$4.78/hr; the 2- and 4-GPU sizes cost 2x and 4x that for a measurement
    that needs one, and this list is a guardrail against exactly the stray
    apply that would reach for them.
  EOT
  type        = list(string)
  default = [
    "Standard_D8als_v6",
    "Standard_D16als_v6",
    "Standard_D32als_v6",
    "Standard_D64als_v6",
    "Standard_D16_v5",
    "Standard_E16-4ads_v5",
    "Standard_NC24ads_A100_v4",
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
