variable "subscription_id" {
  description = "Same sponsorship subscription as the pool and the store; only the lifetime differs."
  type        = string
  default     = "f9c31345-15ac-413f-8841-5d0151baca66"
}

variable "location" {
  description = <<-EOT
    Must match the store's region. The box copies a run and the card abstraction
    off the share at startup, and a cross-region SMB read pays WAN latency on
    every one of ~5,500 checkpoint files.
  EOT
  type        = string
  default     = "swedencentral"
}

variable "resource_group" {
  description = <<-EOT
    Its own group, like the store's and for the same reason: `just destroy` tears
    down the training pool, and it must not be able to reach a box that is
    serving a run to a browser.
  EOT
  type        = string
  default     = "poker-solver-serve-rg"
}

variable "vm_size" {
  description = <<-EOT
    CONSTRAINED BY POLICY, not chosen on merit. The subscription carries an
    `allowed_vm_skus` policy assignment (see infra/variables.tf), and anything
    outside its list is refused at create time with a 403 that names the policy
    rather than the size -- which is how this box first failed to build.

    On merit this wants ~2 vCPU and 8 GB: it is a READER. What has to fit in RAM
    is the run's flat arrays (~2 GB for a production run) plus the 773 MB card
    abstraction, which is mmapped. One request buckets 1326 combos and reads a
    few hundred table rows -- no traversal, no solve, CPU near idle between
    clicks.

    `Standard_D8als_v6` is the smallest thing the policy permits, so the box is
    roughly 4x the machine it needs. That costs nothing at rest, because it
    deallocates when idle; it costs ~$0.40/hr instead of ~$0.10/hr while
    actually in use, which is the cheaper mistake than widening a policy that
    exists to stop a stray apply from creating a 64-core node.

    If a smaller box is ever worth it, add the SKU to `allowed_vm_skus` and
    apply the ROOT infra state -- carefully: that state also owns the live
    training pool, and `pool_vm_size` forces replacement.
  EOT
  type        = string
  default     = "Standard_D8als_v6"
}

variable "data_disk_gb" {
  description = <<-EOT
    Local working disk at /mnt/work, holding the code snapshot, the card
    abstraction and whichever run is being served.

    A MANAGED disk rather than the SKU's ephemeral temp disk, on purpose: it
    survives a deallocate, so stopping the box overnight does not mean re-copying
    ~1.6 GB of small files off SMB before it can answer again.

    One production checkpoint is ~1.9 GB; 128 leaves room for a few, plus the
    abstraction and the repo.
  EOT
  type        = number
  default     = 128
}

variable "admin_username" {
  description = "Login for the SSH tunnel. Password auth is disabled outright in main.tf."
  type        = string
  default     = "solver"
}

variable "ssh_public_key" {
  description = <<-EOT
    Public half of the key that opens the tunnel. Empty means "use
    ~/.ssh/id_ed25519.pub", which is what a laptop actually has -- naming a path
    that does not exist fails at apply with a clearer error than a VM you cannot
    log into.
  EOT
  type        = string
  default     = ""
}

variable "dns_label" {
  description = <<-EOT
    Prefix of the box's name, giving it
    `<label>.<region>.cloudapp.azure.com`. A NAME rather than a bare IP because
    Caddy needs one to obtain a certificate -- Let's Encrypt does not issue for
    IP addresses, and a self-signed cert would mean teaching the console to skip
    verification, which is a setting nobody ever un-skips.

    Globally unique within the region; change it if apply reports a conflict.
  EOT
  type        = string
  default     = "poker-blueprint"
}

variable "idle_timeout_seconds" {
  description = <<-EOT
    How long the server waits with no request before exiting -- which the
    systemd unit escalates into deallocating the whole box.

    30 minutes is long enough to read a chart, think, and come back; short
    enough that a forgotten tab costs about a pound. It is the backstop, not the
    primary control: the console has a stop button, and this is what catches the
    times you close the laptop instead of pressing it.
  EOT
  type        = number
  default     = 1800
}

variable "ssh_source_address" {
  description = <<-EOT
    The ONLY address allowed to reach port 22, and the only inbound rule at all.

    "*" would expose SSH to the internet. Key-only auth makes that survivable
    rather than safe, and this box holds a mount to the durable experiment
    store. Set it to your current public IP (`curl -s ifconfig.me`) and re-apply
    when it changes; that is a 30-second cost against the one thing here worth
    protecting.
  EOT
  type        = string
  default     = "*"
}

variable "store_account_name" {
  description = "Storage account holding the share. Must match infra/store's output."
  type        = string
  default     = "pokersolverstore"
}

variable "store_resource_group" {
  type    = string
  default = "poker-solver-store-rg"
}

variable "share_name" {
  type    = string
  default = "poker-data"
}
