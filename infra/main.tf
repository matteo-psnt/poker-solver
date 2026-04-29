# Cloud training: everything that exists in Azure, declared in one place.
#
# Split of responsibilities, deliberately:
#   Terraform  owns what EXISTS   (Batch account, pool, guardrails)
#   justfile   owns what HAPPENS  (jobs and tasks, via the `az batch` CLI)
#
# Jobs and tasks are NOT declared here on purpose: `azurerm_batch_job` exposes no
# useful properties and there is no Terraform resource for a task at all. A
# submission is a runtime act, not infrastructure.
#
# The pool holds ZERO nodes at rest. Nodes exist only while tasks are queued or
# running, so there is no idle compute cost and nothing to remember to shut down.

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
  features {
    key_vault {
      # Soft-delete reserves a vault name for 90 days. Never purge on destroy:
      # recovering the name is worse than leaving the vault behind.
      purge_soft_delete_on_destroy = false
    }
  }
}

data "azurerm_client_config" "current" {}

# The share lives in infra/store -- a SEPARATE root module and state, so tearing
# down compute cannot touch the experiment record. Read it by name rather than via
# terraform_remote_state: a data lookup keeps the two modules independently
# appliable, so neither can break the other's plan.
data "azurerm_storage_account" "store" {
  name                = var.store_account_name
  resource_group_name = var.store_resource_group
}

locals {
  # Consistent tagging makes cost attribution possible later, when there is more
  # than one thing in the subscription.
  tags = {
    project    = "poker-solver"
    managed_by = "terraform"
  }

  # Node-level setup, run once per node by the pool start task.
  #   /mnt/work    the data disk -- fast local scratch for runs and checkpoints
  #   $AZ_BATCH_NODE_MOUNTS_DIR/shared   the durable share (SMB)
  start_script = <<-EOT
    set -euo pipefail

    # Find the data disk by its PROPERTIES, not by a fixed path. The obvious
    # /dev/disk/azure/scsi1/lun0 symlink is created by the Azure Linux Agent's
    # udev rules and does NOT exist on the Batch node image -- assuming it fails
    # the start task, which fails the whole node (wait_for_success).
    #
    # Three independent guards, because the failure mode of getting this wrong is
    # formatting the OS disk: the candidate must be the expected size, carry no
    # partitions, and be mounted nowhere.
    # A `for` over command substitution, not a pipe into `while`: a piped while
    # runs in a subshell where `return` cannot exit the function.
    find_data_disk() {
      local d
      for d in $(lsblk -dnro NAME,TYPE | awk '$2=="disk"{print $1}'); do
        [ "$(lsblk -dnro SIZE "/dev/$d")" = "${var.data_disk_gb}G" ] || continue
        # Mountpoints of the disk AND any partition on it -- catches a disk whose
        # partition is mounted even though the disk itself is not.
        [ -z "$(lsblk -nro MOUNTPOINT "/dev/$d" | tr -d '[:space:]')" ] || continue
        [ -z "$(lsblk -nro NAME "/dev/$d" | tail -n +2)" ] || continue
        echo "/dev/$d"
        return 0
      done
      return 1
    }

    # IDEMPOTENCE FIRST. Batch RETRIES a failed start task on the same node, and
    # `find_data_disk` deliberately skips disks that are already mounted -- so
    # once attempt 1 has mounted the disk, attempt 2 finds nothing and the FATAL
    # below bricks the node with `starttaskfailed`. Two nodes were lost to
    # exactly this, with the disk sitting healthy at /mnt/work in the very lsblk
    # output printed by the error. Anything already mounted there is the answer,
    # not a problem.
    if mountpoint -q /mnt/work; then
      echo "data disk: already mounted at /mnt/work (start task retry)"
    else
      DEV=""
      for _ in $(seq 30); do
        DEV=$(find_data_disk || true)
        [ -n "$DEV" ] && break
        sleep 2
      done
      if [ -z "$DEV" ]; then
        echo "FATAL: no unpartitioned ${var.data_disk_gb}G data disk found. Block devices:" >&2
        lsblk -o NAME,SIZE,TYPE,MOUNTPOINT >&2
        exit 1
      fi
      echo "data disk: $DEV"
      blkid "$DEV" >/dev/null 2>&1 || mkfs.ext4 -F "$DEV"
      mkdir -p /mnt/work
      mount "$DEV" /mnt/work
    fi
    chmod 1777 /mnt/work
    apt-get update -qq
    apt-get install -y -qq build-essential python3-dev libgomp1 git rsync
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

    # The wrapper's interpreter, installed to a SHARED location and linked onto
    # PATH. Both parts are load-bearing: this start task runs as an elevated
    # POOL-scoped auto-user, tasks run as a different non-elevated one whose
    # HOME is its own working directory, so anything uv leaves under the start
    # user's HOME is invisible to them. Without the explicit dirs the task died
    # in 208ms having recorded nothing.
    export UV_PYTHON_INSTALL_DIR=/opt/uv-python
    export UV_PYTHON_BIN_DIR=/usr/local/bin
    /usr/local/bin/uv python install 3.13
    chmod -R a+rX /opt/uv-python
    SHARE="$AZ_BATCH_NODE_MOUNTS_DIR/shared"
    mkdir -p /mnt/work/data/combo_abstraction /mnt/work/data/runs
    if [ -d "$SHARE/combo_abstraction" ]; then
      cp -ru "$SHARE/combo_abstraction/." /mnt/work/data/combo_abstraction/
    fi

    # The start task runs ELEVATED; tasks do not. Everything created above is
    # therefore root-owned, and a task trying to `mkdir /mnt/work/data/runs` gets
    # "Permission denied" -- the 1777 on /mnt/work itself does not help, because
    # the failure is one level down inside a root-owned `data/`. Open up the whole
    # tree once, here, rather than making every task fight for it.
    chmod -R a+rwX /mnt/work

    echo "node ready: $(df -h /mnt/work | tail -1)"
  EOT
}

# --------------------------------------------------------------------------- #
# Guardrails. Declared FIRST because they constrain everything below: the pool's
# VM size must satisfy the SKU policy it sits alongside.
# --------------------------------------------------------------------------- #

# Preventive and zero-lag: a VM outside this list is rejected at request time.
# This is the only genuinely hard cost control available on this subscription --
# `spendingLimit` is Off and cannot be re-enabled for a Sponsorship offer, so
# budgets below are alerts, not caps.
#
# CRITICAL: this list must contain the pool's SKU family. Batch cannot use
# Dalds_v6, so the list is Dals_v6 -- a mismatch here surfaces as an opaque
# `AllocationFailed` on the pool, not as a policy error.
resource "azurerm_subscription_policy_assignment" "allowed_vm_skus" {
  name                 = "poker-allowed-vm-skus"
  display_name         = "poker-solver: allowed VM sizes"
  subscription_id      = "/subscriptions/${var.subscription_id}"
  policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/cccc23c7-8427-4f53-ad12-b6a63eb452b3"

  parameters = jsonencode({
    listOfAllowedSKUs = { value = var.allowed_vm_skus }
  })
}

# Region restriction is a cost control as much as a compliance one: it stops
# resources appearing somewhere you never look and never think to clean up.
resource "azurerm_subscription_policy_assignment" "allowed_locations" {
  name                 = "poker-allowed-locations"
  display_name         = "poker-solver: allowed locations"
  subscription_id      = "/subscriptions/${var.subscription_id}"
  policy_definition_id = "/providers/Microsoft.Authorization/policyDefinitions/e56962a6-4747-49cd-b67b-bf8b01975c4c"

  parameters = jsonencode({
    listOfAllowedLocations = { value = var.allowed_locations }
  })
}

# Alerts only -- Azure budgets never stop consumption, and cost data lags hours.
# Forecast notifications matter more than actual ones here: they fire while there
# is still time to act.
resource "azurerm_consumption_budget_subscription" "monthly" {
  name            = "poker-solver-monthly"
  subscription_id = "/subscriptions/${var.subscription_id}"
  amount          = var.budget_amount
  time_grain      = "Monthly"

  time_period {
    start_date = var.budget_start_date
  }

  dynamic "notification" {
    for_each = [50, 75, 90, 100]
    content {
      enabled        = true
      threshold      = notification.value
      operator       = "GreaterThanOrEqualTo"
      threshold_type = "Actual"
      contact_emails = [var.alert_email]
    }
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThanOrEqualTo"
    threshold_type = "Forecasted"
    contact_emails = [var.alert_email]
  }
}

# --------------------------------------------------------------------------- #
# Batch account (UserSubscription mode)
# --------------------------------------------------------------------------- #

resource "azurerm_resource_group" "main" {
  name     = var.resource_group
  location = var.location
  tags     = local.tags
}

# UserSubscription mode requires a Key Vault -- Batch stores pool certificates in
# it. `enabled_for_deployment` is mandatory and non-obvious: without it account
# creation fails with "the specified Key Vault is not enabled for deployment".
resource "azurerm_key_vault" "batch" {
  name                   = var.key_vault_name
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  tenant_id              = data.azurerm_client_config.current.tenant_id
  sku_name               = "standard"
  enabled_for_deployment = true
  # 90 = Azure's default, and IMMUTABLE once the vault exists. This vault was
  # created ahead of Terraform (during the UserSubscription-mode probe) and then
  # imported, so the config has to match what is on disk rather than express a
  # preference -- Terraform cannot change it, and a mismatch fails every apply.
  soft_delete_retention_days = 90
  tags                       = local.tags
}

# The operator needs to manage the vault; Batch needs to store secrets in it.
resource "azurerm_key_vault_access_policy" "operator" {
  key_vault_id       = azurerm_key_vault.batch.id
  tenant_id          = data.azurerm_client_config.current.tenant_id
  object_id          = data.azurerm_client_config.current.object_id
  secret_permissions = ["Get", "List", "Set", "Delete", "Recover", "Purge"]
}

resource "azurerm_key_vault_access_policy" "batch" {
  key_vault_id       = azurerm_key_vault.batch.id
  tenant_id          = data.azurerm_client_config.current.tenant_id
  object_id          = var.batch_service_principal_object_id
  secret_permissions = ["Get", "List", "Set", "Delete", "Recover"]
}

# NOTE: the subscription-scope Contributor grant to the Batch service principal is
# deliberately NOT declared here. It is a one-time tenant-level act, and putting a
# subscription-scope role assignment in the same state as the compute would let a
# `terraform destroy` revoke it. See infra/README.md.
resource "azurerm_batch_account" "main" {
  name                 = var.batch_account_name
  resource_group_name  = azurerm_resource_group.main.name
  location             = azurerm_resource_group.main.location
  pool_allocation_mode = "UserSubscription"
  tags                 = local.tags

  key_vault_reference {
    id  = azurerm_key_vault.batch.id
    url = azurerm_key_vault.batch.vault_uri
  }

  depends_on = [azurerm_key_vault_access_policy.batch]
}

# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #

resource "azurerm_batch_pool" "train" {
  name                = "train"
  resource_group_name = azurerm_resource_group.main.name
  account_name        = azurerm_batch_account.main.name
  display_name        = "poker-solver training pool"
  vm_size             = var.pool_vm_size
  node_agent_sku_id   = "batch.node.ubuntu 22.04"

  # `just panic` disables autoscale and forces a resize to zero, which leaves a
  # pending resize operation behind. Without this flag the very next `just create`
  # -- the documented way to re-arm autoscale after a panic -- fails with "because
  # of pending resize operation", stranding the pool with autoscale off. The
  # recovery path has to work unattended, so the pool always yields to Terraform.
  stop_pending_resize_operation = true

  # Gen2 image, non-negotiable: the als_v6 family cannot boot a Generation 1
  # image, and the failure surfaces as a generic `AllocationFailed` whose real
  # cause is buried in resizeErrors[].valuesProperty[].value.
  storage_image_reference {
    publisher = "canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  # Scale to zero at rest. $PendingTasks counts active AND running tasks, so a
  # node is never counted idle while its task is still going; `taskcompletion`
  # then guarantees a node is not deallocated out from under a running task -- both
  # matter here because a task runs for hours.
  # MEASURED CONSTRAINTS, both found by running this against Azure:
  #
  #  * Comments are `//`. A `#` comment is rejected ("Invalid character"), and
  #    an invalid formula means the pool silently stops autoscaling.
  #  * GetSample MUST use the two-argument form. The one-argument form demands
  #    70% sample coverage and THROWS below it, and a thrown formula aborts the
  #    whole evaluation -- so an idle pool (thin data, by definition) refuses to
  #    scale UP. Observed: `InsufficientSampleData: wanted 70%, received 50%`
  #    while the evaluation had already computed the correct target.
  #
  # There is deliberately NO CPU-based stall clause here. One was written and
  # then removed: `$CPUPercent` reports a sample PERCENTAGE on this pool but
  # yields no usable values -- `avg`/`max`/`Count` over it all fail, and `avg`
  # returns NaN even with a live busy node and 99% coverage. A backstop that
  # cannot fire is worse than none, because it reads like protection. Hang
  # protection is the task-level maxWallClockTime instead; see infra/README.md.
  auto_scale {
    evaluation_interval = "PT5M"
    formula             = <<-EOT
      maxNodes = ${var.max_nodes};
      pending = max($PendingTasks.GetSample(5 * TimeInterval_Minute, 20));
      $TargetDedicatedNodes = min(maxNodes, pending);
      $NodeDeallocationOption = taskcompletion;
    EOT
  }

  # The durable share, mounted on every node. Code snapshots, card abstractions,
  # published runs and eval records all travel through here.
  #
  # MOUNT OPTIONS ARE A RELIABILITY CONTROL, not tuning. Two nodes have gone
  # `unusable` with MountConfigurationError MID-LEG (not at startup), stranding a
  # task that Batch then reports as `running` forever. Both happened while
  # publishing multi-GB checkpoint snapshots, which is the only sustained SMB
  # load this pool generates.
  #
  #   vers=3.1.1    Azure Files' recommended dialect; 3.0 predates the reconnect
  #                 and encryption improvements, and this share supports it.
  #   nosharesock   a dedicated TCP connection for this mount rather than one
  #                 shared across mounts to the same server -- one stalled
  #                 operation then cannot take the whole mount down with it.
  #   actimeo=30    caches attributes for 30s. Publishing walks thousands of
  #                 files with cp -u, which stats every one; without this each
  #                 stat is a round trip and the metadata traffic alone can
  #                 exhaust the share's IOPS allowance.
  #   mfsymlinks    symlink support, so a copy cannot fail on one unexpectedly.
  mount {
    azure_file_share {
      account_name        = data.azurerm_storage_account.store.name
      account_key         = data.azurerm_storage_account.store.primary_access_key
      azure_file_url      = "https://${data.azurerm_storage_account.store.name}.file.core.windows.net/${var.store_share_name}"
      relative_mount_path = "shared"
      mount_options       = "-o vers=3.1.1,dir_mode=0777,file_mode=0777,serverino,nosharesock,actimeo=30,mfsymlinks"
    }
  }

  # als_v6 has NO local temp disk, and a run needs the 773 MB abstraction plus
  # multi-GB checkpoints. This disk is the node's working storage; the start task
  # formats and mounts it before anything writes.
  data_disks {
    lun                  = 0
    disk_size_gb         = var.data_disk_gb
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
  }

  # Runs once per node, before any task. Inlined rather than fetched from a
  # resource file so the pool has no external dependency to keep in sync.
  #
  # Three jobs: make the data disk usable, install the interpreter toolchain, and
  # pull the card abstraction down to LOCAL disk. That last one is deliberate --
  # the abstraction is mmapped during training, and every page fault against the
  # SMB mount would be a network round-trip.
  # base64, NOT jsonencode: the script has to survive being a single Batch
  # command-line string. jsonencode turns newlines into a literal `\n` (so the
  # whole script arrives as one physical line), escapes `&&` to `&&`,
  # and any `$`-escaping applied on top is doubly wrong. A base64 blob is
  # alphanumeric, so nothing in it can be reinterpreted by Terraform or by bash.
  start_task {
    command_line       = "/bin/bash -c 'echo ${base64encode(local.start_script)} | base64 -d | bash'"
    task_retry_maximum = 1
    wait_for_success   = true

    user_identity {
      auto_user {
        elevation_level = "Admin"
        scope           = "Pool"
      }
    }
  }
}
