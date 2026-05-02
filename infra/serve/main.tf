# The blueprint host: one long-lived box that serves a trained run for reading.
#
# SEPARATE STATE, ON PURPOSE -- the third of three, and the lifetimes are the
# whole reason there are three:
#
#   infra/         the Batch pool.  Disposable. Scales 0->N->0; `just destroy`
#                  is a normal part of its life.
#   infra/store/   the durable share. Outlives every box.
#   infra/serve/   THIS. Long-lived, but not precious: it holds a *copy* of a
#                  run, so losing it costs a re-copy and nothing else.
#
# WHY THIS IS NOT A BATCH TASK. It was nearly built as one, which would have been
# wrong in a way worth writing down. The pool is for work that finishes: the
# autoscale formula counts pending-and-running tasks to decide how many nodes to
# hold, `taskcompletion` deallocation assumes a task ends, and the task-level
# maxWallClockTime exists specifically to kill anything that does not. A server
# never finishes, so every one of those mechanisms is pointed at it. `TaskName`
# being a closed enum of three training ops is the same statement in code.
#
# WHY A VM AND NOT CONTAINER APPS. Two reasons, and the second is the real one.
# There is no container image anywhere in this project -- deploy is a code
# snapshot plus `uv sync`, which a VM reuses directly. And both artifacts this
# needs must sit on LOCAL disk: a checkpoint is ~5,500 small files that the read
# path mmaps, and SMB turns every page fault into a network round trip (the same
# constraint written up in infra/store/main.tf). A scale-to-zero container would
# re-pay ~1.6 GB of small-file copying on every cold start, so the serverless
# property that would justify the extra machinery is the one property this
# workload cannot use. A warm container app is a VM with a registry attached.
#
# NOTHING IS EXPOSED, AND THE TAILNET IS THE AUTH BOUNDARY. There are no inbound
# NSG rules at all -- not 443, not 22. The box joins a tailnet on boot and the
# console reaches it at `http://<hostname>:8790` over WireGuard.
#
# This replaced a public 443 endpoint with Caddy terminating TLS and checking a
# bearer token, and it deleted every part of that: the certificate, the DNS
# label Let's Encrypt needed, the reverse proxy, and the token itself. The token
# existed only because the port was public. Identity now comes from the tailnet,
# which is the thing that already knows who your devices are -- and a secret you
# do not have is a secret that cannot leak.
#
# Which matters, because this process will tell anyone who reaches it exactly
# what a run's strategy is. Narrow it further with a Tailscale ACL on
# `tag:blueprint` if the tailnet ever has a device you would not hand that to.
#
# BREAK-GLASS IS THE SERIAL CONSOLE. With no port 22, a box whose `tailscale up`
# failed would be unreachable -- so boot diagnostics are on, and Azure's serial
# console works without any networking at all.

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
    virtual_machine {
      # An `-auto-approve` destroy should not leave a 128 GB disk behind billing
      # quietly. The disk holds only copies.
      delete_os_disk_on_deletion = true
    }
  }
}

data "azurerm_storage_account" "store" {
  name                = var.store_account_name
  resource_group_name = var.store_resource_group
}

locals {
  tags = {
    project    = "poker-solver"
    managed_by = "terraform"
    role       = "blueprint-server"
  }

  # Falls back to the key a laptop actually has. Read at plan time, so a missing
  # file fails before anything is created rather than after.
  ssh_key = var.ssh_public_key != "" ? var.ssh_public_key : file("~/.ssh/id_ed25519.pub")

  # Runs once, at first boot. Three jobs, and NOT starting the server: which run
  # to serve is an operational choice that changes far more often than the box
  # does, so it is a command you run over the tunnel rather than baked into an
  # image.
  #
  #   /mnt/work    the managed data disk -- survives a deallocate
  #   /mnt/shared  the durable store, read-only by intent (see the mount below)
  cloud_init = <<-EOT
    #cloud-config
    package_update: true
    packages:
      - cifs-utils
      - git
      - curl
      - apt-transport-https
    write_files:
      # Credentials in a root-only file rather than in /etc/fstab, where the
      # mount options are world-readable.
      - path: /etc/smbcredentials/store.cred
        permissions: "0600"
        owner: root:root
        content: |
          username=${data.azurerm_storage_account.store.name}
          password=${data.azurerm_storage_account.store.primary_access_key}
      # Which run this box serves. An env file rather than a baked-in argument,
      # because the run changes far more often than the box does -- edit this and
      # `systemctl restart blueprint`.
      - path: /etc/blueprint.env
        permissions: "0644"
        content: |
          RUN=
          RUNS_DIR=/mnt/work/runs
          IDLE_TIMEOUT=${var.idle_timeout_seconds}

      # The unit, and the whole on-demand story in six lines.
      #
      # `ExecStopPost` is the hinge: the server exits when it has been idle, and
      # THAT is what deallocates the box. Note the guard -- it fires only on a
      # clean exit, so a crash restarts the service instead of billing you for a
      # box that then sits there having failed. `Restart=on-failure` is the other
      # half of that pair.
      - path: /etc/systemd/system/blueprint.service
        permissions: "0644"
        content: |
          [Unit]
          Description=Blueprint server
          After=network-online.target mnt-work.mount mnt-shared.mount
          Wants=network-online.target

          [Service]
          Type=simple
          User=${var.admin_username}
          EnvironmentFile=/etc/blueprint.env
          Environment=POKER_SOLVER_CACHE=/mnt/work/cache
          WorkingDirectory=/mnt/work/code
          ExecStart=/home/${var.admin_username}/.local/bin/uv run poker-solver blueprint-serve \
            --run ${"$"}{RUN} --runs-dir ${"$"}{RUNS_DIR} --idle-timeout ${"$"}{IDLE_TIMEOUT}
          ExecStopPost=/usr/local/bin/deallocate-if-idle
          Restart=on-failure
          RestartSec=10

          [Install]
          WantedBy=multi-user.target

      # Only on a clean exit, and only via the VM's own managed identity -- there
      # are no credentials on this box to steal, and the identity's role is
      # scoped to this resource group and nothing else.
      - path: /usr/local/bin/deallocate-if-idle
        permissions: "0755"
        content: |
          #!/bin/bash
          if [ "${"$"}{EXIT_STATUS:-1}" != "0" ]; then
            echo "blueprint exited ${"$"}{EXIT_STATUS} -- not deallocating"
            exit 0
          fi
          # The VM's own id from the instance metadata service, NOT from
          # Terraform: interpolating it here would make custom_data depend on
          # the machine custom_data configures, which is a cycle.
          ID=${"$"}(curl -s -H Metadata:true --noproxy "*" \
            "http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text")
          az login --identity >/dev/null 2>&1 || exit 0
          az vm deallocate --ids "${"$"}ID" --no-wait

    runcmd:
      # -- the data disk -------------------------------------------------------
      # By LUN, via the Azure Linux Agent's udev symlink. Unlike the Batch node
      # image, the stock Ubuntu marketplace image DOES ship those rules, so the
      # pool's three-guard disk hunt is not needed here.
      - |
        set -euo pipefail
        DISK=/dev/disk/azure/scsi1/lun0
        for _ in $(seq 1 30); do [ -e "$DISK" ] && break; sleep 2; done
        if ! blkid "$DISK"; then mkfs.ext4 -F -L work "$DISK"; fi
        mkdir -p /mnt/work
        grep -q '/mnt/work' /etc/fstab || echo "LABEL=work /mnt/work ext4 defaults,nofail 0 2" >> /etc/fstab
        mount -a
        chown -R ${var.admin_username}:${var.admin_username} /mnt/work

      # -- the share -----------------------------------------------------------
      # `ro`: this box reads runs and abstractions and publishes nothing. A
      # reader that cannot write cannot corrupt the one thing that is not a copy.
      - |
        set -euo pipefail
        mkdir -p /mnt/shared
        grep -q '/mnt/shared' /etc/fstab || echo "//${data.azurerm_storage_account.store.name}.file.core.windows.net/${var.share_name} /mnt/shared cifs ro,nofail,vers=3.1.1,credentials=/etc/smbcredentials/store.cred,dir_mode=0555,file_mode=0444,serverino,nosharesock,actimeo=30,mfsymlinks 0 0" >> /etc/fstab
        mount -a

      # -- the tailnet ---------------------------------------------------------
      # Joined before anything else needs the network, and with `--ssh`, which is
      # how you get a shell here: port 22 is closed to the world.
      #
      # `--accept-dns=false` on purpose. Tailscale would otherwise take over
      # /etc/resolv.conf, and this box resolves an Azure Files endpoint for its
      # SMB mount -- a DNS change underneath a live CIFS mount is a stall that
      # looks like a hung server.
      - |
        set -euo pipefail
        curl -fsSL https://tailscale.com/install.sh | sh
        tailscale up \
          --authkey='${var.tailscale_auth_key}' \
          --hostname='${var.tailscale_hostname}' \
          --ssh \
          --accept-dns=false
        curl -sL https://aka.ms/InstallAzureCLIDeb | bash
        systemctl daemon-reload
        systemctl enable blueprint

      # -- the toolchain -------------------------------------------------------
      # uv brings its own Python, so the image's interpreter version is not a
      # constraint here the way it is on a Batch node.
      - |
        set -euo pipefail
        su - ${var.admin_username} -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
        mkdir -p /mnt/work/cache
        chown ${var.admin_username}:${var.admin_username} /mnt/work/cache
        echo 'export POKER_SOLVER_CACHE=/mnt/work/cache' >> /home/${var.admin_username}/.bashrc
  EOT
}

resource "azurerm_resource_group" "serve" {
  name     = var.resource_group
  location = var.location
  tags     = local.tags
}

# --------------------------------------------------------------------------- #
# Network: one subnet, one NIC, and exactly one way in
# --------------------------------------------------------------------------- #

resource "azurerm_virtual_network" "serve" {
  name                = "serve-vnet"
  address_space       = ["10.20.0.0/16"]
  location            = azurerm_resource_group.serve.location
  resource_group_name = azurerm_resource_group.serve.name
  tags                = local.tags
}

resource "azurerm_subnet" "serve" {
  name                 = "serve-subnet"
  resource_group_name  = azurerm_resource_group.serve.name
  virtual_network_name = azurerm_virtual_network.serve.name
  address_prefixes     = ["10.20.1.0/24"]
}

resource "azurerm_network_security_group" "serve" {
  name                = "serve-nsg"
  location            = azurerm_resource_group.serve.location
  resource_group_name = azurerm_resource_group.serve.name
  tags                = local.tags

  # NO INBOUND RULES. Not an omission -- the absence IS the design.
  #
  # Azure's default rules already deny inbound from the internet, so leaving this
  # empty means nothing on this box is reachable from outside the tailnet. The
  # server binds loopback and Tailscale carries traffic to it over WireGuard,
  # which is also why there is no rule for 8790 and must never be one.
  #
  # The public IP below is for OUTBOUND only: joining the tailnet, reaching the
  # share, and installing packages.
}

resource "azurerm_public_ip" "serve" {
  name                = "serve-ip"
  location            = azurerm_resource_group.serve.location
  resource_group_name = azurerm_resource_group.serve.name
  # Static, so stopping the box overnight does not invalidate the SSH config
  # line and the tunnel command you saved.
  # Outbound only -- nothing inbound is permitted to reach it. Static so the
  # address in a log or a support ticket stays meaningful across a deallocate.
  allocation_method = "Static"
  sku               = "Standard"
  tags              = local.tags
}

resource "azurerm_network_interface" "serve" {
  name                = "serve-nic"
  location            = azurerm_resource_group.serve.location
  resource_group_name = azurerm_resource_group.serve.name
  tags                = local.tags

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.serve.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.serve.id
  }
}

resource "azurerm_network_interface_security_group_association" "serve" {
  network_interface_id      = azurerm_network_interface.serve.id
  network_security_group_id = azurerm_network_security_group.serve.id
}

# --------------------------------------------------------------------------- #
# The box
# --------------------------------------------------------------------------- #

resource "azurerm_linux_virtual_machine" "serve" {
  name                  = "blueprint-server"
  location              = azurerm_resource_group.serve.location
  resource_group_name   = azurerm_resource_group.serve.name
  size                  = var.vm_size
  admin_username        = var.admin_username
  network_interface_ids = [azurerm_network_interface.serve.id]
  tags                  = local.tags

  # Not a preference. The only inbound rule is SSH from a single address, and a
  # password would make that one guess away from a shell on a box mounting the
  # experiment store.
  disable_password_authentication = true

  # How the box turns itself off. No credential is stored here to be stolen: the
  # token is issued by the platform to this VM and to nothing else.
  identity {
    type = "SystemAssigned"
  }

  admin_ssh_key {
    username   = var.admin_username
    public_key = local.ssh_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = 64
  }

  source_image_reference {
    publisher = "canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts-gen2"
    version   = "latest"
  }

  custom_data = base64encode(local.cloud_init)

  # The only way in if the tailnet join fails. Managed storage, so there is no
  # storage account to create and nothing to pay for beyond the log itself.
  boot_diagnostics {}

  lifecycle {
    # cloud-init runs once at first boot; a re-render (a changed storage key, a
    # tweaked comment) would otherwise silently replace the box and throw away
    # the copied run for no gain. Change it deliberately, by tainting.
    ignore_changes = [custom_data]
  }
}

# Scoped to this resource group and no wider. "Virtual Machine Contributor"
# rather than "Contributor" for the same reason: the box needs to stop itself and
# nothing else, and a compromised reader should not be able to reach the store.
resource "azurerm_role_assignment" "self_deallocate" {
  scope                = azurerm_resource_group.serve.id
  role_definition_name = "Virtual Machine Contributor"
  principal_id         = azurerm_linux_virtual_machine.serve.identity[0].principal_id
}

resource "azurerm_managed_disk" "work" {
  name                 = "serve-work"
  location             = azurerm_resource_group.serve.location
  resource_group_name  = azurerm_resource_group.serve.name
  storage_account_type = "Premium_LRS"
  create_option        = "Empty"
  disk_size_gb         = var.data_disk_gb
  tags                 = local.tags
}

resource "azurerm_virtual_machine_data_disk_attachment" "work" {
  managed_disk_id    = azurerm_managed_disk.work.id
  virtual_machine_id = azurerm_linux_virtual_machine.serve.id
  lun                = 0
  caching            = "ReadOnly"
}
