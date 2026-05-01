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
# NOTHING IS EXPOSED. There is no HTTP ingress and no public port but SSH. The
# server binds loopback on the box and is reached by forwarding a local port over
# SSH, so authentication is Azure's and the key's rather than something invented
# here -- which matters, because this process will happily tell anyone who asks
# what a run's strategy is.

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
    write_files:
      # Credentials in a root-only file rather than in /etc/fstab, where the
      # mount options are world-readable.
      - path: /etc/smbcredentials/store.cred
        permissions: "0600"
        owner: root:root
        content: |
          username=${data.azurerm_storage_account.store.name}
          password=${data.azurerm_storage_account.store.primary_access_key}
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

  # The ONLY inbound rule. The blueprint server's own port is deliberately absent:
  # it binds 127.0.0.1 on the box, so the only route to it is a forwarded port
  # inside this SSH session. Adding 8790 here would publish an unauthenticated
  # read interface to a trained run, which is a mistake worth making impossible
  # rather than merely discouraged.
  security_rule {
    name                       = "ssh"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "22"
    source_address_prefix      = var.ssh_source_address
    destination_address_prefix = "*"
  }
}

resource "azurerm_public_ip" "serve" {
  name                = "serve-ip"
  location            = azurerm_resource_group.serve.location
  resource_group_name = azurerm_resource_group.serve.name
  # Static, so stopping the box overnight does not invalidate the SSH config
  # line and the tunnel command you saved.
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

  lifecycle {
    # cloud-init runs once at first boot; a re-render (a changed storage key, a
    # tweaked comment) would otherwise silently replace the box and throw away
    # the copied run for no gain. Change it deliberately, by tainting.
    ignore_changes = [custom_data]
  }
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
