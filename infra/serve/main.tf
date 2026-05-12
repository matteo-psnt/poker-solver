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
# HOW YOU REACH IT: A PUBLIC HTTPS ENDPOINT BEHIND A BEARER TOKEN. This is the
# third answer to that question and the reasoning for each is worth keeping,
# because the trade is not obvious and the first two both looked right.
#
#   1. An SSH tunnel. Nothing exposed, auth is the key's. Rejected once the box
#      became something the console starts on demand: a button that first needs
#      you to open a terminal and leave it running is not a button.
#   2. A tailnet. Strictly better on every axis -- no open port, no certificate,
#      no secret -- and abandoned for a reason no design review would surface:
#      Tailscale was not reliable enough for the person using it. An auth path
#      that is down is worse than a weaker one that is up.
#   3. This. Caddy terminates TLS and checks a token; the token is the ONLY
#      thing between the internet and a trained run.
#
# So the token is not belt-and-braces here, it is the whole belt. It costs one
# `eval "$(just serve-env)"` and it is the difference between "everyone I send
# the URL to" and "every scanner that walks the certificate transparency log" --
# and this process will tell whoever reaches it exactly what a run plays.
#
# The blast radius if it leaks is bounded but not nil: a reader gets the
# strategy, and can open play sessions. The share is mounted read-only and the
# session store is capped, so neither the experiment record nor the box's memory
# is at risk. Rotate by tainting `random_password.api_token` and re-applying.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
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

# The only thing standing between the internet and a trained run. Generated
# rather than chosen so it cannot be a memorable string someone reuses, and held
# in state rather than in the repo. `terraform output -raw api_token` is the one
# path that prints it.
resource "random_password" "api_token" {
  length  = 48
  special = false
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
      - debian-keyring
      - debian-archive-keyring
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
      # Caddy terminates TLS and checks the token. A reverse proxy rather than
      # doing this in the app for one reason: the app must stay runnable on a
      # laptop with no certificate and no secret, and an auth layer it could not
      # be started without would make every test carry a token.
      #
      # `header_regexp` on the Authorization header, and a bare 404 otherwise --
      # not 401. A 401 advertises that something authenticated lives here; a 404
      # tells a scanner there is nothing at this address at all.
      - path: /etc/caddy/Caddyfile
        permissions: "0644"
        content: |
          ${var.dns_label}.${var.location}.cloudapp.azure.com {
            @authorized header_regexp auth Authorization ^Bearer\s+${random_password.api_token.result}$$
            handle @authorized {
              reverse_proxy 127.0.0.1:8790
            }
            handle {
              respond 404
            }
          }

      # Which run this box serves. An env file rather than a baked-in argument,
      # because the run changes far more often than the box does -- edit this and
      # `systemctl restart blueprint`.
      - path: /etc/blueprint.env
        permissions: "0644"
        content: |
          RUN=
          RUNS_DIR=/mnt/work/runs
          IDLE_TIMEOUT=${var.idle_timeout_seconds}

      # The unit, and the whole on-demand story.
      #
      # `ExecStopPost` is the hinge: the server exits when it has been idle, and
      # THAT is what deallocates the box. The guard fires only on a CLEAN exit,
      # so a crash restarts the service rather than switching off a box that
      # might just have lost a race with its mounts.
      #
      # But a service that can never start -- an unset RUN, a run that is not on
      # this disk, a broken deploy -- would then restart every 10s forever, and
      # the box would bill all weekend having never served a request. That is the
      # expensive failure, so the restart is BOUNDED: three tries in five
      # minutes, after which systemd gives up, the unit enters `failed`, and
      # `OnFailure` deallocates. A misconfigured box costs nothing.
      - path: /etc/systemd/system/blueprint.service
        permissions: "0644"
        content: |
          [Unit]
          Description=Blueprint server
          After=network-online.target mnt-work.mount mnt-shared.mount
          Wants=network-online.target
          OnFailure=blueprint-deallocate.service
          StartLimitIntervalSec=300
          StartLimitBurst=3

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

      # The other half of the bound above: reached when the service has given up
      # restarting, so the box switches off instead of looping.
      - path: /etc/systemd/system/blueprint-deallocate.service
        permissions: "0644"
        content: |
          [Unit]
          Description=Deallocate this box after the blueprint server gave up

          [Service]
          Type=oneshot
          ExecStart=/usr/local/bin/deallocate-box

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
          exec /usr/local/bin/deallocate-box

      - path: /usr/local/bin/deallocate-box
        permissions: "0755"
        content: |
          #!/bin/bash
          # The VM's own id from the instance metadata service, NOT from
          # Terraform: interpolating it here would make custom_data depend on
          # the machine custom_data configures, which is a cycle.
          ID=${"$"}(curl -s -H Metadata:true --noproxy "*" \
            "http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text")
          az login --identity >/dev/null 2>&1 || exit 0
          az vm deallocate --ids "${"$"}ID" --no-wait

      # Provisioning, as ONE script with a real shebang.
      #
      # Not as `runcmd` entries, which is where this first went wrong: cloud-init
      # concatenates those into a file it runs with /bin/sh, and on Ubuntu that is
      # dash -- which has no `pipefail`. Every block aborted on its first line
      # with "Illegal option -o pipefail", cloud-init reported `status: error`,
      # and the box came up with no disk, no mount and no toolchain. A script in
      # `write_files` gets its shebang honoured, so bash is bash.
      - path: /usr/local/bin/blueprint-provision
        permissions: "0755"
        content: |
          #!/bin/bash
          set -euo pipefail

          # Non-interactive, and keep OUR config on a conflict. Both matter:
          # `/etc/caddy/Caddyfile` is written above and the caddy package ships
          # its own, so dpkg stops to ask which to keep -- on a box with no
          # terminal that is a hang, and under `set -e` on a piped rerun it is a
          # silent abort halfway through provisioning.
          export DEBIAN_FRONTEND=noninteractive
          export NEEDRESTART_MODE=a
          APT_KEEP_OURS='-o Dpkg::Options::=--force-confold -o Dpkg::Options::=--force-confdef'

          # -- the data disk ---------------------------------------------------
          # Found by its PROPERTIES, not by a path. `/dev/disk/azure/scsi1/lun0`
          # is the obvious answer and it does not exist here: the als_v6 family
          # presents data disks over NVMe, so the disk arrives as /dev/nvme0n2
          # and no scsi1 symlink is ever created. This cost a rebuild to find,
          # and it is the same shape as the trap in the pool's start task -- a
          # fixed device path is an assumption about the VM family.
          #
          # It also has to WAIT: the attachment is a separate Terraform resource
          # applied after the VM exists, so on a first boot the disk is genuinely
          # not there yet.
          #
          # Three guards, because the failure mode of getting this wrong is
          # formatting the OS disk. The candidate must carry no partitions, be
          # mounted nowhere, and be about the size we asked for.
          # Already done? Then there is nothing to find, and looking would fail:
          # the guards below reject a mounted disk, so a re-run would scan past
          # its own successful first run and exit. Provisioning has to be
          # re-runnable -- it is piped at a box by hand when something needs
          # fixing.
          want_gb=${var.data_disk_gb}
          DISK=""
          if findmnt -n /mnt/work >/dev/null 2>&1; then
            echo "/mnt/work already mounted"
            DISK=skip
          fi
          [ -n "$DISK" ] || for _ in $(seq 1 60); do
            for name in $(lsblk -dn -o NAME,TYPE | awk '$2=="disk"{print $1}'); do
              dev="/dev/$name"
              [ -b "$dev" ] || continue
              [ "$(lsblk -n -o NAME "$dev" | wc -l)" -gt 1 ] && continue
              [ -n "$(lsblk -n -o MOUNTPOINT "$dev" | tr -d ' \n')" ] && continue
              # GiB, not GB: Azure provisions in GiB, so a "128 GB" disk reports
              # 137 GB decimal and a decimal comparison misses it by nine.
              gb=$(( $(blockdev --getsize64 "$dev") / 1073741824 ))
              [ "$gb" -ge $(( want_gb - 2 )) ] && [ "$gb" -le $(( want_gb + 2 )) ] || continue
              DISK="$dev"
              break
            done
            [ -n "$DISK" ] && break
            sleep 5
          done

          if [ -z "$DISK" ]; then
            echo "no unpartitioned disk of about $want_gb GB found -- is it attached?" >&2
            lsblk >&2
            exit 1
          fi
          echo "data disk: $DISK"

          if [ "$DISK" != skip ]; then
            blkid "$DISK" >/dev/null 2>&1 || mkfs.ext4 -F -L work "$DISK"
          fi
          mkdir -p /mnt/work
          grep -q '/mnt/work' /etc/fstab || echo "LABEL=work /mnt/work ext4 defaults,nofail 0 2" >> /etc/fstab
          mount -a
          mkdir -p /mnt/work/cache /mnt/work/data
          chown -R ${var.admin_username}:${var.admin_username} /mnt/work

          # -- the share -------------------------------------------------------
          # `ro`: this box reads runs and abstractions and publishes nothing. A
          # reader that cannot write cannot corrupt the one thing that is not a
          # copy.
          mkdir -p /mnt/shared
          grep -q '/mnt/shared' /etc/fstab || echo "//${data.azurerm_storage_account.store.name}.file.core.windows.net/${var.share_name} /mnt/shared cifs ro,nofail,vers=3.1.1,credentials=/etc/smbcredentials/store.cred,dir_mode=0555,file_mode=0444,serverino,nosharesock,actimeo=30,mfsymlinks 0 0" >> /etc/fstab
          mount -a

          # -- caddy and the azure cli -----------------------------------------
          curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
            | gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
          echo "deb [signed-by=/usr/share/keyrings/caddy-stable-archive-keyring.gpg] https://dl.cloudsmith.io/public/caddy/stable/deb/debian any-version main" \
            > /etc/apt/sources.list.d/caddy-stable.list
          apt-get update
          apt-get install -y $APT_KEEP_OURS caddy
          curl -sL https://aka.ms/InstallAzureCLIDeb | bash
          systemctl enable --now caddy

          # -- the toolchain ---------------------------------------------------
          # uv brings its own Python, so the image's interpreter version is not a
          # constraint here the way it is on a Batch node.
          su - ${var.admin_username} -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
          echo 'export POKER_SOLVER_CACHE=/mnt/work/cache' >> /home/${var.admin_username}/.bashrc

          # -- the service -----------------------------------------------------
          # `enable`, not `enable --now`: RUN is empty until a deploy sets it, and
          # starting now would fail three times and deallocate a box someone is
          # still setting up. From the next boot on it starts by itself, which is
          # what makes waking the box equivalent to starting the server.
          systemctl daemon-reload
          systemctl enable blueprint

          # The passwordless sudo the deploy script needs, and nothing wider: it
          # rewrites /etc/blueprint.env and restarts one unit.
          echo '${var.admin_username} ALL=(ALL) NOPASSWD: /usr/bin/tee /etc/blueprint.env, /usr/bin/systemctl restart blueprint, /usr/bin/systemctl enable blueprint, /usr/bin/systemctl daemon-reload, /usr/bin/journalctl -u blueprint *' \
            > /etc/sudoers.d/blueprint
          chmod 0440 /etc/sudoers.d/blueprint

    runcmd:
      - /usr/local/bin/blueprint-provision
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

  # Two rules, and the asymmetry is the point.
  #
  # 443 is open to the world because the console has to reach this from wherever
  # you are, and a button that first needs an SSH tunnel is not a button. What
  # answers there is Caddy, which checks a bearer token and returns 404 -- not
  # 401 -- to everything else, so a scanner learns nothing.
  #
  # The server's OWN port (8790) is deliberately absent and must stay absent: it
  # binds loopback and speaks to nobody but Caddy. Opening it would publish an
  # unauthenticated read interface to a trained run, which is a mistake worth
  # making impossible rather than merely discouraged.
  security_rule {
    name                       = "https"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "ssh"
    priority                   = 110
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
  # Gives the box a name Let's Encrypt will issue for. Without it Caddy has only
  # an IP, no certificate is possible, and the console would have to be taught to
  # skip verification -- a setting nobody ever turns back on.
  domain_name_label = var.dns_label
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
