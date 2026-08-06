output "public_ip" {
  description = "Outbound address. Nothing inbound is permitted to reach it."
  value       = azurerm_public_ip.serve.ip_address
}

output "resource_group" {
  value = azurerm_resource_group.serve.name
}

output "vm_name" {
  value = azurerm_linux_virtual_machine.serve.name
}

output "url" {
  description = <<-EOT
    What POKER_SOLVER_BLUEPRINT_URL should be set to.

    A MagicDNS name, so it needs MagicDNS enabled on the tailnet -- check with
    `tailscale status`. If short names do not resolve, use the full
    `<hostname>.<tailnet>.ts.net`.
  EOT
  value       = "http://${var.tailscale_hostname}:8790"
}

output "ssh" {
  description = "A shell on the box. Tailscale SSH — port 22 is closed."
  value       = "tailscale ssh ${var.admin_username}@${var.tailscale_hostname}"
}
