output "public_ip" {
  value = azurerm_public_ip.serve.ip_address
}

output "admin_username" {
  value = var.admin_username
}

output "resource_group" {
  value = azurerm_resource_group.serve.name
}

output "vm_name" {
  value = azurerm_linux_virtual_machine.serve.name
}

# The whole operational interface, in one string. The server binds loopback on
# the box, so this forward IS the only route to it -- printing the command rather
# than documenting the parts is what stops someone "simplifying" it into an NSG
# rule that opens 8790 to the internet.
output "tunnel" {
  description = "Run this, then point POKER_SOLVER_BLUEPRINT_URL at http://127.0.0.1:8790."
  value       = "ssh -N -L 8790:127.0.0.1:8790 ${var.admin_username}@${azurerm_public_ip.serve.ip_address}"
}

output "ssh" {
  description = "A shell on the box, for starting the server against a run."
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.serve.ip_address}"
}
