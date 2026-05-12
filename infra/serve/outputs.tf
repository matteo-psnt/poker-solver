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

output "url" {
  description = "What POKER_SOLVER_BLUEPRINT_URL should be set to."
  value       = "https://${azurerm_public_ip.serve.fqdn}"
}

# The one thing between the internet and a trained run. Sensitive, so it stays
# out of logs and plan diffs; `terraform output -raw api_token` is the only path
# that prints it, and `just serve-env` pipes it straight into a shell export.
output "api_token" {
  value     = random_password.api_token.result
  sensitive = true
}

output "ssh" {
  description = "A shell on the box, for pointing it at a run."
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.serve.ip_address}"
}
