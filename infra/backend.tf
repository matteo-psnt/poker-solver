# State lives in the store's `tfstate` container, one blob per root, under
# AAD -- never the account key. Three local files with several sessions
# applying was a corruption waiting to happen; the blob lease is the lock.
terraform {
  backend "azurerm" {
    resource_group_name  = "poker-solver-store-rg"
    storage_account_name = "pokersolverstore"
    container_name       = "tfstate"
    key                  = "compute.tfstate"
    use_azuread_auth     = true
  }
}
