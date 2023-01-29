resource "google_project_service" "project" {
  project  = var.project_id
  for_each = toset(["iamcredentials.googleapis.com", "containerregistry.googleapis.com", "run.googleapis.com"])
  service  = each.key
}
