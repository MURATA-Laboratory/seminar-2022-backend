resource "google_container_registry" "registry" {
  project = var.project_id

  depends_on = [
    google_project_service.project["containerregistry.googleapis.com"],
  ]
}

resource "google_storage_bucket_iam_member" "registry_create" {
  bucket = google_container_registry.registry.id

  role   = "roles/storage.admin"
  member = "serviceAccount:${google_service_account.github_actions.email}"
}
