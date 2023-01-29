resource "google_cloud_run_service" "backend" {
  name     = "murata-lab-seminar-2022-backend"
  location = "asia-northeast1"

  template {
    spec {
      containers {
        image = "gcr.io/cloudrun/hello"
        resources {
          limits = {
            "memory" : "8G",
            "cpu" : "4"
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].spec[0].containers[0].image
    ]
  }

  depends_on = [
    google_project_service.project["run.googleapis.com"],
  ]
}
