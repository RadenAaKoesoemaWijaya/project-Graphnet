#!/usr/bin/env bash
# =============================================================================
# ASTINA - one-shot deploy script for Google Cloud Run
# -----------------------------------------------------------------------------
# Usage:
#   ./deploy.sh                       # deploy with defaults
#   ./deploy.sh my-project my-region  # deploy to a specific project/region
#   ./deploy.sh my-project my-region astina  # customise the Cloud Run service
#
# Requirements:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - The project has billing enabled
#   - APIs enabled: run.googleapis.com, cloudbuild.googleapis.com,
#     artifactregistry.googleapis.com
# =============================================================================
set -euo pipefail

PROJECT_ID="${1:-${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"
REGION="${2:-asia-southeast2}"
SERVICE="${3:-astina}"
REPOSITORY="astina-images"
AR_HOST="${REGION}-docker.pkg.dev"

# --- Pretty logging -----------------------------------------------------------
log() { printf "\033[1;34m[deploy]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  err "No Google Cloud project set. Pass it as the first argument or set GOOGLE_CLOUD_PROJECT."
  exit 1
fi

log "Project : ${PROJECT_ID}"
log "Region  : ${REGION}"
log "Service : ${SERVICE}"

gcloud config set project "${PROJECT_ID}" >/dev/null

# --- Enable the required APIs -------------------------------------------------
log "Enabling required APIs ..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --quiet

# --- Create the Artifact Registry repository if it does not exist ------------
log "Ensuring Artifact Registry repository exists ..."
if ! gcloud artifacts repositories describe "${REPOSITORY}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="ASTINA container images" \
    --quiet
fi

# --- Submit a Cloud Build that builds, pushes and deploys --------------------
log "Submitting Cloud Build (this may take a few minutes) ..."
gcloud builds submit \
  --config=cloudbuild.yaml \
  --region="${REGION}" \
  --substitutions="_REGION=${REGION},_SERVICE=${SERVICE},_REPOSITORY=${REPOSITORY},_AR_HOST=${AR_HOST}" \
  --quiet

# --- Print the URL of the deployed service ------------------------------------
URL=$(gcloud run services describe "${SERVICE}" \
        --region="${REGION}" \
        --format="value(status.url)" 2>/dev/null || true)

log "Deployment finished."
if [[ -n "${URL}" ]]; then
  log "Service URL: ${URL}"
else
  log "Run 'gcloud run services describe ${SERVICE} --region=${REGION} --format=value(status.url)' to get the URL."
fi
