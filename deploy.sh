#!/usr/bin/env bash
# =============================================================================
# ASTINA - one-shot deploy script for Google Cloud Run
# -----------------------------------------------------------------------------
# Usage:
#   ./deploy.sh                       # deploy with defaults
#   ./deploy.sh my-project my-region  # deploy to a specific project/region
#   ./deploy.sh my-project my-region astina bucket-name [runtime-service-account]
#
# Requirements:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - The project has billing enabled
#   - APIs enabled: run.googleapis.com, cloudbuild.googleapis.com,
#     artifactregistry.googleapis.com
#   - A pre-created GCS bucket, runtime service account, and four Secret Manager secrets
# =============================================================================
set -euo pipefail

PROJECT_ID="${1:-${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"
REGION="${2:-asia-southeast2}"
SERVICE="${3:-astina}"
GCS_BUCKET="${4:-${GOOGLE_CLOUD_BUCKET:-}}"
RUNTIME_SERVICE_ACCOUNT="${5:-astina-runtime@${PROJECT_ID}.iam.gserviceaccount.com}"
REPOSITORY="astina-images"
AR_HOST="${REGION}-docker.pkg.dev"
GCS_PREFIX="${GOOGLE_CLOUD_BUCKET_PREFIX:-production/models}"

# --- Pretty logging -----------------------------------------------------------
log() { printf "\033[1;34m[deploy]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; }

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  err "No Google Cloud project set. Pass it as the first argument or set GOOGLE_CLOUD_PROJECT."
  exit 1
fi
if [[ -z "${GCS_BUCKET}" ]]; then
  err "A persistent GCS bucket is required. Pass it as the fourth argument or set GOOGLE_CLOUD_BUCKET."
  exit 1
fi

log "Project : ${PROJECT_ID}"
log "Region  : ${REGION}"
log "Service : ${SERVICE}"

# --- Enable the required APIs -------------------------------------------------
log "Enabling required APIs ..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

# --- Validate production prerequisites before starting the build --------------
log "Validating model bucket, runtime identity, and production secrets ..."
gcloud storage buckets describe "gs://${GCS_BUCKET}" --project="${PROJECT_ID}" >/dev/null
gcloud iam service-accounts describe "${RUNTIME_SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" >/dev/null
for secret in astina-admin-password astina-auditor-password astina-analyst-password astina-viewer-password; do
  gcloud secrets describe "${secret}" --project="${PROJECT_ID}" >/dev/null
done

# --- Create the Artifact Registry repository if it does not exist ------------
log "Ensuring Artifact Registry repository exists ..."
if ! gcloud artifacts repositories describe "${REPOSITORY}" \
      --location="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="ASTINA container images" \
    --project="${PROJECT_ID}" \
    --quiet
fi

# --- Submit a Cloud Build that builds, pushes and deploys --------------------
log "Submitting Cloud Build (this may take a few minutes) ..."
gcloud builds submit \
  --config=cloudbuild.yaml \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --substitutions="_REGION=${REGION},_SERVICE=${SERVICE},_REPOSITORY=${REPOSITORY},_AR_HOST=${AR_HOST},_GCS_BUCKET=${GCS_BUCKET},_GCS_PREFIX=${GCS_PREFIX},_RUNTIME_SERVICE_ACCOUNT=${RUNTIME_SERVICE_ACCOUNT}" \
  --quiet

# --- Verify the deployed revision and attempt authenticated health smoke test -
URL=$(gcloud run services describe "${SERVICE}" \
        --region="${REGION}" --project="${PROJECT_ID}" \
        --format="value(status.url)" 2>/dev/null || true)
READY_REVISION=$(gcloud run services describe "${SERVICE}" \
        --region="${REGION}" --project="${PROJECT_ID}" \
        --format="value(status.latestReadyRevisionName)" 2>/dev/null || true)
CREATED_REVISION=$(gcloud run services describe "${SERVICE}" \
        --region="${REGION}" --project="${PROJECT_ID}" \
        --format="value(status.latestCreatedRevisionName)" 2>/dev/null || true)

log "Deployment finished."
if [[ -n "${URL}" ]]; then
  log "Service URL: ${URL}"
  if [[ -z "${READY_REVISION}" || "${READY_REVISION}" != "${CREATED_REVISION}" ]]; then
    err "The latest revision is not ready (ready=${READY_REVISION:-none}, created=${CREATED_REVISION:-none}). Check Cloud Run logs."
    exit 1
  fi
  TOKEN="$(gcloud auth print-identity-token --audiences="${URL}" 2>/dev/null || true)"
  if [[ -n "${TOKEN}" ]] && curl --fail --silent --show-error --max-time 20 \
      -H "Authorization: Bearer ${TOKEN}" "${URL}/_stcore/health" >/dev/null; then
    log "Authenticated health check: HTTP 200."
  else
    log "Revision is ready; health check could not be authorized from this deployer identity. Verify with a roles/run.invoker identity."
  fi
else
  err "Could not retrieve the deployed service URL."
  exit 1
fi
