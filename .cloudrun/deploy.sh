#!/bin/bash
# ============================================================================
# ASTINA Cloud Run Deployment Script (Bash/Linux/Mac/WSL)
# ============================================================================
# Automated deployment to Google Cloud Run with configuration prompts,
# GCS bucket setup, image building, and deployment verification.
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration defaults
PROJECT_ID="${1:-}"
REGION="${2:-us-central1}"
SERVICE_NAME="${3:-astina}"
BUCKET_NAME="${4:-}"
MEMORY="16"
CPU="4"
REPOSITORY="${REPOSITORY:-astina}"
IMAGE_HOST="${IMAGE_HOST:-${REGION}-docker.pkg.dev}"

# Helper functions
header() {
    echo -e "\n${CYAN}========================================"
    echo "$1"
    echo "========================================${NC}\n"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    header "Checking Prerequisites"
    
    local missing_tools=()
    
    if ! command -v gcloud &> /dev/null; then
        missing_tools+=("gcloud")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
    fi
    
    success "gcloud CLI found"
    success "Docker found"
    
    # Check authentication
    local auth_user
    auth_user=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
    
    if [ -z "$auth_user" ]; then
        error "Not authenticated with gcloud. Run: gcloud auth login"
    fi
    
    success "gcloud authenticated as: $auth_user"
}

# Get configuration
get_configuration() {
    header "Configuration Setup"
    
    if [ -z "$PROJECT_ID" ]; then
        info "Available Google Cloud projects:"
        gcloud projects list --format="table(projectId,name)"
        read -p "Enter Google Cloud Project ID: " PROJECT_ID
    fi
    
    # Validate project
    if ! gcloud projects describe "$PROJECT_ID" &>/dev/null; then
        error "Project '$PROJECT_ID' not found"
    fi
    
    success "Project: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID" 2>&1 | grep -v "Updated" || true
    
    if [ -z "$BUCKET_NAME" ]; then
        local default_bucket="astina-models-$PROJECT_ID"
        read -p "Enter GCS bucket name (default: $default_bucket): " bucket_input
        BUCKET_NAME="${bucket_input:-$default_bucket}"
    fi
    
    success "GCS Bucket: $BUCKET_NAME"
}

# Enable required APIs
enable_apis() {
    header "Enabling Required Google Cloud APIs"
    
    local apis=(
        "run.googleapis.com"
        "artifactregistry.googleapis.com"
        "cloudbuild.googleapis.com"
        "storage.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID" 2>&1 | grep -v "already enabled" || true
    done
    
    success "All required APIs enabled"
}

# Create GCS bucket
create_gcs_bucket() {
    header "Setting up Google Cloud Storage"
    
    if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
        success "Bucket already exists: gs://$BUCKET_NAME"
    else
        info "Creating bucket: gs://$BUCKET_NAME"
        gsutil mb -p "$PROJECT_ID" "gs://$BUCKET_NAME"
        success "Bucket created successfully"
    fi
}

# Build and push Docker image
build_push_image() {
    header "Building and Pushing Docker Image"

    local project_root
    project_root=$(cd "$(dirname "$0")/.." && pwd)

    if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
        info "Creating Artifact Registry repository: $REPOSITORY in $REGION"
        gcloud artifacts repositories create "$REPOSITORY" \
            --repository-format=docker \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --description="ASTINA container repository" >/dev/null
    fi

    local image_url="${IMAGE_HOST}/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:latest"
    info "Building image: $image_url"
    docker build -t "$image_url" -f "$project_root/Dockerfile" "$project_root"

    if [ $? -ne 0 ]; then
        error "Docker build failed"
    fi

    success "Image built successfully"

    info "Pushing to Artifact Registry..."
    docker push "$image_url"

    if [ $? -ne 0 ]; then
        error "Docker push failed"
    fi

    success "Image pushed successfully"
}

# Deploy to Cloud Run
deploy_cloud_run() {
    header "Deploying to Google Cloud Run"

    local image_url="${IMAGE_HOST}/${PROJECT_ID}/${REPOSITORY}/${SERVICE_NAME}:latest"
    local timeout=3600
    
    info "Service name: $SERVICE_NAME"
    info "Region: $REGION"
    info "Memory: ${MEMORY}Gi"
    info "CPU: $CPU"
    info "Timeout: ${timeout}s (60 minutes)"
    
    gcloud run deploy "$SERVICE_NAME" \
        --image "$image_url" \
        --platform managed \
        --region "$REGION" \
        --memory "${MEMORY}Gi" \
        --cpu "$CPU" \
        --port 8501 \
        --concurrency 1 \
        --timeout "$timeout" \
        --no-allow-unauthenticated \
        --max-request-body-size 3Gi \
        --project "$PROJECT_ID" \
        --set-env-vars "GOOGLE_CLOUD_BUCKET=$BUCKET_NAME,ASTINA_LOG_FORMAT=json,STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3072" \
        --min-instances 1 \
        --max-instances 5
    
    if [ $? -ne 0 ]; then
        error "Cloud Run deployment failed"
    fi
    
    success "Deployment completed successfully"
}

# Display deployment summary
deployment_summary() {
    header "Deployment Summary"
    
    local service_url
    service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" \
        --format="value(status.url)" \
        --project="$PROJECT_ID" 2>/dev/null)
    
    if [ -n "$service_url" ]; then
        success "Service deployed successfully!"
        echo ""
        info "Service URL: $service_url"
        info "Service Name: $SERVICE_NAME"
        info "Region: $REGION"
        info "Project: $PROJECT_ID"
        info "GCS Bucket: gs://$BUCKET_NAME"
        echo ""
        info "Next steps:"
        echo -e "${CYAN}  1. Open URL in browser: $service_url"
        echo "  2. View logs: gcloud run logs read $SERVICE_NAME --region=$REGION --project=$PROJECT_ID --limit=50"
        echo "  3. Update: Re-run this script after making code changes${NC}"
        echo ""
    else
        error "Failed to retrieve service URL"
    fi
}

# Main execution
main() {
    echo -e "\n${CYAN}"
    echo "╔════════════════════════════════════════════════════╗"
    echo "║  ASTINA — Cloud Run Deployment Script (Bash)      ║"
    echo "║  Insurance Fraud & Anomaly Detection Platform     ║"
    echo "╚════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_prerequisites
    get_configuration
    enable_apis
    create_gcs_bucket
    build_push_image
    deploy_cloud_run
    deployment_summary
    
    success "Deployment complete!"
}

main "$@"
