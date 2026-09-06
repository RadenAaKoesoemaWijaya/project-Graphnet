#!/usr/bin/env pwsh
<#
.SYNOPSIS
Automated deployment script for ASTINA to Google Cloud Run.

.DESCRIPTION
This script automates the entire deployment process:
1. Validates prerequisites (gcloud, docker)
2. Prompts for configuration (project ID, bucket name, region)
3. Creates GCS bucket for model storage
4. Builds and pushes Docker image to Artifact Registry
5. Deploys to Cloud Run with optimal settings
6. Outputs deployment summary and access URL

.EXAMPLE
.\deploy.ps1

.NOTES
Requires:
- gcloud CLI installed and authenticated
- Docker Desktop running
- Google Cloud project with enabled APIs (Cloud Run, Artifact Registry, Cloud Storage)
#>

param(
    [string]$ProjectId,
    [string]$Region = "us-central1",
    [string]$ServiceName = "astina",
    [string]$BucketName,
    [string]$Repository = "astina",
    [string]$ImageHost,
    [ValidateSet("2", "4", "8")]
    [string]$Memory = "16"
)

$ErrorActionPreference = "Stop"

# Color output functions
function Write-Header {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Cyan
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
    exit 1
}

# Validate prerequisites
function Test-Prerequisites {
    Write-Header "Checking Prerequisites"
    
    $missingTools = @()
    
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        $missingTools += "gcloud"
    }
    
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        $missingTools += "docker"
    }
    
    if ($missingTools.Count -gt 0) {
        Write-Error-Custom "Missing required tools: $($missingTools -join ', ')"
    }
    
    Write-Success "gcloud CLI found"
    Write-Success "Docker found"
    
    # Check gcloud authentication
    $authStatus = & gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if (-not $authStatus) {
        Write-Error-Custom "Not authenticated with gcloud. Run: gcloud auth login"
    }
    
    Write-Success "gcloud authenticated as: $authStatus"
}

# Get or prompt for configuration
function Get-Configuration {
    Write-Header "Configuration Setup"
    
    if (-not $ProjectId) {
        Write-Info "Available projects:"
        $projects = & gcloud projects list --format="table(projectId,name)" 2>$null
        Write-Host $projects
        
        $ProjectId = Read-Host "Enter Google Cloud Project ID"
    }
    
    # Validate project exists
    $projectExists = & gcloud projects describe $ProjectId --format="value(projectId)" 2>$null
    if (-not $projectExists) {
        Write-Error-Custom "Project '$ProjectId' not found"
    }
    
    Write-Success "Project: $ProjectId"
    
    # Set project
    & gcloud config set project $ProjectId 2>&1 | Out-Null
    
    if (-not $BucketName) {
        $defaultBucket = "astina-models-$ProjectId"
        $BucketName = Read-Host "Enter GCS bucket name for model storage (default: $defaultBucket)"
        if ([string]::IsNullOrWhiteSpace($BucketName)) {
            $BucketName = $defaultBucket
        }
    }

    if (-not $ImageHost) {
        $ImageHost = "$Region-docker.pkg.dev"
    }

    Write-Success "GCS Bucket: $BucketName"
    Write-Success "Artifact Registry Host: $ImageHost"

    return @{
        ProjectId     = $ProjectId
        Region        = $Region
        ServiceName   = $ServiceName
        Repository    = $Repository
        BucketName    = $BucketName
        Memory        = $Memory
        ImageHost     = $ImageHost
        ImageTag      = "$ImageHost/$ProjectId/$Repository/$ServiceName"
        ImageURL      = "$ImageHost/$ProjectId/$Repository/${ServiceName}:latest"
    }
}

# Create GCS bucket if it doesn't exist
function Create-GCSBucket {
    param([hashtable]$Config)
    
    Write-Header "Setting up Google Cloud Storage"
    
    $bucketExists = & gsutil ls -b "gs://$($Config.BucketName)" 2>$null
    
    if ($bucketExists) {
        Write-Success "Bucket already exists: gs://$($Config.BucketName)"
    } else {
        Write-Info "Creating bucket: gs://$($Config.BucketName)"
        & gsutil mb -p $Config.ProjectId "gs://$($Config.BucketName)" 2>&1 | Out-Null
        Write-Success "Bucket created successfully"
    }
}

# Enable required APIs
function Enable-APIs {
    param([hashtable]$Config)
    
    Write-Header "Enabling required Google Cloud APIs"
    
    $requiredAPIs = @(
        "run.googleapis.com",
        "artifactregistry.googleapis.com",
        "cloudbuild.googleapis.com",
        "storage.googleapis.com"
    )
    
    foreach ($api in $requiredAPIs) {
        Write-Info "Enabling $api..."
        & gcloud services enable $api --project=$Config.ProjectId 2>&1 | Out-Null
    }
    
    Write-Success "All required APIs enabled"
}

# Build and push Docker image
function Build-PushImage {
    param([hashtable]$Config)
    
    Write-Header "Building and pushing Docker image"
    
    $projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

    try {
        & gcloud artifacts repositories describe $Config.Repository --location=$Config.Region --project=$Config.ProjectId 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Info "Creating Artifact Registry repository: $($Config.Repository) in $($Config.Region)"
            & gcloud artifacts repositories create $Config.Repository --repository-format=docker --location=$Config.Region --project=$Config.ProjectId --description="ASTINA container repository" 2>$null | Out-Null
        }
    } catch {
        # Ignore repository creation errors and continue with image push; the deployment will surface real issues
    }
    
    Write-Info "Building image: $($Config.ImageURL)"
    & docker build -t $Config.ImageURL -f "$projectRoot\Dockerfile" "$projectRoot" 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Docker build failed"
    }
    
    Write-Success "Image built successfully"
    
    Write-Info "Pushing to Artifact Registry..."
    & docker push $Config.ImageURL 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Docker push failed"
    }
    
    Write-Success "Image pushed successfully"
}

# Deploy to Cloud Run
function Deploy-CloudRun {
    param([hashtable]$Config)
    
    Write-Header "Deploying to Google Cloud Run"
    
    $cpuCount = "4"
    $timeout = "3600"  # 60 minutes
    $imageUrl = $Config.ImageURL
    
    Write-Info "Service name: $($Config.ServiceName)"
    Write-Info "Region: $($Config.Region)"
    Write-Info "Memory: $($Config.Memory) GB"
    Write-Info "CPU: $cpuCount"
    Write-Info "Timeout: $timeout seconds"
    
    $deployCmd = @(
        "run", "deploy", $Config.ServiceName,
        "--image", $imageUrl,
        "--platform", "managed",
        "--region", $Config.Region,
        "--memory", "$($Config.Memory)Gi",
        "--cpu", $cpuCount,
        "--port", "8501",
        "--concurrency", "1",
        "--timeout", $timeout,
        "--no-allow-unauthenticated",
        "--max-request-body-size", "3Gi",
        "--project", $Config.ProjectId,
        "--set-env-vars", "GOOGLE_CLOUD_BUCKET=$($Config.BucketName),ASTINA_LOG_FORMAT=json,STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3072,LLM_PROVIDER=heuristic,LLM_MODEL_NAME=gemini-1.5-flash,AUTH_ENABLED=false",
        "--min-instances", "1",
        "--max-instances", "5"
    )
    
    Write-Info "Running: gcloud $($deployCmd -join ' ')"
    & gcloud $deployCmd 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Cloud Run deployment failed"
    }
    
    Write-Success "Deployment completed successfully"
}

# Get deployment summary
function Get-DeploymentSummary {
    param([hashtable]$Config)
    
    Write-Header "Deployment Summary"
    
    $serviceInfo = & gcloud run services describe $Config.ServiceName --region=$Config.Region --format="table(status.url,status.conditions[0].status)" --project=$Config.ProjectId 2>$null
    
    $serviceUrl = & gcloud run services describe $Config.ServiceName --region=$Config.Region --format="value(status.url)" --project=$Config.ProjectId 2>$null
    
    if ($serviceUrl) {
        Write-Success "Service deployed successfully!"
        Write-Host ""
        Write-Info "Service URL: $serviceUrl"
        Write-Info "Service Name: $($Config.ServiceName)"
        Write-Info "Region: $($Config.Region)"
        Write-Info "Project: $($Config.ProjectId)"
        Write-Info "GCS Bucket: gs://$($Config.BucketName)"
        Write-Host ""
        
        Write-Info "Next steps:"
        Write-Host "  1. Open URL in browser: $serviceUrl" -ForegroundColor Cyan
        Write-Host "  2. To view logs: gcloud run logs read $($Config.ServiceName) --region=$($Config.Region) --project=$($Config.ProjectId) --limit=50" -ForegroundColor Cyan
        Write-Host "  3. To update: Re-run this script after making code changes" -ForegroundColor Cyan
        Write-Host ""
        
    } else {
        Write-Error-Custom "Failed to retrieve service URL"
    }
}

# Main execution
function Main {
    Write-Host "`n" -ForegroundColor Cyan
    Write-Host "╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  ASTINA — Cloud Run Deployment Script             ║" -ForegroundColor Cyan
    Write-Host "║  Insurance Fraud & Anomaly Detection Platform     ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    
    Test-Prerequisites
    $config = Get-Configuration
    Enable-APIs $config
    Create-GCSBucket $config
    Build-PushImage $config
    Deploy-CloudRun $config
    Get-DeploymentSummary $config
    
    Write-Host "✅ Deployment complete!" -ForegroundColor Green
}

# Run main
Main
