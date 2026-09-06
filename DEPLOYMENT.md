# ASTINA Deployment Guide

Panduan lengkap untuk deployment aplikasi ASTINA di berbagai lingkungan dengan konfigurasi optimal.

---

## 📋 Prerequisites

### Local Development
- Python 3.11-3.13
- Git
- Virtual environment tool (venv/conda)

### Docker Deployment
- Docker Desktop 4.0+
- Docker Compose v2
- 8GB RAM minimum
- 20GB disk space

### Google Cloud Run Deployment
- Google Cloud account dengan billing enabled
- gcloud CLI terinstall dan terautentikasi
- Docker Desktop
- Project dengan APIs berikut enabled:
  - Cloud Run API
  - Artifact Registry API
  - Cloud Build API
  - Cloud Storage API

---

## 🚀 Deployment Methods

### 1. Local Development Deployment

#### Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd project-Graphnet

# Create virtual environment
python -3.13 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or use your preferred editor
```

#### Run Application
```bash
# Development mode (default authentication bypass)
python run.py

# Production mode (authentication enforced)
export AUTH_ENABLED="true"
python run.py
```

---

### 2. Docker Desktop Deployment

#### Build and Run with Docker Compose
```bash
# Build and start container
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop container
docker-compose down

# Rebuild after code changes
docker-compose up --build -d
```

#### Environment Configuration
Create `.env` file in project root:
```bash
# Server Configuration
PORT=8501
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3072

# LLM Configuration (Optional)
LLM_PROVIDER=heuristic
GEMINI_API_KEY=your-api-key-here

# Authentication
AUTH_ENABLED=false
```

#### Volume Persistence
Docker Compose automatically mounts:
- `./cache` → `/app/cache` (processed data cache)
- `./models` → `/app/models` (trained models)
- `./logs` → `/app/logs` (audit logs)

These directories persist across container restarts.

#### Troubleshooting Docker Issues

**Issue: Container fails to start**
```bash
# Check container logs
docker-compose logs astina-app

# Check if port 8501 is already in use
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # Linux/macOS

# Change port in docker-compose.yml
ports:
  - "8502:8501"
```

**Issue: Permission errors on cache/models directories**
```bash
# Fix directory permissions
chmod -R 777 cache models logs
```

**Issue: Out of memory errors**
```bash
# Increase memory limit in docker-compose.yml
services:
  graphnet-app:
    mem_limit: 8g
```

---

### 3. Google Cloud Run Deployment

#### Prerequisites Setup
```bash
# Install gcloud CLI
# Download from: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set default project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
```

#### Automated Deployment (PowerShell)
```powershell
# Navigate to project directory
cd C:\project-Graphnet

# Run deployment script
.\.cloudrun\deploy.ps1

# Follow prompts for:
# - Project ID
# - Region (default: us-central1)
# - Service name (default: astina)
# - GCS bucket name
# - Memory allocation (default: 16GB)
```

#### Manual Deployment Steps

**1. Create GCS Bucket for Model Storage**
```bash
# Create bucket
gsutil mb -p YOUR_PROJECT_ID gs://astina-models-YOUR_PROJECT_ID

# Set bucket permissions
gsutil iam ch serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com:objectAdmin gs://astina-models-YOUR_PROJECT_ID
```

**2. Build and Push Docker Image**
```bash
# Set project
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=us-central1
export SERVICE_NAME=astina
export REPO_NAME=astina

# Create Artifact Registry repository
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --project=$PROJECT_ID

# Build image
docker build -t $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest .

# Push image
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest
```

**3. Deploy to Cloud Run**
```bash
# Deploy service
gcloud run deploy $SERVICE_NAME \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$SERVICE_NAME:latest \
    --platform managed \
    --region $REGION \
    --memory 16Gi \
    --cpu 4 \
    --port 8501 \
    --concurrency 1 \
    --timeout 3600 \
    --no-allow-unauthenticated \
    --max-request-body-size 3Gi \
    --set-env-vars="GOOGLE_CLOUD_BUCKET=astina-models-$PROJECT_ID,ASTINA_LOG_FORMAT=json,STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3072,LLM_PROVIDER=heuristic,LLM_MODEL_NAME=gemini-1.5-flash,AUTH_ENABLED=false" \
    --min-instances 1 \
    --max-instances 5 \
    --project=$PROJECT_ID
```

**4. Configure IAM Permissions**
```bash
# Allow public access (optional - for demo only)
gcloud run services add-iam-policy-binding $SERVICE_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --member="allUsers" \
    --role="roles/run.invoker"

# For production, use specific service accounts
gcloud run services add-iam-policy-binding $SERVICE_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.invoker"
```

#### Cloud Run Configuration

**Memory and CPU Allocation:**
- Development: 2GB RAM, 1 CPU
- Production: 16GB RAM, 4 CPU
- Large datasets: 32GB RAM, 8 CPU

**Scaling Settings:**
- Min instances: 1 (always available)
- Max instances: 5 (auto-scaling)
- Concurrency: 1 (per instance)

**Timeout Configuration:**
- Default: 3600 seconds (60 minutes)
- For large datasets: Increase to 7200 seconds

#### Cloud Run Environment Variables

Essential environment variables for Cloud Run:
```bash
GOOGLE_CLOUD_BUCKET=astina-models-PROJECT_ID
ASTINA_LOG_FORMAT=json
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3072
LLM_PROVIDER=heuristic
LLM_MODEL_NAME=gemini-1.5-flash
AUTH_ENABLED=false
```

Optional variables for production:
```bash
GEMINI_API_KEY=your-production-api-key
OPENAI_API_KEY=your-production-api-key
AUTH_ENABLED=true
ASTINA_ADMIN_PASSWORD=secure-admin-password
```

#### Cloud Run Monitoring

**View Logs:**
```bash
# Stream logs in real-time
gcloud run logs read astina --region=us-central1 --project=PROJECT_ID --follow

# View recent logs
gcloud run logs read astina --region=us-central1 --project=PROJECT_ID --limit=50
```

**Monitor Service:**
```bash
# Get service status
gcloud run services describe astina --region=us-central1 --project=PROJECT_ID

# Get service URL
gcloud run services describe astina --region=us-central1 --project=PROJECT_ID --format="value(status.url)"
```

**Health Checks:**
Cloud Run automatically checks the health endpoint:
- Health check: `/_stcore/health`
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3

---

## 🔧 Post-Deployment Configuration

### 1. Authentication Setup (Production)

**Enable Authentication:**
```bash
# In Cloud Run deployment
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-env-vars="AUTH_ENABLED=true"
```

**Set Secure Passwords:**
```bash
# Update with secure passwords
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-env-vars="ASTINA_ADMIN_PASSWORD=SecurePass123!,ASTINA_AUDITOR_PASSWORD=SecurePass123!,ASTINA_ANALYST_PASSWORD=SecurePass123!,ASTINA_VIEWER_PASSWORD=SecurePass123!"
```

**Configure Secret Manager (Recommended):**
```bash
# Create secrets
echo "your-api-key" | gcloud secrets create gemini-api-key --project=PROJECT_ID
echo "your-password" | gcloud secrets create admin-password --project=PROJECT_ID

# Grant access
gcloud secrets add-iam-policy-binding gemini-api-key \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --project=PROJECT_ID

# Update service to use secrets
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,ASTINA_ADMIN_PASSWORD=admin-password:latest"
```

### 2. LLM Configuration (Production)

**Configure Gemini API:**
```bash
# Set Gemini as provider
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-env-vars="LLM_PROVIDER=gemini,LLM_MODEL_NAME=gemini-1.5-flash"
```

**Configure API Key via Secret Manager:**
```bash
# Store API key in secret manager
echo "your-gemini-api-key" | gcloud secrets create gemini-api-key --project=PROJECT_ID

# Update service to use secret
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-secrets="GEMINI_API_KEY=gemini-api-key:latest"
```

### 3. Database Configuration (Optional)

**Enable Database Backend:**
```bash
# Enable database mode
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-env-vars="ENABLE_DATABASE=1,DB_HOST=your-db-host,DB_PASSWORD=your-db-password"
```

**Use Cloud SQL:**
```bash
# Create Cloud SQL instance
gcloud sql instances create astina-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1

# Create database
gcloud sql databases create astina --instance=astina-db

# Configure connection
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-env-vars="DB_HOST=/cloudsql/PROJECT_ID:us-central1:astina-db,DB_NAME=astina"
```

---

## 🛡️ Security Best Practices

### 1. API Key Management
- **Never hardcode API keys** in source code
- **Use Secret Manager** for production deployments
- **Rotate API keys** regularly (every 90 days)
- **Limit API key permissions** to minimum required
- **Monitor API usage** for anomaly detection

### 2. Authentication Security
- **Always enable AUTH_ENABLED=true** in production
- **Use strong passwords** (minimum 12 characters, mixed case, numbers, symbols)
- **Implement rate limiting** for login attempts
- **Enable audit logging** for all authentication events
- **Use HTTPS only** (Cloud Run provides this by default)

### 3. Network Security
- **Use VPC connectors** for private database access
- **Implement IP whitelisting** where possible
- **Enable VPC Service Controls** for additional security
- **Use Cloud Armor** for DDoS protection
- **Configure firewall rules** for restricted access

### 4. Data Privacy
- **Enable PII masking** by default
- **Encrypt data at rest** (Cloud Storage provides this)
- **Encrypt data in transit** (HTTPS required)
- **Implement data retention policies**
- **Regular security audits** and compliance checks

---

## 📊 Monitoring and Logging

### 1. Cloud Monitoring

**Setup Cloud Monitoring:**
```bash
# Create monitoring dashboard
gcloud monitoring dashboards create astina-dashboard \
    --display-name="ASTINA Performance Dashboard" \
    --project=PROJECT_ID
```

**Key Metrics to Monitor:**
- Request latency
- Error rates
- Memory usage
- CPU utilization
- Request count
- Instance count

### 2. Error Reporting

**Setup Error Reporting:**
```bash
# Enable error reporting
gcloud beta logging settings update \
    --project=PROJECT_ID \
    --enable-cloud-error-reporting
```

### 3. Cloud Logging

**View Application Logs:**
```bash
# Stream logs
gcloud logging tail "resource.type=cloud_run_revision" \
    --project=PROJECT_ID \
    --filter="resource.labels.service_name=astina"
```

**Log Queries:**
```bash
# Search for errors
gcloud logging read "resource.type=cloud_run_revision" \
    --project=PROJECT_ID \
    --filter="severity>=ERROR" \
    --limit=50
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Configure gcloud
      run: |
        gcloud auth configure-docker
        gcloud config set project ${{ secrets.PROJECT_ID }}
    
    - name: Build and push Docker image
      run: |
        docker build -t us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/astina/astina:latest .
        docker push us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/astina/astina:latest
    
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy astina \
          --image us-central1-docker.pkg.dev/${{ secrets.PROJECT_ID }}/astina/astina:latest \
          --region us-central1 \
          --platform managed \
          --set-env-vars="GOOGLE_CLOUD_BUCKET=astina-models-${{ secrets.PROJECT_ID }}"
```

---

## 🐛 Troubleshooting Deployment Issues

### Common Docker Issues

**Issue: Build fails with "no matching distribution"**
```bash
# Solution: Update base image or use specific Python version
# In Dockerfile, change:
FROM python:3.12-slim
# to:
FROM python:3.13-slim
```

**Issue: Container exits immediately**
```bash
# Check logs
docker logs astina-app

# Common causes:
# 1. Missing dependencies - check requirements.txt
# 2. Port conflict - change port mapping
# 3. Permission issues - check file permissions
```

### Common Cloud Run Issues

**Issue: Service shows "503 Service Unavailable"**
```bash
# Check service status
gcloud run services describe astina --region=us-central1 --project=PROJECT_ID

# Common causes:
# 1. Container starting slowly - increase startup timeout
# 2. Out of memory - increase memory allocation
# 3. Container crashing - check logs for errors
```

**Issue: "502 Bad Gateway"**
```bash
# Check health endpoint
curl https://YOUR_SERVICE_URL/_stcore/health

# Common causes:
# 1. Service not responding on expected port
# 2. Health check failing
# 3. Network configuration issues
```

**Issue: "504 Gateway Timeout"**
```bash
# Increase timeout
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --timeout=7200
```

### Common Configuration Issues

**Issue: Environment variables not working**
```bash
# Verify variable names are correct
gcloud run services describe astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --format="value(spec.template.env)"

# Update if needed
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --set-env-vars="VARIABLE_NAME=value"
```

**Issue: Secrets not accessible**
```bash
# Verify secret exists
gcloud secrets describe SECRET_NAME --project=PROJECT_ID

# Verify IAM permissions
gcloud secrets get-iam-policy SECRET_NAME --project=PROJECT_ID

# Revoke and re-grant if needed
gcloud secrets add-iam-policy-binding SECRET_NAME \
    --member="serviceAccount:SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --project=PROJECT_ID
```

---

## 📈 Performance Optimization

### 1. Container Optimization

**Multi-stage Build:**
The Dockerfile already uses multi-stage build for optimal image size.

**Layer Caching:**
```dockerfile
# In Dockerfile, optimize layer caching
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy application code only after dependencies
COPY --chown=appuser:appuser . .
```

### 2. Runtime Optimization

**Memory Management:**
- Use Polars for large file processing
- Implement chunk-based processing
- Enable garbage collection

**CPU Optimization:**
- Use PyTorch with GPU acceleration when available
- Implement parallel processing with multiprocessing
- Optimize database queries

### 3. Network Optimization

**CDN Integration:**
```bash
# Use Cloud CDN for static assets
gcloud compute backend-services create astina-cdn \
    --global \
    --enable-cdn
```

**Load Balancing:**
Cloud Run automatically handles load balancing across instances.

---

## 🔄 Update and Rollback Procedures

### Update Deployment

**Automatic Update (CI/CD):**
```bash
# Push to main branch triggers automatic deployment
git push origin main
```

**Manual Update:**
```bash
# Build new image
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/astina/astina:v2.0 .

# Push new version
docker push us-central1-docker.pkg.dev/PROJECT_ID/astina/astina:v2.0

# Deploy new version
gcloud run deploy astina \
    --image us-central1-docker.pkg.dev/PROJECT_ID/astina/astina:v2.0 \
    --region=us-central1 \
    --project=PROJECT_ID
```

### Rollback Procedure

**Rollback to Previous Version:**
```bash
# List revisions
gcloud run revisions list astina --region=us-central1 --project=PROJECT_ID

# Rollback to specific revision
gcloud run services update astina \
    --region=us-central1 \
    --project=PROJECT_ID \
    --revision=REVISION_ID
```

**Emergency Rollback:**
```bash
# Stop current deployment
gcloud run services delete astina --region=us-central1 --project=PROJECT_ID

# Redeploy last known good version
gcloud run deploy astina \
    --image us-central1-docker.pkg.dev/PROJECT_ID/astina/astina:stable \
    --region=us-central1 \
    --project=PROJECT_ID
```

---

## 📋 Deployment Checklist

### Pre-Deployment Checklist
- [ ] All dependencies updated in requirements.txt
- [ ] Environment variables configured
- [ ] Database schema updated (if applicable)
- [ ] Model training completed and tested
- [ ] Authentication configured (production)
- [ ] API keys stored in Secret Manager
- [ ] Health checks verified
- [ ] Logging and monitoring setup
- [ ] Backup procedures tested
- [ ] Rollback procedure documented

### Post-Deployment Checklist
- [ ] Service health check passing
- [ ] Authentication working correctly
- [ ] All user roles tested
- [ ] LLM connection tested (if configured)
- [ ] File upload functionality working
- [ ] Model inference tested
- [ ] Audit logs being generated
- [ ] Monitoring dashboards configured
- [ ] Error reporting setup
- [ ] Performance benchmarks met

---

## 🆘 Support and Maintenance

### Monitoring Dashboard Status Codes
- 🟢 **Healthy**: All systems operational
- 🟡 **Degraded**: Partial functionality available
- 🔴 **Critical**: Service unavailable

### Escalation Procedures
1. **Level 1**: Check logs and restart service
2. **Level 2**: Rollback to previous version
3. **Level 3**: Contact infrastructure team
4. **Level 4**: Emergency maintenance mode

### Maintenance Windows
- **Regular Maintenance**: Weekly (Sunday 2-4 AM UTC)
- **Security Updates**: As needed (within 24 hours of patch release)
- **Feature Updates**: Monthly (first Monday of month)

---

## 📞 Additional Resources

- **Cloud Run Documentation**: https://cloud.google.com/run/docs
- **Artifact Registry Documentation**: https://cloud.google.com/artifact-registry/docs
- **Cloud Storage Documentation**: https://cloud.google.com/storage/docs
- **Secret Manager Documentation**: https://cloud.google.com/secret-manager/docs
- **Streamlit Documentation**: https://docs.streamlit.io/

---

## 🎯 Quick Reference Commands

```bash
# Local Development
python run.py

# Docker Local
docker-compose up --build -d

# Cloud Run Deploy
.\.cloudrun\deploy.ps1

# View Logs
gcloud run logs read astina --region=us-central1 --project=PROJECT_ID --follow

# Health Check
curl https://YOUR_SERVICE_URL/_stcore/health

# Rollback
gcloud run services update astina --region=us-central1 --project=PROJECT_ID --revision=REVISION_ID
```

---

**Deployment Guide Version**: 1.0  
**Last Updated**: 2026-09-06  
**Maintained By**: ASTINA Development Team