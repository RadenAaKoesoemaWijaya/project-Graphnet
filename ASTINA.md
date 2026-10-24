# ASTINA - Analisis Sistem Transaksi Identifikasi Nilai Anomali
## Dokumentasi Arsitektur & Alur Kerja Sistem

**Versi Dokumentasi:** 1.0  
**Tanggal:** 24 Oktober 2026  
**Author:** Senior QA Engineering Team  
**Status:** Production Ready

---

## 📋 DAFTAR ISI

1. [Overview Sistem](#1-overview-sistem)
2. [Arsitektur High-Level](#2-arsitektur-high-level)
3. [Komponen Sistem](#3-komponen-sistem)
4. [Alur Kerja Utama](#4-alur-kerja-utama)
5. [Pipeline Deteksi Fraud](#5-pipeline-deteksi-fraud)
6. [Teknologi & Dependencies](#6-teknologi--dependencies)
7. [Keamanan & Compliance](#7-keamanan--compliance)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Monitoring & Logging](#9-monitoring--logging)
10. [Performance Optimization](#10-performance-optimization)
11. [Troubleshooting Guide](#11-troubleshooting-guide)

---

## 1. OVERVIEW SISTEM

### 1.1 Deskripsi Aplikasi

ASTINA (Analisis Sistem Transaksi Identifikasi Nilai Anomali) adalah sistem berbasis AI/ML untuk mendeteksi anomali dan potensi fraud dalam transaksi klaim asuransi kesehatan. Sistem ini menggunakan pendekatan hybrid yang menggabungkan:

- **Rule-Based Detection**: Deteksi berdasarkan aturan bisnis dan domain knowledge
- **Machine Learning**: Ensemble model (Random Forest, XGBoost, LightGBM, CatBoost)
- **Graph Neural Networks (GNN)**: Deteksi pola kolusi antar penyedia layanan
- **Agentic AI Copilot**: Asisten AI untuk investigasi dan penjelasan hasil

### 1.2 Tujuan Utama

1. **Deteksi Dini**: Mengidentifikasi klaim berisiko tinggi sebelum pembayaran
2. **Akurasi Tinggi**: Mengurangi false positive melalui ensemble dan context-aware scoring
3. **Skalabilitas**: Mendukung dataset hingga 3GB dengan out-of-core processing
4. **Audit Trail**: Pelacakan lengkap sesuai regulasi (UU PDP, HIPAA)
5. **Explainability**: SHAP/LIME untuk interpretasi model dan RAG untuk investigasi

### 1.3 Target Pengguna

- **Admin**: Manajemen sistem dan konfigurasi
- **Auditor**: Investigasi klaim berisiko tinggi
- **Analyst**: Pelatihan model dan evaluasi performa
- **Viewer**: Review hasil deteksi tanpa akses modifikasi

---

## 2. ARSITEKTURE HIGH-LEVEL

### 2.1 Diagram Arsitektur

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Home    │ │  Data    │ │ Training │ │ Detection│ │  Status  │  │
│  │  Page    │ │  Collect │ │  Page    │ │  Page    │ │  Page    │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTHENTICATION & RBAC LAYER                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  AuthManager (Session-based, Environment Variable Config)    │  │
│  │  - Role: Admin, Auditor, Analyst, Viewer                    │  │
│  │  - Password: Env Var (Production) / Default (Dev)           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      BUSINESS LOGIC LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  File Handler    │  │  Preprocessing   │  │  Feature Eng.    │ │
│  │  - Upload        │  │  - Schema Harmon.│  │  - Selection     │ │
│  │  - Validation    │  │  - Type Opt.     │  │  - Scaling       │ │
│  │  - Streaming     │  │  - Outlier Det.  │  │  - Imputation    │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │          FRAUD RISK PIPELINE (fraud_risk_pipeline.py)          │  │
│  │  - Repeat Billing Detector  - Phantom Service Rules          │  │
│  │  - Provider Capacity Validator - Upcoding/Unbundling       │  │
│  │  - Inflated Bill/Cloning - Medication/Device Fraud           │  │
│  │  - Prolonged Stay/Readmission - Fuzzy Claim Matcher          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  ML Models       │  │  GNN Model       │  │  Ensemble        │ │
│  │  - RF, XGB, LGB  │  │  - GAT Layers    │  │  - Weighted     │ │
│  │  - CatBoost      │  │  - Graph Pooling │  │  - Voting       │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   AI & EXPLAINABILITY LAYER                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Agentic Copilot │  │  RAG Engine      │  │  Model Explainer│ │
│  │  - LLM Interface │  │  - Knowledge Base│  │  - SHAP         │ │
│  │  - PII Masking   │  │  - Semantic Search│  │  - LIME         │ │
│  │  - Guardrails    │  │  - Context Retrieval│ │  - Feature Imp. │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PERSISTENCE LAYER                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Local Storage   │  │  Cloud Storage   │  │  Database (Opt) │ │
│  │  - models/       │  │  - GCS Bucket    │  │  - PostgreSQL   │ │
│  │  - cache/        │  │  - Audit Logs    │  │  - Claims Table │ │
│  │  - logs/         │  │  - Model Artefacts│ │  - Audit Trail  │ │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Architecture

#### Layer 1: Presentation Layer (Streamlit UI)
- **Entry Point**: `main.py` → `run.py`
- **UI Components**: `ui/pages/`, `ui/sidebar.py`, `ui_components.py`
- **State Management**: `state_manager.py`
- **Routing**: Session state-based navigation

#### Layer 2: Authentication & Authorization
- **Module**: `auth_manager.py`
- **Mechanism**: Session-based authentication with environment variable credentials
- **RBAC**: Role-based access control (Admin, Auditor, Analyst, Viewer)
- **Compliance**: UU PDP No. 27/2022, HIPAA Security Rule

#### Layer 3: Business Logic Layer
- **Data Ingestion**: `file_handler.py` (streaming upload, Parquet conversion)
- **Preprocessing**: `preprocessing_optimized.py`, `large_file_processor.py`
- **Schema Harmonization**: `schema_harmonizer.py`
- **Fraud Detection Pipeline**: `fraud_risk_pipeline.py`
- **Rule Engines**: 8+ specialized rule modules

#### Layer 4: ML/AI Layer
- **Model Training**: `model.py` (CombinedAnomalyDetector)
- **GNN**: Graph Attention Networks for collusion detection
- **Hyperparameter Tuning**: Optuna integration
- **Explainability**: SHAP, LIME, model interpreter

#### Layer 5: AI Copilot & RAG
- **Agentic Copilot**: `agentic_copilot.py` (LLM integration)
- **RAG Engine**: `rag_engine.py` (knowledge base retrieval)
- **PII Masking**: `pii_masker.py` (sensitive data protection)
- **Guardrails**: Prompt injection prevention

#### Layer 6: Data Persistence
- **Local**: Filesystem-based storage (models/, cache/, logs/)
- **Cloud**: Google Cloud Storage mirroring (`cloud_storage.py`)
- **Audit Trail**: Comprehensive logging (`audit_trail.py`)
- **Database**: Optional PostgreSQL integration

---

## 3. KOMPONEN SISTEM

### 3.1 Core Modules

#### 3.1.1 Entry Points
- **`run.py`**: Bootstrap script dengan environment setup
  - Python interpreter resolution (.venv priority)
  - Thread pool configuration (memory optimization)
  - Windows asyncio event loop fix
  - Dependency validation

- **`main.py`**: Streamlit application entry
  - Page routing based on session state
  - Authentication gateway
  - RBAC enforcement
  - Error handling with incident IDs

#### 3.1.2 Data Processing
- **`file_handler.py`**: File upload & streaming
  - Supported formats: CSV, CSV.GZ, Parquet, Excel (XLSX/XLS)
  - Streaming ingestion dengan chunking (8MB chunks)
  - RAM & disk resource checking
  - Polars lazy evaluation for large files
  - Max file size: 3GB (CSV), 1GB (Parquet), 150MB (Excel)

- **`preprocessing_optimized.py`**: Data preprocessing
  - Memory usage monitoring (psutil)
  - Outlier detection (IQR, Z-score)
  - Data range validation
  - Feature scaling & imputation
  - Progressive feature scaler dengan caching

- **`schema_harmonizer.py`**: Schema normalization
  - Column mapping untuk berbagai input formats
  - Type inference & conversion
  - Rule readiness evaluation
  - Missing column handling

#### 3.1.3 Fraud Detection Pipeline
- **`fraud_risk_pipeline.py`**: Orchestration deteksi
  - 8+ rule engines berjalan parallel (ThreadPoolExecutor)
  - Dynamic weight re-normalization (circuit breaker)
  - Context-aware risk scoring
  - Business risk score calculation

**Rule Engines:**
1. **`repeat_billing_detector.py`**: Deteksi billing ganda
   - Temporal window (default 30 hari)
   - Fuzzy matching (threshold 0.85)
   - Whitelist support
   - Adaptive threshold

2. **`phantom_service_rules.py`**: Layanan fiktif
   - Service capacity validation
   - Frequency analysis
   - Rule-based detection

3. **`provider_capacity_validator.py`**: Kapasitas penyedia
   - Daily limits per service type
   - Time-based validation
   - Whitelist support

4. **`upcoding_unbundling_rules.py`**: Upcoding & unbundling
   - Code-level fraud detection
   - Bundling violation detection

5. **`inflated_bill_cloning_rules.py`**: Bill inflation & cloning
   - Amount anomaly detection
   - Duplicate pattern detection

6. **`length_of_stay_rules.py`**: LOS & readmission
   - Prolonged stay detection
   - Readmission pattern analysis

7. **`medication_device_fraud_rules.py`**: Medication & device fraud
   - Drug billing anomalies
   - Medical device fraud patterns

8. **`fuzzy_claim_matcher.py`**: Similarity matching
   - Patient-level fuzzy similarity
   - Temporal window matching
   - Levenshtein distance

#### 3.1.4 ML Models
- **`model.py`**: Model training & inference
  - **CombinedAnomalyDetector**: Ensemble wrapper
  - **Algorithms**: Random Forest, XGBoost, LightGBM, CatBoost, IsolationForest, DBSCAN/HDBSCAN
  - **InsuranceAnomalyGNNModel**: Graph Neural Network untuk collusion detection
  - **Optimization**: Adaptive batch size, GPU memory management
  - **Imbalance Handling**: SMOTE, RandomUnderSampler, SMOTETomek

**Key Features:**
- Progressive feature scaling dengan caching
- Fast rank normalization (O(n log k) approximation)
- Optuna hyperparameter optimization
- Model versioning & registry

#### 3.1.5 AI Copilot & RAG
- **`agentic_copilot.py`**: AI Investigator Assistant
  - LLM integration (Gemini, OpenAI, Ollama, Heuristic)
  - PII masking sebelum LLM call
  - Security guardrails (prompt injection prevention)
  - Investigation dossier generation (BAP format)
  - Retry logic dengan exponential backoff

- **`rag_engine.py`**: Retrieval-Augmented Generation
  - Knowledge base management
  - Semantic similarity search
  - Context retrieval untuk LLM

- **`pii_masker.py`**: PII Protection
  - Regex-based masking (names, NIK, phone, email)
  - Error sanitization
  - Compliance: UU PDP, HIPAA

#### 3.1.6 Security & Compliance
- **`auth_manager.py`**: Authentication & RBAC
  - Environment variable-based credentials
  - Production mode enforcement
  - Session management
  - Role-based page access

- **`audit_trail.py`**: Comprehensive logging
  - Local JSON logs
  - Google Cloud Logging integration
  - PII masking in logs
  - WORM sync to GCS (every 50 events)
  - Event tracking: upload, preprocessing, training, detection, export

- **`cloud_storage.py`**: Cloud persistence
  - Google Cloud Storage mirroring
  - Production validation
  - Health check on startup
  - Automatic retry logic

#### 3.1.7 Utility Modules
- **`config.py`**: Centralized configuration
  - File size limits
  - Memory settings
  - GNN sampling config
  - LLM & RAG config
  - Database config

- **`state_manager.py`**: Session state management
  - Page navigation
  - State persistence
  - Processing flags

- **`cache_manager.py`**: Caching layer
  - Model caching
  - Feature caching
  - Cache invalidation

- **`error_handler.py`**: Error handling
  - Graceful degradation
  - User-friendly error messages
  - Incident ID generation

### 3.2 Advanced Modules (Optional/Phase 2)

#### 3.2.1 Adaptive Threshold
- **`adaptive_threshold_engine.py`**: Dynamic threshold adjustment
  - Historical data analysis
  - Statistical threshold computation
  - Whitelist-aware adaptation

#### 3.2.2 Context-Aware Scoring
- **`context_aware_risk_scorer.py`**: Contextual risk evaluation
  - Patient history tracking
  - Composite risk scoring
  - Flag suppression logic

#### 3.2.3 Historical Baseline
- **`historical_baseline_system.py`**: Baseline comparison
  - Historical data storage
  - Baseline computation
  - Drift detection

#### 3.2.4 Whitelist Management
- **`whitelist_manager.py`**: Whitelist operations
  - Whitelist CRUD
  - Persistence
  - Validation

---

## 4. ALUR KERJA UTAMA

### 4.1 Application Startup Flow

```
User runs: python run.py
    ↓
run.py resolves Python interpreter (.venv priority)
    ↓
Check Python dependencies (subprocess validation)
    ↓
Configure thread pool & environment variables:
  - POLARS_MAX_THREADS = min(4, cpu_count/2)
  - OMP_NUM_THREADS, MKL_NUM_THREADS, etc.
  - POLARS_STREAMING_CHUNK_SIZE = 50000
    ↓
Create directories: models/, plots/, studies/, logs/, cache/
    ↓
Launch Streamlit: streamlit run main.py
    ↓
main.py:
  - Apply Windows asyncio fix (SelectorEventLoop)
  - Set page config
  - Import UI components
  - Validate production auth configuration
  - Validate production storage configuration
    ↓
Check authentication:
  - If not authenticated → render login page
  - If authenticated → render sidebar + current page
    ↓
RBAC check:
  - Can user access current page?
  - If no → show access denied
  - If yes → render page
```

### 4.2 Data Upload & Ingestion Flow

```
User navigates to "Unggah Data" page
    ↓
Upload file via Streamlit file uploader
    ↓
file_handler.py:
  - Check file size limits (CSV: 3GB, Parquet: 1GB, Excel: 150MB)
  - Check disk space (TEMP_DATA_DIR)
  - Check RAM availability (psutil)
    ↓
If RAM < 128 MB and file > 100 MB:
  - Show warning: "RAM tersedia rendah"
  - Continue with memory-safe mode
    ↓
Stream upload to disk (8MB chunks)
    ↓
Determine file type & sniff format:
  - CSV: sniff_csv_dialect() → encoding, delimiter
  - Excel: use openpyxl read_only
  - Parquet: direct read
    ↓
Convert to Parquet (streaming):
  - If CSV: stream_csv_to_parquet()
    * Try Polars scan_csv (if UTF-8)
    * Fallback to pandas chunks (50k rows)
  - If Excel: stream_excel_to_parquet()
    * openpyxl read_only iterator
    * Chunk-wise conversion
    ↓
Save to: TEMP_DATA_DIR/raw_{uuid}.parquet
    ↓
Update session state:
  - df_processed_path = parquet file path
  - preprocessing_metadata = {rows, cols, size, etc.}
    ↓
Log audit event:
  - log_data_upload(file_name, size, rows, cols)
```

### 4.3 Data Preprocessing Flow

```
User navigates to "Pelatihan Model" or auto-preprocess
    ↓
Load Parquet file (polars.read_parquet or pandas)
    ↓
schema_harmonizer.harmonize_claims_schema():
  - Map input columns to standard schema
  - Infer data types
  - Add missing columns with defaults
  - Create _astina_row_id for mapping
    ↓
preprocessing_optimized:
  - check_memory_threshold() every 1000 rows
  - detect_and_handle_outliers()
    * Method: IQR or Z-score
    * Action: cap, flag, or remove
  - validate_data_ranges()
    * Age: 0-120
    * Amount: logical ranges
  - optimize_memory_usage()
    * Downcast int64 → int32/int16/int8
    * Downcast float64 → float32
    * Convert object → category (if cardinality < 50%)
    ↓
Feature Engineering:
  - Select numerical columns
  - Create derived features (if needed)
  - Handle missing values (SimpleImputer)
  - Scale features (StandardScaler)
    ↓
Feature Selection (if training):
  - SelectKBest with mutual_info_classif
  - Or use pre-configured feature list
    ↓
Save processed data:
  - To Parquet (if large)
  - Or keep in session state (if small)
    ↓
Update session state:
  - feature_columns = selected features
  - preprocessing_metadata = {total_rows, outliers, etc.}
    ↓
Log audit event:
  - log_preprocessing(original_rows, processed_rows, time)
```

### 4.4 Model Training Flow

```
User navigates to "Pelatihan Model"
    ↓
Load processed data from session state
    ↓
User selects model configuration:
  - Algorithm(s): RF, XGB, LGB, CatBoost, GNN
  - Training mode: single, ensemble, or all
  - Hyperparameter tuning: manual or Optuna
  - Imbalance handling: SMOTE, undersampling, etc.
    ↓
Split data:
  - Train/test split (default 80/20)
  - Stratified split if labels available
    ↓
Train models (parallel or sequential):
  For each selected algorithm:
    ↓
    Initialize model with hyperparameters
    ↓
    Fit model on training data
    ↓
    If Optuna enabled:
      - Run optimization trials
      - Use pruning for early stopping
      - Log best parameters
    ↓
    Evaluate on validation set
    ↓
    Calculate metrics:
      - F1-score, Precision, Recall
      - ROC-AUC, PR-AUC
      - Confusion matrix
    ↓
    Save model to disk:
      - models/fraud_detector_{algorithm}.pkl
      - models/fraud_detector_gnn.pt (PyTorch)
    ↓
If ensemble mode:
  - Train multiple models
  - Combine predictions (weighted voting)
  - Save ensemble configuration
    ↓
Generate feature importance:
  - SHAP values (if supported)
  - Permutation importance
  - Save plots to plots/
    ↓
Save model metadata:
  - Model registry entry
  - Training timestamp
  - Hyperparameters
  - Performance metrics
    ↓
Update session state:
  - model_trained = True
  - trained_detector = CombinedAnomalyDetector instance
  - evaluation_results = metrics dict
    ↓
Log audit event:
  - log_model_training(model_type, algorithm, samples, time, metrics)
```

### 4.5 Anomaly Detection Flow

```
User navigates to "Deteksi Anomali"
    ↓
Load trained model from session state or model registry
    ↓
Load new data (or use training data)
    ↓
Preprocess data (same pipeline as training)
    ↓
fraud_risk_pipeline.run_integrated_claim_risk_pipeline():
  ↓
  Phase 1: Rule-Based Detection (parallel)
    ↓
    ThreadPoolExecutor(max_workers=4):
      - repeat_billing_detector.detect_repeat_claims()
      - phantom_service_engine.validate_claims_dataframe()
      - capacity_validator.validate_all_providers_batch()
      - compute_patient_level_fuzzy_similarity_scores()
      - detect_upcoding_and_unbundling()
      - detect_inflated_bill_and_cloning()
      - detect_prolonged_stay_and_readmission()
      - detect_medication_and_device_fraud()
    ↓
    Map results to claim rows:
      - repeat_billing_score, repeat_billing_flag
      - phantom_service_score, phantom_service_flag
      - provider_capacity_score, provider_capacity_flag
      - etc.
    ↓
    Calculate business_risk_score:
      - Dynamic weight re-normalization
      - Weighted sum of rule scores
      - business_risk_flag = (score >= 0.6)
  ↓
  Phase 2: Context-Aware Scoring (if enabled)
    ↓
    context_scorer.update_patient_history(claims)
    ↓
    For each claim:
      - Calculate composite_risk_score
      - Determine should_suppress_flag
      - Adjust final_business_risk_flag
  ↓
  Phase 3: ML Model Prediction (if model provided)
    ↓
    Load features from processed data
    ↓
    Predict with ensemble:
      - Get predictions from each model
      - Combine with weights
      - anomaly_prediction, anomaly_score
    ↓
    Phase 4: GNN Graph Analysis (if enabled and graph data available)
    ↓
    Build graph:
      - Nodes: claims, providers, patients
      - Edges: relationships
    ↓
    Run GNN inference:
      - Use NeighborLoader for large graphs
      - Get node-level predictions
      - Detect collusion patterns
    ↓
  Phase 5: Final Risk Aggregation
    ↓
    Combine all signals:
      - business_risk_score
      - ml_anomaly_score
      - gnn_collusion_score
      - composite_risk_score
    ↓
    Calculate final_risk_score:
      - Weighted combination
      - final_risk_flag = (score >= threshold)
    ↓
    Assign risk_category:
      - Repeat Billing, Phantom Service, etc.
      - High Anomaly, Low Risk
    ↓
Return enriched DataFrame with all scores & flags
    ↓
Display results in UI:
  - Summary statistics
  - High-risk claims table
  - Risk distribution plot
  - Feature importance (SHAP)
    ↓
Export options:
  - CSV export
  - PDF report (reportlab)
  - Model export
    ↓
Log audit event:
  - log_anomaly_detection(samples, anomalies, rate, time)
```

### 4.6 AI Copilot Investigation Flow

```
User clicks "Investigate with AI Copilot" on a claim
    ↓
agentic_copilot.generate_investigation_dossier():
  ↓
  Gather context:
    - Claim details (masked PII)
    - Rule violations (flags & scores)
    - ML predictions (probabilities)
    - SHAP feature importance
    - GNN graph patterns (if available)
    ↓
  Retrieve relevant knowledge:
    - rag_engine.get_rag_knowledge_base()
    - Semantic search for similar cases
    - Retrieve investigation guidelines
    ↓
  Construct LLM prompt:
    - System prompt: "You are an insurance fraud investigator..."
    - Context: masked claim data, rules, ML results
    - Knowledge: retrieved RAG snippets
    - Instruction: "Generate investigation dossier (BAP format)"
    ↓
  Apply security guardrails:
    - AIGuardrail.validate_query()
    - Check for prompt injection patterns
    - Block if suspicious
    ↓
  Call LLM API:
    - Provider: Gemini, OpenAI, or Ollama
    - Retry logic with exponential backoff
    - Timeout handling
    ↓
  Parse LLM response:
    - Extract investigation findings
    - Extract recommended actions
    - Extract evidence list
    ↓
  Display dossier:
    - Investigation summary
    - Risk assessment
    - Evidence & patterns
    - Recommended next steps
    ↓
  Log audit event:
    - log_security_event("copilot_investigation", description)
```

---

## 5. PIPELINE DETEKSI FRAUD

### 5.1 Pipeline Architecture

```
Raw Claims Data
    ↓
┌─────────────────────────────────────────────────────────────┐
│                 Schema Harmonization                         │
│  - Column mapping & normalization                           │
│  - Type inference & conversion                              │
│  - Add _astina_row_id for tracking                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│          Rule-Based Detection (Parallel)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Repeat       │  │ Phantom      │  │ Provider     │     │
│  │ Billing      │  │ Service      │  │ Capacity     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Upcoding     │  │ Inflated     │  │ Medication   │     │
│  │ Unbundling   │  │ Bill/Cloning │  │ Device       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                           │
│  │ Fuzzy Match  │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│          Business Risk Score Calculation                     │
│  - Dynamic weight re-normalization                           │
│  - Weighted sum of rule scores                               │
│  - Threshold: 0.6 (configurable)                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│          Context-Aware Scoring (Phase 2)                      │
│  - Patient history tracking                                   │
│  - Composite risk scoring                                    │
│  - Flag suppression based on context                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│          ML Model Prediction (if trained)                     │
│  - Ensemble prediction (RF, XGB, LGB, CatBoost)             │
│  - Anomaly probability                                        │
│  - Feature importance (SHAP)                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│          GNN Graph Analysis (if enabled)                      │
│  - Build graph (claims-providers-patients)                  │
│  - GAT layer inference                                        │
│  - Collusion pattern detection                               │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│          Final Risk Aggregation                              │
│  - Combine: business_risk + ml_score + gnn_score            │
│  - Weighted final risk score                                 │
│  - Risk category assignment                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
Enriched Claims with:
- All rule flags & scores
- ML predictions
- GNN scores
- Final risk score & category
- Context suppression flags
```

### 5.2 Rule Engine Details

#### 5.2.1 Repeat Billing Detector
```python
Configuration:
  - temporal_window_days: 30
  - fuzzy_threshold: 0.85
  - amount_variance_tolerance_pct: 5.0
  - min_risk_score_for_alert: 0.7

Detection Logic:
  1. Group by patient_id
  2. Sort by billing_date
  3. Compare claims within temporal window
  4. Calculate fuzzy similarity (service_code, amount, provider)
  5. Flag if similarity >= threshold
  6. Support adaptive threshold & whitelist
```

#### 5.2.2 Phantom Service Rules
```python
Configuration:
  - rules_update_interval_days: 7
  - enable_rule_validation: True
  - enable_provider_capacity_check: True
  - enable_frequency_check: True

Service Capacity Limits:
  - CT_SCAN: max 2/day, 30 min, requires specialist
  - X_RAY: max 5/day, 10 min
  - BLOOD_TEST: max 20/day, 5 min
  - CONSULTATION: max 15/day, 20 min, requires specialist
  - ULTRASOUND: max 5/day, 25 min, requires specialist
  - LABORATORY: max 10/day, 8 min
  - SURGERY: max 1/day, 180 min, requires specialist

Detection Logic:
  1. Validate service code exists in rules
  2. Check daily frequency per provider
  3. Check service duration constraints
  4. Check specialist requirement
  5. Flag if violations detected
```

#### 5.2.3 Provider Capacity Validator
```python
Detection Logic:
  1. Group by provider_id and service_date
  2. Count services per day
  3. Compare against SERVICE_CAPACITY limits
  4. Flag if exceeding daily limits
  5. Support whitelist for high-volume providers
```

#### 5.2.4 Fuzzy Claim Matcher
```python
Algorithm:
  - Levenshtein distance for string similarity
  - Numerical difference for amounts
  - Temporal proximity for dates

Similarity Calculation:
  - service_code: exact match or hierarchical similarity
  - amount: normalized difference
  - provider: exact match
  - billing_date: temporal window check

Score Range: 0.0 - 1.0
Default Threshold: 0.8
```

### 5.3 Dynamic Weight Re-normalization

```python
Circuit Breaker Logic:
  1. Evaluate rule readiness (schema_harmonizer.evaluate_rule_readiness)
  2. Only include rules with required columns available
  3. Re-normalize weights to sum to 1.0

Default Weights (if all rules ready):
  - repeat_billing: 0.40
  - phantom_service: 0.20
  - provider_capacity: 0.15
  - fuzzy_similarity: 0.15
  - additional_fraud: 0.10

If a rule is not ready:
  - Remove its weight from total
  - Re-normalize remaining weights
  - Continue with reduced set
```

### 5.4 Context-Aware Risk Scoring

```python
Logic:
  1. Track patient history (frequency, patterns)
  2. For each claim:
     - Collect all module flags & scores
     - Calculate composite risk score
     - Check if flag should be suppressed (legitimate pattern)
  3. Suppression criteria:
     - First-time offender with low risk
     - Established pattern with historical approval
     - Whitelisted provider-patient pair
  4. Adjust final flag:
     - final_business_risk_flag = business_risk_flag AND NOT context_suppressed
```

---

## 6. TEKNOLOGI & DEPENDENCIES

### 6.1 Core Dependencies

#### Framework & UI
- **Streamlit 1.61.1**: Web framework untuk UI
- **Python 3.11+**: Runtime environment

#### Data Processing
- **Pandas 2.2.3**: Data manipulation & analysis
- **NumPy 2.1.0**: Numerical computing
- **Polars 1.0.0+**: High-performance dataframe library (lazy evaluation)
- **PyArrow 18.1.0+**: Columnar storage format (Parquet)
- **OpenPyXL 3.1.5+**: Excel file handling
- **Chardet 5.2.0**: Character encoding detection

#### Machine Learning
- **Scikit-learn 1.6.1**: Classical ML algorithms
- **XGBoost 2.1.4+**: Gradient boosting
- **LightGBM 4.5.0+**: Gradient boosting (leaf-wise)
- **CatBoost 1.2.0+**: Gradient boosting (categorical features)
- **Imbalanced-learn 0.14.0+**: SMOTE, undersampling
- **Optuna 4.0.0+**: Hyperparameter optimization
- **HDBSCAN 0.8.40+**: Density-based clustering

#### Deep Learning
- **PyTorch 2.4.0+**: Deep learning framework
- **Torch-Geometric 2.6.0+**: Graph neural networks
- **FAISS-CPU 1.7.4+**: Similarity search (optional)

#### Explainability
- **SHAP 0.48.0+**: Model explanations
- **LIME 0.2.0.0+**: Local explanations

#### Visualization
- **Plotly 6.0.0+**: Interactive plots
- **Matplotlib 3.10.0+**: Static plots
- **Seaborn 0.13.0+**: Statistical visualization
- **NetworkX 3.5.0+**: Graph visualization

#### Utilities
- **Joblib 1.5.3**: Parallel processing
- **TQDM 4.67.1**: Progress bars
- **PSUtil 7.0.0+**: System monitoring
- **ReportLab 4.2.0+**: PDF generation

#### Cloud & Storage
- **Google Cloud Storage 3.0.0+**: Cloud storage integration

#### Validation
- **Pydantic 2.9.2**: Data validation
- **Pydantic-Settings 2.6.1**: Configuration management

#### Testing
- **Pytest 8.0+**: Testing framework

### 6.2 Technology Rationale

#### Polars vs Pandas
- **Polars**: Used for large file ingestion (streaming, lazy evaluation)
- **Pandas**: Used for ML preprocessing & scikit-learn compatibility
- **Hybrid approach**: Best of both worlds

#### GNN Choice
- **Graph Attention Networks (GAT)**: Suitable for claim-provider-patient relationships
- **Multi-head attention**: Captures complex relationship patterns
- **Node-level prediction**: Per-claim anomaly scores

#### Ensemble Strategy
- **Weighted voting**: Combines strengths of multiple algorithms
- **Algorithm diversity**: Tree-based (RF, XGB, LGB, CatBoost) + GNN
- **Fallback mechanism**: Graceful degradation if models unavailable

#### SHAP for Explainability
- **Model-agnostic**: Works with all algorithms
- **Global & local explanations**: Feature importance + per-instance
- **Background sampling**: Efficient for large datasets

---

## 7. KEAMANAN & COMPLIANCE

### 7.1 Authentication & Authorization

#### Authentication Mechanism
```python
Module: auth_manager.py

Development Mode (AUTH_ENABLED=false or not set):
  - Default passwords:
    * admin: AdminAstina2026!
    * auditor: AuditorAstina2026!
    * analyst: AnalystAstina2026!
    * viewer: ViewerAstina2026!
  - Environment variables override defaults if set

Production Mode (AUTH_ENABLED=true):
  - Passwords MUST be set via environment variables:
    * ASTINA_ADMIN_PASSWORD (min 12 chars)
    * ASTINA_AUDITOR_PASSWORD (min 12 chars)
    * ASTINA_ANALYST_PASSWORD (min 12 chars)
    * ASTINA_VIEWER_PASSWORD (min 12 chars)
  - No hardcoded fallbacks
  - Startup fails if passwords not configured
```

#### Role-Based Access Control (RBAC)
```python
Roles:
  - admin: Full access to all pages
  - auditor: Access to detection, evaluation, status
  - analyst: Access to training, evaluation, data collection
  - viewer: Read-only access to detection, status

Page Access Matrix:
  ┌─────────┬───────┬─────────┬─────────┬─────────┬──────────┬────────┬──────────┐
  │ Page    │ Home  │ Collect │ Train   │ Evaluate│ Detect  │ Status │ Settings │
  ├─────────┼───────┼─────────┼─────────┼─────────┼──────────┼────────┼──────────┤
  │ admin   │   ✓   │    ✓    │    ✓    │    ✓    │    ✓    │   ✓    │    ✓     │
  │ auditor │   ✓   │    ✗    │    ✗    │    ✓    │    ✓    │   ✓    │    ✗     │
  │ analyst │   ✓   │    ✓    │    ✓    │    ✓    │    ✗    │   ✓    │    ✗     │
  │ viewer  │   ✓   │    ✗    │    ✗    │    ✗    │    ✓    │   ✓    │    ✗     │
  └─────────┴───────┴─────────┴─────────┴─────────┴──────────┴────────┴──────────┘
```

### 7.2 Data Privacy & PII Protection

#### PII Masking
```python
Module: pii_masker.py

Masked Fields:
  - Names: regex pattern for person names
  - NIK (Indonesian ID): 16-digit identification
  - Phone numbers: Various formats
  - Email addresses: Standard email regex
  - Addresses: Partial masking

Masking Strategy:
  - Full replacement with placeholders (e.g., [MASKED_NAME])
  - Applied before:
    * LLM API calls (agentic_copilot)
    * Audit logs (audit_trail)
    * Export reports
  - Original data never exposed to external systems
```

#### Compliance Standards
```python
Regulations:
  - UU No. 27 Tahun 2022 (Perlindungan Data Pribadi / UU PDP)
    * Explicit consent for data processing
    * Data minimization
    * Purpose limitation
    * Security measures
    * Data subject rights
  
  - HIPAA Security Rule (US Healthcare)
    * Access control (RBAC implemented)
    * Audit controls (comprehensive logging)
    * Integrity controls (hash verification)
    * Transmission security (HTTPS for production)
  
  - GDPR (European Union - optional deployment)
    * Lawful basis for processing
    * Data protection by design
    * Right to explanation (SHAP, LIME)
```

### 7.3 Audit Trail

#### Audit Logging
```python
Module: audit_trail.py

Logged Events:
  - data_upload: File upload details
  - data_preprocessing: Preprocessing metrics
  - model_training: Training parameters & results
  - anomaly_detection: Detection statistics
  - model_export: Model export events
  - security: Security-related events

Log Storage:
  - Local: logs/audit/audit_{date}.log (JSON lines)
  - Cloud: Google Cloud Logging (if enabled)
  - WORM: GCS mirroring every 50 events

Log Format (JSON):
  {
    "timestamp": "2026-10-24T05:30:23.123Z",
    "event_type": "data_upload",
    "action": "upload",
    "resource": "sample_claims.csv",
    "details": {
      "file_size_bytes": 1234567,
      "file_size_mb": 1.18,
      "row_count": 1000,
      "column_count": 15,
      "success": true
    },
    "user_id": "admin",
    "severity": "INFO",
    "environment": "local"
  }

Query Capabilities:
  - get_recent_events(limit=20)
  - get_events_by_type(event_type, limit=50)
  - get_events_by_date(date)
  - cleanup_old_logs(days_to_keep=30)
```

### 7.4 Security Guardrails

#### AI Guardrails
```python
Module: agentic_copilot.py - AIGuardrail class

Protection Against:
  - Prompt injection (ignore instructions, reveal system prompt)
  - Jailbreak attempts (act as unrestricted AI)
  - SQL injection (drop database, delete from)
  - Command injection (rm -rf, eval, os.system)
  - Data exfiltration (reveal secrets, download data)

Patterns Detected:
  - English: "ignore previous instructions", "reveal system prompt"
  - Indonesian: "abaikan instruksi", "tunjukkan prompt sistem"
  - Framework-specific: <im_start>, [SYSTEM], [INST]

Response:
  - Block suspicious queries
  - Log security event
  - Return error message
```

#### Production Validation
```python
Startup Checks (fail closed):

1. Authentication:
   - validate_production_auth_configuration()
   - AUTH_ENABLED must be true
   - All passwords must be set (min 12 chars)

2. Storage:
   - validate_production_storage_configuration()
   - GOOGLE_CLOUD_BUCKET must be set
   - google-cloud-storage must be available
   - Health check: write & delete test file

3. Environment:
   - ASTINA_ENVIRONMENT must be "production" or "prod"
   - Warning in dev mode, error in prod mode
```

---

## 8. DEPLOYMENT ARCHITECTURE

### 8.1 Local Development

#### Requirements
```bash
- Python 3.11+
- 8GB RAM (16GB recommended)
- 10GB disk space
- (Optional) NVIDIA GPU with CUDA 12.4+ for GNN
```

#### Setup Steps
```bash
# 1. Clone repository
git clone <repo-url>
cd project-Graphnet

# 2. Create virtual environment
python -m venv .venv

# 3. Activate environment
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run application
python run.py
```

#### Configuration
```bash
# Environment variables (optional for local dev)
export ASTINA_ENVIRONMENT=development
export AUTH_ENABLED=false  # Use default passwords
export TEMP_DATA_DIR=./temp_data
```

### 8.2 Google Cloud Run Deployment

#### Architecture
```
Internet → Cloud Load Balancer → Cloud Run Service
                                    ↓
                            Container Image (Docker)
                                    ↓
                            Streamlit Application
                                    ↓
                            Google Cloud Storage (Models, Logs)
                                    ↓
                            Cloud Logging (Audit Logs)
```

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set environment variables
ENV ASTINA_ENVIRONMENT=production
ENV AUTH_ENABLED=true
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run application
CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

#### Deployment Steps
```bash
# 1. Build Docker image
gcloud builds submit --tag gcr.io/PROJECT_ID/astina:latest

# 2. Deploy to Cloud Run
gcloud run deploy astina \
  --image gcr.io/PROJECT_ID/astina:latest \
  --platform managed \
  --region REGION \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1 \
  --timeout 3600

# 3. Set environment variables
gcloud run services update astina \
  --set-env-vars ASTINA_ENVIRONMENT=production \
  --set-env-vars AUTH_ENABLED=true \
  --set-env-vars ASTINA_ADMIN_PASSWORD=*** \
  --set-env-vars ASTINA_AUDITOR_PASSWORD=*** \
  --set-env-vars ASTINA_ANALYST_PASSWORD=*** \
  --set-env-vars ASTINA_VIEWER_PASSWORD=*** \
  --set-env-vars GOOGLE_CLOUD_BUCKET=astina-models \
  --set-env-vars GOOGLE_CLOUD_BUCKET_PREFIX=production
```

#### Cloud Storage Setup
```bash
# Create bucket
gsutil mb -p PROJECT_ID -l REGION gs://astina-models

# Set IAM permissions
gsutil iam ch allUsers:objectViewer gs://astina-models
```

### 8.3 Environment Variables

#### Required for Production
```bash
# Authentication
AUTH_ENABLED=true
ASTINA_ADMIN_PASSWORD=<min-12-chars>
ASTINA_AUDITOR_PASSWORD=<min-12-chars>
ASTINA_ANALYST_PASSWORD=<min-12-chars>
ASTINA_VIEWER_PASSWORD=<min-12-chars>

# Storage
GOOGLE_CLOUD_BUCKET=astina-models
GOOGLE_CLOUD_BUCKET_PREFIX=production

# Environment
ASTINA_ENVIRONMENT=production
```

#### Optional Configuration
```bash
# File size limits
ASTINA_MAX_CSV_MB=3072
ASTINA_MAX_PARQUET_MB=1024
ASTINA_MAX_EXCEL_MB=150

# Memory thresholds
ASTINA_MIN_AVAILABLE_RAM_MB=128
ASTINA_CRITICAL_RAM_MB=64

# LLM Configuration
LLM_PROVIDER=gemini  # or openai, ollama, heuristic
LLM_MODEL_NAME=gemini-1.5-flash
LLM_ENDPOINT_URL=http://localhost:11434/api/generate

# Database (optional)
ENABLE_DATABASE=1
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=***
DB_NAME=astina
DB_POOL_SIZE=10
```

---

## 9. MONITORING & LOGGING

### 9.1 Logging Configuration

#### Logging Setup
```python
Module: logging_config.py

Log Levels:
  - DEBUG: Detailed debugging information
  - INFO: General informational messages
  - WARNING: Warning messages for potential issues
  - ERROR: Error messages for failures
  - CRITICAL: Critical errors requiring immediate attention

Log Format:
  - Plain text (default)
  - JSON (if ASTINA_LOG_FORMAT=json)

Log Outputs:
  - Console (Streamlit)
  - File: logs/app.log
  - Cloud Logging (if enabled)
```

#### Log Categories
```python
Main Loggers:
  - graphnet.model: Model training & inference
  - astina.file_handler: File upload & processing
  - astina.fraud_risk_pipeline: Fraud detection pipeline
  - graphnet.storage: Cloud storage operations
  - astina.audit_trail: Audit logging
  - __main__: Application lifecycle
```

### 9.2 System Status Monitoring

#### Status Page
```python
Module: ui/pages/status.py

Displayed Metrics:
  - CPU usage (psutil)
  - Memory usage (psutil)
  - Disk usage (TEMP_DATA_DIR)
  - GPU status (if available)
  - Pipeline readiness (data, model, evaluation, detection)
  - Model registry versions
  - Cloud storage connectivity
  - Recent audit events
```

#### Health Checks
```python
Built-in Health Checks:
  - Dependency availability (torch, sklearn, etc.)
  - Cloud storage connectivity (write & delete test file)
  - Database connectivity (if enabled)
  - Model file integrity (hash verification)
```

### 9.3 Performance Metrics

#### Tracked Metrics
```python
During Training:
  - Training time per algorithm
  - Memory usage peak
  - Feature importance computation time
  - Hyperparameter optimization trials

During Detection:
  - Inference time per claim
  - Pipeline execution time
  - Rule engine parallelism efficiency
  - Memory usage during detection

During Ingestion:
  - Upload speed (MB/s)
  - Conversion time (CSV → Parquet)
  - Chunk processing throughput
```

---

## 10. PERFORMANCE OPTIMIZATION

### 10.1 Memory Optimization

#### Strategies
```python
1. Dtype Downcasting:
   - int64 → int32/int16/int8 (based on value range)
   - float64 → float32
   - object → category (if cardinality < 50%)

2. Lazy Evaluation:
   - Polars lazy frames for large files
   - Stream processing instead of full load
   - Chunk-wise operations

3. Garbage Collection:
   - Explicit gc.collect() after large operations
   - Delete intermediate DataFrames
   - Clear GPU cache after GNN operations

4. Memory Monitoring:
   - check_memory_threshold() every 1000 rows
   - Warning at 6GB, critical at 8GB
   - Automatic garbage collection on threshold
```

#### Configuration
```python
config.py Settings:
  MEMORY_LIMIT_MB = 4096  # 4GB limit
  PANDAS_DTYPE_OPTIMIZATION = True
  USE_CATEGORICAL_OPTIMIZATION = True
  CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks
```

### 10.2 Processing Optimization

#### Thread Pool Configuration
```python
run.py Configuration:
  max_threads = min(4, cpu_count / 2)

Environment Variables:
  - POLARS_MAX_THREADS
  - RAYON_NUM_THREADS
  - OMP_NUM_THREADS
  - OPENBLAS_NUM_THREADS
  - MKL_NUM_THREADS
  - VECLIB_MAXIMUM_THREADS
  - NUMEXPR_NUM_THREADS

Rationale:
  - Prevents thread explosion
  - Reduces memory contention
  - Improves stability on resource-constrained systems
```

#### Parallel Rule Execution
```python
fraud_risk_pipeline.py:
  ThreadPoolExecutor(max_workers=4)
  - 8 rule engines run in parallel
  - Reduces total detection time
  - Thread-safe result aggregation
```

### 10.3 GPU Optimization

#### GNN GPU Acceleration
```python
model.py Functions:
  - get_optimal_batch_size(): Calculate based on GPU memory
  - get_adaptive_gnn_threshold(): Determine sampling strategy
  - clean_gpu_memory(): Explicit cache clearing

GPU Memory Management:
  - Reserve 2GB for overhead
  - Use 80% of available memory
  - Adaptive batch size calculation
  - NeighborLoader for large graphs
```

#### Adaptive Thresholds
```python
GPU Memory < 4GB:
  - GNN node threshold: 3000
  - Batch size: 512 (CPU)

GPU Memory 4-8GB:
  - GNN node threshold: 5000
  - Batch size: 1024 (GPU)

GPU Memory > 8GB:
  - GNN node threshold: 8000
  - Batch size: 2048 (GPU)
```

### 10.4 Large File Processing

#### Streaming Architecture
```python
file_handler.py Flow:
  1. Upload stream (8MB chunks) → disk
  2. Polars scan_csv (lazy) → streaming read
  3. Polars sink_parquet → streaming write
  4. No full DataFrame in memory

Chunk Size Calculation:
  - < 100MB: Process whole file
  - 100-500MB: 50k rows per chunk
  - 500MB-1GB: 50k rows per chunk
  - > 1GB: 25k rows per chunk (more conservative)
```

#### Out-of-Core Processing
```python
large_file_processor.py:
  - Polars lazy evaluation
  - Memory-mapped files
  - Disk-based intermediate storage
  - Progressive aggregation
```

---

## 11. TROUBLESHOOTING GUIDE

### 11.1 Common Issues

#### Issue 1: RAM Warning During Upload
```
Symptom:
  "⚠️ RAM tersedia rendah (84 MB). Proses upload tetap dilanjutkan dengan mode hemat memori."

Cause:
  - Available RAM < 128 MB threshold
  - File size > 100 MB

Solutions:
  1. Close other applications to free RAM
  2. Use smaller file for testing
  3. Adjust threshold:
     export ASTINA_MIN_AVAILABLE_RAM_MB=64
  4. Restart application after freeing RAM
```

#### Issue 2: Import Error for PyTorch
```
Symptom:
  ModuleNotFoundError: No module named 'torch'

Cause:
  - PyTorch not installed or incompatible version
  - Wrong Python interpreter used

Solutions:
  1. Ensure .venv is activated
  2. Reinstall dependencies:
     pip install -r requirements.txt
  3. Check Python version: python --version (should be 3.11+)
  4. Verify torch installation:
     python -c "import torch; print(torch.__version__)"
```

#### Issue 3: Windows AsyncIO Error
```
Symptom:
  ConnectionResetError: [WinError 10054] Connection reset by remote host

Cause:
  - ProactorEventLoop bug in Python 3.12 on Windows

Solution:
  - Fixed in run.py and main.py:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  - Ensure using latest code version
```

#### Issue 4: GPU Not Available
```
Symptom:
  CPU-only training despite having NVIDIA GPU

Cause:
  - CUDA not installed or incompatible
  - PyTorch CPU-only version installed
  - GPU not detected by PyTorch

Solutions:
  1. Check CUDA availability:
     python -c "import torch; print(torch.cuda.is_available())"
  2. Install CUDA version matching PyTorch:
     pip install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu124
  3. Update NVIDIA drivers
  4. Verify GPU: nvidia-smi
```

#### Issue 5: Cloud Storage Connection Failed
```
Symptom:
  RuntimeError: Deployment produksi tidak dapat mengakses bucket GCS

Cause:
  - GOOGLE_CLOUD_BUCKET not set
  - Service account missing permissions
  - Network connectivity issue

Solutions:
  1. Set environment variable:
     export GOOGLE_CLOUD_BUCKET=your-bucket-name
  2. Grant service account Storage Object Admin role
  3. Verify ADC (Application Default Credentials):
     gcloud auth application-default login
  4. Test connectivity:
     gsutil ls gs://your-bucket-name
```

#### Issue 6: Authentication Failure in Production
```
Symptom:
  RuntimeError: Deployment produksi ditolak: kredensial minimum 12 karakter belum dikonfigurasi

Cause:
  - Password environment variables not set
  - Passwords too short (< 12 characters)

Solutions:
  1. Set all required passwords:
     export ASTINA_ADMIN_PASSWORD="SecurePassword123!"
     export ASTINA_AUDITOR_PASSWORD="SecurePassword123!"
     export ASTINA_ANALYST_PASSWORD="SecurePassword123!"
     export ASTINA_VIEWER_PASSWORD="SecurePassword123!"
  2. Ensure AUTH_ENABLED=true
  3. Restart application
```

### 11.2 Debug Mode

#### Enable Debug Logging
```bash
export ASTINA_LOG_LEVEL=DEBUG
export ASTINA_LOG_FORMAT=json
python run.py
```

#### Check System Resources
```python
# In Python console
import psutil
print(f"CPU: {psutil.cpu_count()} cores")
print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
print(f"Available: {psutil.virtual_memory().available / 1024**3:.2f} GB")
print(f"Disk: {psutil.disk_usage('/').free / 1024**3:.2f} GB")
```

#### Verify Dependencies
```bash
# Check all required packages
python -c "
import streamlit, torch, pandas, numpy, sklearn
import plotly, networkx, matplotlib, seaborn
print('All dependencies OK')
"
```

### 11.3 Performance Debugging

#### Profile Memory Usage
```python
# Add to code for debugging
import psutil
import gc

def print_memory_usage():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    print(f"Memory usage: {mem_mb:.2f} MB")
    gc.collect()
    mem_mb_after = process.memory_info().rss / 1024**2
    print(f"After GC: {mem_mb_after:.2f} MB")
```

#### Profile Execution Time
```python
import time

def time_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper

@time_execution
def my_function():
    # Your code here
    pass
```

---

## 12. APPENDIX

### 12.1 File Structure

```
project-Graphnet/
├── run.py                          # Application bootstrap
├── main.py                         # Streamlit entry point
├── config.py                       # Centralized configuration
├── requirements.txt                # Python dependencies
├── ASTINA.md                       # This documentation
│
├── Core Modules/
│   ├── file_handler.py            # File upload & streaming
│   ├── preprocessing_optimized.py  # Data preprocessing
│   ├── schema_harmonizer.py       # Schema normalization
│   ├── fraud_risk_pipeline.py     # Fraud detection orchestration
│   ├── model.py                   # ML models & training
│   ├── state_manager.py           # Session state management
│   ├── auth_manager.py            # Authentication & RBAC
│   ├── audit_trail.py             # Audit logging
│   ├── cloud_storage.py           # Cloud storage integration
│   ├── error_handler.py           # Error handling
│   └── cache_manager.py           # Caching layer
│
├── Rule Engines/
│   ├── repeat_billing_detector.py
│   ├── phantom_service_rules.py
│   ├── provider_capacity_validator.py
│   ├── upcoding_unbundling_rules.py
│   ├── inflated_bill_cloning_rules.py
│   ├── length_of_stay_rules.py
│   ├── medication_device_fraud_rules.py
│   └── fuzzy_claim_matcher.py
│
├── Advanced Modules (Phase 2)/
│   ├── adaptive_threshold_engine.py
│   ├── context_aware_risk_scorer.py
│   ├── historical_baseline_system.py
│   └── whitelist_manager.py
│
├── AI & Explainability/
│   ├── agentic_copilot.py         # AI investigator assistant
│   ├── rag_engine.py              # RAG knowledge base
│   ├── pii_masker.py              # PII protection
│   ├── model_explainer.py         # SHAP/LIME explanations
│   └── model_interpreter.py       # Model interpretation
│
├── Utilities/
│   ├── logging_config.py          # Logging configuration
│   ├── large_file_processor.py    # Large file handling
│   ├── data_validator.py          # Data validation
│   ├── metrics.py                 # Evaluation metrics
│   ├── enhanced_metrics.py        # Advanced metrics
│   ├── pdf_generator.py           # PDF report generation
│   └── system_status.py           # System monitoring
│
├── UI Components/
│   ├── ui_components.py           # Custom UI components
│   ├── ui/
│   │   ├── sidebar.py             # Sidebar navigation
│   │   ├── utils.py               # UI utilities
│   │   └── pages/
│   │       ├── home.py            # Home page
│   │       ├── data_collection.py # Data upload page
│   │       ├── training.py        # Model training page
│   │       ├── evaluation.py      # Model evaluation page
│   │       ├── detection.py       # Anomaly detection page
│   │       ├── status.py          # System status page
│   │       └── settings.py       # Settings page
│
├── Scripts/
│   ├── backup_scheduler.py        # Automated backups
│   ├── cache_cleanup_scheduler.py # Cache cleanup
│   ├── monitoring_setup.py       # Monitoring configuration
│   ├── security_validator.py      # Security validation
│   ├── setup_production_env.py   # Production setup
│   └── setup_secrets_manager.py   # Secrets management
│
├── Tests/
│   ├── conftest.py                # Pytest configuration
│   ├── test_*.py                  # Unit & integration tests
│
├── Directories/
│   ├── models/                    # Saved models
│   ├── plots/                     # Generated plots
│   ├── studies/                   # Optuna studies
│   ├── logs/                      # Application logs
│   ├── cache/                     # Cached data
│   └── logs/audit/               # Audit logs
│
└── Configuration/
    ├── .devin/                    # Devin configuration
    └── .venv/                     # Virtual environment
```

### 12.2 Glossary

| Term | Definition |
|------|------------|
| **ASTINA** | Analisis Sistem Transaksi Identifikasi Nilai Anomali |
| **GNN** | Graph Neural Network - Deep learning on graph structures |
| **GAT** | Graph Attention Network - Attention mechanism for graphs |
| **SHAP** | SHapley Additive exPlanations - Model explainability method |
| **LIME** | Local Interpretable Model-agnostic Explanations |
| **RAG** | Retrieval-Augmented Generation - LLM with knowledge base |
| **PII** | Personally Identifiable Information - Sensitive personal data |
| **RBAC** | Role-Based Access Control - Permission management |
| **WORM** | Write Once, Read Many - Immutable storage pattern |
| **OOB** | Out-of-Bag - Validation method for ensemble models |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **BAP** | Berita Acara Pemeriksaan - Investigation report (Indonesian) |
| **UU PDP** | Undang-Undang Perlindungan Data Pribadi - Indonesian privacy law |

### 12.3 References

#### Regulations
- [UU No. 27 Tahun 2022 tentang Perlindungan Data Pribadi](https://www.dpr.go.id/)
- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html)
- [GDPR](https://gdpr.eu/)

#### Technologies
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Polars](https://pola.rs/)
- [Optuna](https://optuna.org/)

#### ML Concepts
- [Graph Neural Networks](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- [SHAP Values](https://github.com/slundberg/shap)
- [Fraud Detection in Healthcare](https://www.ncbi.nlm.nih.gov/)

---

## 📝 CHANGELOG

### Version 1.0 (2026-10-24)
- Initial documentation release
- Complete architecture overview
- Detailed workflow documentation
- Deployment guide
- Troubleshooting guide

---

**Document End**

*This documentation is maintained by the Senior QA Engineering Team. For questions or updates, please contact the development team.*
