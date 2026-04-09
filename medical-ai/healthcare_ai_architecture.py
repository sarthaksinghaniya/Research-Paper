"""
Healthcare AI System Architecture - Microservices ML Pipeline Visualization

Generates publication-ready architecture diagram with:
- Data ingestion (batch + real-time)
- Data lake and storage
- ETL pipeline
- Feature store
- ML training pipeline
- Model registry
- Real-time inference
- NLP for clinical text
- Monitoring & drift detection
- Feedback loop

Output: SVG and PNG diagrams
"""

import graphviz
import os

def create_healthcare_architecture():
    """Create comprehensive healthcare AI architecture diagram"""
    
    # Initialize directed graph with custom styling
    dot = graphviz.Digraph(
        name='Healthcare_AI_Architecture',
        comment='Microservices-based ML Pipeline for Healthcare',
        engine='dot',
        format='svg',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
            'nodesep': '0.8',
            'ranksep': '1.2',
            'fontname': 'Helvetica',
            'fontsize': '11',
            'bgcolor': 'white',
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fontname': 'Helvetica',
            'fontsize': '10',
            'margin': '0.3,0.2',
        },
        edge_attr={
            'fontname': 'Helvetica',
            'fontsize': '9',
            'penwidth': '1.5',
        }
    )

    # ============================================================================
    # LAYER 1: DATA INGESTION (Sources)
    # ============================================================================
    with dot.subgraph(name='cluster_ingestion') as sg:
        sg.attr(label='Data Ingestion Layer', style='dotted', color='lightgrey')
        sg.node('ehr', 'EHR Systems\n(HL7/FHIR)', fillcolor='#FFE6E6', shape='box')
        sg.node('iot', 'IoT Sensors\n(Wearables)', fillcolor='#FFE6E6', shape='box')
        sg.node('imaging', 'Imaging Systems\n(DICOM)', fillcolor='#FFE6E6', shape='box')
        sg.node('labs', 'Lab Results\n(LOINC)', fillcolor='#FFE6E6', shape='box')

    # ============================================================================
    # LAYER 2: DATA INGESTION SERVICES
    # ============================================================================
    with dot.subgraph(name='cluster_ingestion_svc') as sg:
        sg.attr(label='Ingestion Services', style='dotted', color='lightgrey')
        sg.node('batch_ingestion', 'Batch Ingestion\n(Apache Spark)', 
                fillcolor='#CCE5FF', shape='box')
        sg.node('stream_ingestion', 'Real-time Stream\n(Kafka/Kinesis)', 
                fillcolor='#CCE5FF', shape='box')

    # ============================================================================
    # LAYER 3: DATA LAKE & STORAGE
    # ============================================================================
    with dot.subgraph(name='cluster_storage') as sg:
        sg.attr(label='Data Lake & Storage Layer', style='dotted', color='lightgrey')
        sg.node('raw_data', 'RAW DATA\n(S3/Azure Data Lake)\nPatient Records, Sensors', 
                fillcolor='#E6F3FF', shape='box')
        sg.node('curated_data', 'CURATED DATA\n(Parquet/Delta)\nProcessed & Validated', 
                fillcolor='#E6F3FF', shape='box')

    # ============================================================================
    # LAYER 4: ETL PIPELINE
    # ============================================================================
    with dot.subgraph(name='cluster_etl') as sg:
        sg.attr(label='ETL Pipeline', style='dotted', color='lightgrey')
        sg.node('etl_clean', 'Data Cleaning\n(Great Expectations)', 
                fillcolor='#D4F1D4', shape='box')
        sg.node('etl_transform', 'Transformation\n(Pandas/PySpark)', 
                fillcolor='#D4F1D4', shape='box')
        sg.node('etl_validate', 'Validation & QA\n(dbt/Soda)', 
                fillcolor='#D4F1D4', shape='box')

    # ============================================================================
    # LAYER 5: FEATURE STORE
    # ============================================================================
    with dot.subgraph(name='cluster_feature') as sg:
        sg.attr(label='Feature Store', style='dotted', color='lightgrey')
        sg.node('feature_eng', 'Feature Engineering\n(Feast/Tecton)', 
                fillcolor='#FFF4D4', shape='box')
        sg.node('feature_store', 'Feature Repository\n(PostgreSQL/Redis)\nOnline & Offline', 
                fillcolor='#FFF4D4', shape='box')

    # ============================================================================
    # LAYER 6: PARALLEL PIPELINES - ML & NLP
    # ============================================================================
    
    # ML PIPELINE (LEFT)
    with dot.subgraph(name='cluster_ml') as sg:
        sg.attr(label='ML Model Pipeline', style='dotted', color='#E6D4F7')
        sg.node('ml_preprocess', 'Preprocessing\n(Scikit-learn)', 
                fillcolor='#F0E6FF', shape='box')
        sg.node('ml_train', 'Model Training\n(TensorFlow/PyTorch)\nRF, XGBoost, NN', 
                fillcolor='#F0E6FF', shape='box')
        sg.node('ml_hyperopt', 'Hyperparameter Tuning\n(Optuna/Ray Tune)', 
                fillcolor='#F0E6FF', shape='box')
        sg.node('ml_evaluate', 'Model Evaluation\n(Cross-validation)\nAUC, Precision, Recall', 
                fillcolor='#F0E6FF', shape='box')

    # NLP PIPELINE (RIGHT)
    with dot.subgraph(name='cluster_nlp') as sg:
        sg.attr(label='NLP Pipeline (Clinical Text)', style='dotted', color='#F7E6D4')
        sg.node('nlp_preprocess', 'NLP Preprocessing\n(NLTK/spaCy)', 
                fillcolor='#FFE6CC', shape='box')
        sg.node('nlp_ner', 'Named Entity Recognition\n(BioBERT)\nDiseases, Medications', 
                fillcolor='#FFE6CC', shape='box')
        sg.node('nlp_extract', 'Information Extraction\n(Transformer Models)', 
                fillcolor='#FFE6CC', shape='box')
        sg.node('nlp_features', 'Extract NLP Features\n(Embeddings)', 
                fillcolor='#FFE6CC', shape='box')

    # ============================================================================
    # LAYER 7: MODEL REGISTRY & VERSIONING
    # ============================================================================
    with dot.subgraph(name='cluster_registry') as sg:
        sg.attr(label='Model Management', style='dotted', color='lightgrey')
        sg.node('model_registry', 'Model Registry\n(MLflow/Weights & Biases)\nVersion Control, Metadata', 
                fillcolor='#E0E0E0', shape='box')

    # ============================================================================
    # LAYER 8: REAL-TIME INFERENCE SERVICE
    # ============================================================================
    with dot.subgraph(name='cluster_inference') as sg:
        sg.attr(label='Real-time Inference', style='dotted', color='lightgrey')
        sg.node('inference_api', 'Inference API\n(FastAPI/Flask)\nMicroservice', 
                fillcolor='#D4FFE6', shape='box')
        sg.node('model_serving', 'Model Serving\n(Seldon/Triton)\nGPU-optimized', 
                fillcolor='#D4FFE6', shape='box')
        sg.node('cache', 'Cache Layer\n(Redis)\nLow-latency Predictions', 
                fillcolor='#D4FFE6', shape='box')

    # ============================================================================
    # LAYER 9: MONITORING & DRIFT DETECTION
    # ============================================================================
    with dot.subgraph(name='cluster_monitoring') as sg:
        sg.attr(label='Monitoring & Observability', style='dotted', color='lightgrey')
        sg.node('drift_detect', 'Drift Detection\n(Evidently/WhyLabs)\nData & Model Drift', 
                fillcolor='#FFD4D4', shape='box')
        sg.node('metrics_log', 'Metrics & Logging\n(Prometheus/ELK Stack)', 
                fillcolor='#FFD4D4', shape='box')
        sg.node('alerts', 'Alerting System\n(PagerDuty/Slack)', 
                fillcolor='#FFD4D4', shape='box')

    # ============================================================================
    # LAYER 10: APPLICATIONS & OUTPUT
    # ============================================================================
    with dot.subgraph(name='cluster_output') as sg:
        sg.attr(label='Applications & Output', style='dotted', color='lightgrey')
        sg.node('clinical_app', 'Clinical Decision\nSupport System', 
                fillcolor='#E6FFE6', shape='box')
        sg.node('mobile_app', 'Patient Mobile App\n(React Native)', 
                fillcolor='#E6FFE6', shape='box')
        sg.node('dashboard', 'Analytics Dashboard\n(Grafana/Tableau)', 
                fillcolor='#E6FFE6', shape='box')

    # ============================================================================
    # FEEDBACK LOOP
    # ============================================================================
    dot.node('predictions', 'Predictions & Decisions\n(CSV/JSON)', 
             fillcolor='#FFFFCC', shape='box')

    # ============================================================================
    # EDGES - DATA FLOW
    # ============================================================================
    
    # Data Ingestion Flow
    dot.edge('ehr', 'batch_ingestion', label='Batch (Daily)', style='solid')
    dot.edge('iot', 'stream_ingestion', label='Real-time (Streaming)', style='solid')
    dot.edge('imaging', 'batch_ingestion', label='Batch', style='solid')
    dot.edge('labs', 'batch_ingestion', label='Batch', style='solid')
    
    dot.edge('batch_ingestion', 'raw_data', style='solid')
    dot.edge('stream_ingestion', 'raw_data', style='solid', constraint='false')
    
    # ETL Pipeline Flow
    dot.edge('raw_data', 'etl_clean', style='solid')
    dot.edge('etl_clean', 'etl_transform', style='solid')
    dot.edge('etl_transform', 'etl_validate', style='solid')
    dot.edge('etl_validate', 'curated_data', style='solid')
    
    # Feature Store Flow
    dot.edge('curated_data', 'feature_eng', style='solid', label='Structured Data')
    dot.edge('feature_eng', 'feature_store', style='solid')
    
    # NLP Pipeline Flow
    dot.edge('curated_data', 'nlp_preprocess', style='solid', label='Clinical Notes', color='orange')
    dot.edge('nlp_preprocess', 'nlp_ner', style='solid', color='orange')
    dot.edge('nlp_ner', 'nlp_extract', style='solid', color='orange')
    dot.edge('nlp_extract', 'nlp_features', style='solid', color='orange')
    dot.edge('nlp_features', 'feature_store', style='solid', color='orange', label='NLP Features')
    
    # ML Pipeline Flow
    dot.edge('feature_store', 'ml_preprocess', style='solid', label='Training Data')
    dot.edge('ml_preprocess', 'ml_train', style='solid')
    dot.edge('ml_train', 'ml_hyperopt', style='solid')
    dot.edge('ml_hyperopt', 'ml_evaluate', style='solid')
    dot.edge('ml_evaluate', 'model_registry', style='solid', label='✓ Approved')
    
    # Inference Flow
    dot.edge('model_registry', 'model_serving', style='solid', label='Load Model')
    dot.edge('feature_store', 'inference_api', style='dashed', label='Features (Online)')
    dot.edge('inference_api', 'model_serving', style='solid')
    dot.edge('model_serving', 'cache', style='solid')
    dot.edge('cache', 'predictions', style='solid')
    
    # Output Applications
    dot.edge('predictions', 'clinical_app', style='solid')
    dot.edge('predictions', 'mobile_app', style='solid')
    dot.edge('predictions', 'dashboard', style='solid')
    
    # Monitoring Loop
    dot.edge('predictions', 'drift_detect', style='bold', color='red', label='Monitor Predictions')
    dot.edge('drift_detect', 'metrics_log', style='solid', color='red')
    dot.edge('metrics_log', 'alerts', style='solid', color='red')
    
    # FEEDBACK LOOPS (Bidirectional)
    dot.edge('drift_detect', 'ml_train', style='bold', color='red', constraint='false',
             label='Retrain if Drift', dir='forward')
    dot.edge('alerts', 'etl_clean', style='bold', color='red', constraint='false',
             label='Data Quality Issues')
    dot.edge('clinical_app', 'raw_data', style='dotted', color='blue', constraint='false',
             label='Ground Truth Feedback', dir='back')

    return dot


def create_architecture_documentation():
    """Create comprehensive architecture documentation"""
    
    doc = """# Healthcare AI System Architecture
## Microservices-based ML Pipeline for Clinical Decision Support

**Generated**: April 9, 2026  
**Version**: 1.0  
**Classification**: Educational/Technical Documentation

---

## 1. System Overview

This architecture implements an enterprise-grade machine learning platform for healthcare with:
- Batch and real-time data ingestion
- Centralized feature store
- Parallel ML and NLP processing
- Real-time inference with sub-100ms latency
- Continuous monitoring and drift detection
- Automated retraining loops

---

## 2. Layer-by-Layer Architecture

### Layer 1: Data Sources & Ingestion
```
┌─────────────────────────────────────────────────────────────┐
│ External Data Sources                                       │
├─────────────────────────────────────────────────────────────┤
│ • EHR Systems (HL7/FHIR)           → Batch Ingestion       │
│ • IoT Sensors (Wearables)          → Real-time Stream      │
│ • Imaging (DICOM)                  → Batch Ingestion       │
│ • Lab Results (LOINC)              → Batch Ingestion       │
└─────────────────────────────────────────────────────────────┘
         ↓ (Kafka/Kinesis for real-time, Spark for batch)
      RAW DATA
```

**Technologies**:
- Apache Kafka/AWS Kinesis (streaming)
- Apache Spark (batch processing)
- Data Transfer Services

**SLA**: < 5 minute latency for batch, real-time for stream

---

### Layer 2: Data Lake & Storage
```
┌──────────────────────────────────┐
│ Raw Data Layer                   │
│ (S3/Azure Data Lake)             │
│ - Original, immutable data       │
│ - Compression: Gzip/Snappy       │
│ - Format: Parquet/ORC            │
├──────────────────────────────────┤
│ Curated Data Layer               │
│ (Delta/Iceberg format)           │
│ - Cleaned, validated             │
│ - Schema versioning              │
│ - ACID transactions              │
└──────────────────────────────────┘
```

**Storage Specifications**:
- **Capacity**: 100+ TB
- **Retention**: 7 years (compliance)
- **Replication**: 3x for HA
- **Encryption**: AES-256 at rest, TLS in transit

---

### Layer 3: ETL Pipeline
```
Data Cleaning (Great Expectations)
    ↓
Data Transformation (Pandas/PySpark)
    ├─ Standardization (units, formats)
    ├─ Deidentification (PII removal)
    ├─ Outlier detection
    └─ Aggregation
    ↓
Validation & QA (dbt/Soda)
    ├─ Schema validation
    ├─ Integrity checks
    └─ Quality metrics
```

**Key Components**:
- Data Quality Framework: Great Expectations
- Transform Engine: Spark 3.0+
- Data Testing: dbt + Soda
- Orchestration: Airflow/Databricks Workflows

**Performance**: 
- Processes 1M+ records/minute
- Daily full refresh: ~2 hours

---

### Layer 4: Feature Store
```
┌────────────────────────────────────────────┐
│ Feature Store (Feast/Tecton)               │
├────────────────────────────────────────────┤
│ Online Store (Redis/DynamoDB)              │
│ - Low latency (< 50ms)                     │
│ - Used for real-time inference             │
│                                            │
│ Offline Store (S3/Parquet)                 │
│ - Complete historical data                 │
│ - Used for training                        │
│                                            │
│ Feature Registry (Metadata)                │
│ - Feature definitions                      │
│ - Lineage tracking                         │
│ - Versioning                               │
└────────────────────────────────────────────┘
```

**Feature Categories**:
1. **Patient Demographics**: Age, gender, ethnicity
2. **Clinical History**: Past diagnoses, procedures
3. **Vital Signs**: BP, HR, temperature (time-series)
4. **Lab Values**: Cholesterol, glucose aggregates
5. **NLP Features**: Symptom embeddings from clinical notes
6. **Temporal Features**: Day of week, seasonality

**Total Features**: 500+ (dynamic computation)

---

### Layer 5: Parallel ML & NLP Pipelines

#### ML Pipeline (Structured Data)
```
Feature Store → Preprocessing (Scikit-learn)
    ↓
    Model Training (TensorFlow/PyTorch)
    ├─ Random Forest (baseline)
    ├─ XGBoost (production)
    ├─ Neural Networks (exploration)
    └─ Ensemble (final model)
    ↓
    Hyperparameter Tuning (Optuna/Ray Tune)
    ├─ Bayesian optimization
    ├─ Grid/Random search
    └─ Distributed trials (100+ combinations)
    ↓
    Model Evaluation
    ├─ Cross-validation (5-fold)
    ├─ Stratified splitting
    ├─ Fairness metrics (demographic parity)
    └─ Business metrics (sensitivity/specificity)
```

**Training Infrastructure**:
- Compute: GPU clusters (NVIDIA A100)
- Framework: TensorFlow 2.10+ / PyTorch 2.0+
- Training Time: 2-4 hours per model
- Hyperparameter combinations: 100-500 trials

#### NLP Pipeline (Clinical Text)
```
Clinical Notes (ETL) → NLP Preprocessing (spaCy/NLTK)
    ├─ Tokenization
    ├─ POS tagging
    └─ Sentence segmentation
    ↓
    Named Entity Recognition (BioBERT)
    ├─ Diseases (ICD-10 codes)
    ├─ Medications (RxNorm codes)
    ├─ Procedures (CPT codes)
    └─ Symptoms
    ↓
    Information Extraction (Transformer Models)
    ├─ Negation detection
    ├─ Temporality (acute/chronic)
    └─ Severity levels
    ↓
    Feature Extraction (Embeddings)
    ├─ BioELMo embeddings (1024-dim)
    ├─ Aggregation (mean/max pooling)
    └─ Store in Feature Store
```

**NLP Technologies**:
- Preprocessing: spaCy/NLTK
- NER Models: BioBERT, SciBERT
- Text Embedding: BioELMo, BERT
- Framework: Hugging Face Transformers

**Processing Speed**: 1000 notes/minute on GPU

---

### Layer 6: Model Registry & Versioning
```
┌─────────────────────────────────────┐
│ Model Registry (MLflow/W&B)         │
├─────────────────────────────────────┤
│ Model Metadata:                     │
│ • Version ID (v1.2.3)               │
│ • Training date & author            │
│ • Hyperparameters                   │
│ • Training metrics (AUC, F1, etc.)  │
│ • Dependencies (libraries/data)     │
│ • Validation status                 │
│ • Deployment stage:                 │
│   - Staging                         │
│   - Production                      │
│   - Archived                        │
└─────────────────────────────────────┘
```

**Key Functions**:
- Version control for models
- Model artifacts storage
- Metadata tracking
- Deployment annotations
- A/B testing support

---

### Layer 7: Real-time Inference Service
```
Request (Patient Data)
    ↓
Inference API (FastAPI/Flask)
    ├─ Input validation
    ├─ Authentication/authorization
    └─ Request routing
    ↓
Feature Retrieval (Feature Store - Online)
    ├─ Real-time feature lookup (< 50ms)
    └─ Cache hit optimization
    ↓
Model Serving (Seldon/Triton)
    ├─ Model loading
    ├─ Batch prediction
    └─ GPU optimization
    ↓
Post-processing & Explanation
    ├─ Calibration
    ├─ Confidence intervals
    └─ SHAP explanations
    ↓
Response (Risk Score + Explanation)
```

**Performance Targets**:
- **Latency**: p99 < 100ms
- **Throughput**: 10,000 req/sec
- **Availability**: 99.99% SLA
- **Hardware**: GPU-enabled (NVIDIA A100)

**API Specifications**:
```json
POST /api/v1/predict
{
  "patient_id": "P12345",
  "timestamp": "2026-04-09T10:30:00Z",
  "features": {...}
}

Response:
{
  "risk_score": 0.72,
  "risk_class": "HIGH",
  "confidence": 0.87,
  "top_factors": [
    {"feature": "age", "contribution": 0.23},
    {"feature": "systolic_bp", "contribution": 0.18}
  ],
  "explanation": "..."
}
```

---

### Layer 8: Monitoring & Drift Detection
```
┌───────────────────────────────────────────────────┐
│ Data Drift Monitoring (Evidently/WhyLabs)        │
├───────────────────────────────────────────────────┤
│ • Feature distributions (KL divergence)          │
│ • Statistical tests (Kolmogorov-Smirnov)         │
│ • Data quality metrics (NULL%, outliers%)        │
│ • Alert: Drift detected → Trigger retraining    │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│ Model Drift Monitoring                            │
├───────────────────────────────────────────────────┤
│ • Prediction distribution changes                │
│ • Business metric degradation (AUC < threshold) │
│ • Demographic fairness (group-level metrics)    │
│ • Alert: Model drift → Review/retrain           │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│ System Health & Logging (Prometheus/ELK)         │
├───────────────────────────────────────────────────┤
│ • API latency & error rates                      │
│ • Model serving performance                      │
│ • Feature store hit rates                        │
│ • System resource utilization                    │
└───────────────────────────────────────────────────┘
```

**Alerting Rules**:
- Data drift detected → Slack notification
- Model performance drop > 5% → PagerDuty alert
- Inference latency > 200ms p99 → Dashboard warning
- Feature store unavailable → Critical alert

---

### Layer 9: Applications & Output
```
1. Clinical Decision Support System
   ├─ Web interface for clinicians
   ├─ Real-time risk alerts
   └─ Treatment recommendations

2. Patient Mobile App (React Native)
   ├─ Risk tracking dashboard
   ├─ Personalized recommendations
   └─ Appointment scheduling

3. Analytics Dashboard (Grafana/Tableau)
   ├─ Model performance metrics
   ├─ Population health insights
   └─ Administrative reports
```

---

## 3. Data Flow Overview

### Batch Flow (Daily)
```
EHR/Labs/Imaging → Kafka → Spark Cluster → ETL Jobs → Data Lake → Feature Engineering → Feature Store
                                                                           ↓
                                                                    Store (Offline + Online)
```

**Frequency**: Once daily (2 AM UTC)  
**Duration**: 2-4 hours  
**Trigger**: Automated Airflow DAG

### Real-time Flow (Streaming)
```
IoT Sensors → Kafka Streams → Windowed Aggregation → Feature Store (Online) → Inference API
```

**Latency**: 10-30 seconds  
**Volume**: 1 million events/day

### Training Flow
```
Feature Store (Offline) → Preprocessing → Model Training (100+ trials) → Evaluation → Model Registry
                                                                              ↓
                                                                        Registry update (if approved)
```

**Frequency**: Triggered by data drift or weekly schedule

### Inference Flow
```
Patient Data → Inference API → Feature Store (Online) → Model Serving → Prediction → Application
                                                              ↓
                                                           Cache (Redis)
```

**Latency Target**: < 100ms

---

## 4. Feedback Loops

### Loop 1: Automated Retraining
```
Production Predictions → Drift Detection ──→ Alert
                              ↓
                        Compare to baseline
                             ↓
                        Drift > threshold?
                          YES ↓
                        Retrain Models ──→ Update Registry
                             ↓
                        Validation passed?
                          YES ↓
                        Production Deployment
```

**Trigger Thresholds**:
- Data drift: KL divergence > 0.1
- Model drift: AUC drop > 5%
- Fairness drift: Demographic group performance gap > 3%
- Recency: Weekly automatic retraining

### Loop 2: Continuous Improvement
```
Production Predictions → Ground Truth (6 months later) → Performance Analysis
                              ↓
                        Accuracy < target?
                          YES ↓
                        Investigate Root Causes
                             ↓
                        • Feature engineering
                        • Model selection
                        • Data collection
                             ↓
                        Update Pipeline → Retrain
```

### Loop 3: Clinical Feedback
```
Predictions + Explanations → Clinician Review → Ground Truth Labeling
                                    ↓
                            Feedback captured?
                              YES ↓
                            Store in Data Lake ──→ Improves future models
                                    ↓
                            Model retraining triggered
```

---

## 5. Technology Stack

### Data Processing
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Streaming | Apache Kafka / AWS Kinesis | Distributed, scalable |
| Batch Processing | Apache Spark 3.0+ | Distributed SQL, ML libs |
| Orchestration | Apache Airflow / Databricks | Workflow management |
| Data Quality | Great Expectations / Soda | Data validation |
| Data Transformation | Spark SQL / Pandas / dbt | SQL-based transforms |

### Feature Engineering
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Feature Store | Feast / Tecton | Online + Offline serving |
| Online Cache | Redis / DynamoDB | Low-latency access |
| Offline Storage | S3 / Delta Lake | Historical data |

### ML & NLP
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| ML Framework | TensorFlow 2.10+ / PyTorch | Production-ready |
| NLP | Hugging Face Transformers | SOTA pre-trained models |
| Hyperparameter Optimization | Optuna / Ray Tune | Distributed trials |
| Model Registry | MLflow / Weights & Biases | Version control |

### Serving & APIs
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Web Framework | FastAPI / Flask | Fast, async support |
| Model Serving | Seldon Core / NVIDIA Triton | Production inference |
| Containerization | Docker / Kubernetes | Microservices |
| API Gateway | Kong / Nginx | Request routing |

### Monitoring
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Drift Detection | Evidently / WhyLabs | Automated monitoring |
| Metrics & Logging | Prometheus / ELK Stack | Observability |
| Visualization | Grafana / Kibana | Dashboards |
| Alerting | PagerDuty / Slack | Incident response |

### Infrastructure
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Compute | Kubernetes (EKS/AKS) / Databricks | Scalable |
| GPU | NVIDIA A100 | Training acceleration |
| Storage | S3 / Azure Data Lake | Scalable object storage |
| Networking | VPC / Private Link | Secure connectivity |
| Secrets | AWS Secrets Manager / HashiCorp Vault | Credential management |

---

## 6. Security & Compliance

### Data Security
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Access Control**: Role-based (RBAC)
- **Audit Logging**: All access logged and monitored
- **PII Handling**: Automatic deidentification

### Compliance
- **HIPAA**: Covered entity compliance
- **GDPR**: Right to explanation, data portability
- **FDA**: 21 CFR Part 11 for medical devices
- **Model Validation**: Clinical validation on diverse populations

### Governance
- **Model Approval**: Clinician review + regulatory sign-off
- **Change Management**: Version control for all models
- **Explainability**: SHAP values for each prediction
- **Regular Audits**: Quarterly security assessments

---

## 7. Scalability & Performance

### Throughput Targets
| Component | Target | Capacity |
|-----------|--------|----------|
| Data Ingestion | 1M+ events/min | Kafka clustering |
| Feature Store Queries | 10K queries/sec | Redis clustering |
| Inference Requests | 10K req/sec | Kubernetes auto-scaling |
| Model Training | 100 parallel trials | GPU cluster |

### Latency Targets
| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Feature Store Lookup | 10ms | 30ms | 50ms |
| Inference API | 30ms | 50ms | 100ms |
| End-to-end Prediction | 50ms | 80ms | 150ms |

### Cost Optimization
- Spot instances for training (30% cost savings)
- Auto-scaling based on demand
- Data compression (Parquet reduces 80% storage)
- Batch inference for offline predictions

---

## 8. Deployment Pipeline

```
Local Development → CI/CD → Staging → Canary → Full Production → Monitor
     (Git)          (Jenkins)  (2%)    (5%)    (100%)         (Metrics)
```

**Key Steps**:
1. Code review on GitHub
2. Unit + Integration tests
3. Model performance validation
4. Canary deployment (2% traffic)
5. Monitor for 24 hours
6. Full rollout if healthy
7. Automated rollback on alert

---

## 9. Use Cases

### Primary Use Cases
1. **Cardiovascular Risk Prediction**
   - Input: Patient vitals, history, labs
   - Output: 5-year CVD risk probability
   - SLA: < 100ms latency

2. **Sepsis Early Detection**
   - Input: Real-time vital signs
   - Output: Sepsis probability + alert
   - SLA: < 5 minute detection window

3. **Medication Recommendation**
   - Input: Diagnosis, patient profile
   - Output: Top-3 medication recommendations
   - SLA: < 200ms latency

4. **Clinical Note Analysis**
   - Input: Discharge summary
   - Output: Extracted entities + insights
   - SLA: < 1 second for 5000 character note

---

## 10. Roadmap & Future Enhancements

**Q2 2026**:
- Multi-model ensembling
- Federated learning for privacy-preserving training

**Q3 2026**:
- Reinforcement learning for treatment optimization
- Graph neural networks for patient similarity

**Q4 2026**:
- Causal inference for interventions
- Foundation models for generalist healthcare AI

---

## References

- MLOps Best Practices: https://mlops.community
- Healthcare ML Standards: FDA Software as Medical Device
- Privacy-Preserving ML: Federated Learning surveys
- Model Monitoring: WhyLabs, Evidently documentation

---

**⚠️ DISCLAIMER**: Architecture is for educational/research purposes.  
Implementation requires regulatory approval and clinical validation.

**Document Version**: 1.0 | **Last Updated**: April 9, 2026
"""
    
    return doc


def main():
    """Generate architecture diagrams and documentation"""
    
    print("=" * 80)
    print("HEALTHCARE AI SYSTEM ARCHITECTURE GENERATOR")
    print("=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs('architecture_output', exist_ok=True)
    
    # Generate architecture diagram
    print("\n[1] Generating architecture diagram...")
    dot = create_healthcare_architecture()
    
    # Save as SVG (high-quality)
    svg_path = 'architecture_output/healthcare_ai_architecture'
    dot.render(svg_path, format='svg', cleanup=True)
    print(f"    ✓ Saved: {svg_path}.svg")
    
    # Save as PNG (alternative format)
    dot.render(svg_path, format='png', cleanup=True)
    print(f"    ✓ Saved: {svg_path}.png")
    
    # Generate documentation
    print("\n[2] Generating technical documentation...")
    doc = create_architecture_documentation()
    
    doc_path = 'architecture_output/ARCHITECTURE_DOCUMENTATION.md'
    with open(doc_path, 'w') as f:
        f.write(doc)
    print(f"    ✓ Saved: {doc_path}")
    
    # Create a summary file
    summary = """# Healthcare AI Architecture Summary

## Quick Reference

This folder contains the complete healthcare AI system architecture:

### Files
1. **healthcare_ai_architecture.svg** - High-quality vector diagram (editable)
2. **healthcare_ai_architecture.png** - PNG diagram for presentations
3. **ARCHITECTURE_DOCUMENTATION.md** - Complete technical documentation

### Key Statistics
- **Components**: 25+ microservices
- **Data Ingestion**: Batch + Real-time streams
- **ML Models**: Parallel training pipeline
- **NLP**: Clinical text processing
- **Inference**: < 100ms latency target
- **Monitoring**: Automated drift detection
- **Feedback Loops**: 3 retraining mechanisms

### Quick Start for Reading
1. Open SVG/PNG for visual overview
2. Read ARCHITECTURE_DOCUMENTATION.md for details
3. Reference Technology Stack (Section 5)
4. Review Use Cases (Section 9)

### Main Components
- Data Ingestion Layer (batch + streaming)
- Data Lake & ETL Pipeline
- Feature Store (online + offline)
- ML & NLP Parallel Pipelines
- Real-time Inference Service
- Monitoring & Drift Detection
- Feedback Loops for Continuous Improvement

### Technology Highlights
- Apache Kafka / Spark for data processing
- TensorFlow / PyTorch for ML
- Hugging Face for NLP
- FastAPI for inference
- Kubernetes for orchestration
- Prometheus for monitoring

### Deployment
Publication-ready diagrams suitable for:
- Conference presentations
- Academic papers
- Technical documentation
- System design reviews
- RFP responses

Generated: April 9, 2026
Educational Purpose - Clinical validation required for production use
"""
    
    summary_path = 'architecture_output/README.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"    ✓ Saved: {summary_path}")
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print("\nOutput files:")
    print("  1. healthcare_ai_architecture.svg (High-quality, editable)")
    print("  2. healthcare_ai_architecture.png (Presentation-ready)")
    print("  3. ARCHITECTURE_DOCUMENTATION.md (10-section technical guide)")
    print("  4. README.md (Quick reference)")
    print("\n✓ All files saved to: architecture_output/")


if __name__ == '__main__':
    main()
