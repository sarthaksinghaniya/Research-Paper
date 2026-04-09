"""
Healthcare AI System Architecture - Matplotlib-based Visualization

Generates publication-ready architecture diagrams with:
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

Output: High-resolution PNG diagrams
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from matplotlib.patches import Rectangle
import os

def create_architecture_diagram():
    """Create comprehensive healthcare AI architecture visualization"""
    
    fig = plt.figure(figsize=(20, 24))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # Define colors for different layer types
    colors = {
        'source': '#FFE6E6',
        'ingestion': '#CCE5FF', 
        'storage': '#E6F3FF',
        'etl': '#D4F1D4',
        'feature': '#FFF4D4',
        'ml': '#F0E6FF',
        'nlp': '#FFE6CC',
        'registry': '#E0E0E0',
        'inference': '#D4FFE6',
        'monitoring': '#FFD4D4',
        'output': '#E6FFE6'
    }
    
    # Helper function to create boxes
    def create_box(ax, x, y, width, height, text, color, fontsize=9, weight='normal'):
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color,
                            linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
               weight=weight, wrap=True)
    
    # Helper function for arrows
    def create_arrow(ax, x1, y1, x2, y2, label='', style='->', color='black', lw=1.5, linestyle='solid'):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, color=color, 
                              linewidth=lw, mutation_scale=20,
                              linestyle=linestyle,
                              zorder=1)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # ============================================================================
    # LAYER 1: DATA SOURCES (Y=23)
    # ============================================================================
    ax.text(0.5, 23.5, '1. DATA SOURCES & INGESTION', fontsize=13, weight='bold')
    
    create_box(ax, 2, 22.5, 2.5, 0.8, 'EHR Systems\n(HL7/FHIR)', colors['source'])
    create_box(ax, 5, 22.5, 2.5, 0.8, 'IoT Sensors\n(Wearables)', colors['source'])
    create_box(ax, 8, 22.5, 2.5, 0.8, 'Imaging\n(DICOM)', colors['source'])
    create_box(ax, 11, 22.5, 2.5, 0.8, 'Lab Results\n(LOINC)', colors['source'])
    
    # ============================================================================
    # LAYER 2: INGESTION SERVICES (Y=21)
    # ============================================================================
    ax.text(0.5, 21.3, '2. INGESTION SERVICES', fontsize=12, weight='bold')
    
    create_box(ax, 3.5, 20.5, 3, 0.8, 'Batch Ingestion\n(Apache Spark)', colors['ingestion'])
    create_box(ax, 9.5, 20.5, 3, 0.8, 'Real-time Stream\n(Kafka/Kinesis)', colors['ingestion'])
    
    # Arrows from sources to ingestion
    create_arrow(ax, 2, 22.1, 3.5, 20.9, "Batch", lw=1.2)
    create_arrow(ax, 5, 22.1, 3.5, 20.9, "Batch", lw=1.2)
    create_arrow(ax, 8, 22.1, 3.5, 20.9, "Batch", lw=1.2)
    create_arrow(ax, 11, 22.1, 9.5, 20.9, "Real-time", color='orange', lw=1.2)
    
    # ============================================================================
    # LAYER 3: DATA LAKE & STORAGE (Y=19)
    # ============================================================================
    ax.text(0.5, 19.3, '3. DATA LAKE & STORAGE', fontsize=12, weight='bold')
    
    create_box(ax, 3, 18.5, 4, 0.9, 'RAW DATA LAYER\n(S3/Data Lake)\nImmutable, Original Records',
              colors['storage'], fontsize=9)
    create_box(ax, 10, 18.5, 4, 0.9, 'CURATED DATA LAYER\n(Parquet/Delta)\nCleaned, Validated',
              colors['storage'], fontsize=9)
    
    # Arrows from ingestion to data lake
    create_arrow(ax, 3.5, 20.1, 3, 18.95, lw=1.2)
    create_arrow(ax, 9.5, 20.1, 10, 18.95, lw=1.2)
    
    # ============================================================================
    # LAYER 4: ETL PIPELINE (Y=17)
    # ============================================================================
    ax.text(0.5, 17.3, '4. ETL PIPELINE', fontsize=12, weight='bold')
    
    create_box(ax, 2, 16.5, 2.2, 0.75, 'Data Cleaning\n(Great Exp.)', colors['etl'], fontsize=8)
    create_box(ax, 5, 16.5, 2.2, 0.75, 'Transformation\n(Spark/Pandas)', colors['etl'], fontsize=8)
    create_box(ax, 8, 16.5, 2.2, 0.75, 'Validation\n(dbt/Soda)', colors['etl'], fontsize=8)
    create_box(ax, 11, 16.5, 2.2, 0.75, 'Schema Check\n(Data Quality)', colors['etl'], fontsize=8)
    
    # Arrows in ETL pipeline
    create_arrow(ax, 3, 18.05, 2, 16.88, lw=1.2)
    create_arrow(ax, 3.1, 16.5, 3.9, 16.5, lw=1.2)
    create_arrow(ax, 6.1, 16.5, 6.9, 16.5, lw=1.2)
    create_arrow(ax, 9.1, 16.5, 9.9, 16.5, lw=1.2)
    
    # ============================================================================
    # LAYER 5: FEATURE STORE (Y=15)
    # ============================================================================
    ax.text(0.5, 15.3, '5. FEATURE STORE', fontsize=12, weight='bold')
    
    create_box(ax, 4, 14.5, 3, 0.8, 'Feature Engineering\n(Feast/Tecton)', colors['feature'])
    create_box(ax, 10, 14.5, 3.5, 0.9, 'Feature Repository\n(Redis + S3)\nOnline & Offline Store',
              colors['feature'], fontsize=9)
    
    create_arrow(ax, 11, 16.1, 4, 14.95, lw=1.2)
    create_arrow(ax, 5.5, 14.1, 8.75, 14.1, lw=1.2)
    
    # ============================================================================
    # LAYER 6: PARALLEL ML & NLP PIPELINES (Y=12.5-13)
    # ============================================================================
    ax.text(0.5, 13.8, '6. PARALLEL PROCESSING PIPELINES', fontsize=12, weight='bold')
    
    # ML Pipeline (Left side)
    ax.text(1.5, 13.3, 'ML PIPELINE', fontsize=10, weight='bold', color='purple')
    create_box(ax, 1.5, 12.5, 2, 0.7, 'Preprocessing\n(Scikit-learn)', colors['ml'], fontsize=8)
    create_box(ax, 1.5, 11.5, 2, 0.7, 'Model Training\n(TF/PyTorch)', colors['ml'], fontsize=8)
    create_box(ax, 1.5, 10.5, 2, 0.7, 'Hyperparameter\nTuning', colors['ml'], fontsize=8)
    create_box(ax, 1.5, 9.5, 2, 0.7, 'Evaluation\n(Cross-val)', colors['ml'], fontsize=8)
    
    # ML arrows
    create_arrow(ax, 1.5, 12.15, 1.5, 11.85, lw=1.2)
    create_arrow(ax, 1.5, 11.15, 1.5, 10.85, lw=1.2)
    create_arrow(ax, 1.5, 10.15, 1.5, 9.85, lw=1.2)
    
    # NLP Pipeline (Right side)
    ax.text(12.5, 13.3, 'NLP PIPELINE', fontsize=10, weight='bold', color='darkorange')
    create_box(ax, 12.5, 12.5, 2, 0.7, 'NLP Preprocess\n(spaCy)', colors['nlp'], fontsize=8)
    create_box(ax, 12.5, 11.5, 2, 0.7, 'NER & Extraction\n(BioBERT)', colors['nlp'], fontsize=8)
    create_box(ax, 12.5, 10.5, 2, 0.7, 'Entity Linking\n(Medical Codes)', colors['nlp'], fontsize=8)
    create_box(ax, 12.5, 9.5, 2, 0.7, 'Feature Embedding\n(Transformers)', colors['nlp'], fontsize=8)
    
    # NLP arrows
    create_arrow(ax, 12.5, 12.15, 12.5, 11.85, lw=1.2, color='darkorange')
    create_arrow(ax, 12.5, 11.15, 12.5, 10.85, lw=1.2, color='darkorange')
    create_arrow(ax, 12.5, 10.15, 12.5, 9.85, lw=1.2, color='darkorange')
    
    # Input from feature store to both pipelines
    create_arrow(ax, 8, 14.1, 1.5, 13.2, label='Structured', lw=1.2)
    create_arrow(ax, 12, 14.1, 12.5, 13.2, label='Clinical Text', color='darkorange', lw=1.2)
    
    # ============================================================================
    # LAYER 7: MODEL REGISTRY (Y=8.5)
    # ============================================================================
    ax.text(0.5, 8.8, '7. MODEL MANAGEMENT', fontsize=12, weight='bold')
    
    create_box(ax, 7, 8.2, 4, 0.9, 'MODEL REGISTRY\n(MLflow/W&B)\nVersion Control, Metadata',
              colors['registry'], fontsize=9)
    
    # Arrows from ML and NLP to registry
    create_arrow(ax, 1.5, 9.15, 5, 8.6, lw=1.5)
    create_arrow(ax, 12.5, 9.15, 9, 8.6, lw=1.5, color='darkorange')
    
    # ============================================================================
    # LAYER 8: INFERENCE SERVICE (Y=6.5)
    # ============================================================================
    ax.text(0.5, 7.3, '8. REAL-TIME INFERENCE SERVICE', fontsize=12, weight='bold')
    
    create_box(ax, 3, 6.5, 2.5, 0.8, 'Inference API\n(FastAPI)', colors['inference'], fontsize=9)
    create_box(ax, 7, 6.5, 2.5, 0.8, 'Model Serving\n(Seldon/Triton)', colors['inference'], fontsize=9)
    create_box(ax, 11, 6.5, 2.5, 0.8, 'Cache Layer\n(Redis)', colors['inference'], fontsize=9)
    
    create_arrow(ax, 7, 7.7, 3, 6.9, label='Load', lw=1.2)
    create_arrow(ax, 3, 6.1, 7, 6.1, lw=1.2)
    create_arrow(ax, 9.25, 6.5, 9.75, 6.5, lw=1.2)
    
    # Dashed line from feature store to inference (online features)
    create_arrow(ax, 10, 14.1, 3, 6.9, style='->', color='green', lw=1.5, linestyle='dashed')
    ax.text(7, 10.5, 'Online Features\n(<50ms)', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # ============================================================================
    # LAYER 9: PREDICTIONS OUTPUT (Y=5)
    # ============================================================================
    create_box(ax, 7, 5, 3, 0.8, 'PREDICTIONS & DECISIONS\n(Risk Score + Explanation)',
              '#FFFFCC', fontsize=9, weight='bold')
    
    create_arrow(ax, 7, 6.1, 7, 5.4, lw=1.5, color='red')
    
    # ============================================================================
    # LAYER 10: MONITORING & DRIFT (Y=3.5)
    # ============================================================================
    ax.text(0.5, 4.3, '9. MONITORING & DRIFT DETECTION', fontsize=12, weight='bold')
    
    create_box(ax, 2, 3.5, 2.2, 0.75, 'Drift Detection\n(Evidently)', colors['monitoring'], fontsize=8)
    create_box(ax, 5, 3.5, 2.2, 0.75, 'Metrics Logging\n(Prometheus)', colors['monitoring'], fontsize=8)
    create_box(ax, 8, 3.5, 2.2, 0.75, 'Alerting\n(Slack/Pager)', colors['monitoring'], fontsize=8)
    
    # Arrow from predictions to monitoring
    create_arrow(ax, 5.5, 4.6, 2, 3.85, lw=2, color='red')
    create_arrow(ax, 3.1, 3.5, 3.9, 3.5, lw=1.2)
    create_arrow(ax, 6.1, 3.5, 6.9, 3.5, lw=1.2)
    
    # ============================================================================
    # LAYER 11: APPLICATIONS (Y=2)
    # ============================================================================
    ax.text(0.5, 2.8, '10. OUTPUT APPLICATIONS', fontsize=12, weight='bold')
    
    create_box(ax, 3, 2, 2.2, 0.7, 'Clinical Decision\nSupport System', colors['output'], fontsize=8)
    create_box(ax, 6, 2, 2.2, 0.7, 'Patient\nMobile App', colors['output'], fontsize=8)
    create_box(ax, 9, 2, 2.2, 0.7, 'Analytics\nDashboard', colors['output'], fontsize=8)
    
    create_arrow(ax, 7, 4.6, 3, 2.35, lw=1.5, color='blue')
    create_arrow(ax, 7, 4.6, 6, 2.35, lw=1.5, color='blue')
    create_arrow(ax, 7, 4.6, 9, 2.35, lw=1.5, color='blue')
    
    # ============================================================================
    # FEEDBACK LOOPS (Bidirectional arrows)
    # ============================================================================
    ax.text(0.5, 1.3, 'FEEDBACK LOOPS', fontsize=12, weight='bold')
    
    # Loop 1: Monitoring to retraining (Red)
    create_arrow(ax, 8, 3.1, 1.5, 9.9, style='<->', color='red', lw=2.5)
    ax.text(4, 6.5, 'Retraining Signal\n(if drift detected)', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE6E6', alpha=0.9),
           weight='bold')
    
    # Loop 2: Clinical feedback (Blue)
    create_arrow(ax, 3, 1.65, 3, 14.3, style='<->', color='blue', lw=2, linestyle='dotted')
    ax.text(0.2, 8, 'Ground Truth\nFeedback', fontsize=8,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.9),
           rotation=90, weight='bold')
    
    # ============================================================================
    # LEGEND
    # ============================================================================
    y_legend = 0.5
    ax.text(14, y_legend + 0.7, 'TECHNOLOGIES', fontsize=10, weight='bold')
    
    tech_text = """Data: Apache Spark, Kafka
DTL: Great Expectations, dbt
ML: TensorFlow, PyTorch, XGBoost
NLP: Hugging Face, BioBERT
Serving: FastAPI, Seldon, Triton
Monitor: Evidently, Prometheus
"""
    ax.text(14, y_legend - 0.3, tech_text, fontsize=7, family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Title
    fig.suptitle('Healthcare AI System Architecture\nMicroservices-based ML Pipeline for Clinical Decision Support',
                fontsize=16, weight='bold', y=0.99)
    
    # Footer
    fig.text(0.5, 0.01, 'Educational Purpose | Microservices Architecture | Real-time & Batch Processing | Automated Monitoring & Retraining',
            ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    return fig


def create_data_flow_diagram():
    """Create detailed data flow diagram"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Healthcare AI System - Data Flow Patterns\nBatch Processing, Real-time Streaming, Training, and Inference',
                fontsize=14, weight='bold')
    
    # ============================================================================
    # Flow 1: Batch Ingestion
    # ============================================================================
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Batch Flow (Daily)', fontsize=12, weight='bold', pad=10)
    
    boxes_batch = [
        (1, 8, 'EHR/Labs'),
        (3, 8, 'Extract'),
        (5, 8, 'Transform'),
        (7, 8, 'Load'),
        (9, 8, 'Data Lake'),
        (7, 5, 'Feature\nEngineering'),
        (7, 2, 'Feature Store')
    ]
    
    for x, y, text in boxes_batch:
        box = FancyBboxPatch((x-0.6, y-0.35), 1.2, 0.7,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor='#CCE5FF',
                            linewidth=1)
        ax1.add_patch(box)
        ax1.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows
    for i in range(len(boxes_batch)-1):
        x1, y1 = boxes_batch[i][:2]
        x2, y2 = boxes_batch[i+1][:2]
        if y1 == y2:
            ax1.arrow(x1+0.7, y1, x2-x1-1.4, 0, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    ax1.arrow(7, 7.65, 0, -1.3, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax1.arrow(7, 4.65, 0, -1.3, head_width=0.2, head_length=0.15, fc='green', ec='green')
    
    ax1.text(7, 0.5, 'Frequency: Daily (2 AM UTC)', fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # ============================================================================
    # Flow 2: Real-time Streaming
    # ============================================================================
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Real-time Stream Flow', fontsize=12, weight='bold', pad=10)
    
    boxes_stream = [
        (1, 8, 'IoT Sensors'),
        (3.5, 8, 'Kafka Stream'),
        (6, 8, 'Windowed\nAgg'),
        (8.5, 8, 'Feature Store\n(Online)')
    ]
    
    for x, y, text in boxes_stream:
        box = FancyBboxPatch((x-0.7, y-0.35), 1.4, 0.7,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor='#FFE6CC',
                            linewidth=1)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows
    for i in range(len(boxes_stream)-1):
        x1, y1 = boxes_stream[i][:2]
        x2, y2 = boxes_stream[i+1][:2]
        ax2.arrow(x1+0.8, y1, x2-x1-1.6, 0, head_width=0.2, head_length=0.15, fc='orange', ec='orange')
    
    ax2.arrow(8.5, 7.65, 0, -1.3, head_width=0.2, head_length=0.15, fc='orange', ec='orange')
    
    # Inference connection
    box_inf = FancyBboxPatch((7.5, 5.5), 2, 0.7,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor='#D4FFE6',
                            linewidth=1)
    ax2.add_patch(box_inf)
    ax2.text(8.5, 5.85, 'Inference API', ha='center', va='center', fontsize=9, weight='bold')
    
    ax2.arrow(8.5, 6.2, 0, -0.4, head_width=0.15, head_length=0.1, fc='green', ec='green')
    
    ax2.text(8.5, 0.5, 'Latency: 10-30 seconds\nVolume: 1M events/day',
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # ============================================================================
    # Flow 3: Model Training
    # ============================================================================
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Training Pipeline Flow', fontsize=12, weight='bold', pad=10)
    
    boxes_train = [
        (1, 8, 'Feature Store\n(Offline)'),
        (3.5, 8, 'Preprocessing'),
        (6, 8, 'Model Training\n(100+ trials)'),
        (8.5, 8, 'Evaluation'),
        (8.5, 5, 'Model Registry'),
        (8.5, 2, 'Approved?')
    ]
    
    for x, y, text in boxes_train:
        if y == 2:
            color = '#FFD4D4'
        elif y == 5:
            color = '#E0E0E0'
        else:
            color = '#F0E6FF'
        
        box = FancyBboxPatch((x-0.7, y-0.35), 1.4, 0.7,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color,
                            linewidth=1)
        ax3.add_patch(box)
        ax3.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows
    ax3.arrow(2.2, 8, 0.8, 0, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax3.arrow(4.2, 8, 0.8, 0, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax3.arrow(6.7, 8, 0.8, 0, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax3.arrow(8.5, 7.65, 0, -1.3, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax3.arrow(8.5, 4.65, 0, -1.3, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    # Deploy arrow
    ax3.arrow(9.5, 2, 0.4, 0, head_width=0.2, head_length=0.1, fc='green', ec='green', linewidth=2)
    ax3.text(10.2, 2, 'Deploy', fontsize=8, weight='bold', color='green')
    
    ax3.text(5, 0.5, 'Triggered by: Data drift OR Weekly schedule',
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # ============================================================================
    # Flow 4: Inference to Monitoring Feedback
    # ============================================================================
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Prediction to Feedback Loop', fontsize=12, weight='bold', pad=10)
    
    boxes_feedback = [
        (5, 8, 'Prediction'),
        (2, 5.5, 'Drift\nDetection'),
        (8, 5.5, 'Application'),
        (5, 2.5, 'Ground Truth\n(6 months)'),
        (2, 0.5, 'Retrain?'),
        (8, 0.5, 'Improve')
    ]
    
    colors_fb = ['#FFFFCC', '#FFD4D4', '#E6FFE6', '#E0E0E0', '#FFD4D4', '#D4FFE6']
    
    for (x, y, text), color in zip(boxes_feedback, colors_fb):
        box = FancyBboxPatch((x-0.7, y-0.35), 1.4, 0.7,
                            boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color,
                            linewidth=1)
        ax4.add_patch(box)
        ax4.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')
    
    # Arrows - Feedback loop
    ax4.arrow(4.3, 7.65, -1.8, -1.6, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax4.arrow(5.7, 7.65, 1.8, -1.6, head_width=0.15, head_length=0.1, fc='blue', ec='blue')
    ax4.arrow(5, 5.15, 0, -1.8, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax4.arrow(2, 5.15, -0.3, -4, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax4.arrow(8, 5.15, 0.3, -4, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
    
    ax4.text(3.5, 3.5, 'Retraining\nSignal', fontsize=8, weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='#FFE6E6', alpha=0.7))
    ax4.text(6.5, 3, 'Feedback', fontsize=8, weight='bold', color='blue',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    return fig


def create_technology_stack_table():
    """Create technology stack reference"""
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    title = "Healthcare AI System - Complete Technology Stack"
    fig.suptitle(title, fontsize=14, weight='bold', y=0.98)
    
    # Simple text-based technology stack
    tech_stack_text = """
TECHNOLOGY STACK SUMMARY

DATA INGESTION & STORAGE
  • Apache Kafka / AWS Kinesis (streaming)
  • Apache Spark 3.0+ (batch processing)
  • S3 / Azure Data Lake (data lake)
  • Apache Airflow (orchestration)

FEATURE ENGINEERING & STORE
  • Feast / Tecton (feature store)
  • Redis (online store, <50ms latency)
  • S3 / Delta Lake (offline store)
  • Pandas, PySpark (transformation)

MACHINE LEARNING & TRAINING
  • TensorFlow 2.10+ (deep learning)
  • PyTorch 2.0+ (research frameworks)
  • XGBoost / LightGBM (gradient boosting)
  • Optuna / Ray Tune (hyperparameter optimization)
  • Scikit-learn (preprocessing)

NLP & CLINICAL TEXT PROCESSING
  • Hugging Face Transformers (pre-trained models)
  • BioBERT / SciBERT (biomedical NER)
  • spaCy / NLTK (text preprocessing)
  • BioELMo / Word2Vec (embeddings)

MODEL MANAGEMENT & REGISTRY
  • MLflow (model versioning)
  • Weights & Biases (experiment tracking)
  • Docker (containerization)
  • Git (version control)

SERVING & INFERENCE
  • FastAPI / Flask (API framework)
  • Seldon Core (model serving)
  • NVIDIA Triton (inference server)
  • ONNX Runtime (model optimization)

MONITORING & OBSERVABILITY
  • Evidently AI (drift detection)
  • WhyLabs (ML monitoring)
  • Prometheus (metrics collection)
  • ELK Stack (logging and analysis)

CLOUD INFRASTRUCTURE
  • AWS / Azure / GCP (cloud providers)
  • Kubernetes (container orchestration)
  • NVIDIA A100 GPUs (compute)
  • VPC / Security Groups (networking)


KEY PERFORMANCE TARGETS
  • Data Ingestion: 1M+ events/minute
  • Feature Store: 10K queries/second  
  • Model Training: 100 parallel trials
  • Inference Latency: <100ms p99
  • System Availability: 99.99% SLA
  
DEPLOYMENT PIPELINE
  Code → CI/CD → Staging (2%) → Canary (5%) → Production (100%) → Monitor
"""

    ax.text(0.05, 0.95, tech_stack_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9, pad=1))
    
    fig.text(0.5, 0.01, 'Educational Architecture | Production-Ready Technologies | Microservices Design',
            ha='center', fontsize=9, style='italic', color='gray')
    
    return fig


def main():
    """Generate all architecture diagrams"""
    
    print("=" * 80)
    print("HEALTHCARE AI ARCHITECTURE - DIAGRAM GENERATOR")
    print("=" * 80)
    
    # Create output directory in correct location
    output_dir = r'C:\Users\LOQ\Desktop\research\outputs\plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to output directory
    os.chdir(output_dir)
    
    # Generate Main Architecture Diagram
    print("\n[1] Generating main architecture diagram...")
    fig1 = create_architecture_diagram()
    fig1.savefig('01_Healthcare_AI_Architecture.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: 01_Healthcare_AI_Architecture.png (High-resolution)")
    plt.close(fig1)
    
    # Generate Data Flow Diagrams
    print("\n[2] Generating data flow diagrams...")
    fig2 = create_data_flow_diagram()
    fig2.savefig('02_Data_Flow_Patterns.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: 02_Data_Flow_Patterns.png")
    plt.close(fig2)
    
    # Generate Technology Stack
    print("\n[3] Generating technology stack reference...")
    fig3 = create_technology_stack_table()
    fig3.savefig('03_Technology_Stack.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: 03_Technology_Stack.png")
    plt.close(fig3)
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n✓ All diagrams saved to: {output_dir}/")
    print("\nFiles created:")
    print("  1. 01_Healthcare_AI_Architecture.png (Main system diagram)")
    print("  2. 02_Data_Flow_Patterns.png (Detailed data flows)")
    print("  3. 03_Technology_Stack.png (Technology reference)")


if __name__ == '__main__':
    main()
