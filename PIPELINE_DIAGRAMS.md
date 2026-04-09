# Healthcare ML Pipeline - Visual Architecture Diagrams

---

## 1. Complete Pipeline Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    HEALTHCARE ML PIPELINE ARCHITECTURE                      │
└────────────────────────────────────────────────────────────────────────────┘


PHASE 1: DATA COLLECTION
═══════════════════════════════════════════════════════════════════════════════

    EHR System              Clinical Notes          Wearable Sensors
        ↓                           ↓                      ↓
    Demographics            Unstructured Text        Real-time Data
    Medical History         Diagnoses                Heart Rate
    Medications            Treatment Plans           Blood Pressure
        │                           │                      │
        └───────────────────────────┼──────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │   INPUT DATA LAYER (1)      │
                    │  500+ Features per Patient  │
                    │  Structured & Unstructured  │
                    │  Multiple Data Sources      │
                    └─────────────────────────────┘
                                    │


PHASE 2: DATA PREPROCESSING
═══════════════════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────┐
                    │ PREPROCESSING LAYER (2)     │
                    └────────────┬────────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
        Missing Value      Normalization     Feature Engineering
        Imputation         (StandardScaler)  (New Features)
        ├─ Mean fill       ├─ Z-score       ├─ BP Category
        ├─ Median fill     ├─ Min-Max       ├─ Pulse Pressure
        └─ Forward fill    └─ Robust        ├─ Age Groups
                                            └─ BMI Category
                │                │                │
                └────────────────┼────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │ Categorical Encoding        │
                    │ One-Hot / Label             │
                    └────────────┬────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │ Train-Test Split (80-20)    │
                    │ Stratified by Class          │
                    └─────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
            ┌──────────────┐         ┌──────────────┐
            │ Training Set │         │  Testing Set │
            │ (400 samples)│         │ (100 samples)│
            └───────┬──────┘         └──────────────┘
                    │


PHASE 3: MODEL TRAINING
═══════════════════════════════════════════════════════════════════════════════

            Training Data (Preprocessed)
                    │
                    ▼
        ┌─────────────────────────┐
        │  MODEL TRAINING (3)     │
        └────────┬────────────────┘
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
    
    ╔═════════════════════════╗      ╔═════════════════════════╗
    ║ MODEL 1: RANDOM FOREST  ║      ║ MODEL 2: NEURAL NET     ║
    ╠═════════════════════════╣      ╠═════════════════════════╣
    ║                         ║      ║                         ║
    ║ 100 Decision Trees      ║      ║ Input → 64 → 32 → 16   ║
    ║  ├─ Tree 1              ║      ║ ReLU Activation         ║
    ║  ├─ Tree 2              ║      ║ Softmax Output          ║
    ║  ├─ ...                 ║      ║                         ║
    ║  └─ Tree 100            ║      ║ Epochs: 500             ║
    ║                         ║      ║ Batch Size: 32          ║
    ║ Max Depth: 10           ║      ║ Early Stopping: ON      ║
    ║ Min Samples: 5          ║      ║                         ║
    ║                         ║      ║                         ║
    ║ ✓ Trained               ║      ║ ✓ Trained               ║
    ║ ✓ Saved                 ║      ║ ✓ Saved                 ║
    ║                         ║      ║                         ║
    ╚═════════════════════════╝      ╚═════════════════════════╝
              │                              │
              └──────────────┬───────────────┘


PHASE 4: MODEL EVALUATION
═══════════════════════════════════════════════════════════════════════════════

         ┌─────────────────────────────────────────┐
         │  Apply Models to Testing Data           │
         └──────────┬────────────────────────────┬─┘
                    │                            │
        ┌───────────▼──────────┐    ┌───────────▼──────────┐
        │ RF Predictions       │    │ NN Predictions       │
        │ Probability: 0.73    │    │ Probability: 0.71    │
        │ Class: High Risk     │    │ Class: High Risk     │
        └──────────────────────┘    └──────────────────────┘
                    │                            │
                    └───────────┬────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────┐
                    │ EVALUATION METRICS          │
                    ├─────────────────────────────┤
                    │ Accuracy:      0.87         │
                    │ Precision:     0.85         │
                    │ Recall:        0.79         │
                    │ F1-Score:      0.82         │
                    │ ROC-AUC:       0.91         │
                    └──────────────┬──────────────┘
                                   │


PHASE 5: PREDICTIONS
═══════════════════════════════════════════════════════════════════════════════

         New Patient Data (Preprocessed)
                    │
         ┌──────────▼──────────┐
         │                     │
         ▼                     ▼
    ┌──────────────┐  ┌──────────────┐
    │ Random       │  │ Neural       │
    │ Forest       │  │ Network      │
    │ Model        │  │ Model        │
    └──────┬───────┘  └───────┬──────┘
           │                  │
           ▼                  ▼
      Probability        Probability
       0.72 (72%)        0.70 (70%)
           │                  │
           └──────────┬───────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │   ENSEMBLE VOTING                   │
        │   Average: (0.72 + 0.70) / 2 = 0.71 │
        └─────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │   OUTPUT PREDICTION (4)             │
        ├─────────────────────────────────────┤
        │ Risk Probability:    71%            │
        │ Risk Classification: MEDIUM-HIGH    │
        │ Confidence Score:    0.82           │
        │                                     │
        │ Top Contributing Factors:           │
        │  1. High BP (Systolic 150)  [0.23] │
        │  2. Age (45 years)          [0.18] │
        │  3. Chest Pain Symptoms     [0.16] │
        │  4. High Cholesterol        [0.15] │
        │  5. Elevated Heart Rate     [0.12] │
        └──────────┬────────────────────────┘
                   │


PHASE 6: MONITORING & FEEDBACK
═══════════════════════════════════════════════════════════════════════════════

        ┌─────────────────────────────────┐
        │ MONITORING LAYER (5)            │
        ├─────────────────────────────────┤
        │ • Log Prediction + Timestamp    │
        │ • Store Confidence Score        │
        │ • Track Clinical Outcome        │
        │ • Monitor Model Drift           │
        └──────────┬──────────────────────┘
                   │
                   ▼ (After 6-12 months)
        ┌─────────────────────────────────┐
        │ Actual Medical Outcome          │
        │  Patient developed CAD?         │
        │  YES ✓ or NO ✗                  │
        └──────────┬──────────────────────┘
                   │
                   ▼
        ┌─────────────────────────────────┐
        │ Compare: Prediction vs Actual   │
        │ ✓ Correct Prediction            │
        │ ✗ Incorrect Prediction          │
        │ → Calculate Accuracy            │
        └──────────┬──────────────────────┘
                   │
        ┌──────────³────────────┐
        │                       │
        NO                      YES
        │                       │
        │               ┌───────▼────────┐
        │               │ RETRAINING?    │
        │               │ Thresholds:    │
        │               │ • Accuracy <80%│
        │               │ • 1000+ samples│
        │               │ • Monthly check│
        │               └───────┬────────┘
        │                       │
        │               ┌───────▼────────────┐
        │               │ Retrain Models     │
        │               │ with New Data      │
        │               │ Previous Phase 2   │◄──┐
        │               └────────────────────┘   │
        │                                        │
        └────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
```

---

## 2. Data Transformation Flow

```
BEFORE PREPROCESSING          AFTER PREPROCESSING
═══════════════════════════════════════════════════════════════════════════════

Raw Patient Data              Normalized Features
─────────────────             ──────────────────

Age:                45        age_normalized:        0.15
Gender:             Male      gender_encoded:        [1, 0]
Systolic BP:        150       systolic_normalized:   1.23
Diastolic BP:       95        diastolic_normalized:  0.87
Cholesterol:        280       cholesterol_norm:      1.45
Heart Rate:         110       heart_rate_normalized: 0.92
Glucose:            125       glucose_normalized:    0.66

Missing Glucose     NaN   ──► Imputed with mean:   98
                                glucose_normalized: 0.31

New Features Created:
                              BP_category:           1 (High)
                              pulse_pressure:        55 (150-95)
                              age_group:             [0,1,0,0] (45-50)
                              CHD_risk_score:        0.67


═══════════════════════════════════════════════════════════════════════════════
```

---

## 3. Decision Tree Visualization (Single Tree from Random Forest)

```
                          Feature: Systolic_BP
                         Threshold: 140
                              │
                ┌─────────────┴─────────────┐
                │                           │
            ≤ 140                      > 140
              │                            │
              ▼                            ▼
        Feature: Age               Feature: Cholesterol
        Threshold: 50              Threshold: 200
              │                            │
        ┌─────┴─────┐              ┌──────┴──────┐
        │            │              │             │
      ≤ 50        > 50           ≤ 200        > 200
        │            │              │             │
        ▼            ▼              ▼             ▼
     Class:      Class:         Class:          Class:
     LOW         LOW-           MEDIUM-         HIGH
                 MED            HIGH


Prediction Path Example:
If Systolic_BP > 140 AND Cholesterol > 200  →  HIGH RISK
```

---

## 4. Neural Network Architecture

```
INPUT LAYER              HIDDEN LAYERS                    OUTPUT LAYER
═══════════════════════════════════════════════════════════════════════════════

15 Features              64 Neurons (ReLU)        2 Classes
─────────────            ─────────────────        ──────────
Age                      ●                        ● LOW RISK  (softmax)
Gender              ┌────●                        ● HIGH RISK (softmax)
Systolic BP         │    ●
Diastolic BP        │    ● (64 neurons)      ┌────→ Output probability
Cholesterol         │    ●                   │      [0.28, 0.72]
Heart Rate          │    ●              32   │
Glucose             │────●  32 Neurons   ───┤
BMI                 │    ●  (ReLU)       │   │
Smoking             │    ●                   │
Exercise            │    ●                   │
BP Category         │    ●              16   │
Pulse Pressure      │────●  16 Neurons   ───┤
Age Group (OH)      │    ●  (ReLU)       │   │
CHD Risk Score      │    ●                   │
Risk Factors (OH)   │    ●                   │
                    │────●──────────────────┤
                    │    ●                   │
                    └────●                   │
                         ●                   │
                    64 connections    Softmax
                    to next layer    ├────→ P(High Risk) = 0.72


Forward Pass Example:
INPUT: [45, 1, 150, 95, 280, 110, 122, 28, 1, 55, ......]
           ↓
    Hidden1 = ReLU(W1 × INPUT + b1)    → 64 values
           ↓
    Hidden2 = ReLU(W2 × Hidden1 + b2)  → 32 values
           ↓
    Hidden3 = ReLU(W3 × Hidden2 + b3)  → 16 values
           ↓
    OUTPUT = Softmax(W4 × Hidden3 + b4) → [0.28, 0.72]
           ↓
    Prediction: HIGH RISK (index 1 = 0.72 probability)
```

---

## 5. Model Comparison Metrics

```
RANDOM FOREST vs NEURAL NETWORK
═══════════════════════════════════════════════════════════════════════════════

                Random Forest           Neural Network
                ─────────────           ──────────────

     Accuracy      ┌──────────┐             ┌──────────┐
                   │██████████│ 0.87        │█████████ │ 0.85
                   └──────────┘             └──────────┘

    Precision      ┌──────────┐             ┌──────────┐
                   │██████████│ 0.85        │██████████│ 0.88
                   └──────────┘             └──────────┘

      Recall       ┌──████████┐             ┌█████████ ┐
                   │██████████│ 0.79        │██████████│ 0.82
                   └──────────┘             └──────────┘

    F1-Score       ┌──████████┐             ┌██████████┐
                   │██████████│ 0.82        │██████████│ 0.85
                   └──────────┘             └──────────┘

     ROC-AUC       ┌█████████ ┐             ┌█████████ ┐
                   │██████████│ 0.91        │██████████│ 0.89
                   └──────────┘             └──────────┘


     Speed         ┌──────────┐             ┌──────────┐
   (Inference)     │██████████│ FAST        │██████████│ FAST
                   └──────────┘             └──────────┘


Feature               Random Forest           Neural Network
─────────────────────────────────────────────────────────────────────
Accuracy              ✓ Higher (87%)          Good (85%)
Interpretability      ✓ High (Feature IMP)    Low (Black Box)
Training Time         ✓ Fast                  Moderate
Non-linearity         Good                    ✓ Excellent
Overfitting Risk      Moderate                ✓ Lower (with regularization)
Data Requirements     Moderate                More data preferred
```

---

## 6. Classification Confusion Matrix

```
                    ACTUAL LABELS
                    ──────────────────
                     Low Risk    High Risk
            ┌────────────────────────────────┐
Predicted   │  True      False    │           │
Low Risk    │  Negative  Positive │ 140       │
            │  (TN=125)  (FP=15)  │           │
            ├────────────────────────────────┤
Predicted   │  False     True     │           │
High Risk   │  Negative  Positive │  60       │
            │  (FN=10)   (TP=50)  │           │
            └────────────────────────────────┘
                   150            60    Total: 200


Performance Calculations:
═════════════════════════

Accuracy  = (TP + TN) / Total = (50 + 125) / 200 = 0.875 (87%)
           "Overall correctness"

Precision = TP / (TP + FP) = 50 / (50 + 15) = 0.769 (77%)
           "Of predicted High Risk, how many correct"

Recall    = TP / (TP + FN) = 50 / (50 + 10) = 0.833 (83%)
           "Of actual High Risk cases, how many found"

F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
          = 2 × (0.769 × 0.833) / (0.769 + 0.833) = 0.800 (80%)
           "Balance between Precision & Recall"
```

---

## 7. Feature Importance Ranking

```
TOP 10 MOST IMPORTANT FEATURES FOR PREDICTION
═══════════════════════════════════════════════════════════════════════════════

Rank  Feature Name              Importance Score    Visualization
────  ────────────────────────  ────────────────    ─────────────────────
 1    Systolic_BP               0.245               ████████████████░░░░░
 2    Age                       0.198               █████████████░░░░░░░░
 3    Cholesterol               0.156               ██████████░░░░░░░░░░░
 4    Heart_Rate                0.134               █████████░░░░░░░░░░░░░
 5    Diastolic_BP              0.098               ██████░░░░░░░░░░░░░░░░
 6    Glucose_Level             0.076               █████░░░░░░░░░░░░░░░░░
 7    BMI                       0.054               ███░░░░░░░░░░░░░░░░░░░░
 8    Smoking_Status            0.043               ██░░░░░░░░░░░░░░░░░░░░░
 9    Exercise_Freq             0.032               ██░░░░░░░░░░░░░░░░░░░░░
10    Pulse_Pressure            0.024               █░░░░░░░░░░░░░░░░░░░░░░

Cumulative Importance: 0.860 (86% of decision-making)

Interpretation:
• Top 3 features account for 60% of model's decisions
• Blood pressure is the strongest predictor
• Age and cholesterol are secondary factors
• Exercise has minimal impact (consider for removal)
```

---

## 8. Prediction Probability Distribution

```
HISTOGRAM: DISTRIBUTION OF RISK PROBABILITIES
═══════════════════════════════════════════════════════════════════════════════

Low Risk        Medium Risk       High Risk
(0.0-0.33)      (0.33-0.67)      (0.67-1.0)
││               ││                ││
││  ┌──────┐    ││ ┌──────┐       ││ ┌──────┐
││  │██ 12 │    ││ │██150 │       ││ │██ 38 │
││  │██ mx│    ││ │██ mx│        ││ │██ mx│
││  └──────┘    ││ └──────┘       ││ └──────┘
││               ││                ││
└─┴──────────────┴─┴─────────────┴─┴──────────────┘
0.0            0.33            0.67            1.0


Sample Distribution (200 predictions):
• 12 predicted Low Risk (0.0-0.33)
• 150 predicted Medium Risk (0.33-0.67)
• 38 predicted High Risk (0.67-1.0)

Implications:
✓ Most patients classified as MEDIUM risk
✓ Could indicate balanced sensitivity/specificity
✓ May warrant threshold adjustment
```

---

## 9. Retraining Trigger Decision Tree

```
MODEL PERFORMANCE MONITORING
═══════════════════════════════════════════════════════════════════════════════

START: Monitor Model in Production
        │
        ▼
    ┌───────────────────────────┐
    │ Check Accuracy Drop?       │
    │ (compare to baseline)      │
    └────────┬──────────┬────────┘
             │          │
        YES  │          │  NO
             │          │
             ▼          │
    ┌──────────────┐   │
    │ Drop > 5%?   │   │
    └────┬─────┬──┘   │
         │     │      │
     YES │     │ NO   │
         │     │      ├──┐
         ▼     └─┐    │  │
    ALERT ◄──────┘    │  │
                      │  │
                      ▼  │
                 ┌──────────────┐
                 │ Data Drift?  │
                 │ (Feature     │
                 │ distributions│
                 │ changed?)    │
                 └────┬─────┬───┘
                  YES │     │ NO
                      │     │
                      ▼     └─────┐
                  ALERT ◄──┐      │
                            │     ▼
                            │  ┌──────────────┐
                            │  │ New Samples? │
                            │  │ (> 1000)     │
                            │  └────┬─────┬───┘
                            │   YES │     │ NO
                            │       │     │
                            │       ▼     │
                    ┌───────┴────ALERT   │
                    │                    │
                    ▼                    ▼
            ┌─────────────────┐  ┌──────────────┐
            │ TRIGGER         │  │ MONTHLY      │
            │ RETRAINING      │  │ SCHEDULE     │
            └─────────────────┘  │ RETRAINING?  │
                                 └────┬─────┬───┘
                                  YES │     │ NO
                                      │     │
                                      ▼     │
                            ┌─────────────────┐
                            │ TRIGGER         │
                            │ RETRAINING      │
                            └─────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Collect new data     │
                         │ Retrain models       │
                         │ Validate performance │
                         │ Deploy if improved   │
                         └──────────────────────┘
```

---

## 10. Complete System State Diagram

```
                        ┌─ DEVELOPMENT ─┐
                        │                │
                    ┌───▼────────┐       │
                    │ Raw Data   │       │
                    └────┬───────┘       │
                         │              │
                    ┌────▼────────┐     │
                    │ Preprocess  │     │
                    └────┬───────┘      │
                         │             │
                    ┌────▼─────────┐    │
                    │ Train Models │    │
                    └────┬────────┘     │
                         │             │
                    ┌────▼──────────┐   │
                    │ Evaluate Perf │   │
                    └────┬─────────┘    │
                         │             │
                    ┌────▼──────────┐   │
                    │ Meets QA?     │───┤
                    └────┬────┬─────┘   │
                        NO   YES        │
                         │    │         │
                         │    └────┐    │
                         │         │    │
                         ▼         ▼    │
                    ┌─────────────────┐ │
                    │ NO              │ │
                    │ ├─ Adjust       │ │
                    │ │  hyperparams  │ │
                    │ └─ Retrain      │─┘
                    │                 │
                    │ Loops back      │
                    └────────┬────────┘
                             │ (Approved)
                    ┌────────▼────────┐
                    │  DEPLOYMENT     │
                    │  (Production)   │
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
       ┌────▼──────┐                 ┌────────▼──────┐
       │ Monitoring│                 │ API Serving   │
       │ Dashboard │                 │ Real-time     │
       │           │                 │ Predictions   │
       └────┬──────┘                 └────────┬──────┘
            │                                  │
       ┌────▼──────────────────────────────────▼────┐
       │  New Data Accumulated                      │
       │  Actual Outcomes Recorded                  │
       │  Performance Degradation Detected?         │
       └────┬─────────────────────────┬──────────────┘
           NO                         YES
            │                         │
            │                    ┌────▼──────────────┐
            │                    │ RETRAIN CYCLE     │
            │                    │ (Return to        │
            │                    │  Development)     │
            │                    └───────────────────┘
            │                             │
            └─────────────────────────────┘
```

---

**End of Diagram Suite**

These visual representations show:
- ✓ Complete data flow from input to prediction
- ✓ Data transformation stages
- ✓ Model architectures
- ✓ Performance evaluation
- ✓ Feature importance
- ✓ Deployment lifecycle

**⚠️ EDUCATIONAL PURPOSE ONLY**
