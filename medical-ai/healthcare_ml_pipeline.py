"""
Educational Healthcare ML Pipeline
Demonstrates ML architecture for healthcare data processing
Educational Purpose Only - Not for Medical Diagnosis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HEALTHCARE ML PIPELINE - EDUCATIONAL DEMONSTRATION")
print("=" * 70)

# ============================================================================
# LAYER 1: INPUT DATA GENERATION (Synthetic Data - Educational Purpose)
# ============================================================================
print("\n[LAYER 1] INPUT LAYER - Generating Synthetic Healthcare Data")
print("-" * 70)

np.random.seed(42)
n_samples = 500

data = {
    'age': np.random.randint(20, 80, n_samples),
    'gender': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
    'systolic_bp': np.random.randint(90, 180, n_samples),
    'diastolic_bp': np.random.randint(60, 120, n_samples),
    'cholesterol': np.random.randint(100, 300, n_samples),
    'resting_heart_rate': np.random.randint(60, 130, n_samples),
    'fasting_glucose': np.random.randint(70, 200, n_samples),
    'bmi': np.random.uniform(18, 40, n_samples),
    'smoking_status': np.random.choice([0, 1], n_samples),  # 0=No, 1=Yes
    'exercise_frequency': np.random.randint(0, 7, n_samples),  # days per week
}

df = pd.DataFrame(data)

# Generate synthetic target (educational - not real prediction)
# Risk factors correlate with label for demonstration
risk_score = (
    (df['age'] > 50).astype(int) * 0.3 +
    (df['systolic_bp'] > 140).astype(int) * 0.3 +
    (df['cholesterol'] > 200).astype(int) * 0.2 +
    (df['smoking_status'] == 1).astype(int) * 0.2
)
df['risk_label'] = (risk_score > 0.5).astype(int)

print(f"✓ Generated {n_samples} synthetic records")
print(f"✓ Features: {len(df.columns)-1}")
print(f"Dataset Preview:")
print(df.head(10))
print(f"\nClass Distribution:")
print(df['risk_label'].value_counts())

# ============================================================================
# LAYER 2: DATA PREPROCESSING
# ============================================================================
print("\n[LAYER 2] DATA PREPROCESSING")
print("-" * 70)

# 2.1 Handle Missing Values
print("✓ Missing Value Imputation: Checking for NaN values...")
print(f"  Missing values: {df.isnull().sum().sum()}")

# 2.2 Feature Extraction
print("✓ Feature Engineering:")
df['BP_category'] = ((df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)).astype(int)
df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
print(f"  - Added: BP_category")
print(f"  - Added: pulse_pressure")

# 2.3 Separate features and target
X = df.drop('risk_label', axis=1)
y = df['risk_label']

# 2.4 Data normalization
print("✓ Normalization: StandardScaler")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"  Feature means (after scaling): {X_scaled.mean().mean():.4f}")
print(f"  Feature std (after scaling): {X_scaled.std().mean():.4f}")

# 2.5 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train-Test Split: {len(X_train)} train, {len(X_test)} test")

# ============================================================================
# LAYER 3: MACHINE LEARNING MODELS
# ============================================================================
print("\n[LAYER 3] MODEL TRAINING")
print("-" * 70)

# 3.1 Random Forest Model
print("\n[Model 1] Random Forest Classifier")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

print(f"  Accuracy:  {accuracy_score(y_test, rf_pred):.4f}")
print(f"  Precision: {precision_score(y_test, rf_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, rf_pred, zero_division=0):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, rf_pred_proba):.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  Top 5 Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"    {row['feature']}: {row['importance']:.4f}")

# 3.2 Neural Network Model
print("\n[Model 2] Neural Network (MLP)")
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    max_iter=500,
    random_state=42,
    early_stopping=True
)
nn_model.fit(X_train, y_train)

nn_pred = nn_model.predict(X_test)
nn_pred_proba = nn_model.predict_proba(X_test)[:, 1]

print(f"  Accuracy:  {accuracy_score(y_test, nn_pred):.4f}")
print(f"  Precision: {precision_score(y_test, nn_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test, nn_pred, zero_division=0):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, nn_pred_proba):.4f}")

# ============================================================================
# LAYER 4: OUTPUT PREDICTIONS
# ============================================================================
print("\n[LAYER 4] PREDICTION OUTPUT")
print("-" * 70)

print("\nSample Predictions (First 10 test cases):")
print(f"{'Index':<6} {'Actual':<8} {'RF Prob':<10} {'NN Prob':<10} {'RF Class':<8} {'NN Class':<8}")
print("-" * 60)

for i in range(min(10, len(X_test))):
    actual = y_test.iloc[i]
    rf_prob = rf_pred_proba[i]
    nn_prob = nn_pred_proba[i]
    rf_class = rf_pred[i]
    nn_class = nn_pred[i]
    print(f"{i:<6} {actual:<8} {rf_prob:<10.4f} {nn_prob:<10.4f} {rf_class:<8} {nn_class:<8}")

# ============================================================================
# LAYER 5: FEEDBACK LOOP & MODEL MONITORING
# ============================================================================
print("\n[LAYER 5] MODEL MONITORING & FEEDBACK LOOP")
print("-" * 70)

# Calculate prediction confidence
rf_confidence = np.max(rf_model.predict_proba(X_test), axis=1)
nn_confidence = np.max(nn_model.predict_proba(X_test), axis=1)

print(f"\nModel Confidence Metrics:")
print(f"  Random Forest - Mean Confidence: {rf_confidence.mean():.4f}")
print(f"  Neural Network - Mean Confidence: {nn_confidence.mean():.4f}")

# Model Comparison Summary
print(f"\n{'Metric':<15} {'Random Forest':<15} {'Neural Network':<15}")
print("-" * 45)
print(f"{'Accuracy':<15} {accuracy_score(y_test, rf_pred):<15.4f} {accuracy_score(y_test, nn_pred):<15.4f}")
print(f"{'Precision':<15} {precision_score(y_test, rf_pred, zero_division=0):<15.4f} {precision_score(y_test, nn_pred, zero_division=0):<15.4f}")
print(f"{'Recall':<15} {recall_score(y_test, rf_pred, zero_division=0):<15.4f} {recall_score(y_test, nn_pred, zero_division=0):<15.4f}")
print(f"{'ROC-AUC':<15} {roc_auc_score(y_test, rf_pred_proba):<15.4f} {roc_auc_score(y_test, nn_pred_proba):<15.4f}")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================
print("\n[SAVING RESULTS]")
print("-" * 70)

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Saved: feature_importance.csv")

# Save model predictions
results_df = pd.DataFrame({
    'actual': y_test.values,
    'rf_probability': rf_pred_proba,
    'nn_probability': nn_pred_proba,
    'rf_prediction': rf_pred,
    'nn_prediction': nn_pred
})
results_df.to_csv('model_predictions.csv', index=False)
print("✓ Saved: model_predictions.csv")

# Save model performance metrics
metrics = {
    'Model': ['Random Forest', 'Neural Network'],
    'Accuracy': [accuracy_score(y_test, rf_pred), accuracy_score(y_test, nn_pred)],
    'Precision': [precision_score(y_test, rf_pred, zero_division=0), precision_score(y_test, nn_pred, zero_division=0)],
    'Recall': [recall_score(y_test, rf_pred, zero_division=0), recall_score(y_test, nn_pred, zero_division=0)],
    'ROC-AUC': [roc_auc_score(y_test, rf_pred_proba), roc_auc_score(y_test, nn_pred_proba)]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_metrics.csv', index=False)
print("✓ Saved: model_metrics.csv")

print("\n" + "=" * 70)
print("PIPELINE EXECUTION COMPLETE")
print("=" * 70)
print("\n⚠️  DISCLAIMER: This is an educational demonstration only.")
print("   Not intended for actual medical diagnosis or treatment.")
print("   Always consult qualified healthcare professionals.\n")
