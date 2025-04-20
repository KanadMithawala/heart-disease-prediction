import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import json

# Load the dataset
df = pd.read_csv('framingham.csv')

# Remove unnecessary column
df = df.drop(columns=['education'])

# Handle missing values
num_cols = ['cigsPerDay', 'totChol', 'BMI', 'heartRate', 'glucose']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
df['BPMeds'] = df['BPMeds'].fillna(df['BPMeds'].mode()[0])
df = df.dropna(subset=['TenYearCHD'])

# Define features and target
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Evaluate individual models
models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model}
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

# Evaluate ensemble (average probabilities)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
gb_probs = gb_model.predict_proba(X_test_scaled)[:, 1]
ensemble_probs = (rf_probs + gb_probs) / 2
ensemble_preds = (ensemble_probs > 0.5).astype(int)
accuracy = accuracy_score(y_test, ensemble_preds)
precision = precision_score(y_test, ensemble_preds)
recall = recall_score(y_test, ensemble_preds)
f1 = f1_score(y_test, ensemble_preds)
print(f"Ensemble - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

# Save both models and scaler
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(gb_model, 'gb_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Save feature names for API
with open('feature_names.json', 'w') as f:
    json.dump(X.columns.tolist(), f)