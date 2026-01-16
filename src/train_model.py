import os
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from db_connection import get_engine
from feature_engineering import prepare_features

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# =========================
# Load data
# =========================
engine = get_engine()
df = pd.read_sql("SELECT * FROM transactions", engine)

# =========================
# Train-test split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    stratify=df["is_fraud"],
    random_state=42
)

# =========================
# Feature preparation
# =========================
X_train, y_train, scaler, feature_names = prepare_features(
    train_df, fit=True
)

X_test, y_test, _, _ = prepare_features(
    test_df, scaler=scaler
)

# =========================
# Model
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
y_prob = rf.predict_proba(X_test)[:, 1]
threshold = 0.35
y_pred = (y_prob >= threshold).astype(int)

print("\nRandom Forest – PaySim Dataset")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix – Random Forest (PaySim)")
plt.colorbar()

plt.xticks([0, 1], ["Not Fraud", "Fraud"])
plt.yticks([0, 1], ["Not Fraud", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/confusion_matrix.png")
plt.show()

# =========================
# Save model bundle
# =========================
os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": rf,
    "scaler": scaler,
    "threshold": threshold,
    "features": feature_names
}

with open("models/random_forest_bundle.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

joblib.dump(rf, "models/random_forest_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("\nModel, scaler, and metadata saved successfully.")
