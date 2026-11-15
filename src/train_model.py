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
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# Load data
# =========================
engine = get_engine()
df = pd.read_sql("SELECT * FROM transactions", engine)

# =========================
# Train-test split
# =========================
# Sort transactions chronologically
df = df.sort_values("step").reset_index(drop=True)

# Get unique transaction time steps
unique_steps = sorted(df["step"].unique())

# Select the 80% time boundary
split_index = int(len(unique_steps) * 0.80)
cutoff_step = unique_steps[split_index]

# Split data chronologically
train_df = df[df["step"] < cutoff_step].copy()
test_df = df[df["step"] >= cutoff_step].copy()


# Investigate feature behaviour by fraud class

print("\n========== FEATURE DISTRIBUTION BY FRAUD ==========")

df["orig_balance_delta"] = (
    df["oldbalance_org"] - df["newbalance_orig"]
)

df["dest_balance_delta"] = (
    df["newbalance_dest"] - df["oldbalance_dest"]
)

df["is_amount_mismatch"] = (
    df["amount"] > df["oldbalance_org"]
).astype(int)

features_to_check = [
    "amount",
    "oldbalance_org",
    "newbalance_orig",
    "orig_balance_delta",
    "dest_balance_delta",
    "is_amount_mismatch"
]

for feature in features_to_check:

    print(f"\nFeature: {feature}")

    print(
        df.groupby("is_fraud")[feature]
        .agg(["mean", "median", "min", "max"])
    )

print("\n========== FRAUD COUNTS BY TRANSACTION TYPE ==========")

print(pd.crosstab(df["type"], df["is_fraud"]))

print("======================================================")

print("\n========== FRAUD COUNTS BY TRANSACTION TYPE ==========")

print(
    pd.crosstab(
        df["type"],
        df["is_fraud"]
    )
)

print("======================================================")



# Investigate fraud distribution over time

df["time_period"] = pd.cut(
    df["step"],
    bins=10
)

fraud_analysis = df.groupby(
    "time_period",
    observed=False
).agg(
    total_transactions=("is_fraud", "count"),
    fraud_transactions=("is_fraud", "sum")
)

fraud_analysis["fraud_percentage"] = (
    fraud_analysis["fraud_transactions"]
    / fraud_analysis["total_transactions"]
) * 100

print("\n========== FRAUD DISTRIBUTION OVER TIME ==========")
print(fraud_analysis)
print("==================================================")

# Compare transaction types by fraud status

print("\n========== TRANSACTION TYPE ANALYSIS ==========")

print(
    pd.crosstab(
        df["type"],
        df["is_fraud"],
        normalize="index"
    ) * 100
)

print("================================================")

print("Training period:", train_df["step"].min(),
      "to", train_df["step"].max())

print("Testing period:", test_df["step"].min(),
      "to", test_df["step"].max())

print("Training fraud cases:", train_df["is_fraud"].sum())
print("Testing fraud cases:", test_df["is_fraud"].sum())

# =========================
# Feature preparation
# =========================

# Prepare training features
X_train, y_train, scaler, feature_names = prepare_features(
    train_df,
    fit=True
)

# Prepare testing features using the same scaler and columns
X_test, y_test, _, _ = prepare_features(
    test_df,
    scaler=scaler,
    feature_columns=feature_names
)

# DATA LEAKAGE AND FEATURE AUDIT

print("\n========== FEATURE AUDIT ==========")

print("Training features:", X_train.columns.tolist())
print("Testing features:", X_test.columns.tolist())

# Check whether target-related columns are present
forbidden_columns = [
    "is_fraud",
    "is_flagged_fraud",
    "transaction_id"
]

for column in forbidden_columns:
    print(
        f"{column} present:",
        column in X_train.columns
    )

# Check feature consistency
print(
    "Same training and testing columns:",
    list(X_train.columns) == list(X_test.columns)
)

# Check fraud distribution
print("\nTraining fraud distribution:")
print(y_train.value_counts())

print("\nTesting fraud distribution:")
print(y_test.value_counts())

print("\nTraining fraud percentage:",
      round(y_train.mean() * 100, 4), "%")

print("Testing fraud percentage:",
      round(y_test.mean() * 100, 4), "%")

print("===================================")


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

# Feature Importance Analysis

importances = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Visualize Feature Importance

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"].head(10)[::-1],
    feature_importance["Importance"].head(10)[::-1]
)

plt.title("Top 10 Feature Importances – Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

os.makedirs("reports", exist_ok=True)
plt.savefig("reports/feature_importance.png")
plt.show()

# =========================
# Evaluation
# =========================
y_prob = rf.predict_proba(X_test)[:, 1]
threshold = 0.35
y_pred = (y_prob >= threshold).astype(int)

print("\nRandom Forest – PaySim Dataset")

print(classification_report(
    y_test,
    y_pred,
    digits=6
))

print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("PR-AUC:", average_precision_score(y_test, y_prob))

print("Fraud Precision:", precision_score(y_test, y_pred))
print("Fraud Recall:", recall_score(y_test, y_pred))
print("Fraud F1-Score:", f1_score(y_test, y_pred))

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
