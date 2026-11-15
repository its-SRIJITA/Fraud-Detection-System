
import os
import joblib
import pandas as pd

from db_connection import get_engine
from feature_engineering import prepare_features

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading transaction data...")

engine = get_engine()
df = pd.read_sql("SELECT * FROM transactions", engine)

df = df.sort_values("step").reset_index(drop=True)

# ==========================================
# 2. CHRONOLOGICAL TRAIN / VALIDATION / TEST
# ==========================================

unique_steps = sorted(df["step"].unique())

test_split_index = int(len(unique_steps) * 0.80)
test_cutoff = unique_steps[test_split_index]

development_df = df[df["step"] < test_cutoff].copy()
test_df = df[df["step"] >= test_cutoff].copy()

development_steps = sorted(development_df["step"].unique())

validation_split_index = int(len(development_steps) * 0.80)
validation_cutoff = development_steps[validation_split_index]

train_df = development_df[
    development_df["step"] < validation_cutoff
].copy()

validation_df = development_df[
    development_df["step"] >= validation_cutoff
].copy()

print("\n========== DATA SPLIT ==========")
print("Training:", train_df["step"].min(), "to", train_df["step"].max())
print("Validation:", validation_df["step"].min(), "to", validation_df["step"].max())
print("Testing:", test_df["step"].min(), "to", test_df["step"].max())

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

print("\nPreparing features...")

X_train, y_train, scaler, feature_names = prepare_features(
    train_df, fit=True
)

X_val, y_val, _, _ = prepare_features(
    validation_df,
    scaler=scaler,
    feature_columns=feature_names
)

X_test, y_test, _, _ = prepare_features(
    test_df,
    scaler=scaler,
    feature_columns=feature_names
)

# ==========================================
# 4. HANDLE CLASS IMBALANCE
# ==========================================

majority_indices = y_train[y_train == 0].sample(
    n=min(500000, int((y_train == 0).sum())),
    random_state=42
).index

minority_indices = y_train[y_train == 1].index

selected_indices = majority_indices.union(minority_indices)

X_lr_train = X_train.loc[selected_indices]
y_lr_train = y_train.loc[selected_indices]

print("\nTraining data:")
print("Total:", len(y_lr_train))
print("Fraud:", int((y_lr_train == 1).sum()))
print("Legitimate:", int((y_lr_train == 0).sum()))

# ==========================================
# 5. TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    class_weight="balanced",
    solver="lbfgs",
    max_iter=200,
    tol=0.001
)

lr_model.fit(X_lr_train, y_lr_train)

# ==========================================
# 6. SELECT THRESHOLD USING VALIDATION DATA
# ==========================================

print("\nFinding optimal threshold on validation data...")

val_prob = lr_model.predict_proba(X_val)[:, 1]

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_prob)
)

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-10)
)

best_index = f1_values.argmax()
best_threshold = thresholds[best_index]

print("\n========== VALIDATION RESULTS ==========")
print("Best threshold:", best_threshold)
print("Validation precision:", precision_values[best_index])
print("Validation recall:", recall_values[best_index])
print("Validation F1-score:", f1_values[best_index])
print("=======================================")

# ==========================================
# 7. FINAL EVALUATION ON UNTOUCHED TEST DATA
# ==========================================

print("\nEvaluating on final test data...")

test_prob = lr_model.predict_proba(X_test)[:, 1]

y_pred = (test_prob >= best_threshold).astype(int)

print("\n========== FINAL TEST RESULTS ==========")

print(classification_report(y_test, y_pred))

print("ROC-AUC:", roc_auc_score(y_test, test_prob))
print("PR-AUC:", average_precision_score(y_test, test_prob))

print("Fraud Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Fraud Recall:", recall_score(y_test, y_pred, zero_division=0))
print("Fraud F1-Score:", f1_score(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("========================================")

# ==========================================
# 8. SAVE MODEL AND THRESHOLD
# ==========================================

os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": lr_model,
    "scaler": scaler,
    "feature_names": feature_names,
    "threshold": float(best_threshold)
}

joblib.dump(
    model_bundle,
    "models/logistic_regression_bundle.joblib"
)

print("\nLogistic Regression model bundle saved successfully!")
