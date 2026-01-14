import os
import pickle
import pandas as pd
import joblib
from db_connection import get_engine
from feature_engineering import prepare_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

engine = get_engine()

df = pd.read_sql("SELECT * FROM transactions", engine)

# Train-test split (stratified = crucial)
train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    stratify=df["is_fraud"],
    random_state=42
)

# Features
X_train, y_train, scaler = prepare_features(train_df, fit=True)
X_test, y_test, _ = prepare_features(test_df, scaler=scaler)

# Tuned Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Fraud-friendly threshold
y_prob = rf.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()

tick_marks = np.arange(2)
plt.xticks(tick_marks, ["Not Fraud", "Fraud"])
plt.yticks(tick_marks, ["Not Fraud", "Fraud"])

# Annotate cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

# Save for GitHub / reports
plt.savefig("confusion_matrix.png")
plt.show()

print("Random Forest Results")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))



os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": rf,
    "scaler": scaler,
    "threshold": 0.30,
    "features": X_train.columns.tolist()
}

with open("models/random_forest_bundle.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

joblib.dump(rf, "models/random_forest_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")

print("Model, scaler, and metadata saved successfully.")