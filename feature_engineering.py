import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_features(df, scaler=None, fit=False):
    df = df.copy()

    # Target
    y = df["is_fraud"]

    # ----- Feature Engineering -----
    df["amount_log"] = np.log1p(df["amount"])
    df["is_high_amount"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)

    df["amount_zscore"] = (
        df["amount"] - df["amount"].mean()
    ) / (df["amount"].std() + 1e-6)

    # Feature matrix
    X = df[[
        "amount",
        "amount_log",
        "is_high_amount",
        "amount_zscore"
    ]].copy()

    # ----- Scaling -----
    if scaler is None:
        scaler = StandardScaler()

    scale_cols = ["amount", "amount_log", "amount_zscore"]

    if fit:
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
    else:
        X[scale_cols] = scaler.transform(X[scale_cols])

    return X, y, scaler