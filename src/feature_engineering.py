import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def prepare_features(df, scaler=None, fit=False):
    df = df.copy()

    y = df["is_fraud"]

    df = df.drop(
        columns=["is_fraud", "is_flagged_fraud", "name_orig", "name_dest"],
        errors="ignore"
    )

    df["orig_balance_delta"] = df["oldbalance_org"] - df["newbalance_orig"]
    df["dest_balance_delta"] = df["newbalance_dest"] - df["oldbalance_dest"]

    df["is_amount_mismatch"] = (df["amount"] > df["oldbalance_org"]).astype(int)

    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    numeric_cols = [
        "step", "amount",
        "oldbalance_org", "newbalance_orig",
        "oldbalance_dest", "newbalance_dest",
        "orig_balance_delta", "dest_balance_delta"
    ]

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, y, scaler, df.columns.tolist()
