
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_features(df, scaler=None, fit=False, feature_columns=None):
    df = df.copy()

    # Separate target variable
    y = df["is_fraud"].copy()

    # Remove target, identifiers and unnecessary columns
    df = df.drop(
        columns=[
            "is_fraud",
            "is_flagged_fraud",
            "transaction_id",
            "name_orig",
            "name_dest"
        ],
        errors="ignore"
    )

    # Feature engineering
    df["orig_balance_delta"] = (
        df["oldbalance_org"] - df["newbalance_orig"]
    )

    df["dest_balance_delta"] = (
        df["newbalance_dest"] - df["oldbalance_dest"]
    )

    df["is_amount_mismatch"] = (
        df["amount"] > df["oldbalance_org"]
    ).astype(int)

    # Convert transaction type into numerical columns
    df = pd.get_dummies(
        df,
        columns=["type"],
        drop_first=True,
        dtype=int
    )

    # Scale numerical features
    numeric_cols = [
        "step",
        "amount",
        "oldbalance_org",
        "newbalance_orig",
        "oldbalance_dest",
        "newbalance_dest",
        "orig_balance_delta",
        "dest_balance_delta"
    ]

    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Save the exact training feature order
        feature_columns = df.columns.tolist()

    else:
        if scaler is None or feature_columns is None:
            raise ValueError(
                "Scaler and feature columns are required for test data."
            )

        df[numeric_cols] = scaler.transform(df[numeric_cols])

        # Match test data to training columns
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df, y, scaler, feature_columns
