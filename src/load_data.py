import pandas as pd
from db_connection import get_engine
import os

engine = get_engine()

file_path = "data/payments.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("payments.csv not found")

df = pd.read_csv(file_path)

df.columns = df.columns.str.lower()

df.rename(
    columns={
        "oldbalanceorg": "oldbalance_org",
        "newbalanceorig": "newbalance_orig",
        "oldbalancedest": "oldbalance_dest",
        "newbalancedest": "newbalance_dest",
        "isfraud": "is_fraud",
        "isflaggedfraud": "is_flagged_fraud",
        "nameorig": "name_orig",
        "namedest": "name_dest"
    },
    inplace=True
)

df.to_sql(
    "transactions",
    engine,
    if_exists="replace",
    index=False,
    chunksize=5000
)

print("PaySim data loaded correctly.")
