import pandas as pd
from db_connection import get_engine
import os

engine = get_engine()

file_path = "data/creditcard.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("creditcard.csv not found in data/ folder")

df = pd.read_csv(file_path)
df.columns = df.columns.str.lower()
df.rename(columns={"class": "is_fraud"}, inplace=True)

df.to_sql(
    "transactions",
    engine,
    if_exists="replace",
    index=False,
    chunksize=5000
)

print("Data loaded into MySQL successfully")