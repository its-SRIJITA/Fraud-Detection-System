
import os
from sqlalchemy import create_engine, URL
from dotenv import load_dotenv

load_dotenv()

def get_engine():

    connection_url = URL.create(
        drivername="mysql+pymysql",
        username=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "fraud_db")
    )

    return create_engine(connection_url)
