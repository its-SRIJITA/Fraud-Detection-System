from sqlalchemy import create_engine

def get_engine():
    engine = create_engine(
        "mysql+pymysql://root:@localhost/fraud_db"
    )
    return engine
