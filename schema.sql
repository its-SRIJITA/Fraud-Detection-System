CREATE DATABASE IF NOT EXISTS fraud_db;
USE fraud_db;

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS fraud_predictions;

CREATE TABLE transactions (
    transaction_id BIGINT PRIMARY KEY,
    transaction_time FLOAT,
    amount DECIMAL(10,2),
    v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT,
    v5 FLOAT, v6 FLOAT, v7 FLOAT, v8 FLOAT,
    v9 FLOAT, v10 FLOAT, v11 FLOAT, v12 FLOAT,
    v13 FLOAT, v14 FLOAT, v15 FLOAT,
    v16 FLOAT, v17 FLOAT, v18 FLOAT,
    v19 FLOAT, v20 FLOAT, v21 FLOAT,
    v22 FLOAT, v23 FLOAT, v24 FLOAT,
    v25 FLOAT, v26 FLOAT, v27 FLOAT,
    v28 FLOAT,
    is_fraud TINYINT
);

CREATE TABLE fraud_predictions (
    transaction_id BIGINT,
    fraud_probability FLOAT,
    predicted_label TINYINT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
