CREATE DATABASE IF NOT EXISTS fraud_db;
USE fraud_db;

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS fraud_predictions;

CREATE TABLE transactions (
    transaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    step INT,
    type VARCHAR(20),
    amount DECIMAL(15,2),

    name_orig VARCHAR(50),
    oldbalance_org DECIMAL(15,2),
    newbalance_org DECIMAL(15,2),

    name_dest VARCHAR(50),
    oldbalance_dest DECIMAL(15,2),
    newbalance_dest DECIMAL(15,2),

    is_fraud TINYINT,
    is_flagged_fraud TINYINT
);

CREATE TABLE fraud_predictions (
    transaction_id BIGINT,
    fraud_probability FLOAT,
    predicted_label TINYINT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
