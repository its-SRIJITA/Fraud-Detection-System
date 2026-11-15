# Fraud Detection System

An end-to-end machine learning project for detecting fraudulent financial transactions using Python, MySQL, and scikit-learn. The system includes data preprocessing, feature engineering, model training, chronological evaluation, model persistence, and an interactive Streamlit dashboard.

## Project Overview

This project explores transaction fraud detection using the PaySim synthetic financial transaction dataset. It implements two classification models:

- **Random Forest Classifier**
- **Logistic Regression**

The project focuses on handling imbalanced transaction data and evaluating model performance using precision, recall, F1-score, ROC-AUC, and PR-AUC.

## Key Features

- Transaction data processing and database integration using MySQL.
- Feature engineering and numerical scaling.
- Fraud classification using Random Forest and Logistic Regression.
- Chronological train-validation-test splitting to evaluate performance on later transactions.
- Validation-based threshold selection for Logistic Regression.
- Model persistence using Joblib.
- Interactive fraud prediction dashboard built with Streamlit.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Database | MySQL |
| Database Connectivity | SQLAlchemy, PyMySQL |
| Visualization | Matplotlib |
| Model Persistence | Joblib |
| Dashboard | Streamlit |
| Configuration | Python-dotenv |

## Dataset

This project uses the PaySim Synthetic Financial Dataset.

The dataset is not included in this repository because of its size.

### Download Instructions

1. Download the dataset from [Kaggle – PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1).
2. Extract the downloaded CSV file.
3. Create a folder named `data` in the project root.
4. Rename the dataset file to `payments.csv`.
5. Place it inside the `data` folder.

Your directory should look like this:

Fraud-Detection-System/
├── data/                # Downloaded locally; not uploaded
│   └── payments.csv
├── models/
├── reports/
├── sql/
├── src/
└── README.md

### Load the Dataset

Ensure the downloaded PaySim CSV file is saved as:

`data/payments.csv`

From the project root, run:

```bash
python src/load_data.py

## Project Structure

```text
Fraud-Detection-System/
│
├── models/
│   ├── random_forest_bundle.pkl
│   ├── random_forest_model.joblib
│   ├── scaler.joblib
│   └── logistic_regression_bundle.joblib
│
├── reports/
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── sql/
│   └── schema.sql
│
├── src/
│   ├── app.py
│   ├── db_connection.py
│   ├── feature_engineering.py
│   ├── load_data.py
│   ├── train_model.py
│   └── train_logistic_regression.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Machine Learning Workflow

1. **Data Loading:** Import transaction data and prepare it for analysis.
2. **Preprocessing:** Handle transaction fields and prepare numerical and categorical features.
3. **Feature Engineering:** Generate additional features, including:
   - Origin balance change
   - Destination balance change
   - Amount mismatch indicator
4. **Model Training:** Train Random Forest and Logistic Regression classifiers.
5. **Model Evaluation:** Evaluate performance on chronologically separated data.
6. **Threshold Optimization:** Select the Logistic Regression classification threshold using validation data.
7. **Model Persistence:** Save trained models and associated preprocessing information using Joblib.
8. **Prediction:** Use the saved Logistic Regression model in the Streamlit application.

## Model Evaluation

The models were evaluated using chronological transaction splits rather than relying only on random splitting.

### Logistic Regression

The validation set was used to select the classification threshold. The final test set was kept separate from threshold selection.

| Metric | Test Result |
|---|---:|
| Precision | 91.61% |
| Recall | 62.03% |
| F1-score | 73.97% |
| ROC-AUC | 0.9957 |
| PR-AUC | 0.8570 |

**Selected classification threshold:** 0.9703

### Random Forest

The Random Forest model achieved the following results on its chronological test split:

| Metric | Test Result |
|---|---:|
| Precision | 100% |
| Recall | 100% |
| F1-score | 100% |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |

The unusually high Random Forest results should be interpreted cautiously. They reflect performance on this synthetic dataset and do not establish equivalent performance on real-world financial transactions.

*Note: The two models used different training configurations and training-data sizes, so their scores should not be treated as a strictly controlled head-to-head comparison.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/its-SRIJITA/Fraud-Detection-System.git
cd Fraud-Detection-System
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure MySQL

Create a MySQL database named `fraud_db` and execute the SQL script:

```text
sql/schema.sql
```

Create a `.env` file in the project root:

```env
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_NAME=fraud_db
```

Update the values according to your local MySQL configuration.

### 5. Load the Dataset

Download the PaySim CSV file and place it at the location configured in `src/load_data.py`.

Run:

```bash
python src/load_data.py
```

### 6. Train the Models

Run the Random Forest training script:

```bash
python src/train_model.py
```

Run the Logistic Regression training script:

```bash
python src/train_logistic_regression.py
```

The trained models and preprocessing artifacts are saved in the `models/` directory.

### 7. Launch the Dashboard

```bash
streamlit run src/app.py
```

The Streamlit application allows users to enter transaction details and receive a fraud classification based on the saved Logistic Regression model.

## Limitations

- PaySim is a synthetic dataset and may not represent all real-world fraud patterns.
- Model performance may vary across different transaction distributions.
- The dashboard is a demonstration application, not a production banking fraud detection service.
- Predictions are model scores and should not be interpreted as calibrated real-world fraud probabilities.

## Future Improvements

- Evaluate models on additional real-world or independently generated datasets.
- Explore advanced approaches for imbalanced classification.
- Improve probability calibration and threshold selection.
- Integrate real-time transaction monitoring.
- Deploy the application using a cloud-based infrastructure.

## Author

**Srijita Mukherjee**

GitHub: [its-SRIJITA](https://github.com/its-SRIJITA)

