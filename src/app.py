
import os
import joblib
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "logistic_regression_bundle.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_model()

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    threshold = bundle["threshold"]

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🛡️ Fraud Detection System")

st.markdown(
    """
    ### Transaction Risk Analysis

    Enter transaction details below to evaluate whether
    a transaction may be fraudulent using Machine Learning.
    """
)

st.divider()

# ==========================================
# 4. TRANSACTION INPUT FORM
# ==========================================

st.subheader("Transaction Details")

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input(
            "Transaction Time Step",
            min_value=1,
            max_value=10000,
            value=100,
            step=1
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

        oldbalance_org = st.number_input(
            "Sender's Previous Balance",
            min_value=0.0,
            value=20000.0,
            step=100.0
        )

    with col2:
        newbalance_orig = st.number_input(
            "Sender's New Balance",
            min_value=0.0,
            value=10000.0,
            step=100.0
        )

        oldbalance_dest = st.number_input(
            "Receiver's Previous Balance",
            min_value=0.0,
            value=5000.0,
            step=100.0
        )

        newbalance_dest = st.number_input(
            "Receiver's New Balance",
            min_value=0.0,
            value=15000.0,
            step=100.0
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        use_container_width=True
    )


# ==========================================
# 5. FEATURE ENGINEERING & FRAUD PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalance_org": oldbalance_org,
        "newbalance_orig": newbalance_orig,
        "oldbalance_dest": oldbalance_dest,
        "newbalance_dest": newbalance_dest
    }])

    # Feature engineering
    input_data["orig_balance_delta"] = (
        input_data["oldbalance_org"] -
        input_data["newbalance_orig"]
    )

    input_data["dest_balance_delta"] = (
        input_data["newbalance_dest"] -
        input_data["oldbalance_dest"]
    )

    input_data["is_amount_mismatch"] = (
        input_data["amount"] >
        input_data["oldbalance_org"]
    ).astype(int)

    # Warn about potentially unusual transaction details.
    # Do not reject these inputs: they may be relevant fraud signals.
    if amount > oldbalance_org:
        st.warning(
            "Transaction amount exceeds the sender's previous balance. "
            "Please verify the entered details."
        )

    # Encode transaction type
    input_data = pd.get_dummies(
        input_data,
        columns=["type"],
        drop_first=True,
        dtype=int
    )

    # Align with training features
    input_data = input_data.reindex(
        columns=feature_names,
        fill_value=0
    )

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

    try:
        # Apply the saved scaler
        input_data[numeric_cols] = scaler.transform(
            input_data[numeric_cols]
        )

        # Predict fraud score
        fraud_class_index = list(model.classes_).index(1)

        model_score = model.predict_proba(
            input_data
        )[0][fraud_class_index]

        prediction = int(model_score >= threshold)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # ==========================================
    # 6. DISPLAY PREDICTION RESULTS
    # ==========================================

    st.divider()
    st.header("Transaction Analysis")

    if prediction == 1:
        st.error(
            "Potentially Fraudulent Transaction",
            icon="🚨"
        )
        decision = "Potentially Fraudulent"
    else:
        st.success(
            "Transaction Classified as Legitimate",
            icon="🛡️"
        )
        decision = "Classified as Legitimate"

    # Main metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model Score",
            f"{model_score * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Prediction",
            decision
        )

    # Score visualization
    st.subheader("Model Score Analysis")

    st.progress(float(model_score))

    st.caption(
        f"Model score: {model_score:.4f} | "
        f"Decision threshold: {threshold:.4f}"
    )

    if prediction == 1:
        st.warning(
            "The model score meets or exceeds the selected "
            "fraud detection threshold."
        )
    else:
        st.info(
            "The model score is below the selected "
            "fraud detection threshold."
        )

    # Transaction summary
    st.divider()
    st.subheader("Transaction Summary")

    transaction_summary = pd.DataFrame({
        "Parameter": [
            "Transaction Type",
            "Transaction Amount",
            "Transaction Time Step",
            "Sender's Previous Balance",
            "Sender's New Balance",
            "Receiver's Previous Balance",
            "Receiver's New Balance"
        ],
        "Value": [
            transaction_type,
            f"{amount:,.2f}",
            step,
            f"{oldbalance_org:,.2f}",
            f"{newbalance_orig:,.2f}",
            f"{oldbalance_dest:,.2f}",
            f"{newbalance_dest:,.2f}"
        ]
    })

    st.dataframe(
        transaction_summary,
        use_container_width=True,
        hide_index=True
    )

    # Model information
    st.divider()

    with st.expander("Model Information"):
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Dataset:** PaySim Synthetic Transactions")
        st.write("**Threshold Selection:** Validation dataset")
        st.write(f"**Decision Threshold:** {threshold:.4f}")
        st.write(
            "**Evaluation:** Chronological "
            "train-validation-test split"
        )

    st.caption(
        "Disclaimer: This application uses a machine-learning "
        "model trained on synthetic PaySim data. The model score "
        "is not a calibrated real-world fraud probability. "
        "Predictions should not be treated as financial decisions."
    )

