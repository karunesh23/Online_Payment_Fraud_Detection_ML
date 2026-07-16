import streamlit as st # FOR UI
import pandas as pd # FOR DATA MANIPULATION
import joblib # FOR LOADING MODEL
import numpy as np # FOR NUMERICAL OPERATIONS

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="centered"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
    <style>
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("Online Transaction Fraud Detection")
st.markdown("### Random Forest Model with Hyperparameter Tuning")
st.markdown("---")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource #LOAD MODEL ONCE
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# -------------------------------
# User Input Form
# -------------------------------
st.subheader("Enter Transaction Details")

with st.form("fraud_form"): # CREATE A FORM TO TAKE USER INPUT

    step = st.number_input("Step (Transaction Time Step)", min_value=1, value=200)

    transaction_type = st.selectbox(
        "Transaction Type",
        options=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    )

    # Encode manually (must match training LabelEncoder order)
    type_mapping = {
        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4
    }

    amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)

    oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=20000.0)
    newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=15000.0)

    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=1000.0)
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=6000.0)

    isFlaggedFraud = st.selectbox("Flagged Fraud", [0, 1])

    submitted = st.form_submit_button("🔍 Predict Fraud")

# -------------------------------
# Prediction
# -------------------------------
if submitted:

    input_data = pd.DataFrame([{
        "step": step,
        "type": type_mapping[transaction_type],
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Fraudulent Transaction\n\nProbability: {np.round(probability*100,2)}%")
    else:
        st.success(f"Legitimate Transaction\n\nFraud Probability: {np.round(probability*100,2)}%")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Developed by Karunesh | Fraud Detection ML Project | 2026")
