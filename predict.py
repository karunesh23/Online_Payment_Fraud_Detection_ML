import joblib
import pandas as pd

# Load trained model
model = joblib.load("fraud_model.pkl")

# Example new transaction (same order as training data)
new_transaction = pd.DataFrame([{
    "step": 200,
    "type": 4,                 # encoded type
    "amount": 5000.0,
    "oldbalanceOrg": 20000.0,
    "newbalanceOrig": 15000.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 6000.0,
    "isFlaggedFraud": 0
}])

# Prediction
prediction = model.predict(new_transaction)

if prediction[0] == 1:
    print("🚨 Fraudulent Transaction")
else:
    print("✅ Legitimate Transaction")
