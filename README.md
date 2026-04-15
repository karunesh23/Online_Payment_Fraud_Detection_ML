# 💳 Online Payment Fraud Detection Using Machine Learning

## 📌 Project Overview
This project aims to detect **fraudulent online payment transactions** using Machine Learning.

The system analyzes transaction data and predicts whether a transaction is:

- ✅ Legitimate  
- ⚠️ Fraudulent  

The goal is to help financial institutions identify suspicious activities and prevent fraud.

---

## 📊 Dataset
Dataset used: **paysim_small.csv**

The dataset includes:
- Transaction type (TRANSFER, CASH_OUT, etc.)
- Transaction amount  
- Sender and receiver balances  
- Transaction patterns  

### 🎯 Target Variable:
- 0 → Legitimate Transaction  
- 1 → Fraudulent Transaction  

---

## 🛠 Technologies Used
- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- Joblib  

---

## 🤖 Model Used
- **Random Forest Classifier**
- Hyperparameter tuning using **RandomizedSearchCV**

### ⚙️ Training Details:
- Train-test split: 80-20  
- Stratified sampling  
- Cross-validation: 3-fold  
- Evaluation metric: F1-score  

---

## 📁 Saved Model
- `fraud_model.pkl` → Trained model for prediction  

---

## 📂 Project Structure
Online_Payment_Fraud_Detection_ML/
│
├── train_model.py # Model training script
├── predict.py # Prediction script
├── preprocess.py # Data preprocessing
├── reduce_dataset.py # Dataset optimization
├── paysim_small.csv # Dataset
├── requirements.txt # Dependencies
├── README.md # Project documentation
│
└── fraud_model.pkl # Trained model

---

## ⚙️ Workflow

1. Load and preprocess dataset  
2. Perform feature engineering  
3. Split dataset into training and testing sets  
4. Train model using Random Forest  
5. Perform hyperparameter tuning  
6. Evaluate model using classification metrics  
7. Save trained model for future predictions  

---

## 📈 Model Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  

*(Results are printed in the console after training)*

---

## 👨‍💻 Author

**Karunesh Bansal**

📧 **Email:** karuneshbansal84@gmail.com  

💼 **LinkedIn:**  
[https://www.linkedin.com/in/karunesh-bansal-780828380](https://www.linkedin.com/in/karunesh-bansal)

