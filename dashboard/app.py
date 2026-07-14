import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_v1.pkl')
model = joblib.load(model_path)

st.title("Loan Default / Approval Risk Predictor")

income = st.number_input("Applicant Income", min_value=0, value=5000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=120)
loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60])
credit_history = st.selectbox("Credit History", [1, 0])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
married = st.selectbox("Married", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    total_income = income + coapplicant_income
    emi = loan_amount / loan_term
    balance_income = (total_income / 12) - emi

    input_dict = {
        'Gender': 1 if gender == "Male" else 0,
        'Married': 1 if married == "Yes" else 0,
        'Dependents': dependents,
        'Education': 1 if education == "Graduate" else 0,
        'Self_Employed': 1 if self_employed == "Yes" else 0,
        'ApplicantIncome': income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'TotalIncome': total_income,
        'LoanAmount_to_Income': loan_amount / total_income,
        'EMI': emi,
        'BalanceIncome': balance_income,
        'ApplicantIncome_log': np.log1p(income),
        'TotalIncome_log': np.log1p(total_income),
        'LoanAmount_log': np.log1p(loan_amount),
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.write(f"### Approval Probability: {prob:.2%}")
    st.write("### Result:", "✅ Likely Approved" if prediction == 1 else "❌ Likely Rejected / High Risk")