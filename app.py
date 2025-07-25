import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page settings
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Custom dark theme styling
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
        }
        h1, h2, h3, h4 {
            color: #F39C12;
        }
        .title {
            font-size: 40px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .input-text {
            font-size: 20px !important;
        }
        .css-1v0mbdj button {
            background-color: #F39C12;
            color: black;
        }
        .predict-button {
            background-color: #F39C12;
            color: black;
            font-weight: bold;
            font-size: 20px;
            border-radius: 8px;
            padding: 0.5em 2em;
            margin-top: 1em;
        }
        .footer {
            font-size: 16px;
            color: #cccccc;
            text-align: center;
            margin-top: 3em;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            padding: 1em;
            border-radius: 10px;
        }
        .approved {
            background-color: #27ae60;
            color: white;
        }
        .denied {
            background-color: #c0392b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title section
st.markdown('<div class="title">🏦 Loan Approval Prediction App</div>', unsafe_allow_html=True)
st.markdown("### 📋 Enter your loan details below")

# Input form
credit_score = st.number_input("📊 Credit Score", step=1)
loan_term = st.number_input("📆 Loan Term (in months)", step=1)
income_annum = st.number_input("💰 Annual Income (in ₹)", step=1)
luxury_assets_value = st.number_input("🚗 Luxury Assets Value (in ₹)", step=1)

# Predict button
if st.button("🔍 Predict", type="primary"):
    # Apply flooring rule (business logic outside ML model)
    if income_annum < 500000 and luxury_assets_value < 200000:
        st.markdown('<div class="result denied">❌ Auto-Rejected: Income and Asset values are below minimum criteria.</div>', unsafe_allow_html=True)
    else:
        input_data = np.array([[credit_score, loan_term, income_annum, luxury_assets_value]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.markdown('<div class="result approved">✅ Congratulations! Your loan is likely to be approved.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result denied">❌ Sorry, your loan is not likely to be approved.</div>', unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div class="footer">👨‍💻 Developed by <a href="https://github.com/Muhammad-Ghulam-Ali" style="color: #F39C12;" target="_blank">Muhammad Ghulam Ali</a></div>',
    unsafe_allow_html=True
)

