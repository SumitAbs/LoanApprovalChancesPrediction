import streamlit as st
import pickle
import pandas as pd

class LoanApprovalPrediction:
    def __init__(self, model_filename='loan_approval_model.pkl'):
        with open(model_filename, "rb") as file:
            model_data = pickle.load(file)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']

    def predict_user_input(self, user_data):
        user_data["education"] = self.label_encoders["education"].transform([user_data["education"]])[0]
        user_data["self_employed"] = self.label_encoders["self_employed"].transform([user_data["self_employed"]])[0]

        user_df = pd.DataFrame([user_data])
        missing_cols = set(self.feature_columns) - set(user_df.columns)
        for col in missing_cols:
            user_df[col] = 0

        user_df = user_df[self.feature_columns]
        user_df_scaled = self.scaler.transform(user_df)
        approval_chance = self.model.predict_proba(user_df_scaled)[:, 0][0] * 100
        return approval_chance

# Streamlit UI
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Predictor.")
st.write("Enter your details below to check your loan approval chances.")

# User input form
with st.form("loan_form", clear_on_submit=False):
    income = st.number_input("💰 Applicant Income ($)", min_value=0.0, step=100.0)
    loan_amount = st.number_input("🏠 Loan Amount ($)", min_value=0.0, step=100.0)
    credit_score = st.slider("📊 CIBIL Score", min_value=300, max_value=850, step=10)
    loan_term = st.number_input("📆 Loan Term (in months)", min_value=1, max_value=360, step=1)
    luxury_assets_value = st.number_input("💎 Luxury Assets Value ($)", min_value=0.0, step=100.0)
    education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("💼 Are you self-employed?", ["No", "Yes"])
    submitted = st.form_submit_button("🔍 Check Approval Chances")

if submitted:
    user_data = {
        "cibil_score": credit_score,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "luxury_assets_value": luxury_assets_value,
        "education": education,
        "self_employed": self_employed
    }
    
    loan_prediction = LoanApprovalPrediction("loan_approval_model.pkl")
    approval_chance = loan_prediction.predict_user_input(user_data)
    
    st.subheader(f"Chances of Loan Approval: {approval_chance:.2f}%")
    if approval_chance > 65:
        st.success("You are likely to be approved!")
    else:
        st.error("Unfortunately, your loan application may not be approved.")
    
    st.write("### Your Details:")
    st.write(f"Income: ${income}")
    st.write(f"Loan Amount: ${loan_amount}")
    st.write(f"CIBIL Score: {credit_score}")
    st.write(f"Loan Term: {loan_term} months")
    st.write(f"Luxury Assets Value: ${luxury_assets_value}")
    st.write(f"Education: {education}")
    st.write(f"Self Employed: {self_employed}")
