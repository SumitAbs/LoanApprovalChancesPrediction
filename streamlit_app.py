import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your pre-trained model and functions
class LoanApprovalAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_and_preprocess_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df.columns = self.df.columns.str.strip()
        self.df.drop(columns=["loan_id"], inplace=True)

        # Encoding categorical variables
        for col in ["education", "self_employed", "loan_status"]:
            self.df[col] = self.df[col].str.strip()  
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        X = self.df.drop(columns=["loan_status"])
        y = self.df["loan_status"]

        X_scaled = self.scaler.fit_transform(X)
        return X, X_scaled, y

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def predict_user_input(self, user_data):
        # Encode categorical fields (convert string values to numeric encoding)
        user_data["education"] = self.label_encoders["education"].transform([user_data["education"]])[0]
        user_data["self_employed"] = self.label_encoders["self_employed"].transform([user_data["self_employed"]])[0]

        # Convert user data to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Ensure the column order matches the trained model
        missing_cols = set(self.df.columns) - set(user_df.columns) - {"loan_status"}
        for col in missing_cols:
            user_df[col] = 0  # Add missing columns with default value 0

        user_df = user_df[self.df.drop(columns=["loan_status"]).columns]  # Reorder columns to match training data
        user_df_scaled = self.scaler.transform(user_df)  # Scale the features

        approval_chance = self.model.predict_proba(user_df_scaled)[:, 0][0] * 100
        return approval_chance


# Streamlit UI
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Enter your details below to check your loan approval chances.")

# Input Form for User
with st.form("loan_form", clear_on_submit=False):
    # Ensure all number inputs are consistent types (float where needed)
    income = st.number_input("💰 Applicant Income ($)", min_value=0.0, step=100.0)
    loan_amount = st.number_input("🏠 Loan Amount ($)", min_value=0.0, step=100.0)
    credit_score = st.slider("📊 CIBIL Score", min_value=300, max_value=850, step=10)
    loan_term = st.number_input("📆 Loan Term (in months)", min_value=1, max_value=360, step=1)
    luxury_assets_value = st.number_input("💎 Luxury Assets Value ($)", min_value=0.0, step=100.0)

    # User selects Education and Self Employed Status (String Input)
    education = st.selectbox("🎓 Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("💼 Are you self-employed?", ["No", "Yes"])
    
    submitted = st.form_submit_button("🔍 Check Approval Chances")

# Display Result
if submitted:
    # Collect the inputs in a dictionary to match your backend's expectations
    user_data = {
        "cibil_score": credit_score,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "luxury_assets_value": luxury_assets_value,
        "education": education,  # Keep as strings
        "self_employed": self_employed  # Keep as strings
    }

    # Instantiate the LoanApprovalAnalysis class (assumes CSV path is correct)
    file_path = "ML_Project_Q2.csv"
    loan_analysis = LoanApprovalAnalysis(file_path)
    X, X_scaled, y = loan_analysis.load_and_preprocess_data()
    loan_analysis.train_model(X_scaled, y)

    # Get the prediction
    approval_chance = loan_analysis.predict_user_input(user_data)
    # Display the result to the user
    st.subheader(f"Chances of Loan Approval: {approval_chance:.2f}%")
    if approval_chance > 65:
        st.success("You are likely to be approved!")
    else:
        st.error("Unfortunately, your loan application may not be approved.")
    
    # Display the user inputs for confirmation
    st.write("### Your Details:")
    st.write(f"Income: ${income}")
    st.write(f"Loan Amount: ${loan_amount}")
    st.write(f"CIBIL Score: {credit_score}")
    st.write(f"Loan Term: {loan_term} months")
    st.write(f"Luxury Assets Value: ${luxury_assets_value}")
    
    # Decode the encoded values back into strings for display
    education_display = "Graduate" if user_data["education"] == "Graduate" else "Not Graduate"
    self_employed_display = "Yes" if user_data["self_employed"] == "Yes" else "No"
    
    st.write(f"Education: {education_display}")
    st.write(f"Self Employed: {self_employed_display}")
