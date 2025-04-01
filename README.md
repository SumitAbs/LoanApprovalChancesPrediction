# ![Loan](https://github.com/user-attachments/assets/367ce9ba-a852-4e3e-9e0a-0ddab2fdacec) Loan Approval Chances App

## 📌 Overview

   The Loan Approval Chances App is a machine learning-powered web application built using Streamlit. It helps users estimate the likelihood of loan approval based on financial and personal details. The application utilizes a Random Forest Classifier to analyze key factors such as income, loan amount, credit score, loan term, education, and self-employment status.

## 🚀 Features
    User-Friendly Interface: A simple and intuitive UI built with Streamlit.

    Real-time Predictions: Provides an instant probability of loan approval.

    Machine Learning Model: Uses a Random Forest Classifier trained on historical loan data.

    Data Preprocessing: Handles categorical encoding and numerical scaling for accurate predictions.

    Personalized Results: Displays approval chances along with user-submitted details.

## 🛠️ Installation

   To run the app on your local machine, follow these steps:
   
   1️⃣ **Install Dependencies**
   
      Make sure you have Python installed, then install the required libraries using:
      
      pip install streamlit pandas numpy scikit-learn
   
   2️⃣ **Run the Application** :
   
      Navigate to the project folder and start the Streamlit app:
      
      streamlit run loan_approval.py  (Replace loan_approval.py with the actual filename of your script if different.)
   
   3️⃣ **Access the App**
   
      After running the command, open the displayed localhost URL (e.g., http://localhost:8501) in your web browser.
      
## 🎯 How to Use
      Enter Personal Details : Fill in the required fields like income, loan amount, credit score, and loan term. 

      Submit the Form : Click the "Check Approval Chances" button. 

      View Results : The app will display the predicted approval probability along with a success or failure message. 
   
## 📈 Machine Learning Model
    Algorithm: Random Forest Classifier with 100 trees.

    Feature Engineering: Categorical encoding for education and self_employed, numerical scaling for financial values.

    Training Data: Historical loan application records from ML_Project_Q2.csv.
    
### How to install requirments on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run loan_approval.py
   ```
