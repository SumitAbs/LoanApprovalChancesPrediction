import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

class TrainModel:
    def __init__(self, file_path, model_filename='loan_approval_model.pkl'):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model_filename = model_filename
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_and_preprocess_data(self):
        # Load dataset
        self.df = pd.read_csv(self.file_path)
        self.df.columns = self.df.columns.str.strip()  # Remove extra spacing in column name
        self.df.drop(columns=["loan_id"], inplace=True)  # Remove unwanted column

        # Encode categorical features
        for col in ["education", "self_employed", "loan_status"]:
            self.df[col] = self.df[col].str.strip()  # Remove extra spaces
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        # Define features and target
        X = self.df.drop(columns=["loan_status"])
        y = self.df["loan_status"]

        # Normalize numerical features
        X_scaled = self.scaler.fit_transform(X)
        return X, X_scaled, y

    def train_model(self, X, y):
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Train the model
        self.model.fit(X_train, y_train)

    def save_model(self):
        # Save the model, label encoders, and scaler to a single file
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler
        }, self.model_filename)
        print(f"Model and encoders saved successfully!")

# Usage to train and save the model
file_path = "LoanPredictionTraining.csv"  # Path to your dataset

# Step 1: Load data and train the model
loan_analysis = TrainModel(file_path)
X, X_scaled, y = loan_analysis.load_and_preprocess_data()
loan_analysis.train_model(X_scaled, y)

# Step 2: Save the trained model and everything else in one file
loan_analysis.save_model()
