import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class TrainLoanPredictionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(random_state=42)

    def load_and_preprocess_data(self):
        # Load CSV
        self.df = pd.read_csv(self.file_path)
        
        # Strip spaces from column names
        self.df.columns = self.df.columns.str.strip()
        
        # Drop 'loan_id' as it's not useful for prediction
        self.df.drop(columns=["loan_id"], inplace=True)

        # Encoding categorical variables and storing encoded labels
        for col in ["education", "self_employed", "loan_status"]:
            self.df[col] = self.df[col].str.strip()
            le = LabelEncoder()
            self.df[f"{col}_encoded"] = le.fit_transform(self.df[col])  # Store encoded labels
            self.df[col] = le.transform(self.df[col])  # Keep encoded values in original column
            self.label_encoders[col] = le

        # Features and target variable
        X = self.df.drop(columns=["loan_status"])
        y = self.df["loan_status"]

        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        return X, X_scaled, y

    def train_model(self, X, X_scaled, y):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Save model, scaler, label encoders, and feature columns
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": list(X.columns)  # Save feature names correctly
        }
        model_path = "loan_approval_model.pkl"
        with open(model_path, "wb") as file:
            pickle.dump(model_data, file)

        return model_path

# Train the model using the fixed logic
file_path = "LoanPredictionTraining.csv"
model_fixed = TrainLoanPredictionModel(file_path)
X, X_scaled, y = model_fixed.load_and_preprocess_data()
saved_model_path_fixed = model_fixed.train_model(X, X_scaled, y)
saved_model_path_fixed


