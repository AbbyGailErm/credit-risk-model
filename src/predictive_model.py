# src/predictive_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_data(file_path="data/processed/rfm_clustered.csv"):
    """Load the clustered RFM dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist. Run Task 5 first.")
    df = pd.read_csv(file_path)
    return df

def prepare_features(df):
    """Prepare features (X) and target (y)."""
    X = df[['Recency', 'Frequency', 'Monetary', 'Cluster']]
    y = df['is_high_risk']
    return X, y

def train_model(X_train, y_train):
    """Train Random Forest classifier."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    """Evaluate model performance."""
    y_pred = clf.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def save_model(clf, file_path="data/processed/high_risk_model.pkl"):
    """Save trained model to disk."""
    joblib.dump(clf, file_path)
    print(f"Model saved as {file_path}")

def main():
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    save_model(clf)

if __name__ == "__main__":
    main()
