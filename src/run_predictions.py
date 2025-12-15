import pandas as pd
import joblib
from data_processing import prepare_model_data  # from Task 4

# 1. Load trained model
model = joblib.load("data/processed/high_risk_model.pkl")

# 2. Load new customer transaction data
new_data = pd.read_csv("data/raw/transactions.csv")
# 3. Prepare RFM + cluster features
new_rfm = prepare_model_data(new_data)

# 4. Make predictions
X_new = new_rfm.drop(columns=["CustomerId", "TransactionId", "is_high_risk"], errors="ignore")
new_rfm["predicted_risk"] = model.predict(X_new)

# 5. Save predictions
new_rfm.to_csv("data/processed/new_rfm_predictions.csv", index=False)
print("Predictions saved to data/processed/new_rfm_predictions.csv")
