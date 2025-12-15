# run_pipeline.py
import os
import pandas as pd
from src.data_processing import prepare_model_data


def main():
    # Set the path to your transactions CSV
    csv_path = os.path.join("data", "raw", "transactions.csv")

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        print("Make sure your CSV is in 'data/raw/' folder.")
        return

    # Prepare the RFM dataset with clusters and high-risk labels
    df_rfm = prepare_model_data(csv_path)

    # Show first 5 rows for verification
    print("RFM dataset preview:")
    print(df_rfm.head())

    # Save the prepared dataset
    output_path = os.path.join("data", "processed", "rfm_prepared.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_rfm.to_csv(output_path, index=False)
    print(f"\nPrepared dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
