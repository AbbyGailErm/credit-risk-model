# src/data_processing.py

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        df.groupby("CustomerId")
        .agg(
            total_transaction_amount=("Amount", "sum"),
            avg_transaction_amount=("Amount", "mean"),
            transaction_count=("TransactionId", "count"),
            std_transaction_amount=("Amount", "std"),
        )
        .reset_index()
    )

    return agg_df
def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year

    return df
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_time_features(df)
    agg_df = aggregate_customer_features(df)

    df_final = df.merge(agg_df, on="CustomerId", how="left")

    return df_final
NUMERIC_FEATURES = [
    "total_transaction_amount",
    "avg_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
]

CATEGORICAL_FEATURES = [
    "ProductCategory",
    "ChannelId",
    "ProviderId",
    "PricingStrategy",
]
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)
def build_feature_pipeline():
    return Pipeline(steps=[
        ("preprocessor", preprocessor)
    ])
def save_processed_data(X, path: str):
    pd.DataFrame(X).to_csv(path, index=False)
