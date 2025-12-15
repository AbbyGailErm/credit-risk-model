# src/target_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def calculate_rfm(df: pd.DataFrame, snapshot_date: str = None) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, Monetary (RFM) for each customer.

    Parameters:
        df : pd.DataFrame
            Transaction-level data with CustomerId, Amount, TransactionStartTime
        snapshot_date : str (optional)
            Date to compute recency against. Default: last transaction date in data.

    Returns:
        rfm_df : pd.DataFrame
            Customer-level RFM metrics
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    # Recency = days since last transaction
    recency = df.groupby('CustomerId')['TransactionStartTime'].max().reset_index()
    recency['Recency'] = (snapshot_date - recency['TransactionStartTime']).dt.days

    # Frequency = count of transactions
    frequency = df.groupby('CustomerId')['TransactionId'].count().reset_index()
    frequency.rename(columns={'TransactionId': 'Frequency'}, inplace=True)

    # Monetary = total transaction amount
    monetary = df.groupby('CustomerId')['Amount'].sum().reset_index()
    monetary.rename(columns={'Amount': 'Monetary'}, inplace=True)

    # Merge
    rfm_df = recency.merge(frequency, on='CustomerId').merge(monetary, on='CustomerId')
    rfm_df = rfm_df[['CustomerId', 'Recency', 'Frequency', 'Monetary']]

    return rfm_df
def scale_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    return pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm_df['CustomerId'])
def cluster_customers(rfm_scaled: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm_scaled['Cluster'] = clusters
    return rfm_scaled
def assign_high_risk_label(rfm_scaled: pd.DataFrame) -> pd.DataFrame:
    cluster_centers = rfm_scaled.groupby('Cluster').mean()
    high_risk_cluster = cluster_centers['Recency'].idxmax()
    rfm_scaled['is_high_risk'] = (rfm_scaled['Cluster'] == high_risk_cluster).astype(int)
    return rfm_scaled[['is_high_risk']]
def integrate_target(df: pd.DataFrame, rfm_target: pd.DataFrame) -> pd.DataFrame:
    df_final = df.merge(rfm_target, left_on='CustomerId', right_index=True, how='left')
    return df_final
# Example usage
from src.target_engineering import calculate_rfm, scale_rfm, cluster_customers, assign_high_risk_label, integrate_target

# 1. Load raw transaction data
df = pd.read_csv("data/raw/transactions.csv")

# 2. Calculate RFM
rfm_df = calculate_rfm(df)

# 3. Scale RFM
rfm_scaled = scale_rfm(rfm_df)

# 4. Cluster customers
rfm_scaled = cluster_customers(rfm_scaled)

# 5. Assign high-risk label
rfm_target = assign_high_risk_label(rfm_scaled)

# 6. Merge with main data
df_final = integrate_target(df, rfm_target)
