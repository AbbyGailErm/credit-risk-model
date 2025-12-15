import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, Monetary (RFM) metrics per CustomerId.
    """
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum"),
        )
        .reset_index()
    )

    return rfm


def scale_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize RFM features for clustering.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    return pd.DataFrame(
        rfm_scaled,
        columns=["Recency", "Frequency", "Monetary"],
        index=rfm.index,
    )


def cluster_customers(rfm_scaled: pd.DataFrame, n_clusters: int = 3) -> np.ndarray:
    """
    Cluster customers using KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(rfm_scaled)


def assign_high_risk_label(rfm: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    """
    Assign high-risk label based on least engaged cluster.
    """
    rfm = rfm.copy()
    rfm["cluster"] = clusters

    cluster_summary = (
        rfm.groupby("cluster")[["Frequency", "Monetary"]].mean()
    )

    high_risk_cluster = cluster_summary.sum(axis=1).idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]


def integrate_target(df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge target variable back into main dataset.
    """
    return df.merge(target_df, on="CustomerId", how="left")
