# src/data_processing.py
import pandas as pd
from pathlib import Path


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """Calculate RFM metrics for each customer."""
    df = df.copy()

    # Ensure TransactionStartTime is datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Default snapshot_date
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

    # Group by CustomerId and aggregate
    rfm = df.groupby('CustomerId', as_index=False).agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    )
    return rfm


def assign_high_risk_label(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign high-risk label to customers in the highest cluster."""
    rfm = rfm.copy()
    if 'Cluster' not in rfm.columns:
        # If no cluster, assign 0 as default
        rfm['Cluster'] = 0

    high_risk_cluster = rfm['Cluster'].max()
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm


def prepare_model_data(input_data) -> pd.DataFrame:
    """Prepare RFM dataset with clusters and high-risk labels for modeling."""
    # Load data if input is path
    if isinstance(input_data, (str, Path)):
        df = pd.read_csv(input_data, parse_dates=['TransactionStartTime'])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a CSV path or a pandas DataFrame")

    # Calculate RFM
    rfm = calculate_rfm(df)

    # Dummy clustering: 2 clusters based on Monetary
    if rfm['Monetary'].nunique() > 1:
        rfm['Cluster'] = pd.qcut(rfm['Monetary'], q=2, labels=False)
    else:
        rfm['Cluster'] = 0

    # Assign high-risk label
    rfm = assign_high_risk_label(rfm)

    return rfm
