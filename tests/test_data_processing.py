# tests/test_data_processing.py
import pytest
import pandas as pd
from src.data_processing import calculate_rfm, assign_high_risk_label


@pytest.fixture
def sample_data():
    """Sample transaction data for testing."""
    return pd.DataFrame({
        'TransactionId': [1, 2, 3, 4, 5, 6],
        'CustomerId': [1, 1, 2, 2, 3, 3],
        'Amount': [100, 150, 50, 300, 300, 150],
        'TransactionStartTime': pd.to_datetime([
            '2025-12-01', '2025-12-02', '2025-12-01',
            '2025-12-03', '2025-12-08', '2025-12-09'
        ])
    })


def test_calculate_rfm(sample_data):
    """Test that RFM calculation works correctly."""
    snapshot = pd.Timestamp("2025-12-15")
    rfm = calculate_rfm(sample_data, snapshot_date=snapshot)

    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
    # Customer 1 last transaction: 2025-12-02, snapshot: 2025-12-15 => 13 days
    assert rfm.loc[rfm['CustomerId'] == 1, 'Recency'].values[0] == 13
    # Frequency sums
    assert rfm.loc[rfm['CustomerId'] == 2, 'Frequency'].values[0] == 2
    # Monetary sums
    assert rfm.loc[rfm['CustomerId'] == 3, 'Monetary'].values[0] == 450


def test_assign_high_risk_label():
    """Test high-risk assignment after clustering."""
    df = pd.DataFrame({
        'CustomerId': [1, 2, 3],
        'Recency': [5, 10, 2],
        'Frequency': [2, 1, 3],
        'Monetary': [250, 100, 450],
        'Cluster': [0, 0, 1]
    })
    labeled = assign_high_risk_label(df)

    assert 'is_high_risk' in labeled.columns
    assert labeled.loc[labeled['CustomerId'] == 3, 'is_high_risk'].values[0] == 1
    assert labeled.loc[labeled['CustomerId'] == 1, 'is_high_risk'].values[0] == 0
