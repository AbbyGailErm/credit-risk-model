# tests/test_target_engineering.py
import pytest
import pandas as pd
from src.data_processing import calculate_rfm, assign_high_risk_label, prepare_model_data


@pytest.fixture
def sample_data():
    """Sample transaction data for testing."""
    return pd.DataFrame({
        'TransactionId': [1, 2, 3, 4, 5, 6],
        'CustomerId': [101, 101, 102, 103, 103, 103],
        'Amount': [100, 200, 50, 300, 150, 50],
        'TransactionStartTime': pd.to_datetime([
            '2024-01-25', '2024-01-26', '2024-01-28',
            '2024-01-29', '2024-01-30', '2024-01-31'
        ])
    })


def test_calculate_rfm(sample_data):
    """Test RFM calculation."""
    snapshot = pd.Timestamp("2024-02-01")
    rfm = calculate_rfm(sample_data, snapshot_date=snapshot)

    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns


def test_assign_high_risk(sample_data):
    """Test high-risk assignment after clustering."""
    snapshot = pd.Timestamp("2024-02-01")
    rfm = calculate_rfm(sample_data, snapshot_date=snapshot)

    # Dummy clustering
    rfm['Cluster'] = pd.qcut(rfm['Monetary'], q=2, labels=False, duplicates='drop')
    rfm = assign_high_risk_label(rfm)

    assert 'is_high_risk' in rfm.columns
    assert rfm['is_high_risk'].sum() > 0  # at least one high-risk


def test_prepare_model_data_file_exists(sample_data, tmp_path):
    """Test prepare_model_data with CSV file input."""
    file_path = tmp_path / "transactions.csv"
    sample_data.to_csv(file_path, index=False)

    rfm_target = prepare_model_data(file_path)

    assert 'Cluster' in rfm_target.columns
    assert 'is_high_risk' in rfm_target.columns
    assert not rfm_target.empty
