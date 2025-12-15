import pandas as pd
from src.modeling import cluster_rfm


def test_cluster_rfm():
    rfm = pd.DataFrame(
        {
            "CustomerId": [1, 2, 3, 4],
            "Recency": [1, 10, 5, 20],
            "Frequency": [5, 1, 3, 1],
            "Monetary": [500, 50, 200, 20],
        }
    )

    clustered = cluster_rfm(rfm, n_clusters=2)

    assert "Cluster" in clustered.columns
    assert clustered["Cluster"].nunique() == 2
