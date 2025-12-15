import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """
    Load the processed RFM + cluster + high-risk labeled data.
    Tries multiple possible filenames to avoid FileNotFoundError.
    """
    possible_files = [
        "data/processed/rfm_target.csv",
        "data/processed/rfm_prepared.csv"
    ]

    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            return pd.read_csv(file_path)

    raise FileNotFoundError(
        f"None of the expected files exist: {possible_files}. Run Task 4 first."
    )


def cluster_rfm(df, n_clusters=4):
    """
    Apply KMeans clustering to RFM data.
    """
    rfm_features = df[['Recency', 'Frequency', 'Monetary']].copy()

    # Optional: scale features if needed
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # rfm_scaled = scaler.fit_transform(rfm_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(rfm_features)

    return df


def plot_clusters(df):
    """
    Simple 2D plot of clusters (Recency vs Monetary).
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x='Recency', y='Monetary',
        hue='Cluster', palette='Set2', s=100
    )
    plt.title("RFM Clusters")
    plt.show()


def main():
    df = load_data()
    df = cluster_rfm(df)
    print("Clustered RFM dataset preview:")
    print(df.head())
    plot_clusters(df)

    # Save the clustered data
    output_file = "data/processed/rfm_clustered.csv"
    df.to_csv(output_file, index=False)
    print(f"Clustered dataset saved to: {output_file}")


if __name__ == "__main__":
    main()
