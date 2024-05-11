import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.metrics import silhouette_score
import pandas as pd

def load_sector_data(filepath):
    # Load sectors with Symbol as a string to ensure type consistency
    return pd.read_csv(filepath, header=None, names=['Symbol', 'Sector'], dtype={'Symbol': str})

def merge_sectors(df, sector_df):
    # Ensure both dataframes have the column used for merging
    return df.merge(sector_df, on='Symbol', how='left')

def plot_clusters(reduced_data, labels, file_name):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Cluster visualization with t-SNE')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.savefig(f'{file_name}.png')

def visualize_clusters(embeddings, labels, directory):
    """Generate a t-SNE visualization of the clusters."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_data = tsne.fit_transform(embeddings)
    plot_filename = os.path.join(directory, 'cluster_visualization.png')
    plot_clusters(reduced_data, labels, plot_filename)

def perform_clustering(embeddings, cluster_type="kmeans", n_clusters=11):
    if cluster_type == "kmeans":
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    elif cluster_type == "spectral":
        clustering_model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, n_jobs=-1)
    
    labels = clustering_model.fit_predict(embeddings)
    print(f"Labels from {cluster_type} clustering: {labels[:10]}")  # Print the first 10 labels to check
    return labels

def calculate_silhouette_scores(embeddings, labels):
    score = silhouette_score(embeddings, labels)
    return score

def evaluate_clusters(embeddings, embeddings_df, sectors, directory):
    """Calculate and print silhouette scores, calculate purity, and plot clusters."""
    if 'kmeans_cluster' not in embeddings_df.columns or 'spectral_cluster' not in embeddings_df.columns:
        print("Error: Clustering labels are not assigned to DataFrame.")
        return

    kmeans_silhouette = calculate_silhouette_scores(embeddings, embeddings_df['kmeans_cluster'])
    spectral_silhouette = calculate_silhouette_scores(embeddings, embeddings_df['spectral_cluster'])

    print(f"KMeans Silhouette Score: {kmeans_silhouette}")
    print(f"Spectral Silhouette Score: {spectral_silhouette}")

    # Choose the best clustering method based on silhouette score
    chosen_cluster_label = 'kmeans_cluster' if kmeans_silhouette >= spectral_silhouette else 'spectral_cluster'
    embeddings_df['cluster'] = embeddings_df[chosen_cluster_label]

    print(f"Chosen clustering label: {chosen_cluster_label}")  # Print the chosen label

    # Merge sectors and calculate purity
    all_embeddings_with_sectors = merge_sectors(embeddings_df, sectors)
    purity = calculate_purity(all_embeddings_with_sectors)
    print(f"Clustering completed, average purity: {purity.mean()}")

    # Visualization of clusters
    visualize_clusters(embeddings, embeddings_df['cluster'], directory)

def perform_clustering_and_evaluation(embeddings_file, sectors, directory):
    all_embeddings_df = pd.read_parquet(embeddings_file)
    embeddings = all_embeddings_df.drop('Symbol', axis=1).values

    # Performing Clustering
    kmeans_labels = perform_clustering(embeddings, "kmeans")
    spectral_labels = perform_clustering(embeddings, "spectral")
    all_embeddings_df['kmeans_cluster'] = kmeans_labels
    all_embeddings_df['spectral_cluster'] = spectral_labels

    # Ensuring clustering labels are present before evaluation
    if 'kmeans_cluster' not in all_embeddings_df.columns or 'spectral_cluster' not in all_embeddings_df.columns:
        print("Clustering labels are not assigned properly to DataFrame.")
        return

    all_embeddings_with_sectors = evaluate_clusters(embeddings, all_embeddings_df, sectors,directory)
    if all_embeddings_with_sectors is not None:
        print("Clustering and evaluation completed successfully.")

def calculate_purity(df):
    cluster_sector_count = df.groupby(['cluster', 'Sector']).size().unstack(fill_value=0)
    max_in_cluster = cluster_sector_count.max(axis=1)
    total_in_cluster = cluster_sector_count.sum(axis=1)
    purity = max_in_cluster / total_in_cluster
    return purity

def main():
    base_dir = os.getcwd()
    directory = os.path.join(base_dir, "parquet_output")
    sector_file = os.path.join(base_dir, 'sectors.csv')
    sectors = load_sector_data(sector_file)
    embeddings_file = os.path.join(directory, 'embeddings.parquet')
    perform_clustering_and_evaluation(embeddings_file, sectors, directory)


if __name__ == "__main__":
    main()