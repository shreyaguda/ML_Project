import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
import numpy as np

def load_sector_data(filepath):
    #Load sector data from a CSV file into a pandas DataFrame.
    return pd.read_csv(filepath, header=None, names=['Symbol', 'Sector'], dtype={'Symbol': str})

def perform_pca_and_clustering(embeddings, n_clusters=4, n_components=4, max_clusters=15):
    #Perform PCA on embeddings and then apply KMeans and Spectral Clustering.
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    #Calculate SSD for a range of number of clusters to determine the optimal number
    ssd = []
    for k in range(1, max_clusters + 1):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(reduced_embeddings)
        ssd.append(kmeans_temp.inertia_)

    #Plotting the elbow curve to visualize the optimal number of clusters
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), ssd, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.grid(True)
    plt.savefig(f'elbow_method.png')

    #Perform both KMeans and Spectral Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(reduced_embeddings)
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, n_jobs=-1).fit(reduced_embeddings)
    
    return kmeans.labels_, spectral.labels_, reduced_embeddings

def calculate_silhouette_scores(reduced_embeddings, kmeans_labels, spectral_labels):
    #Calculate and return silhouette scores for KMeans and Spectral clustering.
    kmeans_score = silhouette_score(reduced_embeddings, kmeans_labels)
    spectral_score = silhouette_score(reduced_embeddings, spectral_labels)
    return kmeans_score, spectral_score

def evaluate_and_visualize_clusters(embeddings_df, reduced_embeddings, kmeans_labels, spectral_labels, sectors, directory):
    # Evaluate and visualize the clustering performance.
    # This function chooses the best clustering method based on silhouette scores, merges sector data,
    # calculates purity, and visualizes the clusters.
    kmeans_score, spectral_score = calculate_silhouette_scores(reduced_embeddings, kmeans_labels, spectral_labels)
    chosen_labels = kmeans_labels if kmeans_score >= spectral_score else spectral_labels
    chosen_method = 'KMeans' if kmeans_score >= spectral_score else 'Spectral'

    plot_silhouette(reduced_embeddings, chosen_labels, max(kmeans_score, spectral_score), chosen_method, directory)

    embeddings_df['cluster'] = chosen_labels
    print(f"Chosen Clustering Method: {chosen_method} with Silhouette Score: {max(kmeans_score, spectral_score)}")

    merged_df = merge_sectors(embeddings_df, sectors)
    purity = calculate_purity(merged_df)
    print(f"Clustering completed, average purity: {purity.mean()}")

    visualize_clusters(reduced_embeddings, chosen_labels, directory)

def merge_sectors(df, sector_df):
    #Merge sector data with the main dataframe based on the 'Symbol'.
    return df.merge(sector_df, on='Symbol', how='left')

def calculate_purity(df):
    #Calculate the purity of clusters based on the most common sector in each cluster.
    cluster_sector_count = df.groupby(['cluster', 'Sector']).size().unstack(fill_value=0)
    max_in_cluster = cluster_sector_count.max(axis=1)
    total_in_cluster = cluster_sector_count.sum(axis=1)
    return max_in_cluster / total_in_cluster

def plot_clusters(reduced_data, labels, file_name):
    #Plot and save a scatter plot of clusters.
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Cluster visualization with t-SNE')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.savefig(f'{file_name}.png')

def plot_elbow_method(data, max_clusters, file_path):
    ssd = []
    range_clusters = range(1, max_clusters + 1)
    for k in range_clusters:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        ssd.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range_clusters, ssd, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.grid(True)
    plt.savefig(file_path)
    plt.clf()

def visualize_clusters(reduced_data, labels, directory):
    #Perform t-SNE dimensionality reduction and plot the resulting clusters.
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_data = tsne.fit_transform(reduced_data)
    plot_filename = os.path.join(directory, 'cluster_visualization')
    plot_clusters(tsne_data, labels, plot_filename)

def plot_silhouette(reduced_data, labels, score, method, directory):
    # Plot and save a silhouette diagram to assess the quality of clustering.
    silhouette_vals = silhouette_samples(reduced_data, labels)
    y_lower, y_upper = 0, 0
    ticks = []

    # Plot silhouette values for each cluster
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ticks.append((y_lower + y_upper) / 2)
        y_lower += len(cluster_silhouette_vals)

    plt.axvline(x=score, color="red", linestyle="--")
    plt.yticks(ticks, [f'Cluster {i+1}' for i in range(len(ticks))])
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette Coefficient')
    plt.title(f'Silhouette Plot for {method}')
    plt.savefig(os.path.join(directory, f'{method}_silhouette.png'))
    plt.clf()

def main():
    # Main function to set up paths, load data, perform clustering, and visualize results.
    base_dir = os.getcwd()
    directory = os.path.join(base_dir, "parquet_output")
    sector_file = os.path.join(base_dir, 'sectors.csv')
    sectors = load_sector_data(sector_file)
    embeddings_file = os.path.join(directory, 'embeddings.parquet')

    all_embeddings_df = pd.read_parquet(embeddings_file)
    embeddings = all_embeddings_df.drop('Symbol', axis=1).values

    kmeans_labels, spectral_labels, reduced_embeddings = perform_pca_and_clustering(embeddings)
    evaluate_and_visualize_clusters(all_embeddings_df, reduced_embeddings, kmeans_labels, spectral_labels, sectors, directory)

if __name__ == "__main__":
    main()
