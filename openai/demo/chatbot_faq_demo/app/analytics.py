import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast
import streamlit as st

def analysis():
    # Step 1: Load the CSV file and extract the embeddings
    df = pd.read_csv('data/questions_embeddings.csv')

    # Convert embeddings from strings to lists of floats
    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x))

    # Convert the list of embeddings to a numpy array
    embeddings = np.stack(df['embedding'].values)

    # Step 2: Cluster the embeddings using KMeans
    num_clusters = 3  # Set the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

    # Assign the cluster labels back to the DataFrame
    df['cluster'] = kmeans.labels_

    # Step 3: Reduce dimensionality for visualization
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cluster with a different color
    colors = ['r', 'g', 'b']  # Set the colors for the clusters
    for i in range(num_clusters):
        cluster_i = embeddings_3d[df['cluster'] == i]
        ax.scatter(cluster_i[:, 0], cluster_i[:, 1], cluster_i[:, 2], c=colors[i])
    return fig, df, colors, num_clusters

