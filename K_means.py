# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 23:23:02 2025

@author: Martian
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os
os.environ["OMP_NUM_THREADS"] = "1"

path="D:/OM_PhD_CUBoulder/Classes/25S_MachineLearning/M2/Clustering/DF_Cleaned_clustering.csv"
df = pd.read_csv(path)
columns = list(df.columns)  # Store column names in a list
print(columns)

# Extract features and labels
X = df[['Precipitation','Advance_Purchase','Temperature_Max']]

# Determine best k values using Silhouette Score
sil_scores = {}

# Testing k from 2 to 9
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    sil_scores[k] = silhouette_score(X, cluster_labels)

# Choose the top 3 k values with highest silhouette scores
best_k_values = sorted(sil_scores, key=sil_scores.get, reverse=True)[:3]

print(best_k_values)

# Plot data and centroids for each k
for k in best_k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df[f'cluster_k{k}'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    palette = ["#6528F7", "#7E2553", "#FF004D", "#379237", "#EA047E", "#45FFCA", "#071952"]
    palette = palette[:k]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter plot
    sns.scatterplot(x=df['Precipitation'], y=df['Advance_Purchase'], hue=df[f'cluster_k{k}'], palette=palette, alpha=0.6, ax=ax)
    
    # Scatter plot of cluster centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', label='Centroids')
    
    ax.set_title(f'K-Means Clustering (k={k})')
    ax.legend(title='Cluster', loc='lower right')
    plt.show()

