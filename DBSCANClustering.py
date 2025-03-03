# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 15:33:09 2025

@author: Martian
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Load Data
path = "D:/OM_PhD_CUBoulder/Classes/25S_MachineLearning/M2/Clustering/DF_Cleaned_clustering.csv"
df = pd.read_csv(path)
columns = list(df.columns)  # Store column names in a list
print(columns)

# Select Features for Clustering
X = df[['Precipitation', 'Advance_Purchase', 'Temperature_Max']]

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Best Score
best_score = -1
best_params = {}

# DBSCAN Parameter Search
for eps in np.arange(0.1, 3.0, 0.1):  # Adjusted range for scaled data
    for min_samples in range(2, 10):  # Vary MinPts
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = clustering.labels_

        # Ignore cases where all points are in one cluster (-1 means noise)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}

# Print Best Parameters
print(f"Best Params: {best_params} with Silhouette Score: {best_score}")

####################Best Params: {'eps': 0.5, 'min_samples': 5} with Silhouette Score: 0.1732753743605445

# Apply PCA (Optional: for visualization)
pca = PCA(n_components=3)
df[['PC0', 'PC1', 'PC2']] = pca.fit_transform(X_scaled)

# DBSCAN Clustering
eps = 0.2
min_samples = 7
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)  # Use scaled data

# Copy Data for Display
display_df = df.copy()

# Print Number of Clusters (Excluding Noise)
num_clusters = len(set(df['dbscan_cluster'])) - (1 if -1 in df['dbscan_cluster'] else 0)
print(f"Number of clusters found: {num_clusters}")

# Plot DBSCAN Clustering: PC0 vs PC1
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df['PC0'], y=df['PC1'], hue=df['dbscan_cluster'], palette='Set2', alpha=0.6)
plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
plt.xlabel('PC0')
plt.ylabel('PC1')
plt.legend(title='Cluster', loc='upper right', bbox_to_anchor=(1.20, 1), ncol=2)
plt.show()


###############Optional
# Apply PCA for visualization
pca = PCA(n_components=2)
df[['PC1', 'PC2']] = pca.fit_transform(df[['Advance_Purchase', 'Nightly_Rate', 'Temperature_Max', 'Temperature_Min', 'Precipitation', 'Snowfall', 'Wind_Speed']])

# Plot Clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df['PC1'], y=df['PC2'], hue=df['dbscan_cluster'], palette='Set2', alpha=0.7)
plt.title(f'DBSCAN Clustering (eps={0.5}, min_samples={5})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

print(df.groupby('dbscan_cluster')[['Advance_Purchase', 'Nightly_Rate', 'Temperature_Max', 'Temperature_Min', 'Precipitation', 'Snowfall', 'Wind_Speed']].mean())