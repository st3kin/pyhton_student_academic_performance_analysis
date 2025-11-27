import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Loading the data

df = pd.read_csv('CSV_files/processed_df.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Building cluster dataset

numeric_features = [
    'Hours_Studied',
    'Attendance',
    'Sleep_Hours',
    'Physical_Activity',
    'Tutoring_Sessions'
]

ordinal_features = [
    'Motivation_Level',
    'Parental_Involvement',
    'Access_to_Resources',
    'Peer_Influence'
]

binary_features = [
    'Internet_Access',
    'Extracurricular_Activities'
]

cluster_features = (
    numeric_features +
    ordinal_features +
    binary_features
)

cluster_df = df[cluster_features].copy()

ordinal_mappings = {
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Peer_Influence': ['Negative', 'Neutral', 'Positive']
}

for col, order in ordinal_mappings.items():
    cluster_df[col] = pd.Categorical(
        cluster_df[col],
        categories=order,
        ordered=True
    ).codes

cluster_df[binary_features] = cluster_df[binary_features].astype(int)

# Scaling features

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_df)

# Elbow

inertias = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(cluster_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), inertias, marker='o')
plt.xlabel('Number of clusters(k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.tight_layout()
plt.show()

# Silhouette

sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(cluster_scaled)
    sil = silhouette_score(cluster_scaled, labels)
    sil_scores.append(sil)
    print(f"k={k}: silhouette={sil:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()


# Fitting k

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels = kmeans.fit_predict(cluster_scaled)

cluster_df_with_labels = cluster_df.copy()
cluster_df_with_labels['Cluster'] = labels

df['Cluster'] = labels

cluster_profile = cluster_df_with_labels.groupby('Cluster')[cluster_features].mean()
print("Behavioural feature means by cluster:")
print(cluster_profile)

score_profile = df.groupby('Cluster')['Exam_Score'].mean()
print("\nMean exam score by cluster:")
print(score_profile)

level_profile = df.groupby('Cluster')['Exam_Score_Level'].value_counts(normalize=True)
print("\nExam score level distribution by cluster (proportions):")
print(level_profile)

# PCA scatter plot

pca = PCA(n_components=2)
components = pca.fit_transform(cluster_scaled)

pc1 = components[:, 0]
pc2 = components[:, 1]

plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    pc1, pc2,
    c=df['Cluster'],
    cmap='viridis',
    alpha=0.6,
    edgecolors='k',
    linewidths=0.3
)
plt.title('K-Means Clustering (PCA Projection)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')

handles, labels = scatter.legend_elements()
plt.legend(handles, [f'Cluster {i}' for i in range(len(handles))], title="Clusters")
plt.tight_layout()
plt.show()

'''
K-means clustering was explored to determine whether students naturally grouped into distinct behavioural profiles. However, 
silhouette scores were very low (maximum ≈ 0.09), PCA separation was minimal, and cluster boundaries showed heavy overlap. 
These results indicate that students do not form meaningful discrete clusters in this dataset. 
'''

