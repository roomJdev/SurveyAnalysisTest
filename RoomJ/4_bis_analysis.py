
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from math import pi
import os

# Load data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Create output directory
os.makedirs("outputs/BIS", exist_ok=True)

# Define BIS subscales
bis_subscales = {
    "Attentional": "BIS_Attentional",
    "Motor": "BIS_Motor",
    "NonPlanning": "BIS_NonPlanning"
}
bis_cols = list(bis_subscales.values())

# Descriptive stats
desc = df[bis_cols].describe().round(2)
desc.to_csv("outputs/BIS/bis_descriptive_stats.csv")

# Histograms and Boxplots
for col in bis_cols:
    plt.figure()
    sns.histplot(df[col], bins=10)
    plt.title(f"Histogram - {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/BIS/histogram_{col}.png")
    plt.close()

    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/BIS/boxplot_{col}.png")
    plt.close()

# Cronbach's alpha (using all BIS subscale items, if available)
# Here assuming subscale items are not available. Skip or adapt if item-level data is added.

# Subscale correlation
corr = df[bis_cols].corr(method='pearson').round(2)
corr.to_csv("outputs/BIS/bis_subscale_correlation.csv")

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("BIS Subscale Correlation")
plt.tight_layout()
plt.savefig("outputs/BIS/bis_correlation_heatmap.png")
plt.close()

# Radar chart (mean profile)
values = df[bis_cols].mean().tolist()
values += values[:1]
angles = [n / float(len(bis_cols)) * 2 * pi for n in range(len(bis_cols))]
angles += angles[:1]
plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(list(bis_subscales.keys()))
plt.title("BIS Radar Chart (Mean Profile)")
plt.tight_layout()
plt.savefig("outputs/BIS/bis_radar_chart.png")
plt.close()

# KMeans clustering
X = StandardScaler().fit_transform(df[bis_cols])
kmeans = KMeans(n_clusters=3, random_state=42)
df['BIS_KMeans_Cluster'] = kmeans.fit_predict(X)
sil_score = silhouette_score(X, df['BIS_KMeans_Cluster'])

# Cluster profile (k-means)
profile = df.groupby('BIS_KMeans_Cluster')[bis_cols].mean().round(2)
profile.to_csv("outputs/BIS/bis_kmeans_cluster_profile.csv")

plt.figure(figsize=(6,4))
sns.heatmap(profile, annot=True, cmap="YlGnBu")
plt.title("BIS KMeans Cluster Profiles")
plt.tight_layout()
plt.savefig("outputs/BIS/bis_kmeans_cluster_heatmap.png")
plt.close()

# Cluster scatter plot (k-means)
plt.figure(figsize=(6,5))
sns.scatterplot(x=df["BIS_Motor"], y=df["BIS_NonPlanning"], hue=df["BIS_KMeans_Cluster"], palette="tab10")
plt.title("BIS KMeans Cluster Plot (Motor vs NonPlanning)")
plt.tight_layout()
plt.savefig("outputs/BIS/bis_kmeans_cluster_scatter.png")
plt.close()

# Hierarchical clustering
linked = linkage(X, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("BIS Hierarchical Clustering Dendrogram")
plt.tight_layout()
plt.savefig("outputs/BIS/bis_hierarchical_dendrogram.png")
plt.close()

# Assign hierarchical clusters (3)
df["BIS_Hier_Cluster"] = fcluster(linked, 3, criterion='maxclust')
hier_profile = df.groupby("BIS_Hier_Cluster")[bis_cols].mean().round(2)
hier_profile.to_csv("outputs/BIS/bis_hierarchical_cluster_profile.csv")
# ============================================
# 1. Correlation with Other Psychological Scales
# ============================================

related_cols = ["PQ_Total", "SSS_Total", "MACHI_Total", "VRSQ_Total", "SSQ_Standardized"]
correlations = []

if "BIS_Total" not in df.columns:
    df["BIS_Total"] = df[bis_cols].sum(axis=1)

for scale in related_cols:
    if scale in df.columns:
        r, p = pearsonr(df["BIS_Total"], df[scale])
        correlations.append([scale, r, p])

corr_df = pd.DataFrame(correlations, columns=["Scale", "Pearson_r", "p_value"])
corr_df.to_csv("outputs/BIS/bis_correlation_with_other_scales.csv", index=False)

# ============================================
# 2. PCA Visualization
# ============================================

from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(df[bis_cols])
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["BIS_KMeans_Cluster"]

plt.figure(figsize=(6, 5))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="tab10", data=pca_df)
plt.title("PCA of BIS Subscales by KMeans Cluster")
plt.tight_layout()
plt.savefig("outputs/BIS/bis_pca_kmeans_cluster.png")
plt.close()
# ============================================
# 3. BIS 클러스터별 레이더 차트 (KMeans 기준)
# ============================================
angles = [n / float(len(bis_cols)) * 2 * pi for n in range(len(bis_cols))]
angles += angles[:1]

for cluster_id in profile.index:
    values = profile.loc[cluster_id].tolist() + [profile.loc[cluster_id].tolist()[0]]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(bis_subscales.keys()))
    plt.title(f"BIS Radar Chart - KMeans Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(f"outputs/BIS/bis_radar_kmeans_cluster_{cluster_id}.png")
    plt.close()
# ============================================
# 4. BIS 상하위 25% 그룹 간 비교
# ============================================
q75 = df["BIS_Total"].quantile(0.75)
q25 = df["BIS_Total"].quantile(0.25)

high_group = df[df["BIS_Total"] >= q75]
low_group = df[df["BIS_Total"] <= q25]

with open("outputs/BIS/bis_high_low_comparison.txt", "w") as f:
    f.write("=== High (Top 25%) vs Low (Bottom 25%) BIS_Total Comparison ===\n")
    for col in bis_cols:
        t_stat, p_val = ttest_ind(high_group[col], low_group[col])
        f.write(f"{col}: t={t_stat:.2f}, p={p_val:.3f}\n")
# ============================================
# 5. BIS 계층적 클러스터 간 t-test 분석
# ============================================

from itertools import combinations

with open("outputs/BIS/bis_hierarchical_cluster_ttest.txt", "w") as f:
    f.write("=== Hierarchical Cluster Pairwise t-tests ===\n")
    cluster_ids = df["BIS_Hier_Cluster"].unique()
    for c1, c2 in combinations(cluster_ids, 2):
        f.write(f"\n--- Cluster {c1} vs Cluster {c2} ---\n")
        group1 = df[df["BIS_Hier_Cluster"] == c1]
        group2 = df[df["BIS_Hier_Cluster"] == c2]
        for col in bis_cols:
            t_stat, p_val = ttest_ind(group1[col], group2[col])
            f.write(f"{col}: t={t_stat:.2f}, p={p_val:.3f}\n")
