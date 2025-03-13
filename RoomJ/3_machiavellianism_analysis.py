import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
from math import pi

# Load data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Output folder
os.makedirs("outputs/MACHI", exist_ok=True)

# Define Machiavellianism items
machi_items = [col for col in df.columns if col.startswith("MACHI_Q")]
machi_total = "MACHI_Total"

# Descriptive stats for total
with open("outputs/MACHI/machi_summary.txt", "w") as f:
    stats = df[machi_total].describe().round(2)
    f.write("=== Machiavellianism Total Score ===\n")
    f.write(str(stats) + "\n")

# Top 25% vs Bottom 25%
q75 = df[machi_total].quantile(0.75)
q25 = df[machi_total].quantile(0.25)
high = df[df[machi_total] >= q75]
low = df[df[machi_total] <= q25]

t, p = ttest_ind(high[machi_total], low[machi_total])
with open("outputs/MACHI/machi_group_comparison.txt", "w") as f:
    f.write("=== Top 25% vs Bottom 25% ===\n")
    f.write(f"T-test: t={t:.2f}, p={p:.3f}\n")

# Item-wise analysis
item_stats = df[machi_items].agg(['mean', 'std']).T.round(2)
item_stats.to_csv("outputs/MACHI/machi_item_stats.csv")

# Cronbach's alpha
def cronbach_alpha(items_df):
    items = items_df.dropna()
    item_scores = items.values
    item_variances = item_scores.var(axis=0, ddof=1)
    total_score = item_scores.sum(axis=1)
    n_items = item_scores.shape[1]
    total_variance = total_score.var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return round(alpha, 3)

alpha = cronbach_alpha(df[machi_items])
with open("outputs/MACHI/cronbach_alpha.txt", "w") as f:
    f.write(f"Cronbach's alpha: {alpha}\n")

# KMeans clustering
scaled = StandardScaler().fit_transform(df[machi_items])
kmeans = KMeans(n_clusters=3, random_state=42)
df['MACHI_Cluster'] = kmeans.fit_predict(scaled)
sil_score = silhouette_score(scaled, df['MACHI_Cluster'])

# Cluster profiling
cluster_profile = df.groupby('MACHI_Cluster')[machi_items].mean().round(2)
cluster_profile.to_csv("outputs/MACHI/machi_cluster_profiles.csv")

with open("outputs/MACHI/cluster_silhouette.txt", "w") as f:
    f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")

# Cluster heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm')
plt.title("Machiavellianism Cluster Profiles (Means)")
plt.tight_layout()
plt.savefig("outputs/MACHI/machi_cluster_heatmap.png")
plt.close()

# Item correlation heatmap
item_corr = df[machi_items].corr().round(2)
plt.figure(figsize=(10, 8))
sns.heatmap(item_corr, annot=False, cmap='viridis')
plt.title("Machiavellianism Item Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/MACHI/machi_item_correlation_heatmap.png")
plt.close()

# Radar chart for cluster profiles
angles = [n / float(len(machi_items)) * 2 * pi for n in range(len(machi_items))]
angles += angles[:1]

for cluster_id in cluster_profile.index:
    values = cluster_profile.loc[cluster_id].tolist()
    values += values[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(machi_items, fontsize=8)
    plt.title(f"Machiavellianism Radar - Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(f"outputs/MACHI/machi_radar_cluster_{cluster_id}.png")
    plt.close()

# Distribution plot of total score
sns.histplot(df[machi_total], kde=True, bins=20)
plt.title("Machiavellianism Total Score Distribution")
plt.tight_layout()
plt.savefig("outputs/MACHI/machi_total_distribution.png")
plt.close()

# Boxplot by cluster
sns.boxplot(x='MACHI_Cluster', y=machi_total, data=df)
plt.title("Machiavellianism Total Score by Cluster")
plt.tight_layout()
plt.savefig("outputs/MACHI/machi_boxplot_by_cluster.png")
plt.close()

# ============================================
# 1. Correlation with Other Psychological Scales
# ============================================

related_cols = ["SSS_Total", "BIS_Total", "PQ_Total", "VRSQ_Total", "SSQ_Standardized"]
correlations = []

for scale in related_cols:
    if scale in df.columns:
        r_pearson, p_pearson = pearsonr(df[machi_total], df[scale])
        correlations.append([scale, r_pearson, p_pearson])

corr_df = pd.DataFrame(correlations, columns=["Scale", "Pearson_r", "p_value"])
corr_df.to_csv("outputs/MACHI/machi_correlation_with_other_scales.csv", index=False)

# ============================================
# 2. PCA Visualization (Item-Level Dimensionality)
# ============================================

from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(df[machi_items])
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["MACHI_Cluster"]

plt.figure(figsize=(6,5))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="tab10", data=pca_df)
plt.title("PCA of Machiavellianism Items by Cluster")
plt.tight_layout()
plt.savefig("outputs/MACHI/machi_pca_cluster.png")
plt.close()

#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import ttest_ind
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import os
#
# # Load data
# df = pd.read_excel("GPT_Format_250313.xlsx")
#
# # Output folder
# os.makedirs("outputs/MACHI", exist_ok=True)
#
# # Define Machiavellianism items (assumed Q1~Q16 are used for MACHI_Total)
# machi_items = [col for col in df.columns if col.startswith("MACHI_Q")]
# machi_total = "MACHI_Total"
#
# # Descriptive stats for total
# with open("outputs/MACHI/machi_summary.txt", "w") as f:
#     stats = df[machi_total].describe().round(2)
#     f.write("=== Machiavellianism Total Score ===\n")
#     f.write(str(stats) + "\n")
#
# # Top 25% vs Bottom 25%
# q75 = df[machi_total].quantile(0.75)
# q25 = df[machi_total].quantile(0.25)
# high = df[df[machi_total] >= q75]
# low = df[df[machi_total] <= q25]
#
# t, p = ttest_ind(high[machi_total], low[machi_total])
# with open("outputs/MACHI/machi_group_comparison.txt", "w") as f:
#     f.write("=== Top 25% vs Bottom 25% ===\n")
#     f.write(f"T-test: t={t:.2f}, p={p:.3f}\n")
#
# # Item-wise analysis
# item_stats = df[machi_items].agg(['mean', 'std']).T.round(2)
# item_stats.to_csv("outputs/MACHI/machi_item_stats.csv")
#
# # Cronbach's alpha
# def cronbach_alpha(items_df):
#     items = items_df.dropna()
#     item_scores = items.values
#     item_variances = item_scores.var(axis=0, ddof=1)
#     total_score = item_scores.sum(axis=1)
#     n_items = item_scores.shape[1]
#     total_variance = total_score.var(ddof=1)
#     alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
#     return round(alpha, 3)
#
# alpha = cronbach_alpha(df[machi_items])
# with open("outputs/MACHI/cronbach_alpha.txt", "w") as f:
#     f.write(f"Cronbach's alpha: {alpha}\n")
#
# # KMeans clustering
# scaled = StandardScaler().fit_transform(df[machi_items])
# kmeans = KMeans(n_clusters=3, random_state=42)
# df['MACHI_Cluster'] = kmeans.fit_predict(scaled)
# sil_score = silhouette_score(scaled, df['MACHI_Cluster'])
#
# # Cluster profiling
# cluster_profile = df.groupby('MACHI_Cluster')[machi_items].mean().round(2)
# cluster_profile.to_csv("outputs/MACHI/machi_cluster_profiles.csv")
#
# with open("outputs/MACHI/cluster_silhouette.txt", "w") as f:
#     f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")
#
# # Visualization of cluster means
# plt.figure(figsize=(8,6))
# sns.heatmap(cluster_profile, annot=True, cmap='coolwarm')
# plt.title("Machiavellianism Cluster Profiles (Means)")
# plt.tight_layout()
# plt.savefig("outputs/MACHI/machi_cluster_heatmap.png")
# plt.close()
