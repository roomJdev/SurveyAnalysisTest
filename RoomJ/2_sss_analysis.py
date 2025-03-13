import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
import os
from math import pi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Excel data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Output folder
os.makedirs("outputs/SSS", exist_ok=True)

# SSS columns
sss_cols = ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS", "SSS_Total"]
subscales = sss_cols[:-1]

### 1. Descriptive Stats
with open("outputs/SSS/sss_stats.txt", "w") as f:
    f.write(str(df[sss_cols].describe().round(2)))

### 2. Histogram & Boxplot for Total
sns.histplot(df["SSS_Total"], bins=10)
plt.title("Histogram - SSS_Total")
plt.tight_layout()
plt.savefig("outputs/SSS/histogram_SSS_Total.png")
plt.close()

sns.boxplot(y=df["SSS_Total"])
plt.title("Boxplot - SSS_Total")
plt.tight_layout()
plt.savefig("outputs/SSS/boxplot_SSS_Total.png")
plt.close()

### 3. Subscale Correlation
corr_matrix = df[subscales].corr(method="pearson").round(2)
corr_matrix.to_csv("outputs/SSS/sss_subscale_correlation.csv")
plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("SSS Subscale Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/SSS/sss_subscale_correlation_heatmap.png")
plt.close()

### 4. High vs Low Comparison (Top 25% vs Bottom 25%)
q75 = df["SSS_Total"].quantile(0.75)
q25 = df["SSS_Total"].quantile(0.25)
high_group = df[df["SSS_Total"] >= q75]
low_group = df[df["SSS_Total"] <= q25]

with open("outputs/SSS/sss_high_low_comparison.txt", "w") as f:
    f.write("=== High (Top 25%) vs Low (Bottom 25%) Comparison ===\n")
    for col in subscales:
        t, p = ttest_ind(high_group[col], low_group[col])
        f.write(f"{col}: t={t:.2f}, p={p:.3f}\n")

### 5. Subscale Boxplots
for col in subscales:
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/SSS/boxplot_{col}.png")
    plt.close()

### 6. Radar Chart for Each Participant
for idx, row in df.iterrows():
    values = row[subscales].tolist() + [row[subscales[0]]]
    angles = [n / 4 * 2 * pi for n in range(5)]
    plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["TAS", "ES", "BS", "DIS"])
    plt.title(f"SSS Radar - Participant {idx+1}")
    plt.tight_layout()
    plt.savefig(f"outputs/SSS/radar_participant_{idx+1}.png")
    plt.close()

### 7. Barplot of Subscale Means
means = df[subscales].mean()
sns.barplot(x=means.index, y=means.values)
plt.title("Mean Scores of SSS Subscales")
plt.ylabel("Mean")
plt.tight_layout()
plt.savefig("outputs/SSS/barplot_means.png")
plt.close()

### 8. Overlay Radar of All Participants
angles = [n / 4 * 2 * pi for n in range(5)]
plt.figure(figsize=(6,6))
for _, row in df.iterrows():
    values = row[subscales].tolist() + [row[subscales[0]]]
    plt.polar(angles, values, alpha=0.1)
plt.xticks(angles[:-1], ["TAS", "ES", "BS", "DIS"])
plt.title("Overlay Radar of All Participants")
plt.tight_layout()
plt.savefig("outputs/SSS/overlay_radar_all.png")
plt.close()

### 9. Pairplot of Subscales
sns.pairplot(df[subscales])
plt.savefig("outputs/SSS/subscale_pairplot.png")
plt.close()

### 10. PCA Visualization
X_scaled = StandardScaler().fit_transform(df[subscales])
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
sns.scatterplot(x="PC1", y="PC2", data=pca_df)
plt.title("PCA of SSS Subscales")
plt.tight_layout()
plt.savefig("outputs/SSS/pca_sss.png")
plt.close()

### 11. Correlation with Other Scales
related_cols = ["PQ_Total", "BIS_Total", "MACHI_Total", "VRSQ_Total", "SSQ_Standardized"]
correlations = []
for scale in related_cols:
    if scale in df.columns:
        r_pearson, p_pearson = pearsonr(df["SSS_Total"], df[scale])
        correlations.append([scale, r_pearson, p_pearson])
corr_df = pd.DataFrame(correlations, columns=["Scale", "Pearson_r", "p_value"])
corr_df.to_csv("outputs/SSS/sss_correlation_with_other_scales.csv", index=False)

### 12. Cronbach's Alpha
def cronbach_alpha(df_subscales):
    df_clean = df_subscales.dropna()
    item_scores = df_clean.values
    item_variances = item_scores.var(axis=0, ddof=1)
    total_score = item_scores.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    n_items = item_scores.shape[1]
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return round(alpha, 3)

alpha = cronbach_alpha(df[subscales])
with open("outputs/SSS/sss_cronbach_alpha.txt", "w") as f:
    f.write(f"Cronbach's Alpha: {alpha}\n")

### 13. KMeans Clustering & Cluster Radar
kmeans = KMeans(n_clusters=3, random_state=42)
df["SSS_Cluster"] = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df["SSS_Cluster"])

# Cluster means
cluster_means = df.groupby("SSS_Cluster")[subscales].mean().round(2)
cluster_means.to_csv("outputs/SSS/sss_cluster_means.csv")

# Cluster heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu")
plt.title("SSS Cluster Profiles")
plt.tight_layout()
plt.savefig("outputs/SSS/sss_cluster_heatmap.png")
plt.close()

# Silhouette score 저장
with open("outputs/SSS/sss_silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")

# Cluster radar charts
angles = [n / float(len(subscales)) * 2 * pi for n in range(len(subscales))]
angles += angles[:1]

for cluster_id in cluster_means.index:
    values = cluster_means.loc[cluster_id].tolist() + [cluster_means.loc[cluster_id].tolist()[0]]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subscales)
    plt.title(f"SSS Radar Chart - Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(f"outputs/SSS/sss_radar_cluster_{cluster_id}.png")
    plt.close()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, ttest_ind
# import os
# from math import pi
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
#
# # Load the Excel data
# df = pd.read_excel("GPT_Format_250313.xlsx")
#
# # Output folder
# os.makedirs("outputs/SSS", exist_ok=True)
#
# # SSS columns
# sss_cols = ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS", "SSS_Total"]
# subscales = sss_cols[:-1]
#
# ### 1. Descriptive Stats
# with open("outputs/SSS/sss_stats.txt", "w") as f:
#     f.write(str(df[sss_cols].describe().round(2)))
#
# ### 2. Histogram & Boxplot for Total
# sns.histplot(df["SSS_Total"], bins=10)
# plt.title("Histogram - SSS_Total")
# plt.tight_layout()
# plt.savefig("outputs/SSS/histogram_SSS_Total.png")
# plt.close()
#
# sns.boxplot(y=df["SSS_Total"])
# plt.title("Boxplot - SSS_Total")
# plt.tight_layout()
# plt.savefig("outputs/SSS/boxplot_SSS_Total.png")
# plt.close()
#
# ### 3. Subscale Correlation
# corr_matrix = df[subscales].corr(method="pearson").round(2)
# corr_matrix.to_csv("outputs/SSS/sss_subscale_correlation.csv")
# plt.figure(figsize=(5, 4))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("SSS Subscale Correlation Heatmap")
# plt.tight_layout()
# plt.savefig("outputs/SSS/sss_subscale_correlation_heatmap.png")
# plt.close()
#
# ### 4. High vs Low Comparison
# q75 = df["SSS_Total"].quantile(0.75)
# q25 = df["SSS_Total"].quantile(0.25)
# high_group = df[df["SSS_Total"] >= q75]
# low_group = df[df["SSS_Total"] <= q25]
#
# with open("outputs/SSS/sss_high_low_comparison.txt", "w") as f:
#     f.write("=== High (Top 25%) vs Low (Bottom 25%) Comparison ===\n")
#     for col in subscales:
#         t, p = ttest_ind(high_group[col], low_group[col])
#         f.write(f"{col}: t={t:.2f}, p={p:.3f}\n")
#
# ### 5. Boxplots (Subscales)
# for col in subscales:
#     sns.boxplot(y=df[col])
#     plt.title(f"Boxplot - {col}")
#     plt.tight_layout()
#     plt.savefig(f"outputs/SSS/boxplot_{col}.png")
#     plt.close()
#
# ### 6. Participant Radar Charts
# for idx, row in df.iterrows():
#     values = row[subscales].tolist() + [row[subscales[0]]]
#     angles = [n / 4 * 2 * pi for n in range(5)]
#     plt.figure(figsize=(5,5))
#     ax = plt.subplot(111, polar=True)
#     ax.plot(angles, values)
#     ax.fill(angles, values, alpha=0.3)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(["TAS", "ES", "BS", "DIS"])
#     plt.title(f"SSS Radar - Participant {idx+1}")
#     plt.tight_layout()
#     plt.savefig(f"outputs/SSS/radar_participant_{idx+1}.png")
#     plt.close()
#
# ### 7. Average Subscale Bar Plot
# means = df[subscales].mean()
# sns.barplot(x=means.index, y=means.values)
# plt.title("Mean Scores of SSS Subscales")
# plt.ylabel("Mean")
# plt.tight_layout()
# plt.savefig("outputs/SSS/barplot_means.png")
# plt.close()
#
# ### 8. Overlay Radar for All Participants
# angles = [n / 4 * 2 * pi for n in range(5)]
# plt.figure(figsize=(6,6))
# for _, row in df.iterrows():
#     values = row[subscales].tolist() + [row[subscales[0]]]
#     plt.polar(angles, values, alpha=0.1)
# plt.xticks(angles[:-1], ["TAS", "ES", "BS", "DIS"])
# plt.title("Overlay Radar of All Participants")
# plt.tight_layout()
# plt.savefig("outputs/SSS/overlay_radar_all.png")
# plt.close()
#
# ### 9. Pairplot of Subscales
# sns.pairplot(df[subscales])
# plt.savefig("outputs/SSS/subscale_pairplot.png")
# plt.close()
#
# ### 10. PCA Visualization
# X_scaled = StandardScaler().fit_transform(df[subscales])
# pca = PCA(n_components=2)
# components = pca.fit_transform(X_scaled)
# pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
# sns.scatterplot(x="PC1", y="PC2", data=pca_df)
# plt.title("PCA of SSS Subscales")
# plt.tight_layout()
# plt.savefig("outputs/SSS/pca_sss.png")
# plt.close()
#
# ### 11. Correlation with Other Scales
# related_cols = ["PQ_Total", "BIS_Total", "MACHI_Total", "VRSQ_Total", "SSQ_Standardized"]
# correlations = []
# for scale in related_cols:
#     if scale in df.columns:
#         r_pearson, p_pearson = pearsonr(df["SSS_Total"], df[scale])
#         correlations.append([scale, r_pearson, p_pearson])
# corr_df = pd.DataFrame(correlations, columns=["Scale", "Pearson_r", "p_value"])
# corr_df.to_csv("outputs/SSS/sss_correlation_with_other_scales.csv", index=False)
#
# ### 12. Cronbach's Alpha
# def cronbach_alpha(df_subscales):
#     df_clean = df_subscales.dropna()
#     item_scores = df_clean.values
#     item_variances = item_scores.var(axis=0, ddof=1)
#     total_score = item_scores.sum(axis=1)
#     total_variance = total_score.var(ddof=1)
#     n_items = item_scores.shape[1]
#     alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
#     return round(alpha, 3)
#
# alpha = cronbach_alpha(df[subscales])
# with open("outputs/SSS/sss_cronbach_alpha.txt", "w") as f:
#     f.write(f"Cronbach's Alpha: {alpha}\n")
#
# ### 13. KMeans Clustering
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
#
# X_scaled = StandardScaler().fit_transform(df[subscales])
# kmeans = KMeans(n_clusters=3, random_state=42)
# df["SSS_Cluster"] = kmeans.fit_predict(X_scaled)
# sil_score = silhouette_score(X_scaled, df["SSS_Cluster"])
#
# # Cluster means
# cluster_means = df.groupby("SSS_Cluster")[subscales].mean().round(2)
# cluster_means.to_csv("outputs/SSS/sss_cluster_means.csv")
#
# # Cluster heatmap
# plt.figure(figsize=(6, 4))
# sns.heatmap(cluster_means, annot=True, cmap="YlGnBu")
# plt.title("SSS Cluster Profiles")
# plt.tight_layout()
# plt.savefig("outputs/SSS/sss_cluster_heatmap.png")
# plt.close()
#
# # Silhouette score 저장
# with open("outputs/SSS/sss_silhouette_score.txt", "w") as f:
#     f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")

#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr, ttest_ind
# import os
#
# # Load the Excel data
# df = pd.read_excel("GPT_Format_250313.xlsx")
#
# # Create output folder
# os.makedirs("outputs/SSS", exist_ok=True)
#
# # SSS subscales
# sss_cols = ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS", "SSS_Total"]
# subscales = ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS"]
#
# # 1. Descriptive statistics
# with open("outputs/SSS/sss_stats.txt", "w") as f:
#     desc = df[sss_cols].describe().round(2)
#     f.write("=== Descriptive Statistics ===\n")
#     f.write(str(desc))
#
# # 2. Histogram & Boxplot for Total Score
# plt.figure()
# sns.histplot(df["SSS_Total"], bins=10)
# plt.title("Histogram - SSS_Total")
# plt.tight_layout()
# plt.savefig("outputs/SSS/histogram_SSS_Total.png")
# plt.close()
#
# plt.figure()
# sns.boxplot(y=df["SSS_Total"])
# plt.title("Boxplot - SSS_Total")
# plt.tight_layout()
# plt.savefig("outputs/SSS/boxplot_SSS_Total.png")
# plt.close()
#
# # 3. Subscale correlation matrix + heatmap
# corr_matrix = df[subscales].corr(method="pearson").round(2)
# corr_matrix.to_csv("outputs/SSS/sss_subscale_correlation.csv")
#
# plt.figure(figsize=(5, 4))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("SSS Subscale Correlation")
# plt.tight_layout()
# plt.savefig("outputs/SSS/sss_subscale_correlation_heatmap.png")
# plt.close()
#
# # 4. High vs Low group comparison (top 25% vs bottom 25%)
# q75 = df["SSS_Total"].quantile(0.75)
# q25 = df["SSS_Total"].quantile(0.25)
# high_group = df[df["SSS_Total"] >= q75]
# low_group = df[df["SSS_Total"] <= q25]
#
# with open("outputs/SSS/sss_high_low_comparison.txt", "w") as f:
#     f.write("=== High (Top 25%) vs Low (Bottom 25%) Comparison ===\n")
#     for col in subscales:
#         t, p = ttest_ind(high_group[col], low_group[col])
#         f.write(f"{col}: t={t:.2f}, p={p:.3f}\n")
#
# # 5. Subscale Boxplots
# for col in subscales:
#     plt.figure()
#     sns.boxplot(y=df[col])
#     plt.title(f"Boxplot - {col}")
#     plt.tight_layout()
#     plt.savefig(f"outputs/SSS/boxplot_{col}.png")
#     plt.close()
#
# # 6. Radar chart per participant (4 subscales)
# from math import pi
#
# for idx, row in df.iterrows():
#     values = row[subscales].tolist()
#     values += values[:1]
#     angles = [n / 4 * 2 * pi for n in range(5)]
#     plt.figure(figsize=(5,5))
#     ax = plt.subplot(111, polar=True)
#     ax.plot(angles, values, linewidth=1)
#     ax.fill(angles, values, alpha=0.3)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(["TAS", "ES", "BS", "DIS"])
#     ax.set_title(f"SSS Radar - Participant {idx+1}")
#     plt.tight_layout()
#     plt.savefig(f"outputs/SSS/radar_participant_{idx+1}.png")
#     plt.close()
