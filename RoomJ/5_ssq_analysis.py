
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from scipy.stats import ttest_rel, f_oneway, pearsonr
import os

# Load data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Create output folder
os.makedirs("outputs/SSQ", exist_ok=True)

# Define SSQ question keys if needed (assuming already scored into subcategories)
ssq_subscales = ["SSQ_Nausea", "SSQ_Oculomotor", "SSQ_Disorientation"]
ssq_total = "SSQ_Total"

# Normalize SSQ (standard method from Kennedy et al., 1993)
# SSQ_Total = 100 * (0. nausea*9.54 + oculomotor*7.58 + disorientation*13.92) / 3
df["SSQ_Standardized"] = (
    9.54 * df["SSQ_Nausea"] +
    7.58 * df["SSQ_Oculomotor"] +
    13.92 * df["SSQ_Disorientation"]
) / 3
df["SSQ_Standardized"] *= 100

# Save descriptive statistics
desc = df[ssq_subscales + ["SSQ_Standardized"]].describe().round(2)
desc.to_csv("outputs/SSQ/ssq_descriptive_stats.csv")

# Group comparison if "Condition" exists (ANOVA)
if "Condition" in df.columns:
    groups = df["Condition"].unique()
    with open("outputs/SSQ/ssq_group_comparison.txt", "w") as f:
        f.write("=== One-way ANOVA by Condition ===\n")
        for col in ssq_subscales + ["SSQ_Standardized"]:
            samples = [df[df["Condition"] == g][col] for g in groups]
            f_stat, p = f_oneway(*samples)
            f.write(f"{col}: F={f_stat:.2f}, p={p:.3f}\n")

# Correlation with other scales (if present)
related_cols = ["SSS_Total", "BIS_Total", "VRSQ_Total", "PQ_Avg", "MACHI_Total"]
correlations = []
for rcol in related_cols:
    if rcol in df.columns:
        r, p = pearsonr(df["SSQ_Standardized"], df[rcol])
        correlations.append((rcol, round(r, 3), round(p, 3)))
pd.DataFrame(correlations, columns=["RelatedScale", "PearsonR", "p"]).to_csv("outputs/SSQ/ssq_scale_correlations.csv", index=False)

# Boxplot by Condition (if exists)
if "Condition" in df.columns:
    plt.figure()
    sns.boxplot(x="Condition", y="SSQ_Standardized", data=df)
    plt.title("SSQ Standardized by Condition")
    plt.tight_layout()
    plt.savefig("outputs/SSQ/boxplot_ssq_condition.png")
    plt.close()

# Subscale bar chart (condition-averaged)
if "Condition" in df.columns:
    mean_by_cond = df.groupby("Condition")[ssq_subscales].mean()
    mean_by_cond.plot(kind="bar")
    plt.title("SSQ Subscale Mean by Condition")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("outputs/SSQ/bar_subscale_condition.png")
    plt.close()

# Radar chart (mean subscale values)
means = df[ssq_subscales].mean().tolist()
angles = [n / float(len(ssq_subscales)) * 2 * pi for n in range(len(ssq_subscales))]
means += means[:1]
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, means, linewidth=2)
ax.fill(angles, means, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Nausea", "Oculomotor", "Disorientation"])
plt.title("Mean SSQ Subscale Radar Chart")
plt.tight_layout()
plt.savefig("outputs/SSQ/radar_mean_ssq.png")
plt.close()

# Participant-by-subscale heatmap
heat_data = df[ssq_subscales]
plt.figure(figsize=(10, 6))
sns.heatmap(heat_data, annot=False, cmap="YlOrRd")
plt.title("SSQ Profile per Participant")
plt.xlabel("Subscale")
plt.ylabel("Participant")
plt.tight_layout()
plt.savefig("outputs/SSQ/participant_heatmap.png")
plt.close()
# ============================================
# 1. Spearman 상관분석 추가
# ============================================
from scipy.stats import spearmanr

spearman_corrs = []
for rcol in related_cols:
    if rcol in df.columns:
        r_s, p_s = spearmanr(df["SSQ_Standardized"], df[rcol])
        spearman_corrs.append((rcol, round(r_s, 3), round(p_s, 3)))

pd.DataFrame(spearman_corrs, columns=["RelatedScale", "SpearmanR", "p"]).to_csv(
    "outputs/SSQ/ssq_scale_spearman_correlations.csv", index=False
)

# ============================================
# 2. PCA 시각화 (SSQ 하위척도)
# ============================================
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(df[ssq_subscales])
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])

plt.figure(figsize=(6,5))
sns.scatterplot(x="PC1", y="PC2", data=pca_df)
plt.title("PCA of SSQ Subscales")
plt.tight_layout()
plt.savefig("outputs/SSQ/ssq_pca_scatter.png")
plt.close()

# ============================================
# 3. KMeans 클러스터링 + Radar Chart
# ============================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=3, random_state=42)
df["SSQ_Cluster"] = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df["SSQ_Cluster"])

# 클러스터 평균 저장
cluster_profile = df.groupby("SSQ_Cluster")[ssq_subscales].mean().round(2)
cluster_profile.to_csv("outputs/SSQ/ssq_cluster_profile.csv")

# 클러스터 히트맵
plt.figure(figsize=(6,4))
sns.heatmap(cluster_profile, annot=True, cmap="YlGnBu")
plt.title("SSQ Cluster Profiles")
plt.tight_layout()
plt.savefig("outputs/SSQ/ssq_cluster_heatmap.png")
plt.close()

# Silhouette Score 저장
with open("outputs/SSQ/ssq_silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")

# 클러스터별 Radar Chart
angles = [n / float(len(ssq_subscales)) * 2 * pi for n in range(len(ssq_subscales))]
angles += angles[:1]

for cluster_id in cluster_profile.index:
    values = cluster_profile.loc[cluster_id].tolist()
    values += values[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Nausea", "Oculomotor", "Disorientation"])
    plt.title(f"SSQ Radar - Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(f"outputs/SSQ/ssq_radar_cluster_{cluster_id}.png")
    plt.close()
# ============================================
# 4. SSQ 산점도 시각화 (vs 다른 심리척도들)
# ============================================

for rcol in related_cols:
    if rcol in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x= df[rcol], y= df["SSQ_Standardized"])
        plt.title(f"Scatter: SSQ_Standardized vs {rcol}")
        plt.xlabel(rcol)
        plt.ylabel("SSQ_Standardized")
        plt.tight_layout()
        plt.savefig(f"outputs/SSQ/scatter_ssq_vs_{rcol}.png")
        plt.close()
# ============================================
# 5. SSQ 상하위 25% 그룹 간 비교
# ============================================

from scipy.stats import ttest_ind

q75 = df["SSQ_Standardized"].quantile(0.75)
q25 = df["SSQ_Standardized"].quantile(0.25)

high_group = df[df["SSQ_Standardized"] >= q75]
low_group = df[df["SSQ_Standardized"] <= q25]

with open("outputs/SSQ/ssq_high_low_comparison.txt", "w") as f:
    f.write("=== High (Top 25%) vs Low (Bottom 25%) SSQ_Standardized Comparison ===\n")
    for col in ssq_subscales:
        t_stat, p_val = ttest_ind(high_group[col], low_group[col])
        f.write(f"{col}: t={t_stat:.2f}, p={p_val:.3f}\n")
# ============================================
# 6. Cronbach's Alpha (SSQ 문항 16개 기반)
# ============================================
def cronbach_alpha(df_items):
    df_clean = df_items.dropna()
    item_scores = df_clean.values
    item_variances = item_scores.var(axis=0, ddof=1)
    total_score = item_scores.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    n_items = item_scores.shape[1]
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return round(alpha, 3)

ssq_item_cols = [col for col in df.columns if "SSQ_Q" in col]
alpha = cronbach_alpha(df[ssq_item_cols])

with open("outputs/SSQ/ssq_cronbach_alpha.txt", "w") as f:
    f.write(f"Cronbach's Alpha (SSQ 16 Items): {alpha}\n")
