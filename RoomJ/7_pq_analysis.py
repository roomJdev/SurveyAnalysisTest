
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from scipy.stats import ttest_ind, f_oneway, pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

# Load data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Create output folder
os.makedirs("outputs/PQ", exist_ok=True)

# Define PQ subscales
pq_subscales = {
    "Involvement_Control": ["PQ_PossibilityToAct", "PQ_SelfEvaluationOfPerformance"],
    "Naturalness": ["PQ_Sounds", "PQ_PossibilityToExamine"],
    "Interface_Quality": ["PQ_QualityOfInterface"],
    "Realism": ["PQ_Realism"]
}

# Calculate subscale means
for factor, cols in pq_subscales.items():
    df[factor] = df[cols].mean(axis=1)

# PQ Total Score (average of all items)
df["PQ_Total"] = df[[
    "PQ_Realism", "PQ_PossibilityToAct", "PQ_QualityOfInterface",
    "PQ_PossibilityToExamine", "PQ_SelfEvaluationOfPerformance", "PQ_Sounds"
]].mean(axis=1)

# Descriptive statistics
desc = df[list(pq_subscales.keys()) + ["PQ_Total"]].describe().round(2)
desc.to_csv("outputs/PQ/pq_descriptive_stats.csv")

# Boxplot by Condition
if "Condition" in df.columns:
    plt.figure()
    sns.boxplot(x="Condition", y="PQ_Total", data=df)
    plt.title("PQ Score Distribution by Condition")
    plt.tight_layout()
    plt.savefig("outputs/PQ/boxplot_condition.png")
    plt.close()

# Bar chart: mean of subscales
means = df[list(pq_subscales.keys())].mean()
means.plot(kind="bar", title="PQ Subscale Means", ylabel="Mean Score", ylim=(1,7))
plt.tight_layout()
plt.savefig("outputs/PQ/bar_subscales.png")
plt.close()

# Radar chart
values = means.tolist()
values += values[:1]
labels = list(pq_subscales.keys())
angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
angles += angles[:1]
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Radar Chart - PQ Subscales")
plt.tight_layout()
plt.savefig("outputs/PQ/radar_chart.png")
plt.close()

# Correlation with other scales
related_cols = ["VRSQ_Total", "SSS_Total", "BIS_Total", "MACHI_Total"]
correlations = []
for scale in related_cols:
    if scale in df.columns:
        r_pearson, p_pearson = pearsonr(df["PQ_Total"], df[scale])
        r_spearman, p_spearman = spearmanr(df["PQ_Total"], df[scale])
        correlations.append([scale, r_pearson, p_pearson, r_spearman, p_spearman])
        # Scatterplot
        plt.figure()
        sns.scatterplot(x=scale, y="PQ_Total", data=df)
        plt.title(f"PQ vs {scale}")
        plt.tight_layout()
        plt.savefig(f"outputs/PQ/scatter_PQ_{scale}.png")
        plt.close()
pd.DataFrame(correlations, columns=["Scale", "Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p"])\
    .to_csv("outputs/PQ/pq_correlation_results.csv", index=False)

# Group comparison (e.g., Device or Content Experience if available)
for group_col in ["DeviceExperience", "ContentExperience"]:
    if group_col in df.columns:
        if df[group_col].nunique() == 2:
            lvls = df[group_col].unique()
            t, p = ttest_ind(df[df[group_col] == lvls[0]]["PQ_Total"],
                             df[df[group_col] == lvls[1]]["PQ_Total"])
            with open("outputs/PQ/group_comparison.txt", "a") as f:
                f.write(f"{group_col}: t = {t:.2f}, p = {p:.3f}\n")
        else:
            samples = [df[df[group_col] == lvl]["PQ_Total"] for lvl in df[group_col].unique()]
            f_stat, p_val = f_oneway(*samples)
            with open("outputs/PQ/group_comparison.txt", "a") as f:
                f.write(f"{group_col}: F = {f_stat:.2f}, p = {p_val:.3f}\n")

# Regression (if DeviceExperience or ContentExperience present)
if "DeviceExperience" in df.columns and "ContentExperience" in df.columns:
    df["DeviceExperience"] = df["DeviceExperience"].astype("category")
    df["ContentExperience"] = df["ContentExperience"].astype("category")
    model = smf.ols("PQ_Total ~ DeviceExperience + ContentExperience", data=df).fit()
    with open("outputs/PQ/pq_regression_summary.txt", "w") as f:
        f.write(str(model.summary()))


# ============================================
# 1. KMeans 클러스터링 (PQ 하위척도 기반)
# ============================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

pq_features = list(pq_subscales.keys())
X_scaled = StandardScaler().fit_transform(df[pq_features])

kmeans = KMeans(n_clusters=3, random_state=42)
df["PQ_Cluster"] = kmeans.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df["PQ_Cluster"])

# 클러스터 평균 저장
cluster_means = df.groupby("PQ_Cluster")[pq_features].mean().round(2)
cluster_means.to_csv("outputs/PQ/pq_cluster_profile.csv")

# 클러스터 히트맵
plt.figure(figsize=(6,4))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu")
plt.title("PQ Cluster Profiles")
plt.tight_layout()
plt.savefig("outputs/PQ/pq_cluster_heatmap.png")
plt.close()

# Silhouette Score 저장
with open("outputs/PQ/pq_silhouette_score.txt", "w") as f:
    f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")

# ============================================
# 2. PCA 시각화
# ============================================
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["PQ_Cluster"]

plt.figure(figsize=(6,5))
sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette="tab10")
plt.title("PCA of PQ Subscales by Cluster")
plt.tight_layout()
plt.savefig("outputs/PQ/pq_pca_clusters.png")
plt.close()

# ============================================
# 3. 클러스터별 Radar Chart
# ============================================
angles = [n / float(len(pq_features)) * 2 * pi for n in range(len(pq_features))]
angles += angles[:1]

for cluster_id in cluster_means.index:
    values = cluster_means.loc[cluster_id].tolist() + [cluster_means.loc[cluster_id].tolist()[0]]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pq_features, fontsize=9)
    plt.title(f"PQ Radar - Cluster {cluster_id}")
    plt.tight_layout()
    plt.savefig(f"outputs/PQ/pq_radar_cluster_{cluster_id}.png")
    plt.close()
# ============================================
# 4. PQ Cronbach's Alpha (22개 문항 기반)
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

# PQ 문항 리스트 자동 필터링
pq_item_cols = [col for col in df.columns if col.startswith("PQ_") and "PQ_Total" not in col]
alpha = cronbach_alpha(df[pq_item_cols])

with open("outputs/PQ/pq_cronbach_alpha.txt", "w") as f:
    f.write(f"Cronbach's Alpha (PQ 22 Items): {alpha}\n")
