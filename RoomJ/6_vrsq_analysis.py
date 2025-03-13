
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, ttest_rel, f_oneway, pearsonr, spearmanr
import os

# Load data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Output directory
os.makedirs("outputs/VRSQ", exist_ok=True)

# Boxplot by condition
if "Condition" in df.columns:
    plt.figure()
    sns.boxplot(x="Condition", y="VRSQ_Total", data=df)
    plt.title("VRSQ Score Distribution by Condition")
    plt.tight_layout()
    plt.savefig("outputs/VRSQ/boxplot_condition.png")
    plt.close()

# Bar chart: mean by condition
if "Condition" in df.columns:
    means = df.groupby("Condition")["VRSQ_Total"].mean()
    means.plot(kind="bar", title="Mean VRSQ by Condition", ylabel="VRSQ Score")
    plt.tight_layout()
    plt.savefig("outputs/VRSQ/bar_mean_condition.png")
    plt.close()

# Paired t-test: Pre vs Post (assume columns exist)
if "VRSQ_Pre" in df.columns and "VRSQ_Post" in df.columns:
    t, p = ttest_rel(df["VRSQ_Pre"], df["VRSQ_Post"])
    with open("outputs/VRSQ/paired_ttest.txt", "w") as f:
        f.write(f"Paired t-test (Pre vs Post): t = {t:.2f}, p = {p:.3f}\n")

    plt.figure()
    plt.plot(df[["VRSQ_Pre", "VRSQ_Post"]].T, color='gray', alpha=0.5)
    plt.plot(["VRSQ_Pre", "VRSQ_Post"], [df["VRSQ_Pre"].mean(), df["VRSQ_Post"].mean()], 
             color='red', marker='o', linewidth=2)
    plt.title("VRSQ Change Over Time (Pre vs Post)")
    plt.tight_layout()
    plt.savefig("outputs/VRSQ/line_pre_post.png")
    plt.close()

# Group comparison (e.g., gender, experience)
group_cols = [col for col in df.columns if col.lower() in ["gender", "experience"]]
with open("outputs/VRSQ/group_comparisons.txt", "w") as f:
    for col in group_cols:
        if df[col].nunique() == 2:
            levels = df[col].unique()
            t, p = ttest_ind(df[df[col] == levels[0]]["VRSQ_Total"],
                             df[df[col] == levels[1]]["VRSQ_Total"])
            f.write(f"{col} (2 groups): t = {t:.2f}, p = {p:.3f}\n")
        elif df[col].nunique() > 2:
            samples = [df[df[col] == level]["VRSQ_Total"] for level in df[col].unique()]
            f_stat, p_val = f_oneway(*samples)
            f.write(f"{col} (ANOVA): F = {f_stat:.2f}, p = {p_val:.3f}\n")

# Correlation with other scales
related_cols = ["SSS_Total", "BIS_Total", "PQ_Avg", "NASA_TLX_Total"]
corr_data = []
for scale in related_cols:
    if scale in df.columns:
        r_pearson, p_pearson = pearsonr(df["VRSQ_Total"], df[scale])
        r_spearman, p_spearman = spearmanr(df["VRSQ_Total"], df[scale])
        corr_data.append([scale, r_pearson, p_pearson, r_spearman, p_spearman])
        # Scatter plot
        plt.figure()
        sns.scatterplot(x=scale, y="VRSQ_Total", data=df)
        plt.title(f"VRSQ vs {scale}")
        plt.tight_layout()
        plt.savefig(f"outputs/VRSQ/scatter_VRSQ_{scale}.png")
        plt.close()

# Save correlation table
pd.DataFrame(corr_data, columns=["Scale", "Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p"])\
    .to_csv("outputs/VRSQ/vrsq_correlation_results.csv", index=False)
# ============================================
# 1. KMeans 클러스터링 (VRSQ_Total 기준)
# ============================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

if "VRSQ_Total" in df.columns:
    scaler = StandardScaler()
    X = scaler.fit_transform(df[["VRSQ_Total"]])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["VRSQ_Cluster"] = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, df["VRSQ_Cluster"])

    # Cluster 평균 저장
    cluster_means = df.groupby("VRSQ_Cluster")["VRSQ_Total"].mean().round(2)
    cluster_means.to_csv("outputs/VRSQ/vrsq_cluster_means.csv")

    # Barplot
    plt.figure()
    cluster_means.plot(kind="bar")
    plt.title("VRSQ Total by Cluster")
    plt.ylabel("Mean VRSQ Score")
    plt.tight_layout()
    plt.savefig("outputs/VRSQ/vrsq_cluster_barplot.png")
    plt.close()

    # Silhouette Score 저장
    with open("outputs/VRSQ/vrsq_silhouette_score.txt", "w") as f:
        f.write(f"Silhouette Score (k=3): {sil_score:.3f}\n")

# ============================================
# 2. PCA 시각화 (Pre/Post 존재 시)
# ============================================
from sklearn.decomposition import PCA

if "VRSQ_Pre" in df.columns and "VRSQ_Post" in df.columns:
    pca_input = df[["VRSQ_Pre", "VRSQ_Post"]].dropna()
    pca_scaled = scaler.fit_transform(pca_input)
    pca = PCA(n_components=2)
    components = pca.fit_transform(pca_scaled)
    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])

    # 군집 정보 붙이기 (있는 경우)
    if "VRSQ_Cluster" in df.columns:
        pca_df["Cluster"] = df.loc[pca_input.index, "VRSQ_Cluster"]
        palette = "tab10"
    else:
        palette = None

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster" if "Cluster" in pca_df.columns else None, data=pca_df,
                    palette=palette)
    plt.title("PCA of VRSQ Pre/Post")
    plt.tight_layout()
    plt.savefig("outputs/VRSQ/vrsq_pca.png")
    plt.close()

# ============================================
# 3. 클러스터별 레이더 차트 (Pre/Post)
# ============================================
from math import pi

if "VRSQ_Cluster" in df.columns and "VRSQ_Pre" in df.columns and "VRSQ_Post" in df.columns:
    angles = [0, pi]  # 두 항목이므로 간단한 180도
    labels = ["Pre", "Post"]
    for cluster_id in sorted(df["VRSQ_Cluster"].unique()):
        subset = df[df["VRSQ_Cluster"] == cluster_id][["VRSQ_Pre", "VRSQ_Post"]].mean().tolist()
        subset += subset[:1]
        angles_full = angles + [angles[0]]

        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles_full, subset, linewidth=2)
        ax.fill(angles_full, subset, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)
        plt.title(f"VRSQ Radar - Cluster {cluster_id}")
        plt.tight_layout()
        plt.savefig(f"outputs/VRSQ/vrsq_radar_cluster_{cluster_id}.png")
        plt.close()
# ============================================
# 4. VRSQ 상하위 25% 그룹 간 비교 분석
# ============================================

# 기준: VRSQ_Total
q75 = df["VRSQ_Total"].quantile(0.75)
q25 = df["VRSQ_Total"].quantile(0.25)

high_group = df[df["VRSQ_Total"] >= q75]
low_group = df[df["VRSQ_Total"] <= q25]

with open("outputs/VRSQ/vrsq_high_low_comparison.txt", "w") as f:
    f.write("=== High (Top 25%) vs Low (Bottom 25%) VRSQ_Total Comparison ===\n")

    for col in ["VRSQ_Pre", "VRSQ_Post"]:
        if col in df.columns:
            t_stat, p_val = ttest_ind(high_group[col], low_group[col])
            f.write(f"{col}: t = {t_stat:.2f}, p = {p_val:.3f}\n")
