import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import os

# Load the Excel data
df = pd.read_excel("GPT_Format_250313.xlsx")

# Define scales
scales = {
    "SSS": ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS", "SSS_Total"],
    "Machiavellianism": ["MACHI_Total"],
    "BIS": ["BIS_Attentional", "BIS_Motor", "BIS_NonPlanning", "BIS_Total"],
    "SSQ": ["SSQ_Nausea", "SSQ_Oculomotor", "SSQ_Disorientation", "SSQ_Total"],
    "VRSQ": ["VRSQ_Total"],
    "PQ": [
        "PQ_Realism", "PQ_PossibilityToAct", "PQ_QualityOfInterface",
        "PQ_PossibilityToExamine", "PQ_SelfEvaluationOfPerformance", "PQ_Sounds"
    ]
}

# Create output directory
os.makedirs("outputs", exist_ok=True)

# PQ 평균 추가
df["PQ_Avg"] = df[scales["PQ"]].mean(axis=1)
total_cols = ["SSS_Total", "MACHI_Total", "BIS_Total", "SSQ_Total", "VRSQ_Total", "PQ_Avg"]

# Descriptive stats
with open("outputs/descriptive_stats.txt", "w") as f:
    for name, cols in scales.items():
        f.write(f"\n=== {name} ===\n")
        f.write(str(df[cols].describe().round(2)))
        f.write("\n")

# Basic statistics (mean, std, var) + Histogram + KDE
with open("outputs/statistics_summary.txt", "w") as f:
    f.write("=== Participant Score Distribution Summary ===\n")
    for col in total_cols:
        mean = df[col].mean()
        std = df[col].std()
        var = df[col].var()
        f.write(f"{col}: Mean={mean:.2f}, Std={std:.2f}, Var={var:.2f}\n")

        # Histogram
        plt.figure()
        sns.histplot(df[col], kde=False, bins=10)
        plt.title(f"Histogram - {col}")
        plt.tight_layout()
        hist_path = f"outputs/histogram_{col}.png"
        plt.savefig(hist_path)
        plt.close()

        # Density plot
        plt.figure()
        sns.kdeplot(df[col], fill=True)
        plt.title(f"Density Plot - {col}")
        plt.tight_layout()
        density_path = f"outputs/density_{col}.png"
        plt.savefig(density_path)
        plt.close()

# t-test / ANOVA (if Condition exists)
if "Condition" in df.columns:
    conditions = df["Condition"].dropna().unique()
    with open("outputs/group_comparison.txt", "w") as f:
        for col in total_cols:
            if len(conditions) == 2:
                g1, g2 = conditions
                t, p = ttest_ind(df[df["Condition"]==g1][col], df[df["Condition"]==g2][col])
                f.write(f"{col}: t={t:.2f}, p={p:.3f}\n")
            elif len(conditions) > 2:
                samples = [df[df["Condition"]==g][col] for g in conditions]
                f_stat, p_val = f_oneway(*samples)
                f.write(f"{col}: F={f_stat:.2f}, p={p_val:.3f}\n")

# Boxplots
for col in total_cols:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(f"outputs/boxplot_{col}.png")
    plt.close()

# Radar chart of mean scores
means = df[total_cols].mean().tolist()
labels = total_cols
angles = [n / float(len(labels)) * 2 * 3.14159265 for n in range(len(labels))]
means += means[:1]
angles += angles[:1]
plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, means, linewidth=2)
ax.fill(angles, means, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Radar Chart of Mean Scores")
plt.tight_layout()
plt.savefig("outputs/radar_chart.png")
plt.close()

# Correlation matrix heatmap
corr = df[total_cols].corr().round(2)
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_matrix.png")
plt.close()

# PCA
scaler = StandardScaler()
X = scaler.fit_transform(df[total_cols])
pca = PCA(n_components=2)
comp = pca.fit_transform(X)
pc_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
plt.figure(figsize=(6, 5))
sns.scatterplot(x="PC1", y="PC2", data=pc_df)
plt.title("PCA of Total Scores")
plt.tight_layout()
plt.savefig("outputs/pca_plot.png")
plt.close()

# PDF Report
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Questionnaire Report", ln=1, align="C")

    def chapter(self, title, image_path=None):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=1)
        if image_path and os.path.exists(image_path):
            self.image(image_path, w=180)
        self.ln(5)

pdf = PDF()
pdf.add_page()
pdf.chapter("Radar Chart", "outputs/radar_chart.png")
pdf.chapter("Correlation Matrix", "outputs/correlation_matrix.png")
pdf.chapter("PCA", "outputs/pca_plot.png")

for col in total_cols:
    pdf.chapter(f"Boxplot - {col}", f"outputs/boxplot_{col}.png")
    pdf.chapter(f"Histogram - {col}", f"outputs/histogram_{col}.png")
    pdf.chapter(f"Density - {col}", f"outputs/density_{col}.png")

pdf.output("outputs/questionnaire_report.pdf")


#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_ind, f_oneway
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from fpdf import FPDF
# import os
#
# # Load the Excel data
# df = pd.read_excel("GPT_Format_250313.xlsx")
#
# # Define scales
# scales = {
#     "SSS": ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS", "SSS_Total"],
#     "Machiavellianism": ["MACHI_Total"],
#     "BIS": ["BIS_Attentional", "BIS_Motor", "BIS_NonPlanning", "BIS_Total"],
#     "SSQ": ["SSQ_Nausea", "SSQ_Oculomotor", "SSQ_Disorientation", "SSQ_Total"],
#     "VRSQ": ["VRSQ_Total"],
#     "PQ": [
#         "PQ_Realism", "PQ_PossibilityToAct", "PQ_QualityOfInterface",
#         "PQ_PossibilityToExamine", "PQ_SelfEvaluationOfPerformance", "PQ_Sounds"
#     ]
# }
#
# # Create output directory
# os.makedirs("outputs", exist_ok=True)
#
# # PQ 평균 추가
# df["PQ_Avg"] = df[scales["PQ"]].mean(axis=1)
# total_cols = ["SSS_Total", "MACHI_Total", "BIS_Total", "SSQ_Total", "VRSQ_Total", "PQ_Avg"]
#
# # Descriptive stats
# with open("outputs/descriptive_stats.txt", "w") as f:
#     for name, cols in scales.items():
#         f.write(f"\n=== {name} ===\n")
#         f.write(str(df[cols].describe().round(2)))
#         f.write("\n")
#
# # t-test / ANOVA (if Condition exists)
# if "Condition" in df.columns:
#     conditions = df["Condition"].dropna().unique()
#     with open("outputs/group_comparison.txt", "w") as f:
#         for col in total_cols:
#             if len(conditions) == 2:
#                 g1, g2 = conditions
#                 t, p = ttest_ind(df[df["Condition"]==g1][col], df[df["Condition"]==g2][col])
#                 f.write(f"{col}: t={t:.2f}, p={p:.3f}\n")
#             elif len(conditions) > 2:
#                 samples = [df[df["Condition"]==g][col] for g in conditions]
#                 f_stat, p_val = f_oneway(*samples)
#                 f.write(f"{col}: F={f_stat:.2f}, p={p_val:.3f}\n")
#
# # Boxplots
# for col in total_cols:
#     plt.figure()
#     sns.boxplot(y=df[col])
#     plt.title(f"Boxplot - {col}")
#     plt.tight_layout()
#     plt.savefig(f"outputs/boxplot_{col}.png")
#     plt.close()
#
# # Radar chart of mean scores
# means = df[total_cols].mean().tolist()
# labels = total_cols
# angles = [n / float(len(labels)) * 2 * 3.14159265 for n in range(len(labels))]
# means += means[:1]
# angles += angles[:1]
# plt.figure(figsize=(6,6))
# ax = plt.subplot(111, polar=True)
# ax.plot(angles, means, linewidth=2)
# ax.fill(angles, means, alpha=0.25)
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(labels)
# plt.title("Radar Chart of Mean Scores")
# plt.tight_layout()
# plt.savefig("outputs/radar_chart.png")
# plt.close()
#
# # Correlation matrix heatmap
# corr = df[total_cols].corr().round(2)
# plt.figure(figsize=(6, 5))
# sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Correlation Matrix")
# plt.tight_layout()
# plt.savefig("outputs/correlation_matrix.png")
# plt.close()
#
# # PCA
# scaler = StandardScaler()
# X = scaler.fit_transform(df[total_cols])
# pca = PCA(n_components=2)
# comp = pca.fit_transform(X)
# pc_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
# plt.figure(figsize=(6, 5))
# sns.scatterplot(x="PC1", y="PC2", data=pc_df)
# plt.title("PCA of Total Scores")
# plt.tight_layout()
# plt.savefig("outputs/pca_plot.png")
# plt.close()
#
# # PDF Report
# class PDF(FPDF):
#     def header(self):
#         self.set_font("Arial", "B", 14)
#         self.cell(0, 10, "Questionnaire Report", ln=1, align="C")
#
#     def chapter(self, title, image_path=None):
#         self.set_font("Arial", "B", 12)
#         self.cell(0, 10, title, ln=1)
#         if image_path and os.path.exists(image_path):
#             self.image(image_path, w=180)
#         self.ln(5)
#
# pdf = PDF()
# pdf.add_page()
# pdf.chapter("Radar Chart", "outputs/radar_chart.png")
# pdf.chapter("Correlation Matrix", "outputs/correlation_matrix.png")
# pdf.chapter("PCA", "outputs/pca_plot.png")
# for col in total_cols:
#     pdf.chapter(f"Boxplot - {col}", f"outputs/boxplot_{col}.png")
# pdf.output("outputs/questionnaire_report.pdf")
