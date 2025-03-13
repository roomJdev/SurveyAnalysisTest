
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os

# Load the dataset
df = pd.read_excel("GPT_Format_250313.xlsx")

# Calculate PQ_Total manually (mean of 6 available PQ components)
df["PQ_Total"] = df[[
    "PQ_Realism",
    "PQ_PossibilityToAct",
    "PQ_QualityOfInterface",
    "PQ_PossibilityToExamine",
    "PQ_SelfEvaluationOfPerformance",
    "PQ_Sounds"
]].mean(axis=1)

# Output folder
os.makedirs("outputs/InterScale", exist_ok=True)

# Define available total/average columns
questionnaire_totals = [
    "SSS_Total",        # Zuckerman SSS
    "MACHI_Total",      # Machiavellianism
    "BIS_Total",        # BIS-11
    "SSQ_Total",        # SSQ (sum-based score)
    "VRSQ_Total",       # VRSQ
    "PQ_Total"          # Presence Questionnaire (manually calculated)
]

# Filter and clean
df_corr = df[questionnaire_totals].dropna()

# Pearson correlation
pearson_corr = df_corr.corr(method='pearson').round(2)
pearson_corr.to_csv("outputs/InterScale/pearson_correlation_matrix.csv")

# Spearman correlation
spearman_corr = df_corr.corr(method='spearman').round(2)
spearman_corr.to_csv("outputs/InterScale/spearman_correlation_matrix.csv")

# Heatmaps
plt.figure(figsize=(7, 6))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson Correlation Between Questionnaires")
plt.tight_layout()
plt.savefig("outputs/InterScale/pearson_heatmap.png")
plt.close()

plt.figure(figsize=(7, 6))
sns.heatmap(spearman_corr, annot=True, cmap="viridis", vmin=-1, vmax=1)
plt.title("Spearman Correlation Between Questionnaires")
plt.tight_layout()
plt.savefig("outputs/InterScale/spearman_heatmap.png")
plt.close()

# Save paired correlations with p-values
results = []
for i, q1 in enumerate(questionnaire_totals):
    for j, q2 in enumerate(questionnaire_totals):
        if i < j:
            r_p, p_p = pearsonr(df[q1], df[q2])
            r_s, p_s = spearmanr(df[q1], df[q2])
            results.append([q1, q2, round(r_p, 3), round(p_p, 3), round(r_s, 3), round(p_s, 3)])

pd.DataFrame(results, columns=["Scale1", "Scale2", "Pearson_r", "Pearson_p", "Spearman_r", "Spearman_p"])\
  .to_csv("outputs/InterScale/correlation_pairs.csv", index=False)
