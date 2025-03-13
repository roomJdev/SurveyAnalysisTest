import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load your Excel file
df = pd.read_excel("GPT_Format_250313.xlsx")

# Define subscales
subscales = {
    "SSS": ["SSS_TAS", "SSS_ES", "SSS_BS", "SSS_DIS"],
    "BIS": ["BIS_Attentional", "BIS_Motor", "BIS_NonPlanning"],
    "SSQ": ["SSQ_Nausea", "SSQ_Oculomotor", "SSQ_Disorientation"],
    "PQ": [
        "PQ_Realism", "PQ_PossibilityToAct", "PQ_QualityOfInterface",
        "PQ_PossibilityToExamine", "PQ_SelfEvaluationOfPerformance", "PQ_Sounds"
    ]
}

# Create output directories
output_dir = "outputs/SubscalePlots"
os.makedirs(output_dir, exist_ok=True)

# Radar chart function
def draw_radar(values, labels, title, path):
    values = values.tolist() + [values.iloc[0]]  # ✅ 여기 수정
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))] + [0]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# PDF Report Init
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Subscale Analysis Report", ln=True, align="C")
pdf.ln(10)

# Main Loop
for scale, cols in subscales.items():
    scale_dir = os.path.join(output_dir, scale)
    os.makedirs(scale_dir, exist_ok=True)

    mean_vals = df[cols].mean()

    # Radar
    radar_path = os.path.join(scale_dir, f"{scale}_radar.png")
    draw_radar(mean_vals, cols, f"{scale} - Radar", radar_path)

    # Bar
    plt.figure(figsize=(8, 4))
    sns.barplot(x=cols, y=mean_vals.values)
    plt.title(f"{scale} - Bar Chart")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(scale_dir, f"{scale}_bar.png"))
    plt.close()

    # Line
    plt.figure(figsize=(8, 4))
    plt.plot(cols, mean_vals.values, marker='o')
    plt.title(f"{scale} - Line Chart")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(scale_dir, f"{scale}_line.png"))
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df[cols])
    plt.title(f"{scale} - Boxplot")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(scale_dir, f"{scale}_boxplot.png"))
    plt.close()

    # Overlay per participant
    for i, row in df.iterrows():
        participant_id = row.get("Participant", f"P{i+1}")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df[cols])
        sns.scatterplot(x=np.arange(len(cols)), y=row[cols].values, color='red', s=60, zorder=10)
        plt.xticks(np.arange(len(cols)), cols, rotation=45)
        plt.title(f"{scale} - Overlay: {participant_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(scale_dir, f"{scale}_overlay_{participant_id}.png"))
        plt.close()

    # Correlation Heatmap
    corr = df[cols].corr().round(2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"{scale} - Correlation Matrix")
    plt.tight_layout()
    heatmap_path = os.path.join(scale_dir, f"{scale}_correlation.png")
    plt.savefig(heatmap_path)
    plt.close()

    # PDF Entry
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"{scale} Subscale", ln=True)
    for suffix in ["radar", "bar", "line", "boxplot", "correlation"]:
        img_path = os.path.join(scale_dir, f"{scale}_{suffix}.png")
        if os.path.exists(img_path):
            pdf.image(img_path, w=180)
            pdf.ln(5)

# Save PDF
pdf.output(os.path.join(output_dir, "Subscale_Report_FULL.pdf"))
