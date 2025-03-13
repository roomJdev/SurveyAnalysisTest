import os
import subprocess
import sys
import datetime
import pandas as pd
import unicodedata
from fpdf import FPDF
from jinja2 import Template

# === 유니코드 제거 함수 (PDF용) ===
def sanitize_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# === 1. 설정 ===
current_python = sys.executable
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "outputs")
report_dir = os.path.join(output_dir, "report")
os.makedirs(report_dir, exist_ok=True)

# === 2. 분석 스크립트 실행 ===
scripts = sorted([
    f for f in os.listdir(current_dir)
    if f.endswith(".py") and f[0].isdigit() and f != "10_Final.py"
])

print("🚀 Executing all numbered Python scripts in order:\n")
for script in scripts:
    script_path = os.path.join(current_dir, script)
    print(f"▶ Running {script}...")
    result = subprocess.run([current_python, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n❌ Error in {script}:\n{result.stderr}")
        break
    else:
        print(f"✅ Finished {script}\n")

# === 3. 공통 분석 요약 (텍스트, PDF, HTML, CSV) ===
scales = ["SSS", "MACHI", "BIS", "SSQ", "VRSQ", "PQ"]
summary = {}
core_stats = []
cronbach_alpha = []
html_blocks = []

# === PDF 클래스 정의 ===
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Survey Analysis Summary Report", ln=True, align="C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, title, ln=True, align="L")
        self.ln(2)

    def chapter_body(self, text):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 5, text)
        self.ln()

    def add_section(self, title, text):
        self.add_page()
        self.chapter_title(sanitize_text(title))
        self.chapter_body(sanitize_text(text))

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

for scale in scales:
    scale_path = os.path.join(output_dir, scale)
    if not os.path.exists(scale_path):
        continue

    summary[scale] = {
        "files": sorted([
            f for f in os.listdir(scale_path)
            if f.endswith((".txt", ".csv", ".png"))
        ])
    }

    text_summary = f"[{scale}] Summary Statistics and Cronbach's Alpha\n"
    stat_line = [scale]
    alpha_path = os.path.join(scale_path, f"{scale.lower()}_cronbach_alpha.txt")

    # Cronbach's α
    if os.path.exists(alpha_path):
        with open(alpha_path, "r", encoding="utf-8") as f:
            alpha = f.read().strip().split(":")[-1].strip()
            text_summary += f"- Cronbach's Alpha: {alpha}\n"
            cronbach_alpha.append([scale, alpha])
    else:
        text_summary += "- Cronbach's Alpha: (데이터 없음)\n"
        cronbach_alpha.append([scale, "N/A"])

    # 기술통계 요약
    stat_files = [f for f in os.listdir(scale_path) if "stats" in f.lower() and f.endswith(".txt")]
    for stat_file in stat_files:
        with open(os.path.join(scale_path, stat_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if "mean" in line.lower() or "std" in line.lower():
                    text_summary += f"- {line.strip()}\n"
            stat_line.append(stat_file)

    core_stats.append(stat_line)
    pdf.add_section(f"{scale} Summary", text_summary)
    html_blocks.append(f"<h2>{scale}</h2><pre>{text_summary}</pre>")

# PDF 저장
pdf_path = os.path.join(report_dir, "summary_report.pdf")
pdf.output(pdf_path)

# HTML 대시보드 저장
html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Survey Analysis Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Survey Analysis Dashboard</h1>
    {{ content }}
</body>
</html>
""")
html_content = html_template.render(content="\n".join(html_blocks))
html_path = os.path.join(report_dir, "dashboard.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

# Cronbach 비교 테이블 저장
cronbach_df = pd.DataFrame(cronbach_alpha, columns=["Scale", "Cronbach_Alpha"])
cronbach_df.to_csv(os.path.join(report_dir, "cronbach_comparison.csv"), index=False)

# 핵심 통계 요약 테이블 저장
core_stat_df = pd.DataFrame(core_stats)
core_stat_df.to_csv(os.path.join(report_dir, "core_stats_overview.csv"), index=False)

# 텍스트 리포트도 생성
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
summary_text = f"Survey Analysis Summary Report ({now})\n\n"
for scale, content in summary.items():
    summary_text += f"[{scale}] Output Files:\n"
    for file in content["files"]:
        summary_text += f"  - {file}\n"
    summary_text += "\n"

with open(os.path.join(report_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print("✅ 분석 자동화 및 리포트 생성 완료.")


# import os
# import subprocess
# import sys
# import datetime
# import pandas as pd
# from fpdf import FPDF
# from jinja2 import Template
#
# # === 1. 설정 ===
# current_python = sys.executable
# current_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(current_dir, "outputs")
# report_dir = os.path.join(output_dir, "report")
# os.makedirs(report_dir, exist_ok=True)
#
# # === 2. 분석 스크립트 실행 ===
# scripts = sorted([
#     f for f in os.listdir(current_dir)
#     if f.endswith(".py") and f[0].isdigit() and f != "10_Final.py"
# ])
#
# print("🚀 Executing all numbered Python scripts in order:\n")
# for script in scripts:
#     script_path = os.path.join(current_dir, script)
#     print(f"▶ Running {script}...")
#     result = subprocess.run([current_python, script_path], capture_output=True, text=True)
#     if result.returncode != 0:
#         print(f"\n❌ Error in {script}:\n{result.stderr}")
#         break
#     else:
#         print(f"✅ Finished {script}\n")
#
# # === 3. 공통 분석 요약 (텍스트, PDF, HTML, CSV) ===
# scales = ["SSS", "MACHI", "BIS", "SSQ", "VRSQ", "PQ"]
# summary = {}
# core_stats = []
# cronbach_alpha = []
# pdf_blocks = []
# html_blocks = []
#
# # PDF 정의
# class PDF(FPDF):
#     def header(self):
#         self.set_font("Arial", "B", 12)
#         self.cell(0, 10, "Survey Analysis Summary Report", ln=True, align="C")
#         self.ln(5)
#
#     def chapter_title(self, title):
#         self.set_font("Arial", "B", 11)
#         self.cell(0, 8, title, ln=True, align="L")
#         self.ln(2)
#
#     def chapter_body(self, text):
#         self.set_font("Arial", "", 10)
#         self.multi_cell(0, 5, text)
#         self.ln()
#
#     def add_section(self, title, text):
#         self.add_page()
#         self.chapter_title(title)
#         self.chapter_body(text)
#
# pdf = PDF()
# pdf.set_auto_page_break(auto=True, margin=15)
# pdf.add_page()
#
# for scale in scales:
#     scale_path = os.path.join(output_dir, scale)
#     if not os.path.exists(scale_path):
#         continue
#
#     summary[scale] = {
#         "files": sorted([
#             f for f in os.listdir(scale_path)
#             if f.endswith((".txt", ".csv", ".png"))
#         ])
#     }
#
#     text_summary = f"[{scale}] Summary Statistics and Cronbach's Alpha\n"
#     stat_line = [scale]
#     alpha_path = os.path.join(scale_path, f"{scale.lower()}_cronbach_alpha.txt")
#
#     # Cronbach's α
#     if os.path.exists(alpha_path):
#         with open(alpha_path, "r", encoding="utf-8") as f:
#             alpha = f.read().strip().split(":")[-1].strip()
#             text_summary += f"- Cronbach’s Alpha: {alpha}\n"
#             cronbach_alpha.append([scale, alpha])
#     else:
#         text_summary += "- Cronbach’s Alpha: (데이터 없음)\n"
#         cronbach_alpha.append([scale, "N/A"])
#
#     # 기술통계 요약
#     stat_files = [f for f in os.listdir(scale_path) if "stats" in f.lower() and f.endswith(".txt")]
#     for stat_file in stat_files:
#         with open(os.path.join(scale_path, stat_file), "r", encoding="utf-8") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if "mean" in line.lower() or "std" in line.lower():
#                     text_summary += f"- {line.strip()}\n"
#             stat_line.append(stat_file)
#
#     core_stats.append(stat_line)
#     pdf.add_section(f"{scale} Summary", text_summary)
#     html_blocks.append(f"<h2>{scale}</h2><pre>{text_summary}</pre>")
#
# # PDF 저장
# pdf_path = os.path.join(report_dir, "summary_report.pdf")
# pdf.output(pdf_path)
#
# # HTML 대시보드 저장
# html_template = Template("""
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Survey Analysis Dashboard</title>
#     <style>
#         body { font-family: Arial, sans-serif; margin: 20px; }
#         h1 { color: #2c3e50; }
#         pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
#     </style>
# </head>
# <body>
#     <h1>📊 Survey Analysis Dashboard</h1>
#     {{ content }}
# </body>
# </html>
# """)
# html_content = html_template.render(content="\n".join(html_blocks))
# html_path = os.path.join(report_dir, "dashboard.html")
# with open(html_path, "w", encoding="utf-8") as f:
#     f.write(html_content)
#
# # Cronbach 비교 테이블 저장
# cronbach_df = pd.DataFrame(cronbach_alpha, columns=["Scale", "Cronbach_Alpha"])
# cronbach_df.to_csv(os.path.join(report_dir, "cronbach_comparison.csv"), index=False)
#
# # 핵심 통계 요약 테이블 저장
# core_stat_df = pd.DataFrame(core_stats)
# core_stat_df.to_csv(os.path.join(report_dir, "core_stats_overview.csv"), index=False)
#
# # 텍스트 리포트도 생성
# now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
# summary_text = f"📊 통합 설문 분석 리포트 요약 ({now})\n\n"
# for scale, content in summary.items():
#     summary_text += f"🧪 {scale} 분석 결과:\n"
#     for file in content["files"]:
#         summary_text += f"  - {file}\n"
#     summary_text += "\n"
#
# with open(os.path.join(report_dir, "summary_report.txt"), "w", encoding="utf-8") as f:
#     f.write(summary_text)
#
# print("✅ 분석 자동화 및 리포트 생성 완료.")


# import os
# import subprocess
# import sys
# import datetime
#
# # ✅ 현재 파이썬 가상환경의 실행 경로
# current_python = sys.executable
#
# # ✅ 현재 디렉토리 기준
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# # ✅ 분석 스크립트 목록 (숫자로 시작하고 .py 확장자)
# scripts = sorted([
#     f for f in os.listdir(current_dir)
#     if f.endswith(".py") and f[0].isdigit() and f != "10_Final.py"
# ])
#
# # ✅ 출력 경로 준비
# output_dir = os.path.join(current_dir, "outputs")
# report_dir = os.path.join(output_dir, "report")
# os.makedirs(report_dir, exist_ok=True)
#
# # ✅ 스크립트 실행
# print("🚀 Executing all numbered Python scripts in order:\n")
# for script in scripts:
#     script_path = os.path.join(current_dir, script)
#     print(f"▶ Running {script}...")
#
#     result = subprocess.run([current_python, script_path], capture_output=True, text=True)
#
#     if result.returncode != 0:
#         print(f"\n❌ Error in {script}:\n{result.stderr}")
#         break
#     else:
#         print(f"✅ Finished {script}\n")
#
# # ✅ 통합 리포트 생성
# scales = ["SSS", "MACHI", "BIS", "SSQ", "VRSQ", "PQ"]
# summary = {}
# for scale in scales:
#     scale_path = os.path.join(output_dir, scale)
#     if not os.path.exists(scale_path):
#         continue
#     summary[scale] = {
#         "files": sorted([
#             f for f in os.listdir(scale_path)
#             if f.endswith((".txt", ".csv", ".png"))
#         ])
#     }
#
# now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
# summary_text = f"📊 통합 설문 분석 리포트 요약 ({now})\n\n"
#
# for scale, content in summary.items():
#     summary_text += f"🧪 {scale} 분석 결과:\n"
#     for file in content["files"]:
#         summary_text += f"  - {file}\n"
#     summary_text += "\n"
#
# # ✅ 리포트 저장
# report_path = os.path.join(report_dir, "summary_report.txt")
# with open(report_path, "w", encoding="utf-8") as f:
#     f.write(summary_text)
#
# print(f"\n📄 통합 리포트 생성 완료: {report_path}")



# import os
# import subprocess
# import sys
#
# # 현재 파이썬 실행 경로 (.venv 경로를 자동으로 잡음)
# current_python = sys.executable
#
# # 현재 디렉토리 기준
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# # 실행 대상 스크립트 정렬
# scripts = sorted([
#     f for f in os.listdir(current_dir)
#     if f.endswith(".py") and f[0].isdigit() and f != "10_Final.py"
# ])
#
# # 실행
# print("🚀 Executing all numbered Python scripts in order:\n")
# for script in scripts:
#     script_path = os.path.join(current_dir, script)
#     print(f"▶ Running {script}...")
#     result = subprocess.run([current_python, script_path], capture_output=True, text=True)
#
#     if result.returncode != 0:
#         print(f"\n❌ Error in {script}:\n{result.stderr}")
#         break
#     else:
#         print(f"✅ Finished {script}\n")
#
# print("✅ All scripts completed (or halted on error).")
