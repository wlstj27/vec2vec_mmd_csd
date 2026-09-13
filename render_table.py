import pandas as pd
from tabulate import tabulate

csv_file = "table2_full_reproduced.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"'{csv_file}' 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
    exit()

metrics = ["Cosine", "Top-1", "Rank"]

for metric in metrics:
    if metric in df.columns:
        # 매트릭스 형태로 피벗
        pivot_df = df.pivot(index="M1", columns="M2", values=metric).round(2)
        
        print("\n" + "="*60)
        print(f" [ Metric: {metric} ]")
        print("="*60)
        
        # grid 스타일로 서식 지정 후 출력
        print(tabulate(pivot_df, headers="keys", tablefmt="fancy_grid"))

# 보고서용 HTML 및 Markdown 파일로도 동시 저장
with open("table2_results_pretty.md", "w") as f:
    for metric in metrics:
        if metric in df.columns:
            pivot_df = df.pivot(index="M1", columns="M2", values=metric).round(2)
            f.write(f"### {metric}\n\n")
            f.write(tabulate(pivot_df, headers="keys", tablefmt="github"))
            f.write("\n\n")

print("\n마크다운 표가 'table2_results_pretty.md' 파일로 저장되었습니다.")