import pandas as pd
import wandb

csv_file = "table3_tweettopic_results.csv"

# 1. CSV 파일 읽기
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"'{csv_file}' 파일을 찾을 수 없습니다. 먼저 export_table3.py를 실행해주세요.")
    exit()

# 2. WandB 초기화
run = wandb.init(
    project="vec2vec-evaluation",
    name="table3-tweettopic-summary",
    reinit=True
)

# 3. 전체 결과 테이블 로깅
raw_table = wandb.Table(dataframe=df)
wandb.log({"table3_tweettopic_results": raw_table})

# 4. 각 지표별 매트릭스(피벗 테이블) 생성 및 로깅
# DataFrame의 컬럼명('cos', 'T-1', 'Rank')을 기준으로 피벗을 수행합니다.
metrics = {"cos": "cosine", "T-1": "top-1", "Rank": "rank"}

for col_name, metric_name in metrics.items():
    if col_name in df.columns:
        # M1이 행, M2가 열이 되도록 변환
        pivot_df = df.pivot(index="M1", columns="M2", values=col_name).reset_index()
        metric_table = wandb.Table(dataframe=pivot_df)
        wandb.log({f"matrix_tweettopic_{metric_name}": metric_table})

print("W&B 대시보드로 표가 성공적으로 업로드되었습니다.")
run.finish()