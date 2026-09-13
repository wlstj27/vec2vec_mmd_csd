import os
import json
import subprocess
from pathlib import Path
import pandas as pd

checkpoint_dir = Path("./checkpoints/final_model_release")
subdirs = sorted([x for x in checkpoint_dir.iterdir() if x.is_dir()])

table_data = []

for model_path in subdirs:
    pair_name = model_path.name  # 예: "gte_gtr"
    print(f"==> Processing model pair: {pair_name}")
    
    cmd = ["python", "eval.py", str(model_path)]
    
    try:
        # 1. eval.py 실행 (결과 json은 스크립트 내부 로직에 의해 results/ 폴더에 자동 저장됨)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running eval.py for {pair_name}: {e}")
        continue

    # 2. results/ 폴더에 생성된 json 파일명 추정 및 읽기
    # eval.py의 저장 로직: results/{dataset}_{unsup}_{sup}.json 형태
    # NQ 데이터셋 기본값이므로 보통 nq_*.json 형식으로 저장됨
    results_dir = Path("results")
    json_files = list(results_dir.glob(f"*_{pair_name.replace('_', '_')}.json")) or list(results_dir.glob("*.json"))
    
    # 가장 최근에 생성된 해당 조합의 json 파일을 찾거나 직접 매칭
    target_json = None
    m1, m2 = pair_name.split("_")[0], pair_name.split("_")[1]
    
    for jfile in results_dir.glob("*.json"):
        if m1 in jfile.name and m2 in jfile.name and "baseline" not in jfile.name and "ood" not in jfile.name:
            target_json = jfile
            break
            
    if target_json and target_json.exists():
        with open(target_json, "r") as f:
            metrics = json.load(f)
            
        cos_fwd = metrics["trans"].get(f"{m1}_{m2}_cos", 0.0)
        top1_fwd = metrics["heatmap"].get(f"{m1}_{m2}_top_1_acc (avg. 4 batches)", 0.0)
        rank_fwd = metrics["heatmap"].get(f"{m1}_{m2}_rank (avg. 4 batches)", 0.0)
        
        cos_bwd = metrics["trans"].get(f"{m2}_{m1}_cos", 0.0)
        top1_bwd = metrics["heatmap"].get(f"{m2}_{m1}_top_1_acc (avg. 4 batches)", 0.0)
        rank_bwd = metrics["heatmap"].get(f"{m2}_{m1}_rank (avg. 4 batches)", 0.0)
        
        table_data.append({"M1": m1, "M2": m2, "Cosine": cos_fwd, "Top-1": top1_fwd, "Rank": rank_fwd})
        table_data.append({"M1": m2, "M2": m1, "Cosine": cos_bwd, "Top-1": top1_bwd, "Rank": rank_bwd})

# 판다스로 표 생성
df = pd.DataFrame(table_data)
if not df.empty:
    print("\n" + "="*50)
    print(" [ Table 2 Reconstructed Summary ] ")
    print("="*50)
    
    cos_pivot = df.pivot(index="M1", columns="M2", values="Cosine")
    top1_pivot = df.pivot(index="M1", columns="M2", values="Top-1")
    rank_pivot = df.pivot(index="M1", columns="M2", values="Rank")
    
    print("\n--- Cosine Similarity ---")
    print(cos_pivot.round(2))
    
    print("\n--- Top-1 Accuracy ---")
    print(top1_pivot.round(2))
    
    print("\n--- Mean Rank ---")
    print(rank_pivot.round(2))
    
    df.to_csv("table2_reproduced_results.csv", index=False)
    print("\n결과가 'table2_reproduced_results.csv' 파일로 저장되었습니다.")
else:
    print("수집된 데이터가 없습니다.")