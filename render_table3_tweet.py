import os
import json
import glob

# 논문 표기와 실제 디렉터리 명칭 매핑
display_names = {
    "granite": "gran.",
    "gtr": "gtr",
    "gte": "gte",
    "stella": "stel.",
    "e5": "e5"
}

models = ["granite", "gtr", "gte", "stella", "e5"]

print("\n" + "=" * 65)
print(f"{'M1':<8} | {'M2':<8} | {'cos(·) ↑':<12} | {'T-1 ↑':<8} | {'Rank ↓':<12}")
print("-" * 65)

for m1 in models:
    for m2 in models:
        if m1 == m2:
            continue
        
        # results 폴더 내 JSON 파일 패턴 매칭
        pattern = f"results/*tweettopic*_{m1}_{m2}*.json"
        matched_files = glob.glob(pattern)
        
        if not matched_files:
            pattern_alt = f"results/*tweettopic*_{m2}_{m1}*.json"
            matched_files = glob.glob(pattern_alt)

        disp_m1 = display_names.get(m1, m1)
        disp_m2 = display_names.get(m2, m2)

        if matched_files:
            try:
                with open(matched_files[0], "r") as f:
                    data = json.load(f)
                
                trans = data.get("trans", {})
                
                # cos, top1, rank 키 조회
                cos_val = trans.get(f"{m1}_{m2}_cos", trans.get(f"{m2}_{m1}_cos", 0.0))
                t1_val = trans.get(f"{m1}_{m2}_acc_1", trans.get(f"{m2}_{m1}_acc_1", trans.get(f"{m1}_{m2}_top1", 0.0)))
                rank_val = trans.get(f"{m1}_{m2}_mean_rank", trans.get(f"{m2}_{m1}_mean_rank", trans.get(f"{m1}_{m2}_rank", 0.0)))
                
                cos_str = f"{cos_val:.2f}" if isinstance(cos_val, (int, float)) else str(cos_val)
                t1_str = f"{t1_val:.2f}" if isinstance(t1_val, (int, float)) else str(t1_val)
                rank_str = f"{rank_val:.2f}" if isinstance(rank_val, (int, float)) else str(rank_val)
                
                print(f"{disp_m1:<8} | {disp_m2:<8} | {cos_str:<12} | {t1_str:<8} | {rank_str:<12}")
            except Exception:
                print(f"{disp_m1:<8} | {disp_m2:<8} | Error reading file")
        else:
            print(f"{disp_m1:<8} | {disp_m2:<8} | {'-':<12} | {'-':<8} | {'-':<12}")

print("=" * 65 + "\n")