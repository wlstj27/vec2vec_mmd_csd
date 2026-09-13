import os
import glob
import json
import pandas as pd

# 논문 약어 표기 매핑
display_names = {
    "granite": "gran.",
    "gtr": "gtr",
    "gte": "gte",
    "stella": "stel.",
    "e5": "e5"
}

result_files = sorted(glob.glob("results/*tweettopic*.json"))

if not result_files:
    print("results/ 폴더에 TweetTopic JSON 결과 파일이 없습니다.")
    exit()

records = []
for filepath in result_files:
    try:
        with open(filepath, "r") as f:
            data = json.load(f)

        trans = data.get("trans", {})
        heatmap = data.get("heatmap", {})

        # trans 딕셔너리에서 모든 A_B_cos 패턴(양방향 모두)을 찾습니다.
        for k, v in trans.items():
            if k.endswith("_cos") and not k.endswith("_cos_var") and not k.endswith("_cos_se"):
                pair_name = k[:-4]  # 예: 'gte_e5' 또는 'e5_gte'
                m1, m2 = pair_name.split("_", 1)

                cos_val = v
                t1_val = None
                rank_val = None

                # heatmap에서 해당 pair의 top_1_acc와 rank를 매칭하여 추출합니다.
                for hk, hv in heatmap.items():
                    if hk.startswith(f"{pair_name}_top_1_acc"):
                        t1_val = hv
                    elif hk.startswith(f"{pair_name}_rank") and "_var" not in hk and "_se" not in hk:
                        rank_val = hv

                records.append({
                    "M1_raw": m1,
                    "M2_raw": m2,
                    "M1": display_names.get(m1, m1),
                    "M2": display_names.get(m2, m2),
                    "cos": round(cos_val, 2) if cos_val is not None else None,
                    "T-1": f"{t1_val * 100:.1f}" if t1_val is not None else "-",
                    "Rank": round(rank_val, 2) if rank_val is not None else "-"
                })

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

df = pd.DataFrame(records)

# 중복 방지 (만약 같은 모델 쌍이 여러 번 계산되었을 경우 대비)
df = df.drop_duplicates(subset=["M1", "M2"])

# ==========================================
# 1. CSV 파일 저장
# ==========================================
csv_filename = "table3_tweettopic_results.csv"
df[["M1", "M2", "cos", "T-1", "Rank"]].to_csv(csv_filename, index=False, encoding="utf-8-sig")
print(f"CSV 저장 완료: {csv_filename}\n")

# ==========================================
# 2. 논문 Table 3 스타일 테이블 출력
# ==========================================
model_order = ["gran.", "gtr", "gte", "stel.", "e5"]
df["M1_order"] = df["M1"].map(lambda x: model_order.index(x) if x in model_order else 99)
df["M2_order"] = df["M2"].map(lambda x: model_order.index(x) if x in model_order else 99)
df = df.sort_values(by=["M1_order", "M2_order"]).reset_index(drop=True)

print("=" * 55)
print("             Table 3: TweetTopic Results            ")
print("=" * 55)
print(f"{'M1':<8} | {'M2':<8} | {'cos(·) ↑':<10} | {'T-1 ↑':<8} | {'Rank ↓':<8}")
print("-" * 55)

for _, row in df.iterrows():
    if row['M1'] != row['M2']:
        cos_str = f"{row['cos']:.2f}" if pd.notnull(row['cos']) else "-"
        t1_str = str(row['T-1'])
        rank_str = str(row['Rank'])
        print(f"{row['M1']:<8} | {row['M2']:<8} | {cos_str:<10} | {t1_str:<8} | {rank_str:<8}")

print("=" * 55)