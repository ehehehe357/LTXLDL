"""Chạy 1 lần: xuất ID của nhóm 11 Prague từ file phân công ra CSV nhỏ."""
import sys
import pandas as pd

MA = "DSE2049-C05-G11-prague"
xlsx = sys.argv[1]

L = pd.read_excel(xlsx, sheet_name="Listing")
R = pd.read_excel(xlsx, sheet_name="Review")

L = L[L["Mã dữ liệu"] == MA][["listing_id", "analysis_area"]]
R = R[R["Mã dữ liệu"] == MA].copy()
R["llm_sample"] = R["Mẫu LLM"].notna().astype(int)
R["hand_label"] = R["Nhãn tay"].notna().astype(int)
R = R[["review_id", "listing_id", "llm_sample", "hand_label"]]

L.to_csv("samples/listing_ids.csv", index=False)
R.to_csv("samples/review_ids.csv", index=False)
print(len(L), len(R), R["llm_sample"].sum(), R["hand_label"].sum())