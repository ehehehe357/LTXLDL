import os
import glob
import gzip
import csv
import pandas as pd
from pathlib import Path

# ==============================================================================
# 1. CẤU HÌNH THÔNG SỐ & ĐƠN GIÁ CỦA GEMINI 3.6 FLASH
# ==============================================================================
# Model: gemini-3.6-flash
# Đơn giá dòng Flash: $1.5 / 1M token Input, $7.5 / 1M token Output
PRICE_PER_M_INPUT = 1.5    # USD cho mỗi 1.000.000 token đầu vào
PRICE_PER_M_OUTPUT = 7.5    # USD cho mỗi 1.000.000 token đầu ra

# Prompt mẫu thực tế đang dùng trong src/llm.py
PROMPT_TEMPLATE = """Bạn là chuyên gia phân tích dữ liệu du lịch. Hãy phân tích bài đánh giá phòng Airbnb sau đây:
\"\"\"{review_text}\"\"\"
Trích xuất mã ngôn ngữ, cảm xúc chung (positive, mixed, negative) và các khía cạnh liên quan trong danh sách:
['vị trí', 'sạch sẽ', 'chủ nhà', 'giá trị', 'tiện nghi', 'độ ồn', 'an toàn']."""

# Schema JSON đầu ra chuẩn theo Pydantic trong src/llm.py:
# {"language": "en", "sentiment": "positive", "aspects": ["vị trí", "sạch sẽ"]}
ESTIMATED_OUTPUT_TOKENS_PER_REQ = 30  # Trung bình 30 token cho mỗi output JSON


# ==============================================================================
# 2. HÀM TÍNH TOÁN TOKEN ƯỚC LƯỢNG
# ==============================================================================
def estimate_tokens(text: str) -> int:
    """
    Quy đổi độ dài văn bản sang token (Quy chuẩn NLP quốc tế: 1 token ~ 4 ký tự).
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4))


# ==============================================================================
# 3. THỐNG KÊ TRÊN 1.000 REVIEW MẪU CỦA NHÓM
# ==============================================================================
def analyze_sample_reviews(processed_dir: str = "data/processed"):
    # Đọc tất cả các file reviews đã lọc cho nhóm
    files = sorted(Path(processed_dir).glob("reviews_*.csv.gz"))
    if not files:
        raise FileNotFoundError("Chưa tìm thấy file reviews nào trong data/processed.")

    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["id"], keep="first")
    
    # Lấy đúng 1.000 review mẫu có cờ llm_sample == 1
    sample_df = df_all[df_all["llm_sample"] == 1].copy()
    comments = sample_df["comments"].dropna().astype(str).tolist()

    # Tính lượng token cố định của phần Prompt mẫu (hướng dẫn + danh sách khía cạnh)
    fixed_prompt_tokens = estimate_tokens(PROMPT_TEMPLATE.format(review_text=""))
    # Cộng thêm ~60 token overhead cho JSON Schema do thư viện SDK gắn kèm
    fixed_prompt_tokens += 60  

    # Tính token của từng comment
    comment_tokens = [estimate_tokens(c) for c in comments]
    avg_comment_tokens = sum(comment_tokens) / len(comment_tokens)

    # Input token trung bình = Prompt cố định + Nội dung review
    avg_input_tokens = fixed_prompt_tokens + avg_comment_tokens
    total_sample_input = avg_input_tokens * len(comments)
    total_sample_output = ESTIMATED_OUTPUT_TOKENS_PER_REQ * len(comments)

    return {
        "num_samples": len(comments),
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": ESTIMATED_OUTPUT_TOKENS_PER_REQ,
        "total_sample_input": total_sample_input,
        "total_sample_output": total_sample_output
    }


# ==============================================================================
# 4. THỐNG KÊ TỔNG DỮ LIỆU THỰC TẾ TOÀN TP. PRAGUE QUA 4 MỐC
# ==============================================================================
def count_prague_total_reviews(raw_dir: str = "data/raw"):
    raw_files = sorted(glob.glob(f"{raw_dir}/*/reviews.csv.gz"))
    total_raw_rows = 0
    unique_ids = set()

    for fpath in raw_files:
        with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_raw_rows += 1
                unique_ids.add(row.get("id"))

    return total_raw_rows, len(unique_ids)


# ==============================================================================
# 5. HÀM CHÍNH: PHÓNG CHIẾU CHI PHÍ & IN BẢNG BÁO CÁO
# ==============================================================================
def main():
    print("Đang tính toán số liệu token từ 1.000 mẫu...")
    sample_stats = analyze_sample_reviews()

    print("Đang thống kê dữ liệu toàn TP. Prague trong data/raw/...")
    total_raw_rows, total_unique_reviews = count_prague_total_reviews()

    # Phóng chiếu cho toàn thành phố (Dựa trên số review DUY NHẤT sau khi khử trùng)
    projected_input_tokens = total_unique_reviews * sample_stats["avg_input_tokens"]
    projected_output_tokens = total_unique_reviews * sample_stats["avg_output_tokens"]

    # Tính tiền (USD)
    cost_sample_in = (sample_stats["total_sample_input"] / 1_000_000) * PRICE_PER_M_INPUT
    cost_sample_out = (sample_stats["total_sample_output"] / 1_000_000) * PRICE_PER_M_OUTPUT
    cost_sample_total = cost_sample_in + cost_sample_out

    cost_full_in = (projected_input_tokens / 1_000_000) * PRICE_PER_M_INPUT
    cost_full_out = (projected_output_tokens / 1_000_000) * PRICE_PER_M_OUTPUT
    cost_full_total = cost_full_in + cost_full_out

    # In bảng Markdown kết quả
    print("\n" + "="*80)
    print("BẢNG DỰ TOÁN CHI PHÍ GEMINI 3.6 FLASH - TOÀN THÀNH PHỐ PRAGUE")
    print("="*80)
    print(f"| Hạng mục / Chỉ số | Mẫu thử điểm (1.000 review) | Toàn TP. Prague ({total_unique_reviews:,} review duy nhất) |")
    print(f"|---|---|---|")
    print(f"| **Mô hình sử dụng** | gemini-3.6-flash | gemini-3.6-flash |")
    print(f"| **Số lượng Request (lần gọi)** | {sample_stats['num_samples']:,} | {total_unique_reviews:,} |")
    print(f"| **Input tokens TB / review** | {sample_stats['avg_input_tokens']:.1f} tokens | {sample_stats['avg_input_tokens']:.1f} tokens |")
    print(f"| **Output tokens TB / review** | {sample_stats['avg_output_tokens']} tokens | {sample_stats['avg_output_tokens']} tokens |")
    print(f"| **Tổng lượng Token Input** | {sample_stats['total_sample_input']:,.0f} tokens | {projected_input_tokens:,.0f} tokens (~{projected_input_tokens/1e6:.2f}M) |")
    print(f"| **Tổng lượng Token Output** | {sample_stats['total_sample_output']:,.0f} tokens | {projected_output_tokens:,.0f} tokens (~{projected_output_tokens/1e6:.2f}M) |")
    print(f"| **Chi phí Input (USD)** | ${cost_sample_in:.4f} | ${cost_full_in:.2f} |")
    print(f"| **Chi phí Output (USD)** | ${cost_sample_out:.4f} | ${cost_full_out:.2f} |")
    print(f"| **TỔNG CHI PHÍ ƯỚC TÍNH (USD)** | **${cost_sample_total:.4f}** *(Free Tier)* | **${cost_full_total:.2f}** (~{cost_full_total*25400:,.0f} VNĐ) |")
    print("="*80)

if __name__ == "__main__":
    main()