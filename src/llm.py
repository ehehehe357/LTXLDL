import os
import sys
import json
import time
import yaml
import gzip
import pandas as pd
from pathlib import Path
from typing import List, Literal, Dict, Any, Set
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, Field

# ==============================================================================
# 1. ĐỊNH NGHĨA SCHEMA ĐẦU RA (HỖ TRỢ BATCH NHIỀU REVIEW)
# ==============================================================================
AspectType = Literal["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]

class ReviewAnalysis(BaseModel):
    review_id: str = Field(description="Mã review_id tương ứng")
    language: str = Field(description="Mã ngôn ngữ ISO (ví dụ: 'en', 'cs', 'vi')")
    sentiment: Literal["positive", "mixed", "negative"] = Field(description="Cảm xúc tổng thể")
    aspects: List[AspectType] = Field(description="Danh sách các khía cạnh xuất hiện trong nhận xét")

class BatchReviewAnalysis(BaseModel):
    reviews: List[ReviewAnalysis] = Field(description="Danh sách kết quả phân tích cho từng bài review")

DEFAULT_ASPECTS = ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]

def load_aspects_from_config(config_path: str = "configs/prague.yaml") -> List[str]:
    """Đọc danh sách aspects từ file cấu hình YAML, fallback về DEFAULT_ASPECTS nếu không đọc được."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                loaded = cfg.get("llm", {}).get("aspects")
                if loaded and isinstance(loaded, list):
                    return loaded
        except Exception:
            pass
    return DEFAULT_ASPECTS

ALL_ASPECTS = load_aspects_from_config()

# ==============================================================================
# 2. KHỞI TẠO CLIENT
# ==============================================================================
def get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Chưa tìm thấy GEMINI_API_KEY trong file .env")
    return genai.Client(api_key=api_key)

# ==============================================================================
# 3. QUẢN LÝ CACHE JSONL
# ==============================================================================
def load_existing_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    cache = {} 
    path = Path(cache_path)
    if not path.exists():
        return cache

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    cache[str(data["review_id"])] = data
                except Exception:
                    continue
    return cache

def append_to_cache(cache_path: str, record: Dict[str, Any]):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ==============================================================================
# 4. HÀM GỌI API THEO BATCH 5 REVIEW CÓ RETRY CHỐNG NGHẼN
# ==============================================================================
def analyze_batch_with_retry(
    client: genai.Client,
    model_name: str,
    batch_records: List[Dict[str, Any]],
    max_retries: int = 5
) -> List[ReviewAnalysis]:
    formatted_reviews = "\n\n".join([
        f"--- BÀI ĐÁNH GIÁ (ID: {r['id']}) ---\n\"{r['comments']}\""
        for r in batch_records
    ])

    prompt = f"""Bạn là chuyên gia phân tích dữ liệu du lịch. Hãy phân tích từng bài đánh giá phòng Airbnb dưới đây:

{formatted_reviews}

Với MỖI bài đánh giá (khớp đúng theo ID):
1. review_id: Đúng mã ID của bài đánh giá đó.
2. language: Mã ngôn ngữ ISO (ví dụ: 'en', 'cs', 'vi').
3. sentiment: Cảm xúc chung (positive, mixed, negative).
4. aspects: Các khía cạnh liên quan trong danh sách ['vị trí', 'sạch sẽ', 'chủ nhà', 'giá trị', 'tiện nghi', 'độ ồn', 'an toàn']."""

    delay = 4.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=BatchReviewAnalysis,
                    temperature=0.0,
                ),
            )
            raw_text = response.text.strip()
            # Làm sạch nếu dính markdown ```json
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`").removeprefix("json").strip()
            
            data = json.loads(raw_text)

            # Xử lý linh hoạt cả khi Gemini trả về list [...] lẫn dict {"reviews": [...]}
            if isinstance(data, list):
                return [ReviewAnalysis(**item) for item in data]
            elif isinstance(data, dict):
                if "reviews" in data and isinstance(data["reviews"], list):
                    return [ReviewAnalysis(**item) for item in data["reviews"]]
                return [ReviewAnalysis(**data)]
            return []

        except (ClientError, ServerError) as e:
            if attempt == max_retries - 1:
                raise e
            print(f"    [Thử lại {attempt+1}/{max_retries}] Nghẽn mạng / Rate limit ({e}). Chờ {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"    [Thử lại {attempt+1}/{max_retries}] Lỗi parse ({e}). Chờ {delay:.1f}s...")
            time.sleep(delay)
    return []

# ==============================================================================
# 5. ĐỌC DỮ LIỆU VÀ LẤY MẪU 1.000 REVIEW
# ==============================================================================
def load_and_sample_reviews(data_dir: str = "data/processed", sample_size: int = 1000, seed: int = 42) -> pd.DataFrame:
    processed_path = Path(data_dir)
    review_files = sorted(processed_path.glob("reviews_*.csv.gz"))
    
    if not review_files:
        raise FileNotFoundError(f"Không tìm thấy file reviews nào trong {data_dir}.")

    dfs = []
    for f in review_files:
        snapshot_name = f.name.replace("reviews_", "").replace(".csv.gz", "")
        df_part = pd.read_csv(
            f, 
            usecols=["id", "listing_id", "date", "comments", "llm_sample"],
            low_memory=False
        )
        df_part["snapshot"] = snapshot_name
        dfs.append(df_part)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["comments"])
    df_all["id"] = df_all["id"].astype(str)
    df_all = df_all.drop_duplicates(subset=["id"], keep="first")
    
    df_target = df_all[df_all["llm_sample"] == 1].copy()
    print(f"Đã tải thành công {len(df_target):,} review thuộc tập mẫu LLM của nhóm.")

    if sample_size < len(df_target):
        print(f"Chế độ chạy thử: trích {sample_size}/{len(df_target)} review (seed={seed}).")
        return df_target.sample(n=sample_size, random_state=seed)

    return df_target

# ==============================================================================
# 6. XUẤT BẢNG PHẲNG CSV CUNG CẤP CHO PHƯƠNG TÍNH KPI
# ==============================================================================
def export_sentiment_summary(
    cache_records: List[Dict[str, Any]], 
    output_csv_path: str,
    aspects: List[str] = None
):
    if aspects is None:
        aspects = ALL_ASPECTS

    rows = []
    for r in cache_records:
        row = {
            "review_id": str(r["review_id"]),
            "listing_id": str(r["listing_id"]),
            "snapshot": str(r.get("snapshot", "")),
            "date": str(r.get("date", "")),
            "language": str(r.get("language", "")),
            "sentiment": str(r.get("sentiment", "")),
            "is_positive": 1 if r.get("sentiment") == "positive" else 0,
            "is_mixed": 1 if r.get("sentiment") == "mixed" else 0,
            "is_negative": 1 if r.get("sentiment") == "negative" else 0,
        }
        aspects_set = set(r.get("aspects", []))
        for asp in aspects:
            col_name = "aspect_" + asp.replace(" ", "_")
            row[col_name] = 1 if asp in aspects_set else 0
        rows.append(row)
        
    df_out = pd.DataFrame(rows)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Đã xuất bảng tổng hợp cho KPI: {output_csv_path} ({len(df_out)} dòng)")

# ==============================================================================
# 7. HÀM ĐIỀU PHỐI CHÍNH (GOM NHÓM 5 REVIEWS / REQUEST)
# ==============================================================================
def run_llm_pipeline(config_path: str = "configs/prague.yaml", skip_llm: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    llm_cfg = cfg.get("llm", {})
    model_name = llm_cfg.get("model", "gemini-3.1-flash-lite")
    sample_size = llm_cfg.get("sample_size", 1000)
    seed = llm_cfg.get("seed", 42)
    aspects = llm_cfg.get("aspects", ALL_ASPECTS)
    cache_path = llm_cfg.get("cache", "data/processed/llm_cache.jsonl")
    summary_csv = llm_cfg.get("summary_csv", "data/processed/llm_sentiment_summary.csv")
    processed_dir = cfg.get("paths", {}).get("processed_dir", "data/processed")
    
    print(f"\n--- BẮT ĐẦU PIPELINE LLM BATCHING (Model: {model_name}, Mẫu: {sample_size}) ---")
    
    existing_cache = load_existing_cache(cache_path)
    print(f"Đã có sẵn {len(existing_cache)} bài review trong bộ nhớ đệm (cache).")
    
    if skip_llm:
        print("Tùy chọn skip_llm=True: Xuất kết quả từ cache.")
        export_sentiment_summary(list(existing_cache.values()), summary_csv, aspects=aspects)
        return

    df_samples = load_and_sample_reviews(processed_dir, sample_size, seed)
    client = get_client()

    # Lọc ra những review chưa có trong cache
    pending_df = df_samples[~df_samples["id"].astype(str).isin(existing_cache)].copy()
    print(f"Số review cần gọi mới: {len(pending_df)} / {len(df_samples)} bài.")

    if len(pending_df) == 0:
        print("Tất cả review đã có trong cache! Xuất file tổng hợp...")
        export_sentiment_summary(list(existing_cache.values()), summary_csv, aspects=aspects)
        return

    # Chia nhóm 5 review / 1 batch
    BATCH_SIZE = 5
    records = pending_df.to_dict(orient="records")
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    new_calls = 0

    for b_idx in range(total_batches):
        batch = records[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        batch_ids = [str(r["id"]) for r in batch]
        print(f"[{b_idx + 1}/{total_batches}] Đang gửi Batch {len(batch)} reviews (IDs: {batch_ids[0]}...)...")

        try:
            results = analyze_batch_with_retry(client, model_name, batch)
            # Tạo map kết quả theo review_id để lưu chính xác
            res_dict = {str(item.review_id): item for item in results}

            for row in batch:
                r_id = str(row["id"])
                analysis = res_dict.get(r_id)
                if analysis:
                    rec = {
                        "review_id": r_id,
                        "listing_id": str(row["listing_id"]),
                        "snapshot": str(row["snapshot"]),
                        "date": str(row["date"]),
                        "language": analysis.language,
                        "sentiment": analysis.sentiment,
                        "aspects": analysis.aspects
                    }
                    append_to_cache(cache_path, rec)
                    existing_cache[r_id] = rec
                    new_calls += 1

            # Nghỉ 4.2s giữa các batch để giữ nhịp ~14 request/phút (tuyệt đối an toàn dưới trần 15 RPM)
            time.sleep(4.2)

        except Exception as e:
            print(f"  ❌ Lỗi xử lý Batch {b_idx + 1}: {e}")
            time.sleep(5.0)
            continue

    print(f"\n🎉 Hoàn thành xuất sắc! Đã phân tích mới thành công {new_calls} reviews.")
    export_sentiment_summary(list(existing_cache.values()), summary_csv, aspects=aspects)

if __name__ == "__main__":
    run_llm_pipeline()