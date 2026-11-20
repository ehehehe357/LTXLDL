"""
Đầu vào : DataFrame đã qua QA (đầu ra của qa.run_qa) + bảng LLM phẳng + calendar.
Đầu ra  : data/processed/kpi_*.csv (mỗi KPI một file) và kpi_catalog.csv (bảng mô tả KPI).

Mọi file kpi_*.csv có cùng bố cục cột:
    area_type : cột dùng để nhóm (neighbourhood_cleansed / analysis_area / room_type / city)
    area      : giá trị nhóm ('ALL' = toàn thành phố)
    period    : mốc chụp (snapshot) hoặc năm của review ('ALL' = gộp mọi thời gian)
    value     : giá trị KPI
    n         : số quan sát tạo nên giá trị (để biết mẫu có đủ lớn không)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ENTIRE = "Entire home/apt"
NOISE_COL = "aspect_độ_ồn"          # đúng tên cột trong llm_sentiment_summary.csv
SCORE_MAP = {"positive": 1.0, "mixed": 0.0, "negative": -1.0}


# ----------------------------------------------------------------------------
# Tiện ích dùng chung
# ----------------------------------------------------------------------------
def _agg(df, area_col, period_col, value, n):
    """Nhóm theo (period, area). area_col/period_col = None nghĩa là gộp tất cả ('ALL').

    value, n: tuple (tên_cột, hàm_tổng_hợp) theo kiểu named aggregation của pandas.
    """
    d = df.assign(_pall="ALL", _aall="ALL")
    pcol = period_col or "_pall"
    acol = area_col or "_aall"
    out = (d.groupby([pcol, acol], dropna=True)
             .agg(value=value, n=n)
             .reset_index()
             .rename(columns={pcol: "period", acol: "area"}))
    out.insert(0, "area_type", area_col or "city")
    return out[["area_type", "area", "period", "value", "n"]]


def _kpi(df, area_cols, period_col, value, n, scale=1.0, all_period=False, ndigits=2):
    """Tính KPI cho mọi cách nhóm theo khu vực (+ toàn thành phố), theo thời gian."""
    parts = []
    for a in list(area_cols) + [None]:
        parts.append(_agg(df, a, period_col, value, n))
        if all_period:                       # thêm bản gộp mọi thời gian
            parts.append(_agg(df, a, None, value, n))
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    out["value"] = (out["value"] * scale).round(ndigits)
    return out.sort_values(["area_type", "area", "period"]).reset_index(drop=True)


def _area_cols(df, config):
    """Các cột nhóm khu vực: neighbourhood_cleansed (theo config) và analysis_area (A01-A08)."""
    main = config.get("kpi", {}).get("area_col", "neighbourhood_cleansed")
    cols = []
    for c in (main, "analysis_area"):
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")

def _find_calendar(processed_dir, snap):
    """Tìm calendar của mốc snap, chấp nhận .csv.gz hoặc .csv."""
    for ext in (".csv.gz", ".csv"):
        p = Path(processed_dir) / f"calendar_{snap}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

# ----------------------------------------------------------------------------
# 8 KPI — mỗi KPI một hàm. df = bảng listings ĐÃ LỌC ~qa_any_flag
# ----------------------------------------------------------------------------
def kpi_supply(df, config):
    """KPI 1 — Quy mô nguồn cung: số listing hoạt động (listing) theo khu vực × mốc."""
    return _kpi(df, _area_cols(df, config), "snapshot",
                value=("id", "nunique"), n=("id", "nunique"), ndigits=0)


def kpi_entire_home_share(df, config):
    """KPI 2 — Tỷ lệ nguyên căn (%) = số listing 'Entire home/apt' / tổng listing × 100."""
    d = df.assign(is_entire=(df["room_type"] == ENTIRE).astype(float))
    return _kpi(d, _area_cols(d, config), "snapshot",
                value=("is_entire", "mean"), n=("is_entire", "count"), scale=100)


def kpi_multi_host_share(df, config):
    """KPI 3 — Tỷ lệ listing của chủ nhà thương mại (%) = listing có host sở hữu
    >= ngưỡng (mặc định 2 căn) / tổng listing × 100."""
    thr = config.get("kpi", {}).get("commercial_threshold", 2)
    d = df.assign(host_n=_num(df, "calculated_host_listings_count"))
    d = d.dropna(subset=["host_n"])
    d["is_multi"] = (d["host_n"] >= thr).astype(float)
    return _kpi(d, _area_cols(d, config), "snapshot",
                value=("is_multi", "mean"), n=("is_multi", "count"), scale=100)


def kpi_median_price(df, config):
    """KPI 4 — Giá niêm yết trung vị (CZK/đêm). Chỉ tính dòng sạch VÀ có giá
    (qa_clean_price). Mốc 2 và 3 không có giá nên sẽ không xuất hiện."""
    d = df[df["qa_clean_price"]]
    return _kpi(d, _area_cols(d, config), "snapshot",
                value=("price_clean", "median"), n=("price_clean", "count"))


def kpi_occupancy(df, config):
    """KPI 5 — Số đêm lấp đầy ước tính TB/năm (đêm) = mean(estimated_occupancy_l365d)."""
    if "estimated_occupancy_l365d" not in df.columns:
        logger.warning("Thiếu cột estimated_occupancy_l365d — bỏ qua KPI 5")
        return None
    d = df.assign(occ=_num(df, "estimated_occupancy_l365d")).dropna(subset=["occ"])
    return _kpi(d, _area_cols(d, config), "snapshot",
                value=("occ", "mean"), n=("occ", "count"), ndigits=1)


def kpi_unavailable_rate(df, config, processed_dir="data/processed"):
    """KPI 6 — Tỷ lệ ngày không còn trống trên calendar 365 ngày tới (%)
    = số listing-ngày available='f' / tổng listing-ngày × 100.
    Lưu ý: gồm cả ngày đã đặt lẫn ngày chủ nhà tự khóa lịch."""
    area_cols = _area_cols(df, config)
    keep = ["snapshot", "id"] + area_cols
    frames = []
    for snap in sorted(df["snapshot"].unique()):
        path = _find_calendar(processed_dir, snap)
        if path is None:
            logger.warning(f"Không thấy calendar_{snap}.csv[.gz] — bỏ qua mốc {snap} của KPI 6")
            continue
        cal = pd.read_csv(path, usecols=["listing_id", "available"])
        ids = df.loc[df["snapshot"] == snap, keep].drop_duplicates("id")
        m = cal.merge(ids, left_on="listing_id", right_on="id", how="inner")
        m["unavail"] = (m["available"].astype(str).str.lower() == "f").astype(float)
        frames.append(m.drop(columns=["available", "listing_id"]))
    if not frames:
        return None
    cal_all = pd.concat(frames, ignore_index=True)
    return _kpi(cal_all, area_cols, "snapshot",
                value=("unavail", "mean"), n=("unavail", "count"), scale=100)


def _prepare_llm(df, llm_df):
    """Gắn khu vực/loại phòng vào bảng LLM (chỉ giữ review của listing còn sạch sau QA)."""
    info_cols = [c for c in ("neighbourhood_cleansed", "analysis_area", "room_type")
                 if c in df.columns]
    info = (df.sort_values("snapshot")
              .drop_duplicates("id", keep="last")[["id"] + info_cols])
    llm = llm_df.copy()
    llm["listing_id"] = pd.to_numeric(llm["listing_id"], errors="coerce")
    llm = llm.merge(info, left_on="listing_id", right_on="id", how="inner")
    llm["year"] = pd.to_datetime(llm["date"], errors="coerce").dt.year.astype("Int64").astype(str)
    llm = llm[llm["year"] != "<NA>"]
    return llm


def kpi_noise_rate(df, config, llm_df):
    """KPI 7 (LLM) — Tỷ lệ review nhắc đến độ ồn (%) = mean(aspect_độ_ồn) × 100,
    theo khu vực × năm của review."""
    llm = _prepare_llm(df, llm_df)
    if NOISE_COL not in llm.columns:
        logger.warning(f"Thiếu cột {NOISE_COL} — bỏ qua KPI 7")
        return None
    return _kpi(llm, _area_cols(llm, config), "year",
                value=(NOISE_COL, "mean"), n=(NOISE_COL, "count"),
                scale=100, all_period=True)

def kpi_noise_complaint_rate(df, config, llm_df):
    """KPI 7b (LLM) — Tỷ lệ review có dấu hiệu PHÀN NÀN về độ ồn (%).
    Proxy = review nhắc độ ồn VÀ sentiment là mixed/negative.
    Lý do: aspect_độ_ồn=1 cũng gồm cả lời khen ('quiet'), nên không thể coi là phàn nàn.
    Hạn chế: review mixed có thể khen ồn nhưng chê chỗ khác -> đây là ước lượng cận trên."""
    llm = _prepare_llm(df, llm_df)
    if NOISE_COL not in llm.columns:
        logger.warning(f"Thiếu cột {NOISE_COL} — bỏ qua KPI 7b")
        return None
    llm["noise_complaint"] = ((llm[NOISE_COL] == 1) &
                              (llm["sentiment"] != "positive")).astype(float)
    return _kpi(llm, _area_cols(llm, config), "year",
                value=("noise_complaint", "mean"), n=("noise_complaint", "count"),
                scale=100, all_period=True)

def kpi_sentiment_score(df, config, llm_df):
    """KPI 8 (LLM) — Điểm cảm xúc TB (-1..1) = mean(positive=1, mixed=0, negative=-1),
    theo khu vực và theo loại phòng × năm của review."""
    llm = _prepare_llm(df, llm_df)
    llm["score"] = llm["sentiment"].map(SCORE_MAP)
    llm = llm.dropna(subset=["score"])
    groups = _area_cols(llm, config) + ["room_type"]
    return _kpi(llm, groups, "year",
                value=("score", "mean"), n=("score", "count"),
                all_period=True, ndigits=3)


# ----------------------------------------------------------------------------
# Bảng mô tả KPI (đưa thẳng vào báo cáo): tên → mục tiêu → công thức → đơn vị
# ----------------------------------------------------------------------------
def build_catalog(currency="CZK", threshold=2):
    rows = [
        ("Nguồn cung listing", "kpi_supply.csv",
         "Đo quy mô và biến động nguồn cung Airbnb theo vùng/quý (Sở Quy hoạch)",
         "count(distinct id) sau QA", "listing"),
        ("Tỷ lệ nguyên căn", "kpi_entire_home_share.csv",
         "Phát hiện mức độ chuyển nhà ở thành chỗ lưu trú thương mại",
         "listing Entire home/apt / tổng listing × 100", "%"),
        ("Tỷ lệ chủ nhà thương mại", "kpi_multi_host_share.csv",
         "Đo mức độ tập trung nguồn cung vào chủ sở hữu nhiều căn",
         f"listing có calculated_host_listings_count >= {threshold} / tổng listing × 100", "%"),
        (f"Giá trung vị", "kpi_median_price.csv",
         "Theo dõi mặt bằng giá, so sánh giữa các vùng (chỉ mốc có giá)",
         "median(price_clean) trên dòng qa_clean_price", f"{currency}/đêm"),
        ("Số đêm lấp đầy ước tính", "kpi_occupancy.csv",
         "Nhận diện 'vùng đỏ' khai thác như khách sạn quanh năm",
         "mean(estimated_occupancy_l365d)", "đêm/năm"),
        ("Tỷ lệ ngày không còn trống", "kpi_unavailable_rate.csv",
         "Đo cầu/độ khai thác theo lịch 365 ngày tới",
         "listing-ngày available='f' / tổng listing-ngày × 100", "%"),
        ("Tỷ lệ review nhắc độ ồn (LLM)", "kpi_noise_rate.csv",
         "Xác định điểm nóng tiếng ồn để tuần tra/giám sát",
         "mean(aspect_độ_ồn) × 100", "%"),
        ("Tỷ lệ review phàn nàn độ ồn (LLM, proxy)", "kpi_noise_complaint_rate.csv",
         "Xác định điểm nóng tiếng ồn thực sự để tuần tra/giám sát",
         "review nhắc độ ồn & sentiment ≠ positive / tổng review × 100", "%"),
        ("Điểm cảm xúc trung bình (LLM)", "kpi_sentiment_score.csv",
         "Đo mức hài lòng của du khách theo vùng và loại phòng",
         "mean(positive=1, mixed=0, negative=-1)", "điểm (-1..1)"),
    ]
    return pd.DataFrame(rows, columns=["kpi", "file", "muc_tieu", "cong_thuc", "don_vi"])


# ----------------------------------------------------------------------------
# Hàm điều phối: run_kpi
# ----------------------------------------------------------------------------
def run_kpi(config, df=None, data_dir=None, report_dir="reports"):
    """Tính toàn bộ KPI và ghi data/processed/kpi_*.csv.

    df: bảng đã qua QA. Nếu None, tự gọi qa.run_qa(config).
    """
    processed_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    llm_csv = config.get("llm", {}).get("summary_csv", f"{processed_dir}/llm_sentiment_summary.csv")
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        try:
            from src.qa import run_qa
        except ImportError:
            from qa import run_qa
        df = run_qa(config, data_dir=data_dir, report_dir=report_dir)

    clean = df[~df["qa_any_flag"]].copy()
    logger.info(f"[KPI] {len(clean)}/{len(df)} dòng sạch dùng để tính KPI")

    results = {
        "kpi_supply": kpi_supply(clean, config),
        "kpi_entire_home_share": kpi_entire_home_share(clean, config),
        "kpi_multi_host_share": kpi_multi_host_share(clean, config),
        "kpi_median_price": kpi_median_price(clean, config),
        "kpi_occupancy": kpi_occupancy(clean, config),
        "kpi_unavailable_rate": kpi_unavailable_rate(clean, config, processed_dir),
    }

    llm_path = Path(llm_csv)
    if llm_path.exists():
        llm_df = pd.read_csv(llm_path)
        results["kpi_noise_rate"] = kpi_noise_rate(clean, config, llm_df)
        results["kpi_noise_complaint_rate"] = kpi_noise_complaint_rate(clean, config, llm_df)
        results["kpi_sentiment_score"] = kpi_sentiment_score(clean, config, llm_df)

    else:
        logger.warning(f"Không thấy {llm_path} — bỏ qua 2 KPI dùng LLM")

    for name, res in results.items():
        if res is None or res.empty:
            logger.warning(f"[KPI] {name}: không có dữ liệu, không ghi file")
            continue
        res.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        logger.info(f"[KPI] Đã ghi {name}.csv ({len(res)} dòng)")

    thr = config.get("kpi", {}).get("commercial_threshold", 2)
    build_catalog(config.get("currency", "CZK"), thr).to_csv(
        out_dir / "kpi_catalog.csv", index=False, encoding="utf-8-sig")
    return results


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Tính KPI")
    ap.add_argument("--config", default="configs/prague.yaml")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_kpi(cfg)