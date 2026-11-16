"""
src/qa.py — Bộ quy tắc QA (Quality Assurance) cho dữ liệu Airbnb.
Tác giả : Nam (nhóm 11)
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# Tiện ích: Chuyển đổi cột giá dạng chuỗi sang số thực float
def parse_price(price_series: pd.Series) -> pd.Series:
    clean = (
        price_series.dropna()                                         # Loại bỏ NaN tạm thời để xử lý chuỗi
        .astype(str)                                                  # Ép kiểu dữ liệu về chuỗi ký tự
        .str.replace('$', '', regex=False)                            # Xóa ký hiệu đô la
        .str.replace(',', '', regex=False)                            # Xóa dấu phẩy ngăn cách hàng nghìn
        .str.strip()                                                  # Cắt bỏ khoảng trắng thừa hai đầu
    )
    parsed = pd.to_numeric(clean, errors='coerce')                    # Chuyển thành số thực, giá trị lỗi thành NaN
    return parsed.reindex(price_series.index)                         # Trả về Series theo đúng index ban đầu


# Quy tắc 1: Kiểm tra thiếu dữ liệu cột cấu trúc cốt lõi và cờ thiếu giá
def flag_missing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    essential_cols = [                                                # Danh sách các cột cấu trúc bắt buộc phải có
        'id', 'host_id', 'neighbourhood_cleansed', 'analysis_area',
        'latitude', 'longitude', 'room_type'
    ]

    check_cols = [c for c in essential_cols if c in df.columns]       # Chỉ kiểm tra các cột thực tế có trong bảng

    # Cờ thiếu thông tin cấu trúc cốt lõi
    df['qa_missing'] = df[check_cols].isna().any(axis=1)             # True nếu bất kỳ cột cấu trúc nào bị NaN

    # Cờ thiếu giá riêng biệt (phục vụ KPI giá)
    df['qa_missing_price'] = df['price_clean'].isna()                # True nếu cột giá bị khuyết (NaN)

    n_flagged = df['qa_missing'].sum()                               # Đếm số dòng thiếu thông tin cấu trúc
    n_missing_price = df['qa_missing_price'].sum()                   # Đếm số dòng thiếu giá
    logger.info(f"[QA-missing] Flagged {n_flagged}/{len(df)} rows thiếu cột cấu trúc; "
                f"{n_missing_price}/{len(df)} rows thiếu giá")

    return df                                                        # Trả về DataFrame đã gắn cờ thiếu dữ liệu


# Quy tắc 2: Kiểm tra dữ liệu ngoại lai (giá quá cao, giá <= 0, số đêm thuê > 365)
def flag_outlier(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    qa_config = config.get('qa', {})                                 # Đọc cấu hình QA từ file YAML
    price_max = qa_config.get('price_max_czk', 50000)                # Ngưỡng giá trần (mặc định 50,000 CZK)
    min_nights_cap = qa_config.get('min_nights_cap', 365)            # Ngưỡng số đêm thuê tối đa (mặc định 365 đêm)

    # Điều kiện giá (chỉ so sánh trên dòng có giá)
    outlier_price_high = df['price_clean'] > price_max               # Giá vượt quá ngưỡng tối đa cho phép
    outlier_price_zero = df['price_clean'] <= 0                      # Giá bằng 0 hoặc âm không hợp lệ

    # Điều kiện minimum_nights
    outlier_min_nights = df['minimum_nights'] > min_nights_cap       # Số đêm tối thiểu vượt quá 1 năm

    # Gắn cờ: True nếu bất kỳ điều kiện nào thỏa mãn
    df['qa_outlier'] = outlier_price_high | outlier_price_zero | outlier_min_nights  # Kết hợp các điều kiện ngoại lai

    n_price = (outlier_price_high | outlier_price_zero).sum()         # Đếm số dòng ngoại lai về giá
    n_nights = outlier_min_nights.sum()                              # Đếm số dòng ngoại lai về số đêm
    n_total = df['qa_outlier'].sum()                                 # Tổng số dòng bị gắn cờ ngoại lai
    logger.info(f"[QA-outlier] Flagged {n_total}/{len(df)} rows — "
                f"giá ngoại lai: {n_price}, min_nights>{min_nights_cap}: {n_nights}")

    return df                                                        # Trả về DataFrame đã gắn cờ ngoại lai


# Quy tắc 3: Kiểm tra trùng lặp dữ liệu trong 1 mốc và giữa các mốc
def flag_duplicate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    qa_config = config.get('qa', {})                                 # Đọc cấu hình QA
    dedupe_keys = qa_config.get('dedupe_keys', ['id', 'last_scraped'])  # Các cột khóa kiểm tra trùng qua các mốc

    # Loại 1: Trùng trong 1 mốc
    dup_within = df.duplicated(subset=['id', 'snapshot'], keep='first')  # Trùng lặp cùng ID trong cùng mốc chụp

    # Loại 2: Trùng giữa các mốc
    dup_across = df.duplicated(subset=dedupe_keys, keep='first')     # Trùng lặp cùng ID và cùng ngày cào dữ liệu

    df['qa_duplicate'] = dup_within | dup_across                     # Gắn cờ True nếu trùng lặp loại 1 hoặc loại 2

    n_within = dup_within.sum()                                      # Đếm số dòng trùng trong mốc
    n_across = (~dup_within & dup_across).sum()                      # Đếm số dòng trùng giữa các mốc
    n_total = df['qa_duplicate'].sum()                               # Tổng số dòng bị gắn cờ trùng lặp
    logger.info(f"[QA-duplicate] Flagged {n_total}/{len(df)} rows — "
                f"trùng trong mốc: {n_within}, trùng giữa các mốc: {n_across}")

    return df                                                        # Trả về DataFrame đã gắn cờ trùng lặp


# Quy tắc 4: Kiểm tra sai định dạng tiền và ngày tháng
def flag_format(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Kiểm tra price: có chuỗi khác rỗng nhưng parse_price ra NaN
    price_has_val = df['price'].notna() & (df['price'].astype(str).str.strip().str.lower() != 'nan') & (df['price'].astype(str).str.strip() != '')  # Ô giá có chuỗi hợp lệ
    price_format_bad = price_has_val & df['price_clean'].isna()       # Có chuỗi giá nhưng không parse ra được số

    # Kiểm tra các cột ngày
    date_cols = ['last_scraped', 'host_since', 'first_review', 'last_review']  # Danh sách các cột ngày cần xét
    date_format_bad = pd.Series(False, index=df.index)                # Khởi tạo cờ lỗi ngày toàn False

    for col in date_cols:                                            # Lặp qua từng cột ngày
        if col not in df.columns:                                    # Bỏ qua nếu cột không tồn tại trong bảng
            continue
        has_val = df[col].notna() & (df[col].astype(str).str.strip() != '') & (df[col].astype(str).str.strip().str.lower() != 'nan')  # Ô ngày có chuỗi ký tự
        parsed = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')  # Parse thử theo định dạng chuẩn YYYY-MM-DD
        date_format_bad = date_format_bad | (has_val & parsed.isna())  # Gắn cờ nếu có giá trị nhưng parse thất bại

    df['qa_format'] = price_format_bad | date_format_bad             # True nếu lỗi định dạng giá hoặc ngày

    n_total = df['qa_format'].sum()                                  # Tổng số dòng bị lỗi định dạng
    logger.info(f"[QA-format] Flagged {n_total}/{len(df)} rows — "
                f"price format: {price_format_bad.sum()}, date format: {date_format_bad.sum()}")

    return df                                                        # Trả về DataFrame đã gắn cờ định dạng


# Quy tắc 5: Kiểm tra lệch danh mục room_type và analysis_area
def flag_category(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    valid_room_types = {                                             # Tập hợp 4 loại phòng chuẩn Airbnb
        'Entire home/apt',
        'Private room',
        'Shared room',
        'Hotel room'
    }
    valid_areas = {f'A{i:02d}' for i in range(1, 9)}                 # Tập hợp 8 mã phân vùng chuẩn từ A01 đến A08

    bad_room_type = df['room_type'].notna() & ~df['room_type'].isin(valid_room_types)  # Loại phòng không thuộc tập chuẩn
    bad_area = pd.Series(False, index=df.index)                      # Khởi tạo cờ phân vùng sai là False
    if 'analysis_area' in df.columns:
        bad_area = df['analysis_area'].notna() & ~df['analysis_area'].isin(valid_areas)  # Phân vùng không thuộc A01-A08

    df['qa_category'] = bad_room_type | bad_area                     # Gắn cờ True nếu sai loại phòng hoặc phân vùng

    n_flagged = df['qa_category'].sum()                              # Tổng số dòng lệch danh mục
    logger.info(f"[QA-category] Flagged {n_flagged}/{len(df)} rows — "
                f"room_type sai: {bad_room_type.sum()}, analysis_area sai: {bad_area.sum()}")

    return df                                                        # Trả về DataFrame đã gắn cờ danh mục


# Hàm nội bộ: Tổng hợp bảng báo cáo QA và đo lường tác động KPI trước/sau
def _build_qa_report(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    qa_config = config.get('qa', {})                                 # Đọc cấu hình QA
    price_max = qa_config.get('price_max_czk', 50000)                # Lấy ngưỡng giá trần
    min_nights_cap = qa_config.get('min_nights_cap', 365)            # Lấy ngưỡng số đêm trần
    currency = config.get('currency', 'CZK')                         # Đơn vị tiền tệ (CZK)

    rules = [                                                        # Định nghĩa chi tiết từng quy tắc kiểm tra
        {
            'rule_name': 'missing',
            'flag_col': 'qa_missing',
            'condition': 'Cột cấu trúc cốt lõi (id, host_id, neighbourhood, analysis_area, lat, lon, room_type) bị NaN',
            'reason': 'Thiếu định danh/vị trí → không phân nhóm không gian hay phân loại phòng được',
            'action': 'Gắn cờ qa_missing, loại khỏi phân tích cấu trúc'
        },
        {
            'rule_name': 'outlier',
            'flag_col': 'qa_outlier',
            'condition': f'price > {price_max:,} {currency} HOẶC price ≤ 0 HOẶC minimum_nights > {min_nights_cap}',
            'reason': 'Giá ảo hoặc số đêm thuê > 1 năm làm méo mó các chỉ số thị trường',
            'action': 'Gắn cờ qa_outlier, loại khỏi tính toán KPI giá'
        },
        {
            'rule_name': 'duplicate',
            'flag_col': 'qa_duplicate',
            'condition': 'Cùng id trong 1 mốc hoặc cùng (id, last_scraped) qua các mốc',
            'reason': 'Trùng lặp dữ liệu dẫn đến đếm kép, sai quy mô cung phòng',
            'action': 'Gắn cờ qa_duplicate, chỉ giữ lại bản ghi đầu tiên'
        },
        {
            'rule_name': 'format',
            'flag_col': 'qa_format',
            'condition': 'Price có chuỗi nhưng không parse được số; ngày không đúng chuẩn YYYY-MM-DD',
            'reason': 'Sai định dạng dẫn đến tính toán số học hoặc chuỗi thời gian bị crash',
            'action': 'Gắn cờ qa_format'
        },
        {
            'rule_name': 'category',
            'flag_col': 'qa_category',
            'condition': 'room_type không thuộc 4 loại chuẩn HOẶC analysis_area không thuộc {A01..A08}',
            'reason': 'Lệch danh mục làm sai lệch cơ cấu phòng và phân vùng địa lý',
            'action': 'Gắn cờ qa_category'
        },
        {
            'rule_name': 'missing_price',
            'flag_col': 'qa_missing_price',
            'condition': 'Cột price bị NaN (đặc thù mốc 2 & 3 không có giá; mốc 1 & 4 có một số căn thiếu giá)',
            'reason': 'Không có giá niêm yết trong bảng listings của Inside Airbnb',
            'action': 'Gắn cờ qa_missing_price, chỉ áp dụng cho KPI liên quan đến giá'
        }
    ]

    report_rows = []                                                 # Danh sách chứa từng dòng của bảng báo cáo
    snapshots = df['snapshot'].unique() if 'snapshot' in df.columns else ['all']  # Lấy danh sách các mốc chụp

    for snap in snapshots:                                           # Lặp qua từng mốc chụp
        snap_df = df[df['snapshot'] == snap] if snap != 'all' else df  # Lọc dữ liệu thuộc mốc hiện tại
        total = len(snap_df)                                         # Tổng số dòng trong mốc

        # Baseline KPI giá của snapshot trước khi lọc rule
        prices_raw = snap_df['price_clean'].dropna()                 # Lọc các giá trị giá không bị rỗng
        kpi_base = round(prices_raw.mean(), 2) if len(prices_raw) > 0 else np.nan  # Tính giá trung bình trước lọc

        for rule in rules:                                           # Lặp qua từng quy tắc kiểm tra
            flagged = snap_df[rule['flag_col']].sum()                # Số dòng bị cắm cờ bởi quy tắc này
            # KPI giá sau khi loại bỏ cờ của rule này
            prices_after = snap_df.loc[~snap_df[rule['flag_col']], 'price_clean'].dropna()  # Lọc giá dòng sạch cờ
            kpi_after = round(prices_after.mean(), 2) if len(prices_after) > 0 else np.nan  # Giá trung bình sau lọc

            report_rows.append({                                     # Thêm dòng dữ liệu thống kê vào báo cáo
                'rule_name': rule['rule_name'],
                'condition': rule['condition'],
                'reason': rule['reason'],
                'action': rule['action'],
                'snapshot': snap,
                'total_rows': total,
                'rows_flagged': int(flagged),
                'pct_flagged': round(100 * flagged / total, 2) if total > 0 else 0.0,
                'kpi_metric': f'mean_price ({currency})',
                'kpi_before': kpi_base,
                'kpi_after': kpi_after
            })

        # Dòng tổng hợp ANY (tất cả cờ chất lượng dữ liệu: missing cấu trúc + outlier + duplicate + format + category)
        flagged_any = snap_df['qa_any_flag'].sum()                   # Tổng dòng bị cắm cờ bất kỳ
        clean_prices = snap_df.loc[~snap_df['qa_any_flag'], 'price_clean'].dropna()  # Lọc giá của dòng sạch chất lượng
        kpi_after_any = round(clean_prices.mean(), 2) if len(clean_prices) > 0 else np.nan  # Giá TB dòng sạch

        report_rows.append({                                         # Thêm dòng tổng hợp ANY vào báo cáo
            'rule_name': 'ANY (tổng hợp chất lượng)',
            'condition': 'Bất kỳ quy tắc lỗi nào bị kích hoạt (missing, outlier, dup, format, category)',
            'reason': 'Tập hợp các bản ghi không đạt chuẩn chất lượng dữ liệu',
            'action': 'Loại khỏi KPI (~qa_any_flag)',
            'snapshot': snap,
            'total_rows': total,
            'rows_flagged': int(flagged_any),
            'pct_flagged': round(100 * flagged_any / total, 2) if total > 0 else 0.0,
            'kpi_metric': f'mean_price ({currency})',
            'kpi_before': kpi_base,
            'kpi_after': kpi_after_any
        })

    return pd.DataFrame(report_rows)                                 # Trả về DataFrame báo cáo QA


# Hàm chính: Điều phối chạy toàn bộ quy trình QA và xuất báo cáo
def run_qa(config: dict, data_dir: str = None,
           report_dir: str = 'reports') -> pd.DataFrame:
    snapshots = config.get('snapshots', [])                          # Lấy danh sách các mốc chụp từ file YAML
    if not snapshots:
        raise ValueError("Config thiếu 'snapshots' — cần ít nhất 1 mốc chụp")

    frames = []                                                      # Danh sách chứa DataFrame từng mốc
    for snap in snapshots:                                           # Lặp qua từng mốc chụp
        snap_str = str(snap)

        # Danh sách đường dẫn khả dĩ
        candidate_paths = []                                         # Khởi tạo các đường dẫn tìm kiếm file
        if data_dir:
            candidate_paths.extend([
                os.path.join(data_dir, f'listings_{snap_str}.csv.gz'),
                os.path.join(data_dir, snap_str, 'listings.csv.gz'),
                os.path.join(data_dir, 'listings.csv.gz')
            ])
        candidate_paths.extend([
            os.path.join('data/processed', f'listings_{snap_str}.csv.gz'),
            os.path.join('data/raw', snap_str, 'listings.csv.gz'),
            os.path.join('data/raw', f'listings_{snap_str}.csv.gz')
        ])

        found_path = None
        for p in candidate_paths:                                    # Duyệt tìm file thực sự tồn tại
            if os.path.exists(p) and os.path.getsize(p) > 0:
                found_path = p
                break

        if not found_path:                                           # Bỏ qua nếu không tìm thấy file
            logger.warning(f"Không tìm thấy file listings cho mốc {snap_str} tại các vị trí kiểm tra — bỏ qua")
            continue

        logger.info(f"Đọc {found_path}...")
        snap_df = pd.read_csv(found_path, compression='gzip', low_memory=False)  # Đọc file CSV nén gzip
        snap_df['snapshot'] = snap_str                               # Gán cột snapshot để nhận diện mốc
        frames.append(snap_df)                                       # Thêm vào danh sách frames
        logger.info(f"  → {len(snap_df)} dòng, {len(snap_df.columns)} cột")

    if not frames:                                                   # Báo lỗi nếu không có file nào được nạp
        raise FileNotFoundError(
            "Không tìm thấy bất kỳ file listings nào trong 'data/processed' hoặc 'data/raw'. "
            "Hãy chạy 'python src/download.py --config configs/prague.yaml' trước."
        )

    df = pd.concat(frames, ignore_index=True)                        # Nối tất cả các mốc thành 1 DataFrame duy nhất
    logger.info(f"Tổng sau ghép: {len(df)} dòng × {len(df.columns)} cột, {len(frames)} mốc chụp")

    # ---- Bước 3: Parse price ----
    df['price_clean'] = parse_price(df['price'])                     # Chuyển đổi chuỗi giá thành số thực

    # ---- Bước 4: Chạy 5 quy tắc flag ----
    df = flag_missing(df, config)                                    # Chạy quy tắc thiếu dữ liệu
    df = flag_outlier(df, config)                                    # Chạy quy tắc dữ liệu ngoại lai
    df = flag_duplicate(df, config)                                  # Chạy quy tắc trùng lặp dữ liệu
    df = flag_format(df, config)                                     # Chạy quy tắc lỗi định dạng
    df = flag_category(df, config)                                   # Chạy quy tắc lệch danh mục

    # ---- Bước 5: Cờ tổng hợp chất lượng ----
    df['qa_any_flag'] = (                                            # Tổng hợp các cờ lỗi chất lượng dữ liệu
        df['qa_missing'] | df['qa_outlier'] | df['qa_duplicate'] |
        df['qa_format'] | df['qa_category']
    )

    # Cờ riêng dành cho các phép tính KPI về giá: vừa sạch lỗi, vừa phải có giá hợp lệ
    df['qa_clean_price'] = (~df['qa_any_flag']) & (~df['qa_missing_price'])  # Dòng sạch lỗi và có giá niêm yết

    n_clean = (~df['qa_any_flag']).sum()                             # Đếm tổng dòng sạch chất lượng
    n_clean_price = df['qa_clean_price'].sum()                       # Đếm tổng dòng sạch giá
    logger.info(f"Kết quả QA: {n_clean}/{len(df)} dòng sạch chất lượng ({100*n_clean/len(df):.1f}%), "
                f"{n_clean_price} dòng sạch và có giá đầy đủ")

    # ---- Bước 6: Xuất báo cáo ----
    os.makedirs(report_dir, exist_ok=True)                           # Tạo thư mục xuất báo cáo nếu chưa có
    report_path = os.path.join(report_dir, 'qa_report.csv')          # Đường dẫn lưu file báo cáo QA
    report_df = _build_qa_report(df, config)                         # Tạo DataFrame báo cáo
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig') # Xuất file CSV định dạng UTF-8 có BOM
    logger.info(f"Báo cáo QA đã xuất: {report_path}")

    return df                                                        # Trả về DataFrame đã hoàn tất QA


# Chạy kiểm thử độc lập module
if __name__ == '__main__':
    import yaml
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    parser = argparse.ArgumentParser(description='Chạy QA trên dữ liệu Airbnb')
    parser.add_argument('--config', default='configs/prague.yaml',
                        help='Đường dẫn file config YAML')
    parser.add_argument('--data-dir', default=None,
                        help='Đường dẫn thư mục dữ liệu (mặc định tự tìm processed/raw)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    df = run_qa(config, data_dir=args.data_dir)
    n_clean = (~df['qa_any_flag']).sum()
    n_flag = df['qa_any_flag'].sum()
    print(f"\nSummary: {len(df)} rows across {df['snapshot'].nunique()} snapshots | "
          f"Clean: {n_clean} | Flagged (Quality Issues): {n_flag} | "
          f"Valid Price Listings: {df['qa_clean_price'].sum()}")
