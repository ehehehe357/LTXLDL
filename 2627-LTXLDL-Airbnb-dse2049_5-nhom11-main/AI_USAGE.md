# AI_USAGE.md

## Công cụ đã dùng (vào việc gì)

## Prompt then chốt (5-10): bước nào, vào tệp nào

## AI sai ở đâu (ví dụ thật)

## Nhóm kiểm chứng thế nào

Quỳnh Chi: 
## Công cụ đã dùng
- Claude (Anthropic): hỗ trợ viết khung `src/download.py` và `src/export_ids.py`.

## Prompt then chốt
- "Viết script tải 4 snapshot Inside Airbnb của Prague, lọc theo danh sách ID của nhóm,
  đọc calendar theo chunk."

## AI sai ở đâu
- Không sai

## Nhóm kiểm chứng thế nào
- Chạy `download.py` và đối chiếu số liệu với đề: 1.000 / 365.000 / 5.000 dòng,
  875 + 125 loại phòng, 50-30-25-25 listing chưa có review, 20 listing thiếu giá ở mốc cuối.