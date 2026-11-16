# SỔ TAY QUY CHUẨN GÁN NHÃN TAY (CODEBOOK V3) — NHÓM 11

> **Tài liệu dùng cho:** 5 thành viên nhóm 11 (Kiên, Nam, Quân, Chi, Phương) gán nhãn tập đối chứng Ground Truth 100 review tại `samples/hand_labels.csv`.  
> **Mục tiêu:** Đồng bộ tuyệt đối giữa nhãn gán tay của người, nhãn từ quy tắc Baseline NLP (`src/baseline.py`) và nhãn trích xuất từ Gemini LLM (`src/llm.py`).  
> **Nguyên tắc tối cao:** Chỉ gán nhãn khi có **bằng chứng từ vựng trực tiếp (Explicit Lexical Evidence)** nằm trong Whitelist. Tuyệt đối không suy diễn cảm tính. Với Baseline, mọi khớp từ đơn phải dùng **ranh giới từ nguyên vẹn (`\b...\b`)**, không dùng phép kiểm tra chuỗi con thô (`in`) để tránh nhiễu (`ac` dính `spacious`, `lock` dính `block/clock`).

---

## 1. Danh mục Nhãn Chuẩn (Taxonomy) & Quy tắc Quyết định

### 1.1 Cảm xúc (`sentiment`)
Chỉ chọn đúng **1 trong 3** giá trị sau (chữ thường): `positive`, `mixed`, `negative`.

#### A. Quy tắc cho Người gán nhãn (Human Annotation)
1. **`positive`**:
   - Bài viết **chỉ có khen**, không có lời phàn nàn.
   - **HOẶC Khen áp đảo (≥ 80% nội dung)**, chỉ có 1 góp ý nhỏ/tiểu tiết (minor nitpick — không phải lỗi nghiêm trọng như *dirty, filthy, rude, terrible, worst, misleading, unresponsive*) và cuối bài khách vẫn chốt hài lòng cao / khuyên ở (`recommend`, `10/10`, `will be back`, `stay again`, `can't go wrong` — ví dụ STT 24, STT 59).
   - **Sự cố nhỏ đã được khắc phục ổn thỏa (Resolved Minor Issue):** Thỏa mãn **đồng thời 2 điều kiện**: (1) Sự cố nhỏ về thiết bị/thủ tục được host xử lý ngay (như hỏng máy pha cà phê được mua mới ngay ở STT 69, lỗi internet sửa ngay ở STT 99) **VÀ** (2) Cuối bài khách khẳng định hài lòng / khuyên ở. *(Lưu ý: Nếu sự cố lớn như báo hủy/đổi nhà vào phút chót mà cuối bài không có câu khen chốt lại như STT 77 → gán `mixed`)*.
   - **Phủ định điểm chê (Negated Complaint):** Dùng từ tiêu cực ở dạng phủ định để khen (*"nothing to fault"*, *"not bothered"*, *"no problem"*, *"without problems"*, *"finding anything to criticize was not possible"*).
2. **`negative`**:
   - Bài viết **chỉ có chê / phàn nàn**.
   - **HOẶC Chê áp đảo (≥ 80% nội dung)** hoặc gặp sự cố nghiêm trọng phá hỏng trải nghiệm (bẩn thỉu, mất an toàn, host bỏ mặc), điểm khen (nếu có) chỉ mang tính chiếu lệ.
3. **`mixed`**:
   - Có sự **đánh đổi thực chất (substantial trade-off)** giữa hai mặt tốt và xấu: vừa khen rõ ràng ở khía cạnh này, vừa phàn nàn thực sự ở khía cạnh khác chưa được khắc phục, không bên nào lấn át hoàn toàn (Ví dụ STT 15: *khó tìm & host kém giao tiếp, nhưng nhà sạch và tuyệt vời*; STT 20: *vị trí đẹp, host tốt nhưng ga giường bẩn & thiếu chăn gối*; STT 22: *nhà sạch, gần metro nhưng sofa nằm không thoải mái*).

#### B. Quy tắc định lượng cho Máy (`src/baseline.py`)
1. **Đơn vị tách là Mệnh đề (Clause):** Tách bài viết theo dấu ngắt câu (`.`, `!`, `?`, `;`, `\n`), cặp liệt kê ưu/nhược điểm (`+ ... - ...` như STT 22, không tách dấu gạch ngang ` - ` thông thường trong câu), và các liên từ đối lập (`\bbut\b`, `\bhowever\b`, `\balthough\b`, `\bunfortunately\b`, `\bonly downside\b`, `\bonly issue\b`, `\bonly down part\b`).
2. **Lọc phủ định trước khi đếm:** Các mệnh đề chứa cấu trúc phủ định điểm chê (`not bothered`, `no problem`, `nothing to fault`, `not.*criticize`, `without problems`) không tính vào `N_neg`.
3. **Tính tỷ lệ chê** `neg_ratio = N_neg / (N_pos + N_neg)` kết hợp **Neo khẳng định mạnh (Strong Positive Anchor)**:
   - Nếu `N_neg == 0` → `positive`
   - Nếu `N_pos == 0` và `N_neg > 0` → `negative`
   - Nếu `N_pos > 0` và `N_neg > 0`:
     - Nếu `neg_ratio <= 0.20` → `positive`
     - Nếu `N_neg == 1` (không chứa từ chê nặng: `dirty`, `filthy`, `rude`, `terrible`, `worst`, `poor`, `inadequate`, `unacceptable`, `horrible`, `renovations`) **VÀ** bài có câu chốt khuyên ở (`recommend`, `10/10`, `will be back`, `stay again`, `come back`, `can't go wrong`) → `positive`
     - Nếu `neg_ratio >= 0.80` → `negative`
     - Các trường hợp còn lại (`0.20 < neg_ratio < 0.80`) → `mixed`

---

### 1.2 Ngôn ngữ (`language`)
Gán mã ISO 639-1 (chữ thường) theo **Ngôn ngữ chủ đạo (Dominant Language)** của bài viết (bỏ qua tên riêng địa danh như *Praha, Zizkov* hoặc 1 cụm thành ngữ ngoại lai chêm vào như *"la cerise sur le gâteau"* ở STT 50):
* `en`: Tiếng Anh (*the, and, was, is, very, with, we, apartment, place, stay...*)
* `cs`: Tiếng Séc (*byt, pobyt, čistý, lokace, lokalita, velmi, děkujeme, doporučuji...*)
* `de`: Tiếng Đức (*und, war, die, der, das, sehr, sauber, wohnung, lage, wir...*)
* `es`: Tiếng Tây Ban Nha (*el, la, muy, limpio, ubicación, apartamento, todo, bien...*)
* `fr`: Tiếng Pháp (*le, la, les, très, propre, appartement, bien, situé, séjour, nous...*)
* `other`: Các ngôn ngữ khác (Nga, Hàn, Trung, Nhật, Hà Lan, Ý, Bồ Đào Nha...)

---

### 1.3 Khía cạnh (`aspects`) — Bảng Từ khóa Đóng (Whitelist & Blacklist)
* **Định dạng CSV bắt buộc:** Ghi đúng tên khía cạnh tiếng Việt chữ thường trong cấu hình `configs/prague.yaml`, ngăn cách bởi **dấu phẩy và 1 khoảng trắng** (ví dụ: `vị trí, sạch sẽ, chủ nhà`). Nếu bình luận chung chung không chứa từ khóa của khía cạnh nào thì để trống `""`.

#### 1. `vị trí` (Location & Accessibility)
* **TÍNH KHI CÓ (Whitelist):**
  - Vị trí / Trung tâm: `location`, `located`, `situated`, `central`, `center`, `centre`, `heart of`, `old town`, `old city`, `square`, `castle`, `bridge`, `neighborhood`, `neighbourhood`, `area`, `district`, `zone`.
  - Di chuyển / Khoảng cách: `walk`, `walking`, `on foot`, `close to`, `near`, `nearby`, `proximity`, `minutes away`, `minutes from`, `accessible`, `easy to access`, `easy to get`, `easy to find`, `challenging to find`.
  - Giao thông & Xung quanh: `metro`, `subway`, `tram`, `tramway`, `bus`, `station`, `stop`, `transport`, `transportation`, `transit`, `airport`, `supermarket`, `grocery`, `market`, `restaurant`, `bar`, `pub`, `cafe`, `bistro`, `shop`, `attraction`, `sight`, `sightseeing`, `parks`, `nice park`, `close to.*park`, `beer garden`.
  - Đa ngôn ngữ (`cs/de/es/fr`): `lokalita`, `lokace`, `centrum`, `lage`, `zentral`, `ubicación`, `céntrico`, `emplacement`, `quartier`, `métro`.
* **KHÔNG TÍNH (Blacklist):**
  - Chỉ nhắc tên thành phố/quốc gia đơn thuần (*"Loved our stay in Prague"*, *"Best place in Czech Republic"*).
  - Động từ `park` mang nghĩa đỗ xe (*"easy to park our car"* $\rightarrow$ thuộc `tiện nghi`).

#### 2. `sạch sẽ` (Cleanliness & Hygiene)
* **TÍNH KHI CÓ (Whitelist):**
  - `clean`, `cleaned`, `cleanliness`, `spotless`, `spotlessly`, `tidy`, `hygienic`, `fresh`, `sanitized`, `neat`.
  - `dirty`, `filthy`, `dust`, `dusty`, `stain`, `smell`, `smelled`, `odor`, `smelly`, `mold`, `hair`, `bug`, `insect`, `unclean`, `messy`.
  - Đa ngôn ngữ (`cs/de/es/fr`): `čistý`, `čisté`, `sauber`, `limpio`, `propre`, `sale`.
* **KHÔNG TÍNH (Blacklist):**
  - `clear` (quang đãng/rõ ràng), `beautifully kept` (giữ gìn đẹp), `well maintained`, `new`, `modern`, `bright`, `luminous`, `organized`.

#### 3. `chủ nhà` (Host, Staff & Service)
* **TÍNH KHI CÓ (Whitelist):**
  - Nhân sự đón tiếp: `host`, `hosts`, `owner`, `staff`, `crew`, `manager`, `reception`, `front desk` (hoặc **Tên riêng chủ nhà**) **ĐI KÈM** nhận xét về thái độ/hành vi (`friendly`, `kind`, `nice`, `helpful`, `helpfull`, `welcome`, `welcoming`, `accommodating`, `hospitality`, `hospitable`, `generous`, `obliging`, `patient`, `uncomplicated`, `super host`, `awesome guy`, `care about`, `rude`, `unfriendly`, `responsive`, `flexible`, `sweet`, `courteous`). *(Không tính động từ `to host` trong `"place to host a lot of people"`).*
  - Giao tiếp & Hỗ trợ: `communicate`, `communication`, `communicative`, `communicator`, `response`, `responded`, `reply`, `quick to answer`, `answer questions`, `available`, `on hand`, `reachable`, `easy to deal with`, `helped us`, `for your help`, `tips`, `suggestions`, `recommendations`, `advice`, `guidebook`, `itinerary`, `picked us up`, `pick us up`, `waited for us`, `waited patiently`, `checked in with us`, `store our bags`, `luggage`, `arranged an alternative`.
  - Check-in/out có sự tương tác/hỗ trợ của chủ nhà (`stayed up till midnight to help us check in`, `easy check in` khi đi cùng câu khen host).
  - Đa ngôn ngữ (`cs/de/es/fr`): `hostitel`, `majitel`, `gastgeber`, `anfitrión`, `hôte`, `accueil`.
* **KHÔNG TÍNH (Blacklist):**
  - Lời chào/cảm ơn tên riêng xã giao cuối bài: *"Thanks Vladimir!"*, *"Thank you Maria!"*.
  - Danh từ sở hữu chỉ nơi ở: *"Anna's place"*, *"Niko's flat"*, *"Marketa's apartment"*.
  - Tự check-in không nhắc tới sự hỗ trợ của host (*"check-in with the lockbox worked"*, *"contact free check in"*, *"easy to check in"* đứng độc lập).

#### 4. `giá trị` (Value for Money & Price)
* **TÍNH KHI CÓ (Whitelist):**
  - `value`, `value for money`, `good value`, `great value`, `price`, `priced`, `worth`, `affordable`, `cheap`, `inexpensive`, `expensive`, `overpriced`, `reasonable`, `budget`, `cost`, `money`, `deal`, `bargain`, `without spending a lot`.
  - Đa ngôn ngữ (`cs/de/es/fr`): `cena`, `preis`, `preis-leistungsverhältnis`, `precio`, `calidad-precio`, `prix`, `qualité-prix`.
* **KHÔNG TÍNH (Blacklist):**
  - Các từ chỉ chất lượng/lựa chọn chung (*"worth visiting"*, *"best choice"*), hoặc giá taxi/Uber (*"reasonable Uber"*).

#### 5. `tiện nghi` (Amenities, Equipment & Facilities)
* **TÍNH KHI CÓ (Whitelist):**
  1. **Đồ đạc, nội thất & thiết bị cụ thể:**
     - Phòng ngủ/khách: `bed`, `beds`, `bedding`, `mattress`, `pillow`, `duvet`, `blanket`, `linen`, `linnens`, `sheets`, `sofa`, `couch`, `bedcouch`, `table`, `chair`, `wardrobe`, `closet`, `curtains`, `electric windows`. *(Lưu ý: bỏ qua lỗi chính tả `"slept very bed"` ở STT 91 nhưng câu đó vẫn có `"the bed"`).*
     - Bếp & Pha chế: `kitchen`, `kitchenette`, `fridge`, `refrigerator`, `freezer`, `microwave`, `oven`, `stove`, `cooker`, `kettle`, `coffee`, `coffee machine`, `nespresso`, `tea`, `dishes`, `plates`, `pots`, `pans`, `utensils`.
     - Phòng tắm & Giặt giũ: `bathroom`, `shower`, `bathtub`, `toilet`, `hot water`, `water pressure`, `towel`, `towels`, `shampoo`, `soap`, `body wash`, `hairdryer`, `hair dryer`, `washing machine`, `washer`, `dryer`, `iron`.
     - Điện tử & Nhiệt độ: `wifi`, `wi-fi`, `internet`, `tv`, `television`, `ac`, `a/c`, `air conditioning`, `air conditioner`, `heating`, `heater`, `radiator`, `fan`.
  2. **Tiện ích đi kèm của căn hộ/tòa nhà:**
     - `parking`, `garage`, `charging station`, `elevator`, `lift`, `balcony`, `terrace`, `terace`, `patio`, `garden` *(trừ `beer garden` thuộc vị trí)*, `yard`, `pool`, `gym`.
  3. **Từ khóa & Mẫu câu tổng quát chỉ sự trang bị đầy đủ:**
     - Từ đơn/ghép: `amenities`, `amenity`, `facilities`, `facility`, `equipment`, `equipped`, `eqipped`, `well-equipped`, `well appointed`, `furnished`, `well-furnished`.
     - Mẫu câu đủ đồ dùng trong phòng (Regex): `\b(has|had|have|with|equipped with|provided|got)\b[^.!?;]{0,20}\b(everything|all|what)\b[^.!?;]{0,20}\b(need(ed)?|ask for|wish for|necessary)\b|\bnecessary items\b` *(Chú ý: KHÔNG tính khi đi kèm `information`, `info`, `tips`, `help`, `advice` vì đó là thông tin từ chủ nhà).*
  4. **Đa ngôn ngữ (`cs/de/es/fr`):** `vybavení`, `kuchyně`, `postel`, `ausstattung`, `küche`, `bett`, `equipado`, `cocina`, `cama`, `équipé`, `cuisine`, `lit`.
* **KHÔNG TÍNH (Blacklist — Tuyệt đối loại trừ):**
  - **Diện tích / Kích thước / Số phòng đứng độc lập:** `spacious`, `space`, `big`, `huge`, `large`, `small`, `little`, `tiny`, `roomy`, `bedroom`, `rooms`, `high ceilings` *(chỉ tính khi ghép trực tiếp với thiết bị trong Whitelist như `large table`, `huge bed`)*.
  - **Cảm giác / Thẩm mỹ / Quà tặng / Hộp khóa / Khen chung căn hộ:** `cozy`, `cosy`, `comfortable`, `comfy`, `confortable`, `modern`, `stylish`, `stylist`, `lovely`, `charming`, `gorgeous`, `beautiful`, `nice`, `wonderful`, `amazing`, `perfect`, `decorated`, `decoration`, `decorative touches`, `little details`, `vase`, `view`, `luminous`, `bright`, `beautifully kept`, `functional`, `wine`, `cookies`, `lockbox`, `keybox`.

#### 6. `độ ồn` (Noise Level & Quietness)
* **TÍNH KHI CÓ (Whitelist):**
  - `quiet`, `peaceful`, `silent`, `silence`, `calm`, `tranquil`, `reprieve from the bustle`.
  - `noise`, `noisy`, `loud`, `sound`, `soundproof`, `street noise`, `traffic noise`, `busy street`, `busy road`, `bothersome` (khi gắn với tiếng ồn/quán bar).
  - Đa ngôn ngữ (`cs/de/es/fr`): `klid`, `tichý`, `ruhig`, `leise`, `larm`, `tranquilo`, `ruido`, `calme`, `bruit`.
* **KHÔNG TÍNH (Blacklist):**
  - Khen thư giãn chung chung (*"relaxing stay"*) không đề cập âm thanh.

#### 7. `an toàn` (Safety & Security)
* **TÍNH KHI CÓ (Whitelist):**
  - `safe`, `safety`, `secure`, `security`, `locked gate`, `gated`, `guard`, `cctv`, `camera`, `alarm`, `feel safe`, `felt very safe`, `dangerous`, `sketchy`, `unsafe`.
  - Đa ngôn ngữ (`cs/de/es/fr`): `bezpečný`, `sicher`, `sicherheit`, `seguro`, `sécurité`, `sûr`.
* **KHÔNG TÍNH (Blacklist):**
  - `lockbox` (hộp khóa tự check-in), `half-block`, `clock`, hoặc `parking space` không kèm từ `secure` / `gated`.

---

## 2. Bảng Phân Chia 100 Dòng Độc Lập

Để tránh xung đột Git khi merge code, mỗi thành viên **chỉ điền nhãn ở đúng 20 dòng của mình**, không chạm vào các dòng của thành viên khác:

| Thành viên | Dòng phân công (STT) | Trạng thái |
|---|---|---|
| **Kiên** (Chủ trì) | Dòng **1 — 20** | **ĐÃ HOÀN THÀNH & CHUẨN HÓA THEO CODEBOOK V3** |
| **Nam** | Dòng **21 — 40** | **ĐÃ HOÀN THÀNH & CHUẨN HÓA THEO CODEBOOK V3** |
| **Quân** | Dòng **41 — 60** | Chờ Quân điền |
| **Chi** | Dòng **61 — 80** | Chờ Chi điền |
| **Phương** | Dòng **81 — 100** | Chờ Phương điền |

---

## 3. Cấu Trúc File & Bố Cục Cột (Chuẩn Máy Đọc & Trực Quan)

Thứ tự cột trong `samples/hand_labels.csv`:
`stt` → `assigned_to` → `sentiment` → `aspects` → `language` → `comments` → `review_id` → `listing_id`

* Toàn bộ nội dung bình luận đã được làm sạch thẻ HTML (`<br/>`, `&amp;`), lưu chuẩn mã hóa UTF-8 BOM để mở trực tiếp bằng Microsoft Excel trên Windows mà không bị lỗi phông chữ.
* Khi viết code đọc cột `aspects` trong `evaluate.py`, luôn dùng `[a.strip() for a in str(val).split(",") if a.strip()]` để đảm bảo an toàn khoảng trắng.

---

## 4. Tóm Tắt Các Ca Điển Hình Đã Chuẩn Hóa (Case Studies từ STT 1–40)

1. **Lời chào / cảm ơn xã giao cuối bài (STT 14: *"Thanks Vladimir!"*):**
   * → **KHÔNG gán `chủ nhà`**. Nhưng nếu cảm ơn trực tiếp về sự hỗ trợ/hiếu khách (STT 31: *"Thanks for the pleasant and generous hospitality"*, STT 50: *"Thank you so much for your help"*) → **CÓ gán `chủ nhà`**.
2. **Khen diện tích / cảm giác / trang trí (STT 10, 11, 13, 17, 23, 33, 35, 38, 39: *"spacious"*, *"cozy"*, *"comfortable"*, *"tastefully decorated"*, *"Christmas vase"*):**
   * → **KHÔNG gán `tiện nghi`**. Chỉ gán `tiện nghi` khi có tên thiết bị, tiện ích cụ thể (`kitchen`, `bed`, `shower`, `wifi`, `terrace`, `garden`, `parking`...) hoặc từ khóa trang bị (`amenities`, `equipped`, `furnished`, `facilities`, `has everything you need`).
3. **Khen ngợi bảo quản / từ dễ nhầm (STT 4: *"beautifully kept"*, STT 6: *"clear and cosy"*):**
   * → **KHÔNG gán `sạch sẽ`**.
4. **Khen áp đảo kèm 1 lưu ý nhỏ nhưng câu kết vẫn khẳng định mạnh (STT 24):**
   * → **Gán `positive`** (4/5 câu khen, chốt *"All in all you can't go wrong"*, chỉ nhắc nhẹ quán bar bên dưới *"slightly bothersome"* nếu không hợp gu).
5. **Vừa khen vừa chê thực chất chưa được khắc phục (STT 15, STT 20, STT 22):**
   * → **Bắt buộc gán `mixed`** (khó tìm & host kém giao tiếp nhưng phòng đẹp; vị trí đẹp nhưng chăn ga bẩn; nhà sạch và gần metro nhưng sofa nằm đau lưng).
6. **Nhắc đến chi phí đỗ xe đắt đỏ / đã bao gồm trong giá (STT 40: *"in the price included... parking space in Prague is everywhere expensive"*):**
   * → **Gán `giá trị` và `tiện nghi`**, **KHÔNG gán `an toàn`** khi không có từ khóa an ninh (`safe`, `secure`, `gated`).
