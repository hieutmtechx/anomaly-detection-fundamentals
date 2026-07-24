# Live Prediction — Tầng Serving và Panel 3

Tài liệu này mô tả chi tiết toàn bộ phần code được thêm mới và sửa đổi để đưa mô hình
Isolation Forest ra khỏi notebook, đóng gói thành một dịch vụ chấm điểm trực tiếp, và
nối vào giao diện Streamlit ở Panel 3.

---

## 1. Mục tiêu và bối cảnh

Panel 1 và Panel 2 đều làm việc trên **kết quả đã tính sẵn**: Panel 1 đọc `train.csv` để
khám phá dữ liệu, Panel 2 đọc `scores_*.parquet` và các file `*_config.json` được sinh ra
từ lúc train. Không panel nào tính đặc trưng hay chấm điểm mới — kể cả khối SHAP của
Panel 2 cũng dùng lại bảng `X_test` đã lưu trong artifact.

Panel 3 trả lời một câu hỏi khác hẳn: **mô hình này có dùng được trong vận hành thật
không?** Muốn trả lời được, phải chứng minh bốn điều:

1. Mô hình đóng gói được thành file, nạp lên là chạy, không cần train lại.
2. Có giao diện mạng chuẩn để bất kỳ hệ thống nào cũng gọi được, không chỉ Streamlit.
3. Toàn bộ pipeline feature engineering chạy được **lúc inference** — đây là chỗ dễ sai
   nhất trong ML thực tế, vì lúc train ta có cả chuỗi dài hàng tháng, còn lúc chạy thật
   chỉ có một cửa sổ vài trăm đến vài nghìn điểm gần nhất.
4. Kết quả trả về dùng được để hành động: nhãn, điểm số, ngưỡng, độ tin cậy.

Nguyên tắc xuyên suốt, và cũng là tiêu chí nghiệm thu của toàn bộ tầng này:

> **Điểm số chấm qua API phải trùng khít với điểm số tính được lúc train, tại cùng một
> điểm dữ liệu.** Lệch một chút nghĩa là môi trường serving không tái hiện đúng môi
> trường train, và mọi kết luận sau đó đều không đáng tin.

---

## 2. Kiến trúc tổng thể

Hệ thống gồm hai giai đoạn tách bạch: **giai đoạn đóng gói** chạy một lần ngoại tuyến,
và **giai đoạn phục vụ** chạy liên tục.

```
GIAI ĐOẠN ĐÓNG GÓI (chạy tay, một lần)

  train.csv ──> time_split_per_kpi ──> preprocess_all ──> build_features
                                                               │
                          iffeat_<kpi>.joblib ────┐            │ .attrs["season_prof"]
                          if_per_kpi_config.json ─┼────────────┴──> build_server.py
                          scores_IF_feat.parquet ─┘                       │
                                                                          ▼
                                                        Modeling/Artifacts/serve_<kpi>.joblib
                                                                    (18 file)

GIAI ĐOẠN PHỤC VỤ (chạy liên tục)

  Trình duyệt
      │
      ▼
  App/pages/3_Prediction.py ──(HTTP + JSON)──> Serving/api.py
      │                                              │
      └─> App/common.py: grid_series()               └─> Serving/infer.py
              (cắt cửa sổ trên lưới đều)                     │
                                                             └─> FE_FS/Code/features.py
                                                                   build_features()
```

Điểm đáng chú ý về mặt thiết kế: `features.py` xuất hiện ở **cả hai giai đoạn**. Đây là
lựa chọn có chủ đích — nếu viết một phiên bản tính feature riêng cho lúc serve, hai bản
sẽ trôi dạt khỏi nhau theo thời gian mà không ai phát hiện, và mô hình sẽ nhận đầu vào
khác với thứ nó từng học.

---

## 3. Vấn đề cốt lõi đã phát hiện và sửa

Đây là phần quan trọng nhất của tài liệu, vì nó giải thích vì sao phần lớn code phải
thay đổi.

### 3.1 Triệu chứng ban đầu

Bản đầu tiên của `build_server.py` chỉ đóng gói được **2 trên 18** KPI, với ghi chú
trong code là *"2 KPI dễ (không dùng deseason)"*. Khảo sát bộ feature của cả 18 mô hình
cho kết quả:

| Nhóm | Số KPI | Đặc điểm |
|------|--------|----------|
| Không dùng `deseason_resid` | 2 | `18fbb1d5`, `c58bfcba` — lookback dài nhất là 60 |
| Có dùng `deseason_resid` | 16 | cần lookback bằng cả một chu kỳ ngày |

Bước chọn feature (`select_features`) xếp hạng theo mutual information với nhãn, và với
16 KPI thì `deseason_resid` đủ mạnh để được giữ lại. Đây không phải lựa chọn tùy tiện —
nó phản ánh việc phần lớn KPI trong bộ dữ liệu có nhịp ngày rõ rệt.

### 3.2 Nguyên nhân gốc

Trong `features.py`, phần dư khử mùa vụ được tính như sau:

```python
xi = x.interpolate(limit_direction="both").bfill().ffill()
trend = xi.rolling(period, center=False, min_periods=min(200, period // 2)).median()
mod = ((ts % 86400) // 60).astype(int)                    # phút-trong-ngày 0..1439
prof = pd.Series((xi - trend).values[tr_rows]).groupby(mod[tr_rows]).median()
seasonal = pd.Series(mod).map(prof)
resid = (xi.values - trend.values - seasonal.values)
```

Công thức là `resid = giá trị − trend − seasonal`, trong đó:

- `trend` là **trung vị trượt** trên cửa sổ `period` (1440 điểm với bước 60s, 288 điểm
  với bước 300s), chỉ nhìn về quá khứ.
- `seasonal` là **profile theo phút-trong-ngày**, lấy trung vị của `giá trị − trend` gom
  theo từng mốc phút trên các hàng thuộc tập train.

Vấn đề nằm ở dòng tính `prof`. Nó lấy trung vị **từ chính dữ liệu được truyền vào**.

Lúc train, `p` là cả chuỗi dài khoảng 90 ngày, nên mỗi mốc phút-trong-ngày có khoảng 90
mẫu để lấy trung vị — con số đủ để coi là quy luật. Lúc serve, nếu truyền vào một cửa sổ
đúng một ngày thì **mỗi mốc phút chỉ xuất hiện đúng một lần**. Trung vị của một số chính
là số đó, nên:

```
seasonal = xi − trend
resid    = xi − trend − (xi − trend) = 0
```

Feature sụp về đúng 0 cho mọi điểm, và **không có lỗi nào được ném ra**. Mô hình vẫn
nhận đủ vector đặc trưng, vẫn trả về điểm số, chỉ là điểm số đó vô nghĩa.

### 3.3 Bằng chứng thực nghiệm

Đo trên KPI `1c35dbf5` tại điểm cuối chuỗi, so giá trị tính trên cả chuỗi (đúng như lúc
train) với giá trị tính trên cửa sổ:

```
Giá trị ĐÚNG (tính trên toàn chuỗi):  2.484375

Tính trên cửa sổ:
    201 điểm  →   0.000000
  1.000 điểm  →   0.000000
  1.440 điểm  →   0.000000
  2.000 điểm  → -20.390625
  2.880 điểm  → -21.031250
 10.000 điểm  →   0.000000
```

Không kích thước cửa sổ nào cho ra giá trị đúng. Số lần mỗi mốc phút xuất hiện giải
thích rõ nguyên nhân:

```
cửa sổ 1.440 điểm  : mỗi mốc phút xuất hiện   1,0 lần
cửa sổ 2.880 điểm  : mỗi mốc phút xuất hiện   2,0 lần
toàn chuỗi (train) : mỗi mốc phút xuất hiện  89,5 lần
```

### 3.4 Hướng sửa

Tách ba thành phần của công thức và xét khả năng tính lại từ một cửa sổ:

| Thành phần | Tính được từ cửa sổ? | Lý do |
|-----------|---------------------|-------|
| `xi` (giá trị) | Có | dữ liệu gửi lên chính là nó |
| `trend` | Có, nếu cửa sổ ≥ `period` | là cửa sổ trượt **trailing**, chỉ nhìn quá khứ; đủ `period` điểm là cửa sổ đầy |
| `seasonal` | **Không bao giờ** | cần hàng chục ngày để trung vị có ý nghĩa |

Kết luận: **profile mùa vụ phải được tính một lần lúc đóng gói và cất vào bundle**, còn
`trend` thì tính lại được lúc serve miễn là cửa sổ đủ dài.

Kiểm chứng hướng sửa bằng cách so toàn bộ vector đặc trưng và điểm số cuối cùng:

```
N điểm gửi   số feature lệch   sai lệch score
       201                 1        +0.002133
     1.000                 1         0.000000
     1.440                 0         0.000000    ← khớp tuyệt đối
     2.880                 0         0.000000
```

Từ đây rút ra quy tắc đặt `min_pts`: KPI dùng `deseason_resid` cần đúng `period` điểm;
KPI không dùng thì chỉ cần vượt lookback dài nhất trong tên feature.

---

## 4. Chi tiết từng file

### 4.1 `FE_FS/Code/features.py` — sửa

Đây là file dùng chung giữa lúc train và lúc serve, nên mọi thay đổi phải **tương thích
ngược tuyệt đối**.

**Thay đổi 1 — thêm tham số `season_prof`:**

```python
def build_features(p, period=None, season_prof=None):
```

Và trong thân hàm:

```python
mod = ((ts % 86400) // 60).astype(int)
if season_prof is None:
    season_prof = pd.Series((xi - trend).values[tr_rows]).groupby(mod[tr_rows]).median()
seasonal = pd.Series(mod).map(season_prof)
```

Khi `season_prof` là `None` (mặc định), code đi đúng nhánh cũ — tự tính profile từ dữ
liệu truyền vào. Notebook train gọi `build_features(p)` không truyền tham số này, nên
hành vi không đổi một chút nào. Chỉ khi `infer.py` chủ động truyền profile đóng băng vào
thì mới đi nhánh mới.

**Thay đổi 2 — đính profile vào kết quả trả về:**

```python
    # metadata đính kèm (không phải feature) — build serve bundle lấy ra đóng băng
    F.attrs["season_prof"] = season_prof
    F.attrs["period"] = period
    return F
```

Dùng `.attrs` của DataFrame thay vì đổi kiểu giá trị trả về, để mọi lời gọi cũ không bị
ảnh hưởng. Nhờ đó `build_server.py` lấy được profile mà không phải lặp lại năm dòng
logic tính trend — tránh nguy cơ hai nơi trôi dạt khỏi nhau.

**Kiểm chứng tính tương thích ngược:** nạp lại bảng `X_test` (2000 hàng × 25 cột đặc
trưng) đã lưu trong artifact từ lúc train, tính lại bằng code mới, so từng ô. Sai lệch
lớn nhất trên 50.000 con số là `0.0000000000`. Mô hình, ngưỡng, và mọi số liệu ở Panel 2
đều giữ nguyên giá trị — **không cần train lại**.

---

### 4.2 `Modeling/Code/build_server.py` — viết lại

File này chạy ngoại tuyến, sinh ra các file `serve_<kpi>.joblib`.

**Trước:** danh sách KPI viết cứng hai phần tử, `min_pts=65` cố định cho cả hai.

```python
KPIS = ["18fbb1d5", "c58bfcba"]        # 2 KPI dễ (không dùng deseason); thêm dần sau
...
min_pts=65,                            # đủ cho lookback 60 của 2 KPI này
```

**Sau:** quét toàn bộ artifact, tái tạo đúng pipeline lúc train để lấy profile mùa vụ,
tính `min_pts` riêng cho từng KPI.

Phần tái tạo pipeline phải khớp chính xác với notebook train, kể cả tham số:

```python
gk = df[df.k8 == kpi].drop(columns="k8").copy()   # bỏ k8: khớp đúng input lúc train
gk = time_split_per_kpi(gk, train_frac=0.6, val_frac=0.2)
p = preprocess_all(gk, max_gap_points=5, norm_method="robust") \
    .sort_values("timestamp").reset_index(drop=True)
F = build_features(p)
prof, period = F.attrs["season_prof"], F.attrs["period"]
```

Việc `drop(columns="k8")` là cần thiết vì notebook train duyệt `df.groupby("kpi")` trên
DataFrame chưa có cột `k8`; giữ cột thừa sẽ làm `reindex_regular_grid` mang theo một cột
lạ qua bước merge.

Quy tắc tính số điểm tối thiểu:

```python
BASE_MIN_PTS = 65        # đủ cho lookback 60 (lag_60, rmean_60, ...) + đệm

def lookback_from_names(features):
    """Lookback dài nhất suy từ hậu tố số trong tên feature (lag_60 -> 60)."""
    lb = 0
    for c in features:
        tail = c.rsplit("_", 1)[-1]
        if tail.isdigit():
            lb = max(lb, int(tail))
    return lb

needs_deseason = "deseason_resid" in feats
min_pts = max(BASE_MIN_PTS, lookback_from_names(feats) + 5)
if needs_deseason:
    min_pts = max(min_pts, period)     # trend cần đầy cửa sổ `period`
```

Lưu ý `deseason_resid` không có hậu tố số trong tên nên `lookback_from_names` không bắt
được nó — đó là lý do phải xét riêng bằng câu lệnh `if`.

Cấu trúc bundle sinh ra gồm chín trường:

```python
bundle = dict(
    kpi=kpi,
    model=d["model"],              # IsolationForest đã train, 100 cây
    features=feats,                # tên VÀ thứ tự — mô hình học theo vị trí cột
    thr=float(cfg[kpi]["b_thr_pw"]),
    step_s=step,
    min_pts=int(min_pts),
    period=int(period),
    season_prof=prof,              # ĐÓNG BĂNG — serve tra bảng, không tự tính
    score_ref=np.quantile(sc, np.linspace(0, 1, 101)) if len(sc) else None,
)
```

Trong đó `score_ref` là 101 phân vị của toàn bộ điểm số lịch sử, dùng để quy đổi một
điểm số mới thành thứ hạng phân vị (`confidence`).

**Kết quả chạy:** 18 bundle, trong đó 16 KPI cần `deseason_resid`. KPI `07927a9a` có bước
lấy mẫu 300s nên `period` tự ra 288 chứ không phải 1440 — công thức `86400 // step` xử lý
việc này mà không cần viết trường hợp riêng.

---

### 4.3 `Serving/infer.py` — sửa

Lõi inference, không chứa một dòng nào liên quan HTTP. Ba hàm:

```python
def available_kpis():
    return sorted(p.stem.replace("serve_", "") for p in ART.glob("serve_*.joblib"))

def load_bundle(kpi):
    f = ART / f"serve_{kpi}.joblib"
    return joblib.load(f) if f.exists() else None

def predict_one(values, ts_end, bundle):
    ...
```

`available_kpis()` không có danh sách viết cứng — nó quét thư mục. Thêm một file bundle
là tự động có thêm một KPI được phục vụ. `load_bundle()` trả `None` thay vì ném lỗi khi
không tìm thấy, nhường quyền quyết định mã lỗi cho tầng trên.

**Thay đổi chính** nằm trong `predict_one`:

```python
# TRƯỚC
row = build_features(p)[bundle["features"]].iloc[-1]

# SAU
F = build_features(p, period=bundle.get("period"),
                   season_prof=bundle.get("season_prof"))
row = F[bundle["features"]].iloc[-1]
```

Dùng `.get()` thay vì truy cập trực tiếp để bundle cũ (chưa có hai trường này) vẫn nạp
được — khi đó `None` được truyền vào và `build_features` quay về nhánh tự tính.

**Năm bước của `predict_one`:**

*Bước 1 — kiểm tra đủ điểm.* Ngưỡng đọc từ bundle nên mỗi KPI một yêu cầu khác nhau.

```python
if n < bundle["min_pts"]:
    return {"error": f"Cần ≥{bundle['min_pts']} điểm, nhận {n}."}
```

*Bước 2 — dựng lại trục thời gian.*

```python
ts = int(ts_end) - (n - 1 - np.arange(n)) * step
p = pd.DataFrame({"value_filled": values, "timestamp": ts,
                  "masked": False, "split": "train"})
```

Client chỉ gửi một mốc thời gian (`ts_end`, ứng với điểm cuối cùng); server suy ngược ra
các mốc còn lại bằng cách trừ dần `step`. Cần timestamp vì bốn đặc trưng `tod_sin`,
`tod_cos`, `dow_sin`, `dow_cos` mã hóa giờ-trong-ngày và thứ-trong-tuần, và vì việc tra
bảng `season_prof` dựa trên phút-trong-ngày.

Hai cột `masked=False` và `split="train"` là **lời khai báo giả định**: cửa sổ liền mạch,
và coi mọi hàng là dữ liệu train. Chúng chỉ tồn tại để `build_features` — vốn viết cho
lúc train — chạy được nguyên vẹn.

Công thức này giả định các điểm **cách đều tuyệt đối**. Trách nhiệm bảo đảm điều đó nằm ở
phía gọi, và đó là lý do Panel 3 phải cắt cửa sổ trên lưới đều.

*Bước 3 — tính đặc trưng*, chỉ lấy hàng cuối cùng: `F[bundle["features"]].iloc[-1]`. Thứ
tự cột lấy từ bundle, vì mô hình học theo vị trí chứ không theo tên.

*Bước 4 — kiểm tra rỗng.* Chỉ cần một đặc trưng là `NaN` thì dừng, kèm theo tên các cột
bị rỗng để dễ chẩn đoán.

```python
if row.isna().any():
    return {"error": "Thiếu lịch sử hoặc dữ liệu không đều.",
            "nan_features": list(row.index[row.isna()])}
```

*Bước 5 — chấm điểm và kết luận.*

```python
score = float(-bundle["model"].decision_function(row.values.reshape(1, -1))[0])  # ĐẢO dấu
thr = float(bundle["thr"])
ref = bundle.get("score_ref")
conf = float((np.asarray(ref) < score).mean()) if ref is not None else None
```

`decision_function` của scikit-learn quy ước **càng cao càng bình thường**; đảo dấu để
thành **càng cao càng bất thường**, trực giác hơn khi hiển thị. `confidence` là tỷ lệ
phân vị lịch sử thấp hơn điểm số hiện tại — nó là **thứ hạng**, không phải xác suất.

---

### 4.4 `Serving/api.py` — sửa

Lớp vỏ HTTP mỏng, 51 dòng, hai endpoint. Không tính toán gì, chỉ phiên dịch giữa thế
giới HTTP và thế giới Python.

Bundle được nạp một lần ở cấp module:

```python
BUNDLES = {k: load_bundle(k) for k in available_kpis()}
```

Vì dòng này chỉ chạy lúc khởi động, **thêm bundle mới bắt buộc phải khởi động lại
server**. Cờ `--reload` của uvicorn chỉ theo dõi file `.py`, không theo dõi `.joblib`.

Hợp đồng dữ liệu vào khai báo bằng Pydantic:

```python
class PredictRequest(BaseModel):
    kpi_id: str
    values: list[float]
    ts_end: int
```

Bốn dòng này thay cho toàn bộ code kiểm tra đầu vào: gửi sai kiểu hay thiếu trường đều
tự động nhận mã `422` kèm chỉ dẫn sai ở đâu.

**Thay đổi:** endpoint `/kpis` trước chỉ trả danh sách tên; nay trả thêm yêu cầu dữ liệu
của từng KPI, để phía giao diện biết cần gửi bao nhiêu điểm mà không phải đoán.

```python
return {
    "kpis": list(BUNDLES),
    "info": {k: {"min_pts": int(b["min_pts"]), "step_s": int(b["step_s"]),
                 "n_features": len(b["features"]),
                 "needs_deseason": "deseason_resid" in b["features"]}
             for k, b in BUNDLES.items()},
}
```

Endpoint `/predict` giữ nguyên không đổi một chữ — bằng chứng cho việc tách tầng có hiệu
quả: toàn bộ thay đổi về cách tính đặc trưng diễn ra bên dưới mà không lan lên đây.

```python
@app.post("/predict")
def predict(req: PredictRequest):
    bundle = BUNDLES.get(req.kpi_id)
    if bundle is None:
        raise HTTPException(404, f"KPI '{req.kpi_id}' chưa có serve bundle. ...")
    result = predict_one(req.values, req.ts_end, bundle)
    if "error" in result:
        raise HTTPException(422, result["error"])
    return result
```

Phân biệt mã lỗi có chủ đích: `404` nghĩa là thứ được hỏi không tồn tại (sai KPI), `422`
nghĩa là thứ được hỏi có tồn tại nhưng dữ liệu gửi lên không dùng được (thiếu điểm, đặc
trưng rỗng). Bên gọi nhờ đó biết phải sửa gì.

---

### 4.5 `App/common.py` — thêm `grid_series()`

Dữ liệu thô thiếu khoảng 2% điểm, và file CSV không lưu dòng trống mà bỏ qua luôn. Hệ
quả: cắt N dòng thô sẽ được một cửa sổ trải **hơn** N bước thời gian.

Mức độ nghiêm trọng phụ thuộc kích thước cửa sổ:

```
KPI        min_pts   % điểm thiếu   % vị trí cắt cho cửa sổ KHÔNG liền mạch
71595dd7      1440          2,25%                                    99,8%
8c892e55      1440          3,13%                                    99,8%
07927a9a       288         10,86%                                    92,0%
18fbb1d5        65          2,02%                                     1,7%
```

Chỉ 2% điểm thiếu nhưng vì cửa sổ dài 1440 điểm nên xác suất dính ít nhất một lỗ lên tới
99,8%. Với cửa sổ 201 điểm trước đây thì vấn đề gần như không lộ ra.

```python
@st.cache_data(show_spinner="Đang dựng lưới thời gian đều…")
def grid_series(k8: str, source: str = "train") -> pd.DataFrame:
    df = load_data(source)
    g = df[df["k8"] == k8][["timestamp", "value", "label", "kpi"]].sort_values("timestamp")
    g = reindex_regular_grid(g, kpi_col="kpi")
    g = mask_long_gaps(g, max_gap_points=5)
    g["dt"] = pd.to_datetime(g["timestamp"], unit="s")
    return g.reset_index(drop=True)
```

Hàm dùng lại đúng `reindex_regular_grid` và `mask_long_gaps` của `preprocess.py` với cùng
tham số `max_gap_points=5` như lúc train: lỗ ngắn từ 5 điểm trở xuống được nội suy tuyến
tính, lỗ dài hơn bị đánh dấu `masked` và giữ `NaN` để tầng trên từ chối chấm.

Kiểm chứng: với cả 18 KPI, lưới do `grid_series()` sinh ra **trùng khít** lưới do
`preprocess_all()` sinh ra lúc train, cả về mốc thời gian lẫn giá trị.

Về mức thiệt hại thực tế nếu cắt thô: đo trên 133 vị trí thuộc 4 KPI, sai lệch điểm số
trung bình nằm trong khoảng 0,000–0,003 và lớn nhất là 0,014; **không có trường hợp nào
đổi nhãn**. Nghĩa là cắt thô cho kết quả xấp xỉ chấp nhận được, nhưng cắt trên lưới đều
cho kết quả **chính xác tuyệt đối** và thêm khả năng từ chối cửa sổ hỏng thay vì bịa giá
trị.

---

### 4.6 `App/pages/3_Prediction.py` — viết lại

**Thay đổi 1 — cửa sổ động thay vì cố định.** Trước đây hằng số `WIN = 201` áp cho mọi
KPI. Nay kích thước lấy từ API:

```python
meta_api = requests.get(f"{API}/kpis", timeout=5).json()
kpis, info = meta_api["kpis"], meta_api.get("info", {})

need = int(info.get(kpi, {}).get("min_pts", 201))
step_s = int(info.get(kpi, {}).get("step_s", 60))
deseason = info.get(kpi, {}).get("needs_deseason", False)
```

Giao diện hiển thị con số này ngay trong danh sách chọn (`1c35dbf5 · 1440 điểm/lần`) và
giải thích lý do dài hay ngắn ở phần chú thích bên dưới.

**Thay đổi 2 — cắt trên lưới đều.** Thay `load_data("train")` bằng `grid_series(kpi)`, và
gửi cột `value_filled` thay vì `value`.

**Thay đổi 3 — chỉ nhảy tới vị trí liền mạch.** Trước đây hai nút ngẫu nhiên có thể ném
người dùng vào một cửa sổ dính lỗ. Nay tính trước tập vị trí hợp lệ:

```python
@st.cache_data(show_spinner=False)
def valid_ends(k8: str, win: int) -> np.ndarray:
    """Các vị trí kết thúc mà cửa sổ `win` điểm không dính gap dài (không có NaN)."""
    g = grid_series(k8)
    bad = g["value_filled"].isna().values.astype(int)
    if len(bad) < win:
        return np.array([], dtype=int)
    c = np.convolve(bad, np.ones(win, dtype=int), mode="valid")
    return np.nonzero(c == 0)[0] + win - 1
```

Dùng tích chập với cửa sổ toàn số 1 để đếm số điểm `NaN` trong mọi cửa sổ cùng lúc — chỗ
nào tổng bằng 0 là cửa sổ sạch. Cách này chạy trong một phép tính vector thay vì vòng lặp
qua hàng trăm nghìn vị trí. Số vị trí hợp lệ tìm được nằm trong khoảng 9.500–146.000 tùy
KPI, và được hiển thị ngay trên nút bấm.

**Thay đổi 4 — chặn cửa sổ hỏng khi chọn tay.** Nếu người dùng kéo thanh trượt tới vị trí
dính lỗ dài, giao diện cảnh báo và dừng, **không gọi API**:

```python
if win["value_filled"].isna().any():
    n_nan = int(win["value_filled"].isna().sum())
    st.warning(f"Cửa sổ này dính **gap dài** ({n_nan:,}/{need:,} điểm mất, ...)")
    st.stop()
```

**Thay đổi 5 — nhớ vị trí riêng theo từng KPI.** Khóa `session_state` gắn tên KPI vào:

```python
key = f"pos_{kpi}"
```

Nhờ đó đổi KPI rồi quay lại không mất chỗ đang xem. Điều này cần thiết vì Streamlit chạy
lại toàn bộ file mỗi lần tương tác.

**Thay đổi 6 — giảm điểm khi vẽ.** Cửa sổ 1440 điểm vẽ vẫn nhanh, nhưng đặt sẵn ngưỡng
để an toàn:

```python
show = win if len(win) <= 3000 else win.iloc[:: len(win) // 3000 + 1]
```

**Thay đổi 7 — chế độ dán giá trị kiểm tra theo từng KPI.** Trước chỉ báo "cần ≥65";
nay báo đúng con số của KPI đang chọn và từ chối nếu thiếu.

---

## 5. Kiểm chứng

| Nội dung kiểm | Cách làm | Kết quả |
|---------------|----------|---------|
| Sửa `features.py` có phá mô hình cũ không | Tính lại 2000×25 ô đặc trưng đã lưu trong artifact, so từng ô | sai lệch tối đa `0.0000000000` |
| `grid_series()` có khớp lưới lúc train không | So mốc thời gian và giá trị với `preprocess_all()`, cả 18 KPI | **18/18 trùng khít** |
| Điểm số qua API có khớp lúc train không | Gọi HTTP thật, so với `build_features` trên toàn chuỗi, cả 18 KPI | **18/18 khớp**, lệch `0.0e+00` |
| Đường code Panel 3 thực dùng | Lặp lại phép trên nhưng đi qua `grid_series` + `valid_ends` | **18/18 khớp**, lệch `0.0e+00` |
| Xử lý lỗi | Gọi API với 5 tình huống sai | đúng mã `404` / `422` ở đúng tầng |

Điểm cần lưu ý khi đọc kết quả: trong bảng kiểm thử, một số KPI trả về nhãn `normal` cho
điểm có nhãn thật là `anomaly`. Đây **không phải lỗi của tầng serving** — điểm số khớp
tuyệt đối với lúc train. Đó là **recall thấp của mô hình**, đã thấy rõ trong
`if_per_kpi_config.json` (ví dụ `71595dd7` có `b_PW_R = 0.189`). Bài kiểm thử này chứng
minh phần triển khai đúng, không chứng minh mô hình tốt.

---

## 6. Giới hạn hiện tại

1. **Chỉ phục vụ Isolation Forest.** ARIMA, STL và RL không đóng gói được thành mô hình
   chấm lẻ từng điểm — chúng fit trên cả chuỗi và cần giữ trạng thái. Vì Panel 2 cho thấy
   mô hình tốt nhất theo `AP_val` khác nhau tùy KPI, nên Panel 3 hiện **chưa dùng mô hình
   tốt nhất cho mọi KPI**.

2. **18 trên 26 KPI.** Tám KPI còn lại bị loại ngay từ bước train vì tập val hoặc test
   không có điểm bất thường nào, nên không chọn được ngưỡng và không đánh giá được. Cách
   cứu khả thi là đặt ngưỡng theo phân vị thay vì tối ưu F1, đổi lại không có số liệu tin
   cậy kèm theo.

3. **Không lưu lịch sử dự đoán.** Mỗi lời gọi `/predict` là độc lập, chấm xong không để
   lại dấu vết. Panel 4 (alert log) và Panel 5 (incident timeline) sẽ cần bổ sung nơi lưu
   trữ — với quy mô này thì SQLite là đủ.

4. **Chấm một điểm mỗi lời gọi.** Chưa có chấm hàng loạt nhiều điểm hoặc nhiều KPI cùng
   lúc, thứ mà Panel 5 sẽ cần khi so sánh các KPI để tìm nguyên nhân gốc.

5. **Ngưỡng cố định trong bundle**, chưa điều chỉnh được từ giao diện.

6. **Chỉ đọc `train.csv`**, chưa nối vào nguồn dữ liệu chạy thật.

7. **Bẫy tiềm ẩn với tập test.** `grid_series(kpi, "test")` sẽ biến nhãn `NaN` (tập test
   không có nhãn) thành `0`, do `reindex_regular_grid` có sẵn `fillna(0).astype(int)`.
   Panel 3 chỉ dùng `train` nên chưa gặp, nhưng cần chặn trước khi Panel 4 hoặc 5 dùng
   tới tập test.

8. **Artifact không nằm trong git.** `.gitignore` loại `Modeling/Artifacts/`, nên người
   clone repo về phải chạy lại `Modeling/Code/build_server.py` mới có 18 bundle.

---

## 7. Cách chạy

Cần hai tiến trình chạy song song.

```bash
# Terminal 1 — API
cd anomaly-detection-fundamentals
python -m uvicorn api:app --app-dir Serving --reload --port 8000

# Terminal 2 — giao diện
python -m streamlit run App/Home.py
```

Bật API trước rồi mới mở Streamlit, vì Panel 3 kiểm tra kết nối ngay khi nạp trang. Nếu
mở ngược thì chỉ cần tải lại trang sau khi API đã lên.

Thử API trực tiếp không cần giao diện: mở `http://localhost:8000/docs`, hoặc chạy
`Serving/try_api.py` (kiểm tra qua HTTP) và `Serving/try_infer.py` (kiểm tra `predict_one`
mà không cần bật server).

Lưu ý: hai script `try_*.py` chỉ kiểm tra KPI `18fbb1d5` với cửa sổ 201 điểm cắt thô —
tức là một trong hai KPI **không** dùng `deseason_resid`. Chúng chạy thành công không
chứng minh được điều gì về 16 KPI còn lại.

Khi sinh thêm bundle mới, phải khởi động lại tiến trình API thì danh sách KPI mới cập
nhật.
