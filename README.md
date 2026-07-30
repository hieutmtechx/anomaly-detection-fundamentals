# Anomaly Detection trên KPI time series

Phát hiện bất thường trên chuỗi KPI hệ thống, từ EDA → feature engineering → model → evaluation  → dashboard.

## Mục lục

- [Bài toán](#bài-toán)
- [Dữ liệu](#dữ-liệu)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Tech stack](#tech-stack)
- [Cách chạy](#cách-chạy)
- [Pipeline hoạt động thế nào](#pipeline-hoạt-động-thế-nào)
- [Kết quả](#kết-quả)
- [Hạn chế](#hạn-chế)
- [Tài liệu](#tài-liệu)


## Bài toán

Cho một chuỗi giá trị KPI theo thời gian (CPU, lưu lượng, tỷ lệ lỗi… đã ẩn danh), xác định mỗi điểm là **bình thường** hay **bất thường**. Đầu ra phải dùng được trong vận hành: có điểm số, có ngưỡng, có giải thích.

Ba đặc thù khiến đây không phải bài toán phân loại nhị phân thông thường:

**Mất cân bằng** — 2,16% điểm là anomaly, KPI ít nhất chỉ 0,02%. Tỷ lệ giữa KPI cao nhất và thấp nhất chênh 364 lần. Accuracy vô nghĩa; phải dùng PR-AUC, và tune ngưỡng riêng từng KPI.

**Bất thường theo cụm, không phải điểm đơn lẻ** — đo được 933 đoạn anomaly, trung vị 5 điểm, chỉ 1% là đoạn một điểm và 30% dài từ 10 điểm trở lên.

## Dữ liệu

| | |
|---|---|
| Số KPI | 26 (ID đã hash, không có tên nghiệp vụ) |
| `Data/train.csv` | 2.476.315 dòng · cột `timestamp, value, label, KPI ID` |
| `Data/test.csv` | 2.345.211 dòng · không có cột `label` |
| Khoảng thời gian | 2016-06-30 → 2017-08-11 |
| Tỷ lệ anomaly | 2,16% (53.500 điểm) |
| Bước lấy mẫu | 19 KPI ở 60s · 7 KPI ở 300s |

Vì `test.csv` không có nhãn nên toàn bộ đánh giá được làm bên trong `train.csv`, chia theo thời gian 60/20/20. `test.csv` chỉ dùng để xem chuỗi ở Panel 1.

26 KPI không cùng khoảng thời gian, chúng thuộc nhiều cohort rời nhau (một nhóm 2016 ở 300s, phần còn lại 2017 ở 60s). Vì vậy correlation heatmap chỉ tính trên nhóm 16 KPI có lưới 60s khớp nhau.

18/26 KPI đi được đến cuối pipeline, 8 KPI bị loại vì val hoặc test của chúng không chứa anomaly nào sau khi chia.

**Nguồn dữ liệu.** Bộ Preliminary dataset của [NetManAIOps / KPI-Anomaly-Detection](https://github.com/NetManAIOps/KPI-Anomaly-Detection/tree/master/Preliminary_dataset) — dữ liệu KPI thật từ các công ty Internet, dùng trong AIOps Challenge.


## Cấu trúc repo

```
anomaly-detection-fundamentals/
├─ Problem/                    # Lý thuyết: định nghĩa anomaly, time series vs tabular
├─ EDA/
│  ├─ Code/EDA.ipynb
│  └─ Docs/EDA.md
├─ FE_FS/
│  ├─ Code/features.py         # build_features (27 feature) + select_features (3 bộ lọc)
│  └─ Docs/
├─ Modeling/
│  ├─ Code/
│  │  ├─ preprocess.py         # lưới đều → mask gap dài → segment → chuẩn hoá
│  │  ├─ eval_protocol.py      # chia split, PR/F1, best-F1 threshold, TTD, PR curve
│  │  └─ build_server.py       # đóng gói serve bundle cho từng KPI
│  ├─ Stats/
│  │  ├─ RL/RL_Zscore.ipynb    # rolling z-score (baseline)
│  │  ├─ STL/all_kpis.ipynb    # STL decomposition
│  │  └─ ARIMA/all_kpis.ipynb  # ARIMA residual
│  ├─ ML/IsolationForest/all_kpis.ipynb
│  ├─ Artifacts/               # 18 model + 18 serve bundle + 4 file score
│  ├─ Docs/
│  └─ compare_models.ipynb
├─ Serving/
│  ├─ infer.py                 # predict_one — dùng chung cho API và App
│  ├─ api.py                   # FastAPI: GET /kpis, POST /predict
│  └─ Docs/Live_Prediction.md
├─ App/
│  ├─ Home.py                  # điểm vào Streamlit
│  ├─ common.py                # nạp data, lưới đều, metadata, ACF, tương quan
│  ├─ modellab.py              # tầng dữ liệu Panel 2
│  ├─ panel4.py / panel5.py    # tầng dữ liệu Panel 4 / 5
│  └─ pages/                   # 5 panel
└─ Data/                       # train.csv, test.csv
```

## Tech stack

| Nhóm | Thư viện | Dùng để làm gì |
|---|---|---|
| Xử lý dữ liệu | `pandas`, `numpy` | Toàn bộ pipeline |
| Model thống kê | `statsmodels` | STL decomposition, ARIMA |
| Model ML | `scikit-learn` | Isolation Forest, mutual information, PR-AUC / ROC |
| Giải thích | `shap` | TreeExplainer cho Isolation Forest (Panel 2C, 5B) |
| API | `fastapi`, `uvicorn`, `pydantic` | `GET /kpis`, `POST /predict` |
| Dashboard | `streamlit`, `plotly` | 5 panel tương tác |
| Lưu trữ | `joblib`, `pyarrow` | Model bundle (.joblib), điểm số (.parquet) |
| Notebook | `jupyter`, `matplotlib` | Biểu đồ trong notebook |

## Cách chạy

### Bước 0 — Môi trường

```bash
git clone https://github.com/hieutmtechx/anomaly-detection-fundamentals.git
cd anomaly-detection-fundamentals

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
```

Dữ liệu và model đã nằm sẵn trong repo, **xuống [Bước 2](#bước-2--bật-api)** là chạy được dashboard.

### Bước 1 — Sinh lại artifacts (không bắt buộc)

Chỉ cần khi bạn muốn train lại từ đầu, hoặc đã sửa code trong `FE_FS/` hay `Modeling/`. Nếu chỉ muốn xem hệ thống chạy thì bỏ qua bước này.

```bash
jupyter notebook Modeling/ML/IsolationForest/all_kpis.ipynb    # chạy toàn bộ cell
python Modeling/Code/build_server.py                            # đóng gói 18 serve bundle
```

`build_server.py` in ra bảng xác nhận:

```
[OK] 02e99bd4: 19 feat · thr=  0.0327 · step= 60s · period=1440 · min_pts=1440  (cần deseason)
[OK] c58bfcba: 17 feat · thr=  0.0207 · step= 60s · period=1440 · min_pts=  65
...
Xong: 18 bundle -> Modeling/Artifacts
```

**Đầy đủ 4 model** để Panel 2 so sánh được cả bảng, chạy thêm 3 notebook, thứ tự bất kỳ:

```
Modeling/Stats/RL/RL_Zscore.ipynb
Modeling/Stats/STL/all_kpis.ipynb
Modeling/Stats/ARIMA/all_kpis.ipynb
```

### Bước 2 — Bật API

```bash
uvicorn api:app --app-dir Serving --reload --port 8000
```

Kiểm tra nhanh:

```bash
curl http://localhost:8000/kpis
```

```json
{
  "kpis": ["02e99bd4", "07927a9a", "..."],
  "info": {
    "c58bfcba": {"min_pts": 65,   "step_s": 60, "n_features": 17, "needs_deseason": false},
    "02e99bd4": {"min_pts": 1440, "step_s": 60, "n_features": 19, "needs_deseason": true}
  }
}
```

`min_pts` khác nhau giữa các KPI và client bắt buộc phải đọc số này. KPI dùng feature `deseason_resid` cần đủ một chu kỳ ngày (1440 điểm ở bước 60s) vì trend là trung vị trượt trên cửa sổ đó; KPI không dùng thì 65 điểm là đủ.

Gọi thử `POST /predict` — 65 giá trị liên tiếp, cách nhau 60s:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"kpi_id": "c58bfcba", "values": [0.12, 0.13, ...], "ts_end": 1502265600}'
```

```json
{
  "kpi": "c58bfcba",
  "score": 0.00720823,
  "threshold": 0.0207,
  "label": "normal",
  "confidence": 0.9504,
  "n_points_used": 65,
  "features_used": 17
}
```

### Bước 3 — Bật dashboard

Mở terminal thứ hai, giữ API đang chạy:

```bash
streamlit run App/Home.py
```

| Panel | Nội dung | Cần gì để chạy |
|---|---|---|
| **1 · Data Explorer** | Chuỗi thời gian, phân bố nhãn, tương quan giữa KPI, phân bố giá trị, đoạn anomaly, ACF + heatmap giờ×thứ | `Data/*.csv` |
| **2 · Model Lab** | So sánh 4 model, trực quan phát hiện trên trục thời gian, SHAP | Khối A: config JSON (có sẵn trong repo) · Khối B/C: Artifacts |
| **3 · Live Prediction** | Chọn KPI → cắt cửa sổ thật hoặc tự dán → gọi API → nhãn + score + gauge | **API đang chạy** + serve bundle |
| **4 · System Health** | Nhật ký alert (đúng/báo động giả), alert theo KPI và theo ngày, chỉ báo drift | `scores_IF_feat.parquet` |
| **5 · Root Cause** | Gom alert thành sự cố, xếp hạng KPI nghi là nguyên nhân gốc, SHAP | `scores_IF_feat.parquet` |

## Pipeline hoạt động thế nào

```
train.csv
   ↓  time_split_per_kpi      chia 60/20/20 theo thời gian, riêng từng KPI
   ↓  preprocess_all          lưới đều → nội suy gap ngắn (≤5 điểm) → mask gap dài
   │                          → đánh segment id (không bắc cầu) → chuẩn hoá robust
   ↓  build_features          27 feature: lag, rolling stats, độ lệch baseline,
   │                          tốc độ thay đổi, deseason_resid, mã hoá thời gian
   ↓  select_features         suy biến → tương quan → mutual information
   ↓  IsolationForest         tune max_samples trên val
   ↓  evaluate_protocol       ngưỡng max-F1 chọn trên VAL, mọi chỉ số báo cáo trên TEST
   ↓
   ├→ scores_*.parquet + config JSON   →  Panel 2, 4, 5
   └→ build_server.py → serve_<kpi>.joblib → API → Panel 3
```

## Kết quả

Macro trung bình trên 18 KPI, tập test, ngưỡng chọn trên val:

| Model | AP (PR-AUC) | ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Isolation Forest** | **0.458** | **0.877** | 0.426 | **0.415** | **0.352** |
| ARIMA residual | 0.381 | 0.795 | **0.447** | 0.383 | 0.312 |
| STL residual | 0.280 | 0.732 | 0.405 | 0.302 | 0.223 |
| Rolling z-score | 0.148 | 0.542 | 0.308 | 0.368 | 0.167 |

Một phát hiện đáng nói: phần lớn chiến thắng thuộc về feature engineering, không thuộc về thuật toán. Cùng Isolation Forest nhưng đổi đầu vào từ feature đã tuyển sang shingle thô, macro AP tụt từ 0.458 xuống 0.252.

## Hạn chế

**Tập validation không đại diện.** Cách chia 60/20/20 theo số hàng đẩy cả cụm sự cố về một phía. KPI `18fbb1d5` có val chứa 25,77% anomaly nhưng test chỉ 0,18%, lệch 141 lần. Hậu quả đo được: nếu dùng val để chọn giữa hai biến thể model, macro AP giảm từ 0.458 xuống 0.413. Ngoài ra 8/26 KPI bị loại hoàn toàn vì val hoặc test của chúng không có anomaly nào.

**2/18 KPI có recall bằng 0** trên test — ngưỡng tune trên val không bao giờ bị vượt.

**API không kiểm tra giả định đầu vào** — không xác minh dữ liệu gửi lên có thật sự nằm trên lưới đều hay không.

**Model nạp một lần lúc khởi động API** — train lại thì phải restart server.

## Tài liệu

| File | Nội dung |
|---|---|
| `Modeling/Docs/Evaluation_Metrics.md` | Metric, point-wise vs point-adjust, PR curve, chọn ngưỡng |
| `Modeling/Docs/Stats_Model.md` · `ML_Model.md` | Lý thuyết từng model |
| `FE_FS/Docs/Feature_Engineering.md` · `Feature_Selection.md` | 6 nhóm feature · 3 bộ lọc |
| `EDA/Docs/EDA.md` | Checklist EDA cho time series |
| `Problem/` | Định nghĩa anomaly, time series vs tabular |

