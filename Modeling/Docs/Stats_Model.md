## 1. Rolling Z-score

### 1.1 Ý tưởng

Z-score đo độ lệch của một điểm dữ liệu so với phân phối "bình thường" gần đó theo đơn vị độ lệch chuẩn. Bản chất đây là phương pháp đơn giản nhất, dùng như baseline trước khi thử các model phức tạp hơn — nếu model phức tạp không vượt được baseline này thì không đáng dùng.

Rolling nghĩa là mean và std được tính trên một **cửa sổ trượt** (sliding window) thay vì toàn bộ lịch sử, để thích nghi với baseline thay đổi theo thời gian (trend, seasonality chậm).

### 1.2 Công thức

```
mean_t = mean(x[t-w : t-1])
std_t  = std(x[t-w : t-1])
z_t    = (x_t - mean_t) / std_t
```

Trong đó `w` là kích thước cửa sổ (ví dụ 60 điểm nếu dữ liệu phút và muốn nhìn lại 1 giờ).

**Lưu ý quan trọng**: window phải dùng dữ liệu **quá khứ tại thời điểm t** (không bao gồm x_t), tránh leakage — nếu đưa x_t vào tính mean/std của chính nó, anomaly cực đoan sẽ tự kéo mean/std giãn ra, làm giảm z-score của chính nó (self-masking).

### 1.3 Biến thể Robust Z-score (MAD-based)

Mean và std nhạy cảm với outlier — một vài điểm anomaly trong window có thể kéo lệch cả mean và std, làm giảm khả năng phát hiện các anomaly tiếp theo trong cùng window (masking effect). Robust Z-score dùng median và MAD (Median Absolute Deviation) thay thế:

```
median_t = median(x[t-w : t-1])
MAD_t    = median(|x[t-w:t-1] - median_t|)
robust_z_t = 0.6745 · (x_t - median_t) / MAD_t
```

Hằng số 0.6745 là hệ số chuẩn hoá để MAD tương đương std trong phân phối chuẩn, giúp robust z-score có thang đo tương đương z-score thường.

→ **Khuyến nghị**: dùng Robust Z-score làm baseline mặc định cho hệ thống production vì dữ liệu CPU/latency hiếm khi phân phối chuẩn hoàn hảo và thường có outlier lịch sử.

### 1.4 Pseudocode Train + Predict

```
Pseudocode rolling_zscore_fit_predict(x, window_size=60, use_robust=True):
    scores = []
    for t in range(window_size, len(x)):
        window = x[t-window_size : t]
        if use_robust:
            center = median(window)
            scale = median(abs(window - center)) * 1.4826  # 1/0.6745
        else:
            center = mean(window)
            scale = std(window)

        if scale == 0:  # tránh chia 0 khi window phẳng tuyệt đối
            scale = epsilon

        score = abs(x[t] - center) / scale
        scores.append(score)
    return scores  # anomaly score liên tục, càng lớn càng bất thường
```

Đây là mô hình **không cần "train" theo nghĩa học tham số** — chỉ cần chọn window_size phù hợp, nên không có giai đoạn fit riêng biệt, predict luôn là online.

### 1.5 Đánh giá theo PR Curve và Tune Threshold

```
1. Chạy rolling_zscore_fit_predict trên toàn bộ chuỗi → có scores liên tục
2. Chia scores/labels thành validation (để tune) và test (để đánh giá final)
3. precision_recall_curve(y_val, scores_val) → chọn threshold theo best-F1
4. Áp threshold lên scores_test → tính Precision/Recall/F1 (PW và PA)
```

**Hyperparameter cần tune thêm ngoài threshold**: `window_size`. Nên thử lưới giá trị (ví dụ 15, 30, 60, 120 điểm) và chọn theo AP (Average Precision) trên validation set trước khi tune threshold cuối cùng — window quá nhỏ → nhạy nhiễu (Precision thấp), window quá lớn → chậm thích nghi với thay đổi baseline thật (Recall thấp với anomaly dạng "level shift").

### 1.6 Ưu / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Cực nhanh, dễ triển khai real-time | Không xử lý được seasonality (z-score cao giả vào giờ cao điểm bình thường) |
| Diễn giải trực quan (bao nhiêu độ lệch chuẩn) | Nhạy với window_size, cần tune thủ công |
| Không cần training data lớn | Không bắt được anomaly dạng thay đổi pattern/trend dần dần (gradual drift) |
| Tốt cho spike/dip đột ngột | Self-masking nếu dùng mean/std thường thay vì robust |

---

## 2. STL Decomposition + Residual Thresholding

### 2.1 Ý tưởng

STL (Seasonal-Trend decomposition using LOESS) tách time series thành 3 thành phần:

```
x_t = Trend_t + Seasonal_t + Residual_t   (additive)
hoặc
x_t = Trend_t · Seasonal_t · Residual_t   (multiplicative, thường log-transform trước rồi dùng additive)
```

Ý tưởng cốt lõi: sau khi loại bỏ trend (xu hướng dài hạn) và seasonal (chu kỳ lặp lại — ví dụ pattern theo giờ trong ngày của traffic web), phần còn lại (residual) phải "nhiễu ngẫu nhiên" nếu hệ thống hoạt động bình thường. Bất thường thực sự sẽ để lại residual lớn bất thường so với phân phối nhiễu nền.

→ Đây là cải tiến trực tiếp so với Rolling Z-score thuần vì **giải quyết được vấn đề seasonality** — z-score thường sẽ báo động giả vào giờ cao điểm hợp lệ, STL residual thì không vì seasonal pattern đã bị trừ ra trước.

### 2.2 Quy trình

```
Bước 1 — STL decomposition:
    period = chu kỳ chính của dữ liệu (ví dụ 1440 nếu data theo phút, chu kỳ ngày;
             hoặc 24 nếu data theo giờ, chu kỳ ngày; 7 nếu data theo ngày, chu kỳ tuần)
    trend, seasonal, residual = STL(x, period=period).fit()

Bước 2 — Tính ngưỡng trên residual:
    center = median(residual)   # dùng median thay mean vì residual có thể bị méo bởi
                                  # chính các anomaly trong dữ liệu train
    MAD = median(|residual - center|)
    threshold_band = center ± k · MAD · 1.4826   (k thường 3)

Bước 3 — Gắn nhãn:
    anomaly nếu |residual_t - center| > k · MAD · 1.4826
    anomaly_score_t = |residual_t - center| / (MAD · 1.4826)   # dùng làm score liên tục cho PR curve
```

### 2.3 Vấn đề thực tế: Residual bị méo bởi chính anomaly (quan trọng)

Nếu trong dữ liệu dùng để fit STL đã tồn tại sẵn các đoạn anomaly kéo dài (ví dụ outage nhiều giờ), STL có thể "hấp thụ" anomaly đó vào thành phần **trend** (vì LOESS coi đó là thay đổi xu hướng), khiến residual tại chính đoạn anomaly đó lại nhỏ bất thường — gọi là hiện tượng "spurious anomaly" / trend bị méo. Hậu quả: model bỏ sót chính anomaly mà nó cần phát hiện, đồng thời sinh ra false positive ở các điểm lân cận do trend bị lệch.

**Cách giảm thiểu**:
- Dùng **median của toàn chuỗi** làm baseline ổn định thay vì để trend tự do hấp thụ mọi biến động (kỹ thuật tương tự thuật toán S-H-ESD của Twitter).
- Hoặc dùng Robust STL (loại biến thể chống outlier khi fit LOESS).
- Hoặc đảm bảo tập dùng để fit STL ban đầu tương đối "sạch" (giai đoạn hệ thống hoạt động bình thường), chỉ áp model lên dữ liệu mới để phát hiện.

### 2.4 Pseudocode Train + Predict

```
Pseudocode stl_fit_predict(x, period, k=3):
    stl_result = STL(x, period=period, robust=True).fit()
    residual = stl_result.resid

    center = median(residual)
    mad = median(abs(residual - center))
    scale = mad * 1.4826

    scores = abs(residual - center) / scale   # anomaly score liên tục
    return scores, stl_result  # giữ lại trend/seasonal để debug/visualize
```

Lưu ý: STL truyền thống (statsmodels) là **batch**, không phải online — cần fit lại định kỳ (ví dụ mỗi ngày) trên cửa sổ dữ liệu gần nhất khi dùng cho production, không fit một lần rồi dùng mãi mãi vì seasonal pattern có thể trôi theo thời gian.

### 2.5 Đánh giá theo PR Curve và Tune Threshold

Tương tự mục 1.5, nhưng có thêm hyperparameter `period` cần xác định trước (thường biết trước từ domain knowledge — ví dụ traffic web luôn có chu kỳ ngày) thay vì tune tự động. Có thể thử nghiệm thêm:
- `k` (số MAD) qua lưới giá trị 2, 2.5, 3, 3.5 → chọn theo AP trên validation.
- So sánh additive vs multiplicative decomposition nếu biên độ seasonal tỉ lệ thuận với mức nền (ví dụ traffic tăng theo % chứ không theo số tuyệt đối → nên multiplicative).

### 2.6 Ưu / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Xử lý tốt seasonality — giảm false positive ở giờ cao điểm | Cần biết trước `period`, không tự động phát hiện chu kỳ |
| Diễn giải được (xem trực tiếp residual plot) | Batch, cần refit định kỳ, không thật sự online |
| Robust version giảm ảnh hưởng của anomaly trong tập fit | Anomaly kéo dài có thể bị "hấp thụ" vào trend (mục 2.3) |
| Tốt cho dữ liệu có chu kỳ rõ ràng (daily/weekly traffic) | Kém hơn nếu dữ liệu không có seasonality rõ (ví dụ latency ngẫu nhiên) |

---

## 3. ARIMA (AutoRegressive Integrated Moving Average)

### 3.1 Ý tưởng

ARIMA mô hình hoá giá trị tại thời điểm t dựa trên các giá trị quá khứ (AR), độ sai phân để loại trend (I — Integrated), và sai số dự báo quá khứ (MA). Áp dụng cho anomaly detection bằng cách: **dự báo giá trị kỳ vọng tại mỗi điểm, so sánh với giá trị thực tế** — nếu sai số dự báo (residual) vượt ngưỡng, coi là anomaly.

Đây là cách tiếp cận khác về bản chất so với Z-score/STL (vốn dựa trên thống kê mô tả) — ARIMA là **mô hình dự báo (forecasting)**, anomaly được định nghĩa là "điểm mà mô hình dự báo tốt cũng không đoán trúng".

### 3.2 Công thức ARIMA(p, d, q)

```
Bước differencing (d lần) để chuỗi dừng (stationary):
    x'_t = x_t - x_{t-1}        (d=1, có thể lặp lại nếu cần d=2)

Phương trình ARMA(p, q) trên chuỗi đã sai phân:
    x'_t = c + Σ(i=1..p) φ_i · x'_{t-i} + Σ(j=1..q) θ_j · ε_{t-j} + ε_t

Trong đó:
    p = bậc AR (số lag tự hồi quy)
    d = bậc differencing
    q = bậc MA (số lag của sai số)
    φ_i, θ_j = hệ số học được khi fit model
    ε_t = sai số (residual) tại thời điểm t — đây chính là phần dùng để phát hiện anomaly
```

### 3.3 Chọn (p, d, q)

- `d`: xác định bằng kiểm định **ADF (Augmented Dickey-Fuller)** — nếu chuỗi gốc không dừng (p-value > 0.05), lấy sai phân và kiểm định lại đến khi dừng. Thường d=1 là đủ cho metrics hệ thống.
- `p, q`: xác định qua biểu đồ **ACF (Autocorrelation Function)** và **PACF (Partial Autocorrelation Function)**, hoặc đơn giản hơn — dùng `auto_arima` (thư viện `pmdarima`) để grid-search tự động theo tiêu chí AIC/BIC.

AIC = Akaike Information Criterion — một điểm số để so sánh các model với nhau, cân bằng giữa "khớp dữ liệu tốt" và "đừng quá phức tạp".

Công thức và ý nghĩa

AIC = 2k − 2·ln(L)
- L = likelihood (độ khớp của model với dữ liệu) — càng khớp, −2·ln(L) càng nhỏ.
- k = số tham số của model (số hạng AR + MA + ...) — càng nhiều tham số, 2k càng lớn (phạt).

→ AIC thưởng cho model khớp tốt, nhưng phạt model dùng nhiều tham số. Càng thấp càng tốt. Đây chính là cách chống overfitting: một model nhồi thêm tham số sẽ khớp train tốt hơn (L tăng), nhưng nếu cải thiện không đủ bù phần phạt 2k thì AIC sẽ tăng → báo rằng thêm tham số không đáng.

Điểm quan trọng: AIC chỉ có nghĩa TƯƠNG ĐỐI

Con số AIC tuyệt đối (14679, 52339...) tự nó vô nghĩa — không có "AIC tốt" hay "AIC xấu". Nó chỉ dùng để so sánh các model trên cùng một bộ dữ liệu. Chỉ hiệu số ΔAIC mới mang thông tin:

┌──────────────────────────────┬────────────────────────────────────┐
│ ΔAIC (so với model tốt nhất) │              Ý nghĩa               │
├──────────────────────────────┼────────────────────────────────────┤
│ < 2                          │ Gần như tương đương, khó phân biệt │
├──────────────────────────────┼────────────────────────────────────┤
│ 2 – 10                       │ Model kia kém hơn rõ rệt           │
├──────────────────────────────┼────────────────────────────────────┤
│ > 10                         │ Model kia kém hơn hẳn              │
└──────────────────────────────┴────────────────────────────────────┘

- Với dữ liệu có seasonality rõ (traffic theo giờ/ngày), nên dùng **SARIMA(p,d,q)(P,D,Q,s)** — mở rộng có thêm thành phần seasonal, tránh ARIMA thường hiểu nhầm chu kỳ là trend.

### 3.4 Pseudocode Train + Predict

```
Pseudocode arima_fit_predict(x_train, x_eval, order=(p,d,q), seasonal_order=None):
    model = ARIMA(x_train, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()

    # One-step-ahead forecast trên tập eval (rolling forecast)
    predictions = []
    residuals = []
    for actual_t in x_eval:
        forecast_t, conf_int = fitted_model.forecast(steps=1, return_conf_int=True)
        residual_t = actual_t - forecast_t
        predictions.append(forecast_t)
        residuals.append(residual_t)

        fitted_model = fitted_model.append(actual_t)  # cập nhật model với điểm mới (không refit từ đầu)

    # Chuẩn hoá residual thành anomaly score
    std_resid = std(residuals)
    scores = abs(residuals) / std_resid
    return scores, predictions
```

**Lưu ý hiệu năng quan trọng**: refit ARIMA từ đầu sau mỗi điểm mới là rất tốn kém (O(n) mỗi lần fit). Trong production nên dùng `model.append()` (cập nhật incremental, statsmodels hỗ trợ) hoặc chỉ refit định kỳ (ví dụ mỗi giờ) thay vì mỗi điểm.

### 3.5 Dynamic Thresholding cho Residual (thay vì ngưỡng cố định)

Residual của ARIMA cũng có thể dùng confidence interval của chính forecast để xác định ngưỡng động:

```
upper_bound_t = forecast_t + z_alpha · sqrt(forecast_variance_t)
lower_bound_t = forecast_t - z_alpha · sqrt(forecast_variance_t)
anomaly nếu actual_t nằm ngoài [lower_bound_t, upper_bound_t]
```

`z_alpha` tương ứng mức tin cậy mong muốn (ví dụ 1.96 cho 95%). Cách này tốt hơn ngưỡng MAD cố định vì confidence interval của ARIMA tự nới rộng khi model kém chắc chắn (ví dụ đầu chuỗi, hoặc sau khi vừa có biến động lớn) — giảm false positive trong giai đoạn model chưa "ổn định" lại.

### 3.6 Đánh giá theo PR Curve và Tune Threshold

```
1. Fit ARIMA trên tập train (giai đoạn hệ thống hoạt động bình thường, càng sạch càng tốt)
2. Forecast rolling trên validation → residuals_val, sinh anomaly score
3. precision_recall_curve(y_val, scores_val) → chọn threshold (mục 6.2, doc 1)
   HOẶC chọn z_alpha nếu dùng dynamic thresholding (mục 3.5) — thử lưới 1.65/1.96/2.58
4. Áp lên test set → Precision/Recall/F1 (PW và PA)
```

Cần tune thêm `(p, d, q)` — nên chọn theo AIC/BIC trên tập train trước, sau đó coi như cố định, chỉ tune threshold/z_alpha trên validation để tránh không gian tìm kiếm quá lớn.

### 3.7 Ưu / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Mô hình hoá được autocorrelation — bắt anomaly dạng "phá vỡ pattern" tinh vi hơn z-score | Giả định tuyến tính — kém với pattern phi tuyến phức tạp |
| Confidence interval cho dynamic threshold tự nhiên | Chi phí tính toán cao hơn nhiều khi refit thường xuyên |
| SARIMA xử lý được seasonality như STL | Cần chuỗi tương đối dừng sau differencing — nhạy với non-stationary mạnh |
| Diễn giải được qua hệ số AR/MA | Khó scale cho nhiều time series song song (mỗi series 1 model riêng) |

---