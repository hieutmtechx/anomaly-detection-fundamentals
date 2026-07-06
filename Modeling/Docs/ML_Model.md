### 0.1 Feature engineering từ time series sang vector

Hai cách phổ biến:

**(a) Sliding window / Shingling**: ghép `s` điểm liên tiếp thành 1 vector đặc trưng, giúp model nhìn thấy được pattern cục bộ thay vì chỉ 1 giá trị rời rạc.

```
shingle_t = [x_{t-s+1}, x_{t-s+2}, ..., x_t]   # vector s chiều
```

Ví dụ s=10: mỗi điểm dữ liệu đưa vào model là 1 cửa sổ 10 giá trị gần nhất, giúp phát hiện cả anomaly dạng "pattern lạ" (không chỉ giá trị tức thời lạ) — ví dụ CPU dao động bất thường dù không vượt ngưỡng tuyệt đối nào.

**(b) Feature thủ công (hand-crafted)**: tính các đặc trưng thống kê trên cửa sổ trượt làm input, ví dụ:

```
features_t = [
    x_t,
    rolling_mean(x, w),
    rolling_std(x, w),
    x_t - rolling_mean(x, w),        # độ lệch so với baseline
    rolling_mean(x, w) - rolling_mean(x, 2w),  # độ dốc trend ngắn hạn
    hour_of_day(t), day_of_week(t)   # đặc trưng thời gian theo chu kỳ (có thể encode sin/cos)
]
```

→ Cách (a) đơn giản, ít giả định, là cách chuẩn của RCF gốc. Cách (b) cho phép đưa domain knowledge vào nhưng cần thiết kế cẩn thận.

---

## 1. Isolation Forest

### 1.1 Ý tưởng cốt lõi

Isolation Forest dựa trên quan sát: **anomaly là điểm "ít và khác"**, nên dễ bị cô lập (isolate) hơn điểm bình thường khi phân vùng dữ liệu ngẫu nhiên. Thuật toán xây nhiều cây nhị phân (isolation tree), mỗi cây chia dữ liệu bằng cách chọn ngẫu nhiên 1 feature và 1 giá trị cắt ngẫu nhiên trong khoảng [min, max] của feature đó, lặp lại đệ quy đến khi mỗi điểm bị cô lập vào 1 leaf riêng.

**Trực giác**: điểm anomaly nằm xa cụm dữ liệu chính → chỉ cần vài lần cắt ngẫu nhiên là đã bị tách riêng (path length ngắn). Điểm bình thường nằm trong vùng dày đặc → cần rất nhiều lần cắt mới tách được (path length dài).

### 1.2 Công thức Anomaly Score

```
s(x, n) = 2^( -E[h(x)] / c(n) )

Trong đó:
    h(x)    = path length của điểm x trong 1 cây (số lần cắt để cô lập x)
    E[h(x)] = trung bình h(x) trên toàn bộ T cây trong forest
    c(n)    = path length trung bình kỳ vọng của 1 tìm kiếm không thành công
              trong Binary Search Tree với n điểm — dùng để chuẩn hoá theo cỡ mẫu:

    c(n) = 2·H(n-1) - 2(n-1)/n
    H(i) = ln(i) + 0.5772156649   (xấp xỉ harmonic number qua hằng số Euler-Mascheroni)
```

**Diễn giải giá trị score**:
- `s → 1`: path length rất ngắn (E[h(x)] → 0) → **anomaly rõ ràng**.
- `s → 0.5`: path length xấp xỉ c(n), tức bình thường như một điểm ngẫu nhiên trong BST → **không có dấu hiệu bất thường**.
- `s → 0` (hiếm xảy ra, khi E[h(x)] → n-1): điểm nằm rất sâu trong vùng dày đặc → **rất bình thường**.

→ Ngưỡng mặc định kinh điển trong paper gốc là 0.5, nhưng trong thực hành **không nên dùng cứng 0.5** — phải tune theo PR Curve trên dữ liệu thực tế (mục 1.6) vì phân phối score thực tế phụ thuộc nhiều vào đặc tính dữ liệu.

### 1.3 Tham số quan trọng

| Tham số | Ý nghĩa | Khuyến nghị |
|---|---|---|
| `n_estimators` (T) | Số cây trong forest | 100 (mặc định) thường đủ, tăng thêm cải thiện ít |
| `max_samples` (ψ) | Số điểm subsample để train mỗi cây | không nên dùng toàn bộ dataset, subsample nhỏ giúp cây "nông" hơn, tăng độ tương phản path length giữa anomaly/normal |
| `contamination` | Tỉ lệ anomaly giả định, dùng để suy ra threshold mặc định của thư viện | Đặt = tỉ lệ anomaly thực tế ước lượng nếu biết, hoặc để `'auto'` rồi tự tune lại bằng PR Curve |
| `max_features` | Số feature xét khi chọn split | 1.0 (toàn bộ) cho shingle vector, có thể giảm nếu vector dài để tăng đa dạng cây |

### 1.4 Pseudocode Train + Predict

```
Pseudocode isolation_forest_fit_predict(x_series, shingle_size=10, n_estimators=100, max_samples=256):
    # Bước 1: chuyển time series thành ma trận shingle
    X = build_shingles(x_series, shingle_size)   # shape: (n_samples, shingle_size)

    # Bước 2: train (chỉ cần dữ liệu "đa số bình thường", không cần label)
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination='auto'
    )
    model.fit(X_train)

    # Bước 3: predict — sklearn trả score càng THẤP càng bất thường (ngược dấu so với công thức gốc)
    raw_scores = model.decision_function(X_eval)   # cao = bình thường trong sklearn
    anomaly_scores = -raw_scores                   # đảo dấu để "cao = bất thường", nhất quán với các model khác

    return anomaly_scores
```

**Lưu ý**: thư viện `sklearn.ensemble.IsolationForest` trả `decision_function` theo chiều "cao = bình thường". Cần đảo dấu khi ghép chung pipeline đánh giá với các model khác để đảm bảo "score cao = nghi ngờ anomaly" thống nhất trên toàn bộ benchmark.

### 1.5 Vấn đề riêng khi áp dụng cho Time Series

- **Mất thông tin thời gian giữa các shingle**: Isolation Forest coi mỗi shingle là một vector độc lập, không biết shingle nào xảy ra trước/sau — phù hợp phát hiện pattern bất thường cục bộ nhưng không nắm bắt xu hướng dài hạn (trend) như ARIMA.
- **Train trên dữ liệu "bẩn"**: nếu tập train chứa sẵn anomaly, các điểm đó có thể vô tình làm path length trung bình của vùng bình thường bị lệch nhẹ. Nên cố gắng dùng giai đoạn dữ liệu sạch nhất có thể để train, tương tự lưu ý ở ARIMA/STL.
- **Subsample (max_samples) nhỏ là chủ đích, không phải hạn chế**: trái với trực giác ML thông thường ("train trên nhiều data hơn luôn tốt hơn"), Isolation Forest cố tình dùng subsample nhỏ (256) để giữ cây nông — nếu dùng toàn bộ dữ liệu lớn, path length giữa anomaly và normal sẽ co lại gần nhau hơn (swamping effect), giảm khả năng phân biệt.

### 1.6 Đánh giá theo PR Curve và Tune Threshold

```
1. Build shingle matrix từ time series
2. Train IsolationForest trên tập train
3. anomaly_scores_val = -model.decision_function(X_val)
4. precision_recall_curve(y_val, anomaly_scores_val) → chọn threshold theo best-F1
5. Áp threshold lên test set → Precision/Recall/F1 (PW và PA)
```

**Hyperparameter cần tune thêm ngoài threshold**: `shingle_size`, `max_samples`, `contamination`. Nên grid-search trên validation theo AP (Average Precision) trước khi cố định threshold cuối cùng — đặc biệt `shingle_size` ảnh hưởng lớn đến việc model bắt được anomaly dạng "giá trị đơn lẻ bất thường" (shingle nhỏ) hay "pattern dao động bất thường" (shingle lớn hơn).

### 1.7 Ưu / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Không cần giả định phân phối dữ liệu | Mất thông tin thứ tự thời gian (mỗi shingle độc lập) |
| Train nhanh, scale tốt với dữ liệu lớn (O(n log n)) | Cần feature engineering (shingling) thủ công cho time series |
| Không cần label (unsupervised) | Kém với anomaly "tinh vi" nằm gần ranh giới phân phối bình thường (path length không tách biệt rõ) |
| Phát hiện tốt anomaly đa biến nếu ghép nhiều metrics cùng lúc | Batch — cần refit định kỳ để thích nghi với baseline mới, không streaming tự nhiên |

---

## 2. Random Cut Forest (RCF)

### 2.1 Ý tưởng cốt lõi

Random Cut Forest là họ thuật toán cùng gốc rễ với Isolation Forest (random space partitioning để cô lập điểm), nhưng được thiết kế chuyên biệt cho **dữ liệu streaming** với 2 khác biệt quan trọng:

1. **Cơ chế chọn điểm cắt theo tỉ lệ không gian (probability proportional to range)**: thay vì chọn feature ngẫu nhiên đều như Isolation Forest, RCF chọn feature để cắt với xác suất tỉ lệ thuận với độ trải rộng (range) của feature đó — feature có range lớn hơn có xác suất bị chọn cắt cao hơn. Điều này giúp RCF nhạy hơn với outlier theo nhiều chiều cùng lúc (multivariate).
2. **Cập nhật động (streaming-native) với cơ chế reservoir sampling**: cây của RCF có thể thêm/xoá điểm theo thời gian thực mà không cần xây lại từ đầu — khi cây đầy, điểm cũ bị loại bỏ ngẫu nhiên có trọng số để nhường chỗ cho điểm mới, giúp model "quên" dần dữ liệu cũ và thích nghi với concept drift tự nhiên hơn Isolation Forest (vốn là batch thuần).

### 2.2 Công thức: CoDisp (Collusive Displacement)

Thay vì dùng path length trực tiếp như Isolation Forest, RCF dùng độ đo **CoDisp** — đo lường mức độ **cấu trúc cây thay đổi bao nhiêu** khi chèn (hoặc xoá) một điểm:

```
CoDisp(x) = E[ Displacement gây ra khi chèn x vào cây, tính trung bình trên mọi cách x có thể "kéo theo" 1 nhóm điểm khác ]
```

Diễn giải trực giác:
- Nếu chèn x vào cây mà cấu trúc cây gần như không đổi (x "hoà nhập" tự nhiên vào vùng dữ liệu hiện có) → CoDisp thấp → bình thường.
- Nếu chèn x làm thay đổi đáng kể cấu trúc cây (x buộc phải tách riêng một vùng mới, hoặc "kéo theo" nhiều điểm lân cận bị dịch chuyển vị trí trong cây) → CoDisp cao → bất thường.

Điểm khác biệt cốt lõi so với Isolation Forest's path length: CoDisp đo *ảnh hưởng của 1 điểm lên toàn bộ cấu trúc*, trong khi path length chỉ đo *độ sâu của riêng điểm đó*. Điều này giúp RCF phát hiện tốt hơn các **anomaly theo nhóm (collusive anomalies)** — ví dụ một cụm nhỏ vài điểm bất thường xuất hiện gần nhau, điều mà Isolation Forest path length đơn thuần có thể bỏ sót nếu cụm đó đủ lớn để "tự tạo vùng riêng".

```
Score cuối cùng = trung bình CoDisp(x) trên toàn bộ forest (giống nguyên lý ensemble như Isolation Forest)
```

### 2.3 Shingling — cơ chế bắt buộc của RCF cho Time Series

RCF nguyên bản được thiết kế để nhận **shingle** (cửa sổ trượt các điểm liên tiếp) làm input chuẩn, không phải optional như gợi ý ở mục 0 — đây là cách RCF "nhìn thấy" được tính tuần tự thời gian dù bản thân cây không có khái niệm thời gian. Tham số `shingle_size` ảnh hưởng trực tiếp đến loại anomaly RCF phát hiện được:

```
shingle_size nhỏ (ví dụ 1-4): nhạy với spike/dip giá trị tức thời
shingle_size lớn (ví dụ 20-50): nhạy với thay đổi pattern/hình dạng dao động,
                                  kém nhạy hơn với spike đơn lẻ ngắn
```

### 2.4 Pseudocode Train + Predict (streaming, đúng tinh thần RCF)

```
Pseudocode rcf_fit_predict_streaming(x_series, num_trees=100, tree_size=256, shingle_size=8):
    forest = RandomCutForest(num_trees=num_trees, tree_size=tree_size, shingle_size=shingle_size)
    scores = []

    for x_t in x_series:
        forest.update([x_t])   # chèn điểm mới, tự động loại điểm cũ nếu cây đầy (reservoir sampling)

        if forest.n_samples >= shingle_size:   # cần đủ điểm để tạo 1 shingle hoàn chỉnh
            score_t = forest.codisp()          # CoDisp của shingle hiện tại
        else:
            score_t = 0   # chưa đủ dữ liệu để đánh giá

        scores.append(score_t)

    return scores   # anomaly score liên tục theo thời gian, cập nhật online
```

**Lưu ý quan trọng về thứ tự update/score**: trong cách triển khai streaming chuẩn, `forest.update(x_t)` nên được gọi **trước** khi tính score nếu muốn đánh giá "điểm này khác thường thế nào so với phần còn lại của forest sau khi đã có mặt nó" (self-inclusive); một số implementation tính score **trước** khi update để tránh chính điểm đó ảnh hưởng đến model đang đánh giá nó (tránh self-masking, tương tự nguyên tắc ở mục 1.2 doc 2). Cần thống nhất rõ ràng cách nào đang dùng khi báo cáo kết quả vì ảnh hưởng đáng kể đến độ nhạy.

### 2.5 Đánh giá theo PR Curve và Tune Threshold

```
1. Build shingle theo shingle_size đã chọn (qua streaming update hoặc batch tương đương)
2. Sinh CoDisp scores trên tập validation
3. precision_recall_curve(y_val, codisp_scores_val) → chọn threshold theo best-F1
4. Áp threshold lên test set → Precision/Recall/F1 (PW và PA)
```

**Hyperparameter cần tune thêm ngoài threshold**: `num_trees`, `tree_size`, `shingle_size`. Vì RCF là streaming, nên đặc biệt chú ý đánh giá theo **rolling/walk-forward validation** (chia nhiều cửa sổ thời gian liên tiếp, train trên đoạn trước, test trên đoạn sau, lặp lại trượt dần) thay vì chia train/test tĩnh một lần — phản ánh đúng hơn cách RCF vận hành thực tế trong production.

### 2.7 Ưu / Nhược điểm

| Ưu điểm | Nhược điểm |
|---|---|
| Native streaming — không cần refit định kỳ như Isolation Forest/ARIMA | Phức tạp hơn để triển khai từ đầu, ít thư viện hỗ trợ tốt như sklearn |
| Tự thích nghi concept drift qua cơ chế reservoir sampling | Khó diễn giải hơn Isolation Forest (CoDisp trừu tượng hơn path length) |
| CoDisp bắt được cả anomaly đơn lẻ lẫn anomaly theo cụm nhỏ (collusive) | Cần hiểu rõ cơ chế update/score để tránh self-masking khi triển khai |
| Phù hợp tự nhiên cho giám sát hệ thống real-time, nhiều metric liên tục đổ về | Tuning nhiều tham số hơn (num_trees, tree_size, shingle_size) so với Isolation Forest |

---