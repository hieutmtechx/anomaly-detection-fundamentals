## 1. Vì sao không thể dùng Accuracy

Trong anomaly detection, tỉ lệ anomaly thường rất thấp (1-5% tổng số điểm dữ liệu). Một model dự đoán "tất cả đều bình thường" có thể đạt accuracy 95-99% nhưng vô dụng hoàn toàn — nó không phát hiện được anomaly nào (Recall = 0).

Vì vậy bộ metrics chuẩn cho bài toán này là **Precision, Recall, F1** thay vì Accuracy, vì chúng phản ánh đúng trade-off giữa "bắt được bao nhiêu anomaly thật" và "báo động giả bao nhiêu lần".

## 2. Định nghĩa Confusion Matrix cho Anomaly Detection

| | Predicted Anomaly | Predicted Normal |
|---|---|---|
| **Actual Anomaly** | True Positive (TP) | False Negative (FN) |
| **Actual Normal** | False Positive (FP) | True Negative (TN) |

Trong bối cảnh giám sát hệ thống:
- **TP**: model báo động đúng lúc CPU/latency thực sự bất thường (ví dụ: sự cố thật, outage thật).
- **FP**: model báo động nhầm khi hệ thống vẫn hoạt động bình thường — gây alert fatigue, người vận hành dần bỏ qua cảnh báo.
- **FN**: model bỏ sót một sự cố thật — hậu quả nghiêm trọng nhất vì không ai được cảnh báo.
- **TN**: model đúng khi im lặng lúc hệ thống bình thường.

## 3. Precision, Recall, F1

### 3.1 Công thức

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · (Precision · Recall) / (Precision + Recall)
```

- **Precision** trả lời: "Trong số các điểm tôi báo là anomaly, bao nhiêu % là đúng?" → Precision thấp = nhiều false alarm, on-call engineer mệt mỏi vì bị gọi nhầm.
- **Recall** trả lời: "Trong số các anomaly thật sự tồn tại, tôi bắt được bao nhiêu %?" → Recall thấp = bỏ sót sự cố, rủi ro vận hành cao.
- **F1** là trung bình điều hòa (harmonic mean) của hai đại lượng trên, dùng khi cần một con số duy nhất cân bằng cả hai.

### 3.2 F-beta: khi Precision và Recall không quan trọng ngang nhau

```
F_beta = (1 + beta^2) · (Precision · Recall) / (beta^2 · Precision + Recall)
```

- `beta > 1` (ví dụ F2): ưu tiên Recall — phù hợp hệ thống critical (payment, core infra) nơi bỏ sót sự cố tốn kém hơn nhiều so với báo động giả.
- `beta < 1` (ví dụ F0.5): ưu tiên Precision — phù hợp hệ thống có đội ngũ on-call nhỏ, không chịu được nhiều false alarm.

Lựa chọn beta là một quyết định nghiệp vụ, không phải thuần kỹ thuật — cần thống nhất với team vận hành trước khi tối ưu.

## 4. Bài toán đặc thù của Time Series: Point-wise vs Point-Adjust vs Range-based

Đây là phần khác biệt quan trọng nhất giữa đánh giá anomaly detection cho time series so với classification thông thường (tabular, ảnh...). Ground truth anomaly trong time series thường là **một đoạn (segment)** liên tục chứ không phải một điểm rời rạc (ví dụ: sự cố CPU spike kéo dài từ phút 10 đến phút 25).

### 4.1 Point-wise (PW) — cách tính cơ bản nhất

Tính TP/FP/FN trực tiếp theo từng điểm thời gian, so khớp 1-1 giữa nhãn dự đoán và ground truth tại từng timestamp.

- Ưu điểm: đơn giản, không thiên vị.
- Nhược điểm: model chỉ phát hiện được 1-2 điểm trong một segment anomaly dài 50 điểm vẫn có Recall thấp dù về mặt vận hành, việc bắt được điểm đầu tiên của segment đã đủ để cảnh báo người vận hành.

### 4.2 Point-Adjust (PA)

Nguyên tắc: nếu model phát hiện **ít nhất một điểm** trong một ground-truth anomaly segment, toàn bộ segment đó được tính là TP (kể cả các điểm model thực sự bỏ sót trong segment). Các điểm dự đoán dương nằm ngoài mọi segment ground truth vẫn tính là FP bình thường.

```
Pseudocode point_adjust(y_true, y_pred):
    for mỗi segment liên tục các điểm anomaly trong y_true:
        if tồn tại ít nhất 1 điểm trong segment mà y_pred = 1:
            đặt y_pred = 1 cho TOÀN BỘ segment đó
    return y_pred đã điều chỉnh
```

- Ưu điểm: phản ánh đúng hơn cách vận hành thực tế — chỉ cần 1 cảnh báo trong cửa sổ sự cố là đủ để con người phản ứng.
- **Nhược điểm nghiêm trọng**: PA có xu hướng **overestimate** hiệu năng nghiêm trọng. Một model dự đoán ngẫu nhiên, hoặc một model chỉ cần trúng 1 điểm/segment, có thể đạt F1 rất cao sau khi point-adjust dù bản chất không "hiểu" anomaly.
- **Thực hành**: luôn báo cáo **cả hai** — số liệu PW và số liệu PA — và ghi rõ đang dùng protocol nào khi so sánh giữa các mô hình. Không nên chỉ dùng PA một mình để tránh ảo tưởng về hiệu năng.

### 4.3 Range-based Precision/Recall

Mở rộng khái niệm Precision/Recall sang các range thay vì điểm rời rạc, có tham số hoá mức độ thưởng cho:
- **Existence**: phát hiện được sự tồn tại của anomaly (dù chỉ 1 điểm).
- **Overlap size**: phần trăm overlap giữa predicted range và true range.
- **Cardinality**: phạt nếu một true segment bị "vỡ" thành nhiều predicted segment rời rạc (gây nhiều alert trùng lặp cho 1 sự cố).
- **Position bias**: có thể ưu tiên phát hiện sớm (đầu segment) hơn phát hiện muộn — quan trọng với giám sát hệ thống vì phát hiện sớm = giảm thời gian downtime (MTTD).

Đây là lựa chọn phù hợp nhất về mặt học thuật cho time series, nhưng phức tạp hơn để triển khai. Trong phạm vi thực hành, có thể bắt đầu bằng PW + PA, và nâng cấp lên range-based khi cần so sánh sâu hơn.

### 4.4 Áp dụng

1. Luôn tính **PW metrics** làm baseline tham chiếu (không bias).
2. Tính thêm **PA metrics** để phản ánh giá trị vận hành thực tế (1 cảnh báo/segment là đủ).
3. Khi so sánh nhiều mô hình ở doc benchmark, trình bày cả hai bộ số, không chỉ chọn bộ nào "đẹp" hơn.
4. Cân nhắc thêm metric vận hành: **Time-to-Detect (TTD)** = độ trễ từ lúc anomaly segment bắt đầu đến lúc model phát hiện điểm đầu tiên — quan trọng ngang Recall trong hệ thống production.

## 5. PR Curve (Precision-Recall Curve)

### 5.1 Vì sao dùng PR Curve thay vì ROC Curve

Hầu hết các model trong phạm vi này (Z-score, STL residual, ARIMA residual, Isolation Forest, RCF) đều xuất ra một **anomaly score liên tục**, sau đó áp threshold để ra nhãn nhị phân. PR Curve cho thấy toàn bộ trade-off Precision/Recall khi threshold thay đổi, không cố định ở một điểm.

ROC Curve (TPR vs FPR) bị "lạc quan giả" khi class imbalance cao (anomaly hiếm) vì FPR = FP/(FP+TN) — mẫu số TN rất lớn nên FPR luôn trông nhỏ, làm ROC curve trông đẹp dù Precision thực tế tệ. PR Curve nhạy hơn nhiều với imbalance vì Precision có FP ở tử số trực tiếp so với TP, không bị pha loãng bởi TN khổng lồ.

→ **Trong anomaly detection (luôn imbalance), PR Curve là lựa chọn chuẩn, ROC chỉ nên dùng bổ sung.**

### 5.2 Cách dựng PR Curve

```
Pseudocode build_pr_curve(y_true, scores):
    thresholds = sorted(unique(scores), giảm dần)
    results = []
    for t in thresholds:
        y_pred = (scores >= t) → 1, ngược lại → 0
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        results.append((r, p, t))
    return results  # vẽ Recall (trục x) vs Precision (trục y)
```

Mỗi điểm trên đường cong tương ứng với một threshold khác nhau.

### 5.3 Average Precision (AP)

```
AP = sum over k của (R_k - R_{k-1}) · P_k
```

Là diện tích xấp xỉ dưới PR Curve, dùng để so sánh tổng thể giữa các mô hình mà không cần chọn một threshold cụ thể — rất hữu ích ở bước benchmark vì cho phép so sánh các model có anomaly score ở thang đo khác nhau.

## 6. Threshold Tuning

### 6.1 Vì sao cần tune threshold

Mọi mô hình trong scope này đều trả về **score liên tục** (z-score, residual, anomaly score của Isolation Forest/RCF...), không tự sinh ra nhãn nhị phân. Threshold là tham số quyết định ranh giới "bao nhiêu thì coi là bất thường" — ảnh hưởng trực tiếp đến Precision/Recall, và **không có giá trị threshold nào đúng tuyệt đối**, chỉ có giá trị phù hợp với mục tiêu nghiệp vụ.

### 6.2 Chiến lược 1: Best F1 trên PR Curve

```
Pseudocode tune_threshold_best_f1(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + eps)
    best_idx = argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]
```

Đây là cách phổ biến nhất để so sánh các mô hình "công bằng" (mỗi model tự chọn threshold tối ưu cho chính nó).

**Lưu ý quan trọng**: best-F1 trên tập test/validation có rủi ro overfit threshold vào chính tập đó. Nên tune threshold trên **validation set riêng**, rồi đánh giá final metrics trên **test set** chưa từng dùng để chọn threshold — tương tự nguyên tắc train/val/test trong ML thông thường.

### 6.3 Chiến lược 2: Threshold theo target Precision hoặc target Recall

Khi nghiệp vụ có ràng buộc cứng (ví dụ: "đội on-call chỉ chấp nhận tối đa 10% false alarm" → cần Precision ≥ 0.9):

```
Pseudocode tune_threshold_for_target_precision(y_true, scores, target_precision):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    candidates = [(t, r) for p, r, t in zip(...) if p >= target_precision]
    return candidate có recall cao nhất trong số candidates
```

### 6.4 Chiến lược 3: Threshold thống kê (không cần label) — dùng khi thiếu ground truth

Các mô hình thống kê (Z-score, STL residual) thường dùng ngưỡng dựa trên phân phối thay vì tối ưu trên label, hữu ích khi dữ liệu chưa có nhãn:
- `mean ± k·std` (k thường 2-3 cho phân phối gần chuẩn)
- `median ± k·MAD` (Median Absolute Deviation — robust hơn với chính các anomaly làm méo mean/std)
- Percentile-based (ví dụ top 1% score cao nhất bị coi là anomaly, dựa trên tỉ lệ contamination giả định)

Khi có label, nên dùng các threshold thống kê này làm **điểm khởi tạo**, sau đó fine-tune bằng PR Curve (mục 6.2/6.3) để đạt kết quả tối ưu thực sự theo ground truth.

### 6.5 Vấn đề Threshold trôi theo thời gian (Concept Drift)

Hệ thống production có pattern thay đổi theo thời gian (traffic tăng dần, seasonality thay đổi theo mùa kinh doanh...). Một threshold tối ưu tại thời điểm train có thể không còn phù hợp sau vài tuần/tháng. Cần:
- Định kỳ re-tune threshold trên dữ liệu gần nhất (rolling re-calibration).
- Hoặc dùng threshold tương đối (theo độ lệch chuẩn động — rolling std) thay vì giá trị tuyệt đối cố định, để threshold tự thích nghi.

## 7. Quy trình

```
1. Train/fit model trên tập train (chỉ dữ liệu normal, hoặc toàn bộ tuỳ model)
2. Sinh anomaly score liên tục trên tập validation
3. Dựng PR Curve trên validation set, tính AP
4. Tune threshold bằng best-F1 (mục 6.2) hoặc target-precision (mục 6.3)
5. Áp threshold đã chọn lên tập test (chưa từng thấy)
6. Báo cáo Precision / Recall / F1 trên test — cả bản Point-wise và Point-Adjust
7. Ghi nhận Time-to-Detect trung bình nếu có thể
```
