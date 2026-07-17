## 1. Missing values, data gaps, sampling consistency

### 1.1 Phân biệt 2 loại thiếu dữ liệu
Có hai dạng thiếu khác hẳn nhau, cần xử lý khác nhau.

- **Missing value (NaN) trong record có sẵn**: timestamp tồn tại nhưng `value` là null. Thường do lỗi ghi nhận, lỗi parser.
- **Gap trong timeline**: timestamp bị nhảy cóc — ví dụ sampling 1 phút nhưng có đoạn nhảy từ `10:05` sang `10:20`. Đây là dấu hiệu downtime của hệ thống thu thập — bản thân gap có thể chính là một dạng anomaly (collecting failure), không nên âm thầm lấp đầy.

### 1.2 Sampling consistency
- Verify interval giữa các timestamp liên tiếp có đúng là hằng số không, vì trong dataset thực tế có tần suất không hoàn toàn đều.
- Check trùng timestamp (duplicate index) — cần quyết định giữ bản nào hoặc aggregate.
- Nếu trong dataset có nhiều tần suất khác nhau, cần resample về cùng grid trước khi làm cross-metric analysis.

### 1.3 Quyết định xử lý

Tài liệu chung về xử lý missing trong time series khuyên nên cân nhắc theo độ dài gap: gap ngắn (1-2 điểm) dùng linear interpolation hoặc rolling-statistic imputation là hợp lý; gap dài cần phương pháp tôn trọng seasonality (ví dụ seasonal-aware imputation) hoặc đơn giản là loại bỏ đoạn đó khỏi phân tích thống kê thay vì cố lấp. Method đơn giản (mean/forward-fill) rủi ro làm giảm variance giả tạo, ảnh hưởng tới ADF test và STL ở bước sau.

Linear interpolation: là cách lấp giá trị thiếu bằng cách nối thẳng đường giữa điểm biết trước và điểm biết sau, rồi lấy giá trị trên đường thẳng đó tại vị trí cần điền; Nếu biết value tại thời điểm t0 là v0, và tại t1 là v1, thì giá trị tại thời điểm t nằm giữa (t0 < t < t1) được tính bằng v(t) = v0 + (v1 - v0) * (t - t0) / (t1 - t0)

Seasonal-aware imputation: là cách lấp giá trị sao cho vẫn giữ được hình dạng chu kỳ thay vì một đường thẳng như linear interpolation/forward-fill. Có nghĩa là vào đúng thời điểm trong chu kỳ này, series thường có giá trị là bao nhiêu.

**Output cần có**: bảng tổng hợp theo từng KPI — số điểm, % NaN, số gap, gap dài nhất, sampling interval mode.

---

## 2. Stationarity: ADF test + rolling mean/variance

### 2.1 Vì sao quan tâm stationarity

Phần lớn pipeline thống kê cổ điển (ARIMA, một số baseline forecast-based anomaly detector) giả định series stationary. Ngay cả khi dùng deep learning, biết series có stationary hay không vẫn giúp chọn đúng cách feature engineering (cần differencing hay không, window size cho rolling feature).

### 2.2 ADF test và KPSS test
- Null hypothesis: series có unit root (non-stationary). p-value < 0.05 → bác bỏ null → series stationary.
- **Pitfall quan trọng**: p-value > 0.05 không tự động nghĩa là "non-stationary chắc chắn" — đó chỉ là "không đủ bằng chứng bác bỏ null". Nên kết hợp thêm **KPSS test** vì KPSS đảo ngược null hypothesis (null: series stationary). Bốn kịch bản kết hợp ADF + KPSS:
  - ADF bác bỏ null + KPSS không bác bỏ → series stationary (đồng thuận).
  - ADF không bác bỏ + KPSS bác bỏ → series non-stationary (đồng thuận).
  - Cả hai không bác bỏ → không đủ data để kết luận.
  - Hai test mâu thuẫn nhau → thường gợi ý series có **trend xác định (deterministic trend)** chứ không phải random walk — cần detrend trước khi kết luận.
- Trước khi chạy ADF, nên loại bỏ tạm thời các đoạn có gap lớn đã impute thô, vì imputation thô có thể tạo ra artificial stationarity hoặc artificial trend.
- Ghi lại đầy đủ: ADF statistic, p-value, lag được chọn, critical values 1%/5%/10%, và quyết định transform nếu series non-stationary (differencing bậc mấy).

### 2.3 Rolling mean/variance plot
- Vẽ rolling mean và rolling std (window nên chọn theo chu kỳ nghi ngờ — ví dụ nếu data 1 phút/điểm và nghi ngờ seasonality theo ngày, dùng window = 1440).
- Mean trôi dần theo thời gian → trend. Std thay đổi theo thời gian (đặc biệt tăng theo giai đoạn) → heteroscedasticity, dấu hiệu cần log-transform hoặc variance-stabilizing transform trước khi model.
- Với KPI có nhãn anomaly, nên overlay luôn vùng anomaly lên plot rolling mean/std để xem anomaly có trùng với giai đoạn variance bất thường không — đây là tín hiệu sớm cho biết liệu detector dựa trên rolling-zscore đơn giản có khả thi hay không.

---

## 3. Seasonality: STL decomposition + FFT

### 3.1 STL decomposition
- **Bắt buộc set `robust=True`** khi áp dụng cho KPI có anomaly: STL chuẩn (non-robust) bị chính các anomaly làm méo seasonal component (xuất hiện pattern "lởm chởm" giả) và trend bị làm mượt quá mức, mất luôn các đoạn thay đổi đột ngột thật sự đang muốn quan sát.
- Ngay cả robust STL cũng có giới hạn: nếu trong data có **collective anomaly** (một đoạn dài bất thường, không phải spike đơn lẻ), STL có xu hướng "nuốt" đoạn đó vào trend hoặc seasonal component thay vì đẩy ra residual — nghĩa là residual-based anomaly detection sau STL sẽ bỏ sót đúng loại anomaly nguy hiểm nhất. Vì vậy nên luôn visual-check 3 component (trend/seasonal/residual) so với raw series và nhãn anomaly thật, không tin tưởng tuyệt đối residual.
- `period` cần xác định trước chứ không nên đoán mò theo "ngày = 1440 điểm" nếu chưa chắc data có đúng chu kỳ ngày.
- Diễn giải: spike/dip ngắn → thể hiện rõ ở residual lớn bất thường. Thay đổi mức trung bình kéo dài (level shift) → thể hiện ở trend component đổi dốc đột ngột.

### 3.2 FFT để tìm dominant frequency
- Trước khi chạy STL, dùng FFT (`numpy.fft.rfft`) trên series đã loại trend thô (hoặc differencing bậc 1) để tìm tần số có biên độ lớn nhất → suy ra `period` cho STL thay vì đoán.
- Cách làm: tính power spectrum, lấy các đỉnh (peak) có biên độ vượt ngưỡng (ví dụ > mean + 3*std của spectrum), convert tần số → chu kỳ (period = sampling_rate / frequency).
- Lưu ý: nếu series có nhiều seasonality lồng nhau (ví dụ vừa theo giờ vừa theo ngày trong tuần — pattern weekday/weekend khác nhau), FFT đơn sẽ cho nhiều đỉnh — cần xem xét `MSTL` (multiple seasonal-trend decomposition) thay vì STL đơn period nếu việc này xảy ra phổ biến trong dataset.
- Anomaly (đặc biệt point anomaly dày đặc) có thể tạo nhiễu broadband trong FFT, làm các đỉnh seasonal bị "chìm" — nên thử chạy FFT cả trên bản gốc và bản đã loại outlier thô (ví dụ clip theo IQR) để so sánh, không kết luận vội nếu phổ tần không rõ đỉnh.

---

## 4. ACF/PACF trên 5+ key metrics

- Chọn tối thiểu 5 KPI đại diện cho các nhóm hành vi khác nhau quan sát được ở bước 1-3 (ví dụ: 1 KPI rất stationary, 1 KPI có trend rõ, 1 KPI seasonal mạnh, 1 KPI nhiều anomaly dày đặc, 1 KPI sparse/ít data) — không chọn ngẫu nhiên 5 cái đầu tiên vì sẽ không đại diện.
- Diễn giải:
  - ACF decay chậm (tail dài) → dấu hiệu non-stationary / có trend, củng cố thêm cho ADF test.
  - ACF có đỉnh lặp lại tại lag = period → xác nhận seasonality tìm được từ FFT/STL.
  - PACF cắt đột ngột tại lag p → gợi ý bậc AR(p) phù hợp nếu sau này cân nhắc baseline ARIMA-based detector.
- Vẽ ACF/PACF trên **residual sau STL** (không chỉ raw series) cũng hữu ích — nếu residual sau decomposition vẫn còn autocorrelation rõ ràng, nghĩa là decomposition chưa lấy hết structure, cần tăng độ phức tạp model (MSTL, hoặc thêm exogenous regressor).
- Số lag tối đa nên vẽ: tối thiểu 2-3 lần period nghi ngờ để thấy rõ pattern lặp lại, không chỉ vài chục lag mặc định.

---

## 5. Anomaly label distribution

### 5.1 Thống kê cơ bản (per KPI và toàn dataset)
- Count: tổng số điểm label = 1 vs 0.
- Density: tỉ lệ % anomaly trên tổng — đây là con số quyết định chiến lược: dataset thực tế thường có anomaly rate rất thấp (dưới 1-5%), nghĩa là bài toán cực kỳ imbalanced, ảnh hưởng trực tiếp đến chọn metric (không nên dùng accuracy, nên dùng F1/precision-recall, hoặc best-F1-with-point-adjustment).
- Phân bố density theo từng KPI riêng — vì rate có thể chênh lệch rất lớn giữa các KPI (có KPI gần như không có anomaly, có KPI nhiều).

### 5.2 Point vs Collective
Phân loại từng vùng anomaly liên tiếp (segment) theo độ dài:
- **Point/global outlier**: 1 điểm đơn lẻ, deviate mạnh so với cả series.
- **Contextual outlier**: giá trị nằm trong range bình thường của toàn series nhưng bất thường so với context cục bộ (ví dụ window xung quanh).
- **Collective/subsequence anomaly**: chuỗi nhiều điểm liên tiếp cùng được gắn nhãn, từng điểm riêng lẻ có thể không bất thường nhưng cả đoạn tạo thành pattern lạ.

Cách đo: group các label=1 liên tiếp thành segment (`(label != label.shift()).cumsum()` rồi filter label=1), tính độ dài mỗi segment → vẽ histogram độ dài. Nếu phần lớn segment dài = 1 → point-dominant dataset, phù hợp baseline đơn giản (z-score, IQR). Nếu nhiều segment dài → cần detector nhận diện pattern/shape.

### 5.3 Vị trí anomaly tương quan với các thành phần đã phân tích
- Overlay nhãn anomaly lên: rolling mean/std plot, STL residual — xem các anomaly đã biết rơi vào residual lớn hay bị decomposition "nuốt" mất.
- Đây là bước kiểm tra chéo quan trọng: nếu phần lớn anomaly thật KHÔNG hiện rõ ở residual STL, nghĩa là pipeline residual-based sẽ miss nhiều case → cần cân nhắc thêm hướng tiếp cận khác (forecasting-based, hoặc multivariate).

---

## 6. Cross-metric correlation heatmap

- Resample toàn bộ KPI về cùng time grid trước khi tính correlation, nếu không sẽ bị lệch alignment.
- Tính Pearson correlation matrix giữa các KPI ở mức giá trị gốc, **và** ở mức "anomaly indicator" (binary label) để xem liệu nhiều KPI có thường xuyên bất thường cùng lúc không (co-occurrence của anomaly, gợi ý root cause chung hoặc dependency giữa services).
- Pearson chỉ bắt linear relationship — nên bổ sung thêm Spearman (rank correlation) cho các cặp có khả năng phi tuyến, vì nhiều cặp metric vận hành (vd CPU vs latency) không nhất thiết tuyến tính.
- Với data nhiều KPI (vài chục), nên cluster heatmap (hierarchical clustering trên ma trận correlation) thay vì nhìn ma trận thô, để nhóm các KPI có hành vi tương tự — hữu ích sau này nếu định làm multivariate anomaly detection theo nhóm thay vì univariate riêng lẻ từng KPI.
- Lưu ý: correlation tính trên toàn bộ chuỗi có thể bị che lấp nếu quan hệ giữa hai metric chỉ xuất hiện trong giai đoạn anomaly (correlation tăng đột biến lúc incident) — có thể bổ sung rolling correlation theo thời gian để bắt hiện tượng này, hữu ích cho hướng root-cause-analysis sau này.

---