# Time Series Data

## 1. Định Nghĩa

Time series là dạng dữ liệu mà mỗi quan sát được ghi nhận tại một thời điểm cụ thể. Khác với dữ liệu dạng bảng, thứ tự xuất hiện của các quan sát trong time series mang ý nghĩa quan trọng và không thể thay đổi một cách tùy ý.

**Các ứng dụng phổ biến:**

- Giám sát hệ thống
- Dữ liệu tài chính
- Dữ liệu cảm biến IoT
- Dự báo nhu cầu sản phẩm
- Phát hiện bất thường

---

## 2. Cấu Trúc của Dữ Liệu Time Series

Một chuỗi thời gian thường bao gồm:

| Thành phần | Mô tả |
|------------|-------|
| **Timestamp** | Thời điểm ghi nhận dữ liệu. |
| **Value** | Giá trị được quan sát tại thời điểm đó. |

Tùy vào số lượng biến được theo dõi, time series được chia thành hai loại:

| Loại | Mô tả |
|------|-------|
| **Univariate time series** | Chỉ theo dõi một biến theo thời gian. |
| **Multivariate time series** | Theo dõi nhiều biến theo thời gian. |

---

## 3. Các Đặc Điểm Cơ Bản của Time Series

### Temporal Dependency
Giá trị hiện tại thường phụ thuộc vào các giá trị trong quá khứ — các quan sát trong time series không độc lập với nhau.

### Trend
Xu hướng tăng hoặc giảm của dữ liệu trong thời gian dài, phản ánh sự thay đổi dài hạn của hệ thống.

### Seasonality
Các mẫu lặp lại theo chu kỳ cố định. Các chu kỳ thường gặp: theo giờ, ngày, tuần, tháng, năm,...

### Noise
Các biến động ngẫu nhiên không mang nhiều ý nghĩa thực tế, làm giảm độ chính xác của mô hình.

### Autocorrelation
Thể hiện mức độ tương quan giữa các giá trị trong chuỗi tại các thời điểm khác nhau.

- Hai giá trị có mối quan hệ mạnh → autocorrelation cao.
- Hai giá trị không liên quan → autocorrelation thấp.

Điều này giúp xác định dữ liệu quá khứ ảnh hưởng tới hiện tại như thế nào.

### Stationarity
Tính chất cho biết các đặc trưng thống kê của chuỗi không thay đổi theo thời gian. Các đặc trưng cần ổn định:

- Mean
- Variance
- Covariance

---

## 4. Các Vấn Đề Thường Gặp trong Dữ Liệu Time Series

| Vấn đề | Mô tả |
|--------|-------|
| **Missing values** | Dữ liệu bị mất tại một số thời điểm. |
| **Outliers** | Các điểm dữ liệu bất thường so với phần lớn dữ liệu còn lại. |
| **Concept drift** | Phân phối dữ liệu thay đổi theo thời gian. |
| **Imbalanced anomalies** | Làm cho bài toán phát hiện bất thường trở nên khó khăn. |

---

## 5. Time Series vs Tabular Data

| Đặc điểm | Mô tả |
|----------|-------|
| **Phụ thuộc thời gian** | Các quan sát không độc lập; mô hình cần học được các quy luật dài hạn và chu kỳ lặp lại. |
| **Không thể shuffle** | Việc chia train/test phải tuân thủ thứ tự thời gian. |
| **Concept drift** | Dữ liệu thay đổi theo thời gian khiến mô hình nhanh chóng bị lỗi thời. |
| **Nhiều bước tiền xử lý** | Dữ liệu thường chứa missing value và noise, đòi hỏi xử lý nhiều hơn dữ liệu tabular. |