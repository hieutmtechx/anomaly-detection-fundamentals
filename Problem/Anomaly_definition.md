# Anomaly Detection trong Hệ Thống

## 1. Khái niệm Anomaly

Anomaly hay outlier là những điểm dữ liệu hoặc hành vi của hệ thống có sự khác biệt đáng kể so với trạng thái hoạt động bình thường.

Đối với dữ liệu time series, anomaly có thể xuất hiện dưới nhiều dạng:

- Giá trị tăng hoặc giảm đột ngột.
- Hành vi không phù hợp với xu hướng hiện tại.
- Hành vi không tuân theo tính chu kỳ của dữ liệu.
- Sự thay đổi bất thường trong mẫu hoạt động của hệ thống.

Anomaly thường chia thành ba nhóm:

| Loại | Mô tả |
|------|-------|
| **Point anomaly** | Một điểm dữ liệu đơn lẻ có giá trị khác biệt đáng kể so với các điểm còn lại. |
| **Contextual anomaly** | Một giá trị bình thường trong ngữ cảnh này nhưng bất thường trong ngữ cảnh khác. |
| **Collective anomaly** | Một nhóm điểm dữ liệu liên tiếp tạo thành hành vi bất thường, dù từng điểm riêng lẻ có thể hoàn toàn bình thường. |

---

## 2. Cách Xác Định Anomaly

### Dựa trên ngưỡng
Định nghĩa trước các ngưỡng cho từng chỉ số. Khi giá trị vượt quá ngưỡng, hệ thống sẽ cảnh báo.

- **Ưu điểm:** Dễ triển khai và giải thích.
- **Nhược điểm:** Khó thích ứng với các hệ thống có thay đổi liên tục.

### Dựa trên thống kê
Giả định dữ liệu tuân theo một phân phối nhất định. Các điểm dữ liệu nằm quá xa giá trị trung bình sẽ được xem là anomaly.

### Dựa trên dự báo
Hệ thống học hành vi bình thường của chuỗi thời gian và dự đoán giá trị tương lai. Nếu sai số giữa giá trị thực tế và giá trị dự đoán vượt quá ngưỡng cho phép, hệ thống coi đó là anomaly.

### Dựa trên hành vi hoặc mẫu dữ liệu
Anomaly không nhất thiết là giá trị quá lớn hoặc quá nhỏ, mà có thể là một hành vi khác thường so với lịch sử.

---

## 3. Anomaly ở Mức Metric

Metric-level anomaly là các bất thường xuất hiện trên từng chỉ số riêng lẻ.

### 3.1. CPU Utilization
CPU tăng đột ngột từ 45% lên 95%.

Nguyên nhân có thể bao gồm:
- Tải hệ thống tăng cao.
- Tiến trình chạy bất thường.
- Vòng lặp vô hạn trong ứng dụng.

### 3.2. Memory Usage
Bộ nhớ tăng liên tục và không được giải phóng.

### 3.3. Latency
Thời gian phản hồi tăng từ vài trăm mili giây lên vài giây — thường cho thấy hệ thống đang bị nghẽn hoặc quá tải.

### 3.4. Error Rate
Tỷ lệ lỗi tăng bất thường.

---

## 4. Anomaly ở Mức System

System-level anomaly xảy ra khi nhiều chỉ số cùng biểu hiện bất thường trong cùng một khoảng thời gian.

Ví dụ điển hình — các dấu hiệu xuất hiện đồng thời:

- CPU tăng cao.
- Memory tăng cao.
- Latency tăng mạnh.
- Error rate tăng đột biến.

Tập hợp các dấu hiệu này cho thấy hệ thống đang gặp sự cố nghiêm trọng thay vì chỉ một chỉ số riêng lẻ bất thường.

**Ví dụ:** Một cơ sở dữ liệu bị nghẽn có thể dẫn tới:

- CPU tăng do phải xử lý nhiều truy vấn chờ.
- Memory tăng do hàng đợi nhiều.
- Latency tăng do phản hồi chậm.
- Error rate tăng do timeout.

Trong trường hợp này, anomaly không còn là vấn đề của một metric đơn lẻ mà đã trở thành sự cố ở cấp độ toàn hệ thống.

---

## 5. Hậu Quả Khi Bỏ Sót Anomaly (False Negative)

False negative xảy ra khi hệ thống thực sự có anomaly nhưng mô hình hoặc hệ thống giám sát không phát hiện được.

**Ví dụ:** Một cơ sở dữ liệu đang quá tải — CPU tăng lên 98%, latency tăng liên tục — nhưng hệ thống không phát hiện và không sinh cảnh báo.

**Hậu quả:**

- Sự cố không được xử lý kịp thời.
- Hiệu năng hệ thống tiếp tục suy giảm.
- Dịch vụ có thể bị gián đoạn.
- Doanh nghiệp mất doanh thu.

---

## 6. Hậu Quả Khi Cảnh Báo Nhầm (False Positive)

False positive xảy ra khi hệ thống phát cảnh báo mặc dù không có sự cố thực sự.

**Ví dụ:** CPU tăng do lượng người dùng tăng tự nhiên nhưng hệ thống vẫn coi đây là anomaly.

**Hậu quả:**

Đội ngũ vận hành phải xử lý quá nhiều cảnh báo không cần thiết. Kỹ sư vận hành có xu hướng:
- Bỏ qua cảnh báo.
- Giảm mức độ tin tưởng vào hệ thống giám sát.

Ngoài ra, false positive còn làm tăng:
- Chi phí vận hành.
- Thời gian điều tra sự cố.