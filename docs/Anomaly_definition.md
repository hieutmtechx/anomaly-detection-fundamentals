1. Khái niệm Anomaly:

Anomaly hay outlier là những điểm dữ liệu hoặc hành vi của hệ thống có sự khác biệt đáng kể so với trạng thái hoạt động bình thường.

Đối với dữ liệu time series, anomaly có thể xuất hiện dưới nhiều dạng:
- Giá trị tăng hoặc giảm đột ngột.
- Hành vi không phù hợp với xu hướng hiện tại.
- Hành vi không tuân theo tính chu kỳ của dữ liệu.
- Sự thay đổi bất thường trong mẫu hoạt động của hệ thống.

Anomaly thường chia thành ba nhóm: Point anomaly, contextual anomaly và collective anomaly. 

Point anomaly là trường hợp một điểm dữ liệu đơn lẻ có giá trị khác biệt đáng kể so với các điểm dữ liệu còn lại. Contextual anomaly là trường hợp một giá trị dữ liệu có thể bình thường trong một ngữ cảnh nhưng lại trở thành bất thường trong một ngữ cảnh khác. Collective anomaly xảy ra khi một nhóm điểm dữ liệu liên tiếp tạo thành một hành vi bất thường, mặc dù từng điểm dữ liệu riêng lẻ có thể hoàn toàn bình thường.

2. Cách xác định anomaly:

- Dựa trên ngưỡng:
Đây là phương pháp đơn giản nhất, sẽ định nghĩa trước các ngưỡng cho từng chỉ số. Khi giá trị vượt quá ngưỡng đã định nghĩa thì hệ thống sẽ cảnh báo. Ưu điểm của phương pháp này là dễ triển khai và giải thích. Tuy nhiên, khó thích ứng với các hệ thống có thay đổi liên tục.

- Dựa trên thống kê:
Phương pháp này giả định dữ liệu tuân thủ theo một phân phối nhất định. Các điểm dữ liệu nằm quá xa giá trị trung bình sẽ được xem là anomaly.

- Dựa trên dự báo:
Thay vì chỉ xem xét giá trị hiện tại, hệ thống học hành vi bình thường của chuỗi thời gian và dự đoán giá trị tương lai. Nếu sai số giữa giá trị thực tế và giá trị dự đoán vượt quá ngưỡng cho phép thì hệ thống coi đó là anomaly.

- Dựa trên hành vi hoặc mẫu dữ liệu:
Trong nhiều trường hợp, anomaly không phải là một giá trị quá lớn hoặc quá nhỏ mà là một hành vi khác thường so với lịch sử.

3. Anomaly ở mức metric:

Metric-level anomaly là các bất thường xuất hiện trên từng chỉ số riêng lẻ.

Một số ví dụ phổ biến:

3.1. CPU Utilization:

CPU tăng đột ngột từ 45% lên 95%.

Nguyên nhân có thể bao gồm:

- Tải hệ thống tăng cao.
- Tiến trình chạy bất thường.
- Vòng lặp vô hạn trong ứng dụng.

3.2. Memory Usage:

Bộ nhớ tăng liên tục và không được giải phóng.

3.3. Latency:

Thời gian phản hồi tăng từ vài trăm mili giây lên vài giây, điều này thường cho thấy hệ thống đang bị nghẽn hoặc quá tải.

3.4. Error Rate:

Tỷ lệ lỗi tăng bất thường.

4. Anomaly ở mức system:

System-level anomaly xảy ra khi nhiều chỉ số cùng biểu hiện bất thường trong cùng một khoảng thời gian.

Ví dụ:

- CPU tăng cao.
- Memory tăng cao.
- Latency tăng mạnh.
- Error rate tăng đột biến.

Tập hợp các dấu hiệu này cho thấy hệ thống đang gặp một sự cố nghiêm trọng thay vì chỉ một chỉ số riêng lẻ bất thường.

Một cơ sở dữ liệu bị nghẽn có thể dẫn tới:

- CPU tăng do phải xử lý nhiều truy vấn chờ.
- Memory tăng do hàng đợi nhiều.
- Latency tăng do phản hồi chậm.
- Error rate tăng do timeout.

Trong trường hợp này, anomaly không còn là vấn đề của một metric đơn lẻ mà đã trở thành sự cố ở cấp độ toàn hệ thống.

5. Hậu quả khi bỏ sót anomaly:

False negative xảy ra khi hệ thống thực sự có anomaly nhưng mô hình hoặc hệ thống giám sát không phát hiện được.

Ví dụ:

Một cơ sở dữ liệu đang quá tải:

- CPU tăng lên 98%.
- Latency tăng liên tục.

Tuy nhiên hệ thống không phát hiện và không sinh cảnh báo.

Hậu quả:

- Sự cố không được xử lý kịp thời.
- Hiệu năng hệ thống tiếp tục suy giảm.
- Dịch vụ có thể bị gián đoạn.
- Doanh nghiệp mất doanh thu.

6. Hậu quả khi cảnh báo nhầm:

False positive xảy ra khi hệ thống phát cảnh báo mặc dù không có sự cố thực sự.

Ví dụ:

CPU tăng do lượng người dùng tăng tự nhiên nhưng hệ thống vẫn coi đây là anomaly.

Hậu quả:

Đội ngũ vận hành phải xử lý quá nhiều cảnh báo không cần thiết.

Khi đó kỹ sư vận hành có xu hướng:

- Bỏ qua cảnh báo.
- Giảm mức độ tin tưởng vào hệ thống giám sát.

Ngoài ra, false positive còn làm tăng:

- Chi phí vận hành.
- Thời gian điều tra sự cố.
