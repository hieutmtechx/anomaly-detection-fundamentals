# Feature Engineering

## 1. Lag Features

Lag features là kỹ thuật được sử dụng để nắm bắt các mối quan hệ và mô hình theo thời gian trong dữ liệu chuỗi thời gian.

Cụ thể lag feature là việc lấy dữ liệu của quá khứ và ghép vào dòng của hiện tại để làm đầu vào cho mô hình học máy dự đoán.

---

## 2. Rolling Statistics

Rolling statistics là kỹ thuật biến đổi dữ liệu thời gian, giúp mô hình nhìn vào bức tranh tổng thể của một khoảng thời gian vừa qua.

### Định nghĩa Rolling Window

Rolling window là một khung cửa sổ dịch chuyển theo thời gian. Khi thời gian trôi về phía trước, cửa sổ này sẽ tự động cuộn theo để lấy N dữ liệu gần nhất, bỏ lại những dữ liệu cũ phía sau.

- Cửa sổ 5m: Tại mỗi phút hiện tại, mô hình gom dữ liệu của 5 phút ngay trước đó để tính toán.
- Cửa sổ 15m / 1h: Tương tự nhưng với khoảng thời gian rộng hơn.

### Các Đại Lượng Tính Toán

Sau khi gom dữ liệu trong cửa sổ, 4 đại lượng sau được tính toán:

| Đại lượng | Ý nghĩa |
|-----------|---------|
| **Rolling Mean** | Trung bình cộng tất cả giá trị trong cửa sổ. Đại diện cho trend hiện tại, làm mượt các biến động nhiễu để thấy được xu hướng tăng/giảm. |
| **Rolling Std** | Đo mức độ phân tán/sai lệch so với mức trung bình. Đại diện cho độ biến động — nếu std đột ngột tăng cao, dữ liệu đang bất ổn. |
| **Rolling Min** | Điểm thấp nhất mà dữ liệu chạm tới trong cửa sổ. Dùng để xác định các ngưỡng dưới. |
| **Rolling Max** | Điểm cao nhất dữ liệu đạt được trong cửa sổ. Dùng để xác định các ngưỡng trên. |

---

## 3. Rate of Change

Rate of Change là cách đo lường sự biến động của một đại lượng theo thời gian. Thay vì nhìn vào giá trị tuyệt đối, ta nhìn vào sự chuyển dịch, ví dụ: thay vì nhìn vào giá, ta nhìn xem hôm nay tăng hay giảm so với hôm trước.

### First-order Difference

Phép tính trừ giữa giá trị tại thời điểm hiện tại và giá trị ngay trước đó.

### Percentage Change

Đo lường sự thay đổi dưới dạng tỷ lệ phần trăm so với giá trị gốc. Cho biết tốc độ tăng trưởng tương đối, điều này quan trọng khi muốn so sánh biến động của hai đại lượng có quy mô khác nhau.

---

## 4. STL Residual

STL Residual là kết quả sau khi lấy dữ liệu gốc và loại bỏ đi thành phần trend và seasonal, chỉ còn lại những tín hiệu gốc hoặc nhiễu ngẫu nhiên.

Residual phản ánh những biến động bất thường mà các quy luật thông thường không giải thích được. Bất kỳ điểm dữ liệu nào có giá trị residual quá cao hoặc quá thấp đều là dấu hiệu của một sự cố hoặc một sự kiện đặc biệt.

---

## 5. Cross-metric Ratios

Cross-metric ratios là kỹ thuật tạo ra một chỉ số mới bằng cách thực hiện phép chia giữa hai chỉ số thuộc hai phạm trù hoặc hai nguồn khác nhau.

Thay vì nhìn vào từng chỉ số một cách độc lập, việc kết hợp chúng thành một tỷ lệ giúp hiểu được mối quan hệ tương quan và hiệu suất tương đối của hệ thống.

### Lợi ích khi đưa vào mô hình

Khi đưa dữ liệu hệ thống vào mô hình để dự báo anomalies hoặc tự động mở rộng quy mô, các chỉ số gốc thường không đủ:

- Normalization: Tỷ lệ phần trăm hoặc tỷ số luôn nằm trong một khoảng dễ kiểm soát, giúp mô hình không bị nhầm khi lượng truy cập đột ngột tăng cao.
- Contextualization: Cross-metric ratios cung cấp ngữ cảnh cho mô hình.

---

## 6. Time Encoding

### Time-of-day Encoding

Kỹ thuật này giúp mô hình bắt được các quy luật sinh hoạt theo giờ của con người.

### Time-of-week Encoding

Kỹ thuật này giúp mô hình phân biệt được hành vi giữa ngày đi làm và ngày nghỉ.