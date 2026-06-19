# Feature Selection

## 1. Variance Threshold

Variance threshold là phương pháp loại bỏ những features có biến động thấp hoặc không thay đổi trong suốt toàn bộ tập dữ liệu.

- Nếu một cột giá trị có lúc cao lúc thấp → variance lớn, chứa nhiều thông tin.
- Nếu một cột từ trên xuống dưới đều có giá trị giống hệt nhau → variance gần bằng 0, cột này chứa ít thông tin.

---

## 2. Correlation Filter

### Hệ số tương quan Pearson

Hệ số tương quan Pearson là chỉ số đo lường mối quan hệ tuyến tính giữa hai biến, có giá trị nằm trong khoảng từ -1 đến 1.

| Giá trị | Ý nghĩa |
|---------|---------|
| `r = 1` | Hai biến tăng cùng nhau theo tỷ lệ |
| `r = -1` | Biến này tăng thì biến kia giảm theo tỷ lệ |
| `r = 0` | Hai biến không có mối quan hệ tuyến tính nào |

Khi lấy giá trị tuyệt đối `|r|`, ta không quan tâm đến chiều tương quan (dương/âm) mà chỉ quan tâm đến độ mạnh của mối quan hệ.

Ví dụ: `|r| > 0.95` nghĩa là hai cột gần như là bản sao của nhau, nếu biết giá trị cột A thì có thể đoán được giá trị cột B.

### Multicollinearity

Hiện tượng hai hoặc nhiều cột đầu vào phụ thuộc chặt chẽ vào nhau được gọi là multicollinearity. Giữ lại cả hai cột có tương quan cao sẽ gây ra nhiều vấn đề:

- Redundancy: Không cung cấp thêm bất kỳ thông tin mới nào cho mô hình nhưng lại tốn thêm tài nguyên tính toán cho một cột bị trùng lặp ý nghĩa.
- Làm nhiễu các mô hình tuyến tính: Toán học đằng sau mô hình sẽ bị bất ổn, mô hình không thể xác định được đâu mới là cột thực sự có tầm ảnh hưởng, dẫn đến việc phân bổ trọng số sai.

---

## 3. Mutual Information

Trong lý thuyết thông tin, Mutual Information là đại lượng đo lường mức độ phụ thuộc giữa hai biến ngẫu nhiên. Nó trả lời câu hỏi:

Nếu biết thông tin của cột X thì sẽ hiểu thêm được bao nhiêu phần trăm về cột Y.

| Giá trị | Ý nghĩa |
|---------|---------|
| `MI = 0` | Hai biến hoàn toàn độc lập. Biết X không giúp đoán được gì về Y |
| `MI càng cao` | X và Y càng liên quan chặt chẽ. Biết X sẽ giúp đoán được Y |

### So sánh với Correlation Filter

Khác với correlation filter chỉ bắt được các mối quan hệ tuyến tính, Mutual Information bắt được mọi loại quan hệ phức tạp. Nếu cột X biến động theo một kiểu bất thường mà nhãn Y cũng biến động theo, MI sẽ phát hiện ra được.

### Ứng dụng trong bài toán phát hiện bất thường

Trong bài toán phát hiện bất thường, dữ liệu có cột target variable (anomaly label) thường ở dạng nhị phân. Do đó, ta tính điểm số MI giữa từng cột feature đầu vào với cột nhãn bất thường này.

**Quy trình sau khi tính MI:**

1. **Ranking**: Sắp xếp các cột dữ liệu theo thứ tự điểm MI từ cao xuống thấp.
2. **Giữ lại features tốt nhất**: Những cột đứng đầu bảng chứa nhiều thông tin nhất, có khả năng giải thích và dự báo xem một dòng dữ liệu là bình thường hay bất thường.
3. **Loại bỏ features kém**: Những cột đứng cuối bảng không có giá trị dự báo sẽ bị drop.

---

## 4. Curated Feature Set

Curated feature set là kết quả sau khi đã áp dụng tất cả các bộ lọc, một danh sách gồm các cột dữ liệu sạch và có tính dự báo cao.

---

## 5. Written Justification

Written justification là một tài liệu chứng minh bằng lập luận khoa học và toán học, giải thích tại sao giữ lại cột này và xóa cột kia.

Tài liệu bao gồm hai phần:

- **Lý do kỹ thuật**:
  - *"Cột A bị loại vì có phương sai bằng 0."*
  - *"Cột B bị loại vì tương quan `|r| > 0.95` với cột C."*
  - *"Cột D được giữ lại vì nằm trong top điểm số Mutual Information cao nhất với nhãn bất thường."*

- **Bối cảnh thực tế (Domain Knowledge)**:
  - Ví dụ cột `error_rate / request_rate` được giữ lại vì nó trực tiếp phản ánh tỷ lệ lỗi hệ thống theo thời gian thực, giúp mô hình nhạy bén hơn với các sự cố sập nguồn.