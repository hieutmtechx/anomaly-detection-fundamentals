1. Tổng quan về dataset:

Dataset được sử dụng là AIOps 2018 KPI Anomaly Detection Dataset, bao gồm các chuỗi thời gian KPI (Key Performance Indicator) được thu thập từ nhiều hệ thống thực tế. Các KPI thuộc hai nhóm chính:

- Machine KPI: CPU Utilization, Memory Utilization, Disk I/O, Network Throughput.
- Service KPI: Response Time, Error Rate, Throughput.

Mỗi quan sát trong tập huấn luyện bao gồm:

- KPI ID: Mã định danh KPI
- Timestamp: Thời điểm ghi nhận dữ liệu
- Value: Giá trị KPI tại thời điểm đó
- Label: Nhãn bất thường


2. Dataset statistics:

- Tổng số KPI: 26
- Tổng số điểm dữ liệu: 2,476,315
- Số lượng dữ liệu bình thường: 2,422,815
- Số lượng dữ liệu bất thường: 53,500

Tỷ lệ phân bố nhãn: Normal là 97.84% còn anomaly là 2.16%.

mean      95242.884615
std       57377.365298
min        8247.000000
25%       24579.000000
50%      128673.000000
75%      135807.000000
max      147689.000000

3. Seasonality:

Phân tích được thực hiện trên KPI có ID: 02e99bd4f6cfb33f

Khi quan sát dữ liệu theo từng ngày, KPI thể hiện chu kỳ lặp lại rất rõ ràng.

Giá trị trung bình theo giờ:

| Hour  | Mean Value |
| ----- | ---------- |
| 00:00 | 2.40       |
| 03:00 | 2.44       |
| 06:00 | 1.95       |
| 09:00 | 1.10       |
| 12:00 | 1.05       |
| 15:00 | 1.95       |
| 18:00 | 2.46       |
| 21:00 | 2.35       |

Giá trị KPI cao nhất xuất hiện vào khoảng 02:00–03:00 và thấp nhất vào khoảng 10:00–12:00. Điều này cho thấy KPI có tính chu kỳ trong 24 giờ.

4. Trend:

Quan sát toàn bộ chuỗi dữ liệu trong khoảng thời gian từ tháng 05/2017 đến tháng 08/2017 không cho thấy xu hướng tăng hoặc giảm dài hạn rõ ràng. KPI này không có long-term trend đáng kể.

5. Noise:

Dữ liệu xuất hiện các dao động ngẫu nhiên xung quanh mức hoạt động bình thường. Những dao động này không tuân theo một quy luật cố định và được xem là noise của chuỗi thời gian.
 
6. Value distribution:

Histogram của KPI cho thấy dữ liệu không tuân theo phân phối chuẩn.
Phân phối xuất hiện nhiều cụm giá trị khác nhau:

- Nhóm giá trị quanh 1.0
- Nhóm giá trị quanh 2.0–2.8

Điều này cho thấy sự tồn tại của nhiều trạng thái vận hành khác nhau trong hệ thống.

5. Phân tích anomaly:

Các điểm được gán nhãn bất thường xuất hiện dưới nhiều dạng khác nhau:

- Positive anomaly: Giá trị tăng đột biến so với trạng thái bình thường.
- Negative anomaly: Giá trị giảm mạnh bất thường.
- Collective anomaly: Nhiều điểm bất thường liên tiếp xuất hiện trong một khoảng thời gian.

