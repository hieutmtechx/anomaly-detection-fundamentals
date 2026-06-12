1. Định nghĩa:

Time series là dạng dữ liệu mà mỗi quan sát được ghi nhận tại một thời điểm cụ thể. Khác với dữ liệu dạng bảng, thứ tự xuất hiện của các quan sát trong time series mang ý nghĩa quan trọng và không thể thay đổi một cách tùy ý.

Các ứng dụng phổ biến của time series:
- Giám sát hệ thống
- Dữ liệu tài chính
- Dữ liệu cảm biến IoT
- Dự báo nhu cầu sản phẩm
- Phát hiện bất thường

2. Cấu trúc của dữ liệu time series:

Một chuỗi thời gian thường bao gồm:
- Timestamp: thời điểm ghi nhận dữ liệu.
- Value: giá trị được quan sát tại thời điểm đó.

Tùy vào số lượng biến được theo dõi, time series được chia thành hai loại: Univariate time series và multivariate time series với univariate time series chỉ theo dõi một biến theo thời gian còn multivariate time series là theo dõi nhiều biến theo thời gian.

3. Các đặc điểm cơ bản của time series:

- Temporal dependency: Là giá trị hiện tại thường phụ thuộc vào các giá trị trong quá khứ, các quan sát trong time series không độc lập với nhau.
- Trend: Là xu hướng tăng hoặc giảm của dữ liệu trong thời gian dài, giúp phản ánh sự thay đổi dài hạn của hệ thống.
- Seasonality: Là các mẫu lặp lại theo chu kỳ cố định, các chu kỳ thường gặp: Theo giờ, ngày, tuần , tháng, năm,...
- Noise: Là các biến động ngẫu nhiên không mang nhiều ý nghĩa thực tế và làm giảm độ chính xác của mô hình.
- Autocorrelation: Thể hiện mức độ tương quan giữa các giá trị trong chuỗi tại các thời điểm khác nhau. Nếu hai giá trị có mối quan hệ mạnh autocorrelation cao còn nếu không liên quan autocorrelation thấp. Điều này sẽ giúp xác định dữ liệu quá khứ ảnh hưởng tới hiện tại như thế nào.
- Stationarity: Là tính chất cho biết các đặc trưng thống kê của chuỗi không thay đổi theo thời gian. Các đặc trưng cần ổn định là:
+ Mean
+ Variance
+ Covariance
Nhiều mô hình thống kê truyền thống như ARIMA yêu cầu dữ liệu phải gần stationary trước khi huấn luyện.

4. Các vấn đề thường gặp trong dữ liệu time series:

- Missing values: Dữ liệu bị mất tại một số thời điểm.
- Outliers: Các điểm dữ liệu bất thường so với phần lớn dữ liệu còn lại.
- Concept drift: Phân phối dữ liệu thay đổi theo thời gian.
- Imbalanced anomalies: Điều này làm cho bài toán phát hiện bất thường trở nên khó khăn.

5. Time series vs tabular data:

- Dữ liệu có phụ thuộc thời gian, các quan sát không độc lập với nhau. Mô hình cần học được các quy luật dài hạn và chu kỳ lặp lại.
- Không thể shuffle dữ liệu, việc chia train và test phải tuân thủ thứ tự thời gian.
- Dữ liệu thay đổi theo thời gian như hiện tượng concept drift sẽ khiến mô hình nhanh chóng bị lỗi thời.
- Dữ liệu thường chứa missing value và noise, đòi hỏi nhiều bước tiền xử lý hơn dữ liệu tabular.
