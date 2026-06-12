1. Missing values, data gaps, sampling consistency:

Missing value được định nghĩa là giá trị bị thiếu trong dữ liệu. Đối với dữ liệu timeseries nếu bị mất một giá trị ta sẽ không biết hệ thống thực sự diễn ra như thế nào ở giữa. Có thể điểm bị thiếu chính là anomaly nhưng vì dữ liệu bị mất nên là anomaly biến mất.

Data gaps là khoảng thời gian không có dữ liệu.

Sampling consistency là xem dữ liệu được ghi nhận như thế nào. Ví dụ là được ghi nhận 1p một hay là 5p một.

2. Stationarity:

Stationary là trạng thái mà các đặc điểm thống kê của dữ liệu như mean, variance luôn giữ nguyên, không bị thay đổi theo thời gian. Một chuỗi stationarity sẽ dao động ổn định quanh một đường nằm ngang, không có trend tăng giảm hay là có tính seasonality. Nếu có các yếu tố này thì sẽ làm giá trị của chuỗi thay đổi ở các thời điểm khác nhau.  

Ngược lại, white noise là một chuỗi các dữ liệu hoàn toàn ngẫu nhiên và không có bất kỳ quy luật nào có thể dự đoán được trong dài hạn. Bởi vì tính chất ngẫu nhiên, nên quan sát chuỗi ở bất kỳ khoảng thời gian nào thì nhìn cũng giống nhau, không hề bị ảnh hưởng bởi trend hay seasonality.

Differencing: Đây là kỹ thuật để biến một chuỗi non-stationarity thành chuỗi stationarity. Bằng cách tính độ chênh lệch giữa các điểm dữ liệu liên tiếp, diff giúp làm ổn định mức trung bình của chuỗi từ đó giảm thiểu trend và seasonal.

3. ADF Test (Augmented Dickey-Fuller)

Là một bài kiểm tra thống kê giúp đánh giá xem chuỗi dữ liệu có stationarity hay không. Thay vì nhìn vào biểu đồ thì sẽ nhìn vào một điểm số được gọi là p-value. Ví dụ:

- Nếu p-value < một ngưỡng: Thì dữ liệu đang ở trạng thái stationarity.
- Nếu p-value > một ngưỡng: Thì dữ liệu non-stationarity. Khi này sẽ thực hiện bước differencing và chạy lại ADF test.

4. Rolling mean và rolling variance

Đây là phương pháp để kiểm tra xem dữ liệu có phải stationarity hay không.

Ý tưởng chính là thay vì tính trên toàn bộ dataset thì sẽ tính trên một cửa sổ trượt. 

Rolling mean là tính trung bình cộng của các dữ liệu trong khung, rồi trượt khung tiến lên và lại tính tiếp. Biểu đồ của các điểm này sẽ tạo ra một đường mượt và giúp loại bỏ nhiễu ngẫu nhiên ngắn hạn để nhìn rõ chuyển động của dữ liệu.

Rolling variance cũng tương tự nhưng thay vì tính trung bình thì tính variance trong khung cửa sổ để xem độ phân tán của dữ liệu. 

Một chuỗi thời gian stationarity thì biểu đồ sẽ nằm ngang ổn định và có mức độ biến động không đổi theo thời gian.

5. Seasonality và cyclic:

Seasonality xảy ra khi dữ liệu xuất hiện những dao động lặp lại do ảnh hưởng của các yếu tố thời gian, chẳng hạn như thời điểm trong năm, ngày trong tuần hoặc giờ trong ngày. Đặc điểm quan trọng của seasonality là luôn lặp lại một chu kỳ cố định và biết trước.

Cyclic xảy ra khi dữ liệu xuất hiện những đợt tăng giảm nhưng không theo một tần suất hay chu kỳ thời gian cố định. Khác với seasonality luôn lặp lại đều đặn, một chu kỳ sẽ kéo dài hơn và có biên độ dao động biến động mạnh hơn so với seasonality.

Nếu tần suất lặp lại là cố định thì đó là thì đấy là seasonality còn nếu độ dài các đợt biến động thay đổi ngẫu nhiên thì sẽ được gọi là cyclic.

6. STL Decomposite:

Decomposite là phương pháp chia một chuỗi dữ liệu thành 3 thành phần: Trend, seasonal và residual. Việc tách từng thành phần giúp nhìn rõ bản chất bên trong của dữ liệu.

STL (Seasonal and Trend decomposition using Loess) là một trong những phương pháp chia chuỗi dữ liệu tốt hiện nay. STL xử lý được các loại tần số seasonal, cho phép seasonal biến đổi theo thời gian chứ không bắt buộc phải cố định. Có thể tự động điều chỉnh độ mượt của trend và không bị sai lệch bởi các dữ liệu outlier.

STL mặc định chỉ thực hiện phân rã theo dạng additive. Nếu dữ liệu có seasonal tăng dần theo thời gian dạng multiplicative thì cần lấy logarit của dữ liệu trước khi chạy STL.

Additive vs multiplicative time series components: 2 cách để kết hợp các thành phần time series. Additive là các thành phần riêng lẻ (Trend, seasonal, residual) được cộng lại với nhau. Xu hướng cộng biểu thị một xu hướng tuyến tính và tính mùa vụ cộng tính biểu thị tần số (độ rộng) và biên độ (chiều cao) giống nhau của các chu kỳ mùa vụ.  Multiplicative có nghĩa là từng thành phần nhân với nhau, multiplicative trend thể hiện xu hướng phi tuyến tính và phép nhân cho thấy tần số (chiều rộng) và/hoặc biên độ (chiều cao) của các chu kỳ theo mùa tăng/giảm.

7. FFT (Fast Fourier Transform)

FFT là một thuật toán giúp chuyển đổi chuỗi dữ liệu từ dạng các giá trị thay đổi theo thời gian sang mức độ lặp lại của các giá trị.

Sử dụng FFT để tìm dominant frequencies giúp xác định chính xác những cyclic hoặc seasonal nào đang hoặt động mạnh nhất và lặp đi lặp lại rõ nét nhất trong dữ liệu.

Phương pháp này phân tích chuỗi thời gian phức tạp ban đầu thành một tập hợp các sóng sine và cosine đơn giản. Bên cạch việc tìm cyclic, FFT cũng được ứng dụng để tăng tốc độ tính toán khi cần tìm sự tương đồng giữa các chuỗi dữ liệu lớn.

8. ACF (Autocorrelation Function), PACF (Partial Autocorrelation Function)

ACF đo mức độ tương quan tuyến tính giữa giá trị ở thời điểm hiện tại và các giá trị trong quá khứ của chính chuỗi dữ liệu đó tại một khoảng lag nhất định.

- Lag: là khoảng bước lùi về quá khứ so với thời điểm hiện tại đang xét. 

PACF đo lường mối quan hệ trực tiếp giữa giá trị hiện tại và một giá trị trong quá khứ sau khi đã loại bỏ ảnh hưởng của các mốc thời gian nằm ở giữa.

ACF và PACF được dùng để xác định thông số p cho phần AR và q cho phần MA sau khi dữ liệu đã được diff ở phần I. 

9. Cross-metric correlation heatmap:

Đây là công cụ trực quan giúp đánh giá mối quan hệ giữa nhiều metric hoặc nhiều chuỗi thời gian cùng một lúc. 

- Correlation: Khi A tăng và B tăng thì giá trị = +1, khi giá trị = 0 thì A và B không liên quan, còn nếu A tăng và B giảm thì giá trị = -1.
- Heatmap: Khi đỏ đậm thì sẽ là tương quan cao, xanh đậm là tương quan âm và còn trắng thì gần 0.

10. Anomaly label distribution:

- Count: có bao nhiêu anomaly.
- Density: mật độ anomaly, được tính bằng số nhãn anomaly trên tổng số điểm quan sát được.
- Point anomaly: đây là loại anomaly đơn giản nhất, một điểm riêng lẻ bất thường.
- Collective anomaly: một điểm riêng lẻ có thể không bất thường nhưng một chuỗi liên tiếp thì bất thường.