## 1. Vai trò của dữ liệu trong AI

Khác với các phần mềm truyền thống — nơi lập trình viên trực tiếp viết ra các quy tắc xử lý — các mô hình AI **rút ra quy luật từ dữ liệu** thông qua quá trình huấn luyện. Thay vì được cung cấp trước các luật, mô hình quan sát một tập dữ liệu đã được gán nhãn và điều chỉnh tham số sao cho dự đoán khớp với nhãn thực tế.

> Dữ liệu càng phù hợp và có chất lượng cao, mô hình càng có khả năng dự đoán tốt.

## 2. Từ dữ liệu đến mô hình

```
Raw data → Preprocessing → Feature Selection / Feature Engineering → Model Training → Prediction
```

Dữ liệu thô thường không sẵn sàng để đưa thẳng vào mô hình. Trước đó cần được:

- **Làm sạch**: xử lý giá trị thiếu, loại bỏ duplicate, xử lý outlier
- **Mã hóa**: chuyển các biến phân loại sang dạng số mà mô hình hiểu được
- **Chuẩn hóa**: đưa các cột số về cùng một thang đo

## 3. Dữ liệu tabular vs time-series

### 3.1. Dữ liệu có cấu trúc

Dữ liệu có cấu trúc được tổ chức thành các **trường (cột)** và **bản ghi (hàng)**. Khác với dữ liệu phi cấu trúc (ảnh, video, văn bản tự do), dữ liệu có cấu trúc thường được lưu trong bảng, CSDL hoặc file dạng CSV, v.v.

### 3.2. Dữ liệu tabular

Dữ liệu tabular là dữ liệu dạng bảng, trong đó:
- Mỗi **hàng** đại diện cho một quan sát
- Mỗi **cột** đại diện cho một đặc trưng (feature)

Đây là dạng dữ liệu có cấu trúc phổ biến nhất, thường được lưu dưới dạng CSV, Excel, Parquet hoặc trong CSDL.

Dữ liệu thường được biểu diễn dưới dạng ma trận 2 chiều với shape `(n_samples × n_features)`.

### 3.3. Dữ liệu time series

Dữ liệu time series là chuỗi các observation được ghi nhận theo thứ tự thời gian, trong đó mỗi quan sát gắn với một mốc thời gian cụ thể, gọi là **timestamp**. Khác với dữ liệu tabular (nơi các hàng độc lập với nhau), dữ liệu time series **phụ thuộc mạnh vào thứ tự trước sau** của các quan sát.

Dữ liệu time series thường được biểu diễn dưới dạng `(T × n_features)`, trong đó:
- `T` là số bước thời gian
- `n_features` là số biến được quan sát tại mỗi thời điểm

**Phân loại:**
- **Univariate time series**: tại mỗi thời điểm chỉ ghi nhận một biến
- **Multivariate time series**: tại mỗi thời điểm ghi nhận nhiều biến

#### Đặc trưng quan trọng của dữ liệu time series

| Đặc trưng | Mô tả |
|---|---|
| **Timestamp** | Mốc thời gian gắn với từng quan sát, xác định thứ tự trước sau của dữ liệu |
| **Time step** | Khoảng cách thời gian giữa hai quan sát liên tiếp (1 phút, 1 giờ, 1 ngày...) |
| **Frequency** | Tần suất lấy mẫu của toàn bộ chuỗi |

#### Các thành phần chính trong time series

- **Trend (xu hướng)**: thể hiện chiều hướng dài hạn của dữ liệu
- **Seasonality (tính mùa vụ)**: thể hiện các mẫu lặp lại theo chu kỳ
- **Residual (phần dư / nhiễu)**: thành phần biến động còn lại sau khi đã tách trend và seasonality, thường chứa nhiễu ngẫu nhiên hoặc các biến động khó giải thích

**Stationarity (tính dừng)**: một chuỗi được gọi là dừng khi các đặc trưng thống kê như trung bình và phương sai không thay đổi nhiều theo thời gian.

#### Nhiễu và bất thường trong dữ liệu time series

Cần phân biệt rõ giữa **noise** và **anomaly**:

- **Noise**: những dao động ngẫu nhiên nhỏ quanh giá trị thực, thường xuất hiện liên tục và không mang lại thông tin quan trọng. Khi xử lý nhiễu, mục tiêu thường là làm mượt dữ liệu để xu hướng chính hiện ra rõ hơn.
- **Anomaly**: những điểm hoặc đoạn dữ liệu lệch rõ rệt so với mẫu thông thường, thường xuất hiện ít hơn nhưng lại quan trọng.

→ Nhiễu thường cần được **giảm bớt hoặc lọc bỏ**, còn anomaly cần được **phát hiện và phân tích kỹ hơn**.

#### Các loại xu hướng (trend) trong time series data

1. **No trend (approximately stationary)**: chuỗi dao động quanh một giá trị trung bình ổn định, không tăng hoặc giảm rõ rệt theo thời gian
2. **Linear trend**: chuỗi tăng hoặc giảm với tốc độ tương đối đều
3. **Exponential trend**: chuỗi tăng hoặc giảm với tốc độ ngày càng nhanh
4. **Polynomial trend**: chuỗi có dạng cong, có thể tăng rồi giảm hoặc giảm rồi tăng
5. **Piecewise trend / structural break**: chuỗi có nhiều giai đoạn xu hướng khác nhau, thường do sự kiện lớn gây ra như thay đổi chính sách

> Việc nhận diện đúng loại trend giúp ta chọn cách xử lý và mô hình phù hợp hơn:
> - Chuỗi có mùa vụ rõ ràng → cần **decomposition**
> - Chuỗi có xu hướng dài hạn → cần mô hình dự báo có **thành phần trend**
> - Chuỗi có nhiều điểm bất thường → cần mô hình **anomaly detection**

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

# Quy Trình Xử Lý Dữ Liệu Time Series

Quá trình xử lý gồm các bước: đọc dữ liệu và parse thời gian, thiết lập index thời gian, trực quan hóa, tiền xử lý, phân rã chuỗi thời gian, tạo đặc trưng và chia tập huấn luyện/kiểm thử.

## Bước 1: Đọc dữ liệu và parse thời gian

Khi làm việc với dữ liệu time series, bước đầu tiên là đọc dữ liệu và chuyển đổi cột thời gian về đúng kiểu **datetime**. Nếu cột thời gian vẫn ở dạng chuỗi văn bản, pandas sẽ không thể xử lý chính xác các thao tác như sắp xếp theo thời gian, lọc theo khoảng ngày, resampling hoặc trực quan hóa theo trục thời gian.

> Ở bước này cần kiểm tra xem cột thời gian đã được chuyển sang kiểu `datetime64` hay chưa.

## Bước 2: Thiết lập index thời gian và kiểm tra

Sau khi cột thời gian đã đúng kiểu dữ liệu, ta đặt cột này làm **index** của dataframe. Việc này giúp thao tác với time series dễ hơn.

Dữ liệu đọc từ file chưa chắc đã được sắp xếp đúng thứ tự thời gian, nên cần **sắp xếp lại index** ngay sau khi thiết lập, để các bước rolling, lag features và chia train/test không bị sai.

Đồng thời, cần kiểm tra:
- Chuỗi bắt đầu từ đâu, kết thúc ở đâu
- Có bao nhiêu bước thời gian
- Có timestamp bị thiếu hay không

## Bước 3: Trực quan hóa

Với dữ liệu time series, trực quan hóa bằng **biểu đồ đường** là bước quan trọng Biểu đồ giúp quan sát xu hướng, mùa vụ, phần dư và điểm bất thường nhanh hơn so với việc chỉ nhìn vào bảng số liệu.

## Bước 4: Tiền xử lý

Tiền xử lý time series thường bao gồm ba bước: xử lý timestamp thiếu, resampling và làm mượt dữ liệu.

| Thao tác | Mô tả | Ví dụ |
|---|---|---|
| **Xử lý timestamp thiếu** | Nội suy tuyến tính (interpolate), điền giá trị trước đó (forward-fill), hoặc giá trị sau đó (bfill) | `df.interpolate()`, `df.ffill()`, `df.bfill()` |
| **Resampling** | Chuyển dữ liệu sang tần số khác — gộp lên hoặc chia nhỏ | `df.resample('D').mean()` (giờ → ngày), `df.resample('H').interpolate()` |
| **Làm mượt (smoothing)** | Dùng trung bình trượt để giảm nhiễu và làm rõ xu hướng | `df.rolling(window=...).mean()` |

## Bước 5: Phân rã chuỗi thời gian (Decomposition)

Đây là bước giúp tách chuỗi gốc thành ba thành phần chính: **trend**, **seasonality** và **residual**. Đây là một trong những bước quan trọng để hiểu cấu trúc của chuỗi.

Statsmodels cung cấp `seasonal_decompose` để thực hiện thao tác này.

## Bước 6: Tạo đặc trưng và chia tập

Với time series, đặc trưng thường được tạo từ chính **giá trị quá khứ** của chuỗi. Các đặc trưng phổ biến:

- **Lag features**: giá trị tại các bước thời gian trước đó
- **Rolling statistics**: trung bình, độ lệch chuẩn, min/max trong cửa sổ trượt
- **Time features**: các đặc trưng trích xuất từ thời gian (giờ, thứ, tháng, mùa...)

Sau khi tạo đặc trưng, chia dữ liệu thành tập **train** và **test**.