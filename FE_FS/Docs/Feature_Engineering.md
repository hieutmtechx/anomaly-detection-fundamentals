# Feature Engineering cho Anomaly Detection trên Time-Series Metrics

## Mục tiêu và bối cảnh

Một điểm dữ liệu thô tại thời điểm t gần như vô nghĩa khi đứng một mình — nó chỉ trở nên có ý nghĩa khi được đặt trong ngữ cảnh lịch sử (nó khác gì so với 1 phút trước, 1 giờ trước), trong ngữ cảnh thống kê cục bộ (nó có nằm ngoài biên độ dao động bình thường gần đây không), và trong ngữ cảnh chu kỳ (đây có phải giờ cao điểm, có phải cuối tuần không). Năm nhóm đặc trưng dưới đây được thiết kế để phủ đầy đủ các ngữ cảnh này.

## 1. Lag Features (t-1, t-5, t-10, t-60)

Lag feature là giá trị của chính metric đó tại các thời điểm lùi về quá khứ, được đưa vào làm input ở thời điểm hiện tại. Đây là kỹ thuật nền tảng nhất trong feature engineering cho time-series vì nó cho phép mô hình nhìn thấy trực tiếp lịch sử gần nhất mà không cần qua bất kỳ phép biến đổi nào.

Lý do chọn bộ lag t-1, t-5, t-10, t-60 (giả định đơn vị là phút):

- t-1 nắm bắt sự liên tục tức thời (immediate continuity) — phần lớn metrics hệ thống có tính tự tương quan (autocorrelation) rất cao ở độ trễ 1 bước, nên đây gần như luôn là feature có sức mạnh dự đoán cao nhất.
- t-5 và t-10 nắm bắt xu hướng ngắn hạn (short-term momentum) — liệu metric đang tăng/giảm dần trong vài phút gần đây, hữu ích để phân biệt một cú tăng đột biến thật sự với nhiễu tức thời.
- t-60 nắm bắt ngữ cảnh trễ dài hơn, thường tương ứng với chu kỳ theo giờ — hữu ích để so sánh trạng thái hiện tại với trạng thái cách đây một giờ, một mốc thời gian có ý nghĩa vận hành rõ ràng (ví dụ so sánh tải hệ thống đầu giờ và cuối giờ).

Khoảng cách giữa các lag (1, 5, 10, 60) không tuyến tính mà giãn dần theo cấp số — đây là lựa chọn có chủ đích để tránh dư thừa thông tin: lag liền kề nhau (t-1, t-2, t-3...) thường tương quan rất cao với nhau và không bổ sung nhiều tín hiệu mới, nên việc chọn các mốc thưa dần giúp mỗi lag đóng góp một lớp ngữ cảnh khác biệt. Trong thực hành, một cách khoa học hơn để xác định lag tối ưu là dùng hàm tự tương quan (ACF — Autocorrelation Function) để tìm các độ trễ có tương quan thống kê có ý nghĩa với biến mục tiêu, thay vì chọn hoàn toàn theo kinh nghiệm.

Lag feature yêu cầu dữ liệu phải được căn chỉnh trên một lưới thời gian đều đặn (regular time grid). Nếu dữ liệu gốc có khoảng trống (missing timestamps) hoặc tần suất lấy mẫu không đều, cần resample/interpolate trước khi tính lag, nếu không các giá trị t-1, t-5... sẽ không phản ánh đúng khoảng cách thời gian thực tế.

## 2. Rolling Statistics (mean, std, min, max trên cửa sổ 5m / 15m / 1h)

Trong khi lag feature cho biết giá trị tại một điểm cụ thể trong quá khứ, rolling statistics tổng hợp hành vi của metric trên cả một cửa sổ thời gian, giúp làm mượt nhiễu (smoothing) và làm lộ rõ xu hướng cục bộ cũng như độ biến động.

Bốn thống kê được tính cho mỗi cửa sổ:

- Rolling mean: mức nền (baseline) gần đây của metric, dùng làm điểm tham chiếu để so sánh giá trị hiện tại có lệch bất thường hay không.
- Rolling std (độ lệch chuẩn): đo độ biến động/nhiễu cục bộ. Đây là feature cốt lõi cho các phương pháp phát hiện bất thường dựa trên Z-score, vì biên độ "bình thường" sẽ co giãn theo từng giai đoạn (ví dụ giờ cao điểm dao động mạnh hơn giờ thấp điểm) — rolling std giúp mô hình thích nghi với điều này thay vì dùng một ngưỡng cố định toàn cục.
- Rolling min/max: nắm bắt giá trị cực trị gần đây, hữu ích để phát hiện các điểm phá vỡ biên độ dao động đã thiết lập (range breakout), một dấu hiệu kinh điển của sự kiện bất thường.

Ba kích thước cửa sổ 5 phút / 15 phút / 1 giờ được chọn để tạo một hệ đa độ phân giải (multi-resolution view):

- Cửa sổ 5m nhạy với các đột biến ngắn (spike), phù hợp phát hiện sự cố tức thời như lỗi triển khai (deploy) gây lỗi ngay lập tức.
- Cửa sổ 15m cân bằng giữa độ nhạy và độ ổn định, giảm bớt false positive do nhiễu ngẫu nhiên trong khi vẫn phản ứng đủ nhanh với sự cố đang hình thành.
- Cửa sổ 1h nắm bắt xu hướng trôi dạt chậm (slow drift) — ví dụ rò rỉ bộ nhớ (memory leak) tăng dần không đột ngột, sẽ không hiện rõ ở cửa sổ ngắn nhưng rất rõ ở cửa sổ dài.

Việc dùng nhiều kích thước cửa sổ song song, thay vì chỉ một, là thực hành phổ biến trong các hệ thống giám sát thực tế (ví dụ theo kinh nghiệm triển khai của các đội kỹ thuật giám sát mạng), vì các loại bất thường khác nhau bộc lộ rõ nhất ở các tần số khác nhau — không có một kích thước cửa sổ nào tối ưu cho mọi loại sự cố.

## 3. Rate of Change (sai phân bậc một và phần trăm thay đổi)

Rate of change đo tốc độ biến thiên của metric thay vì giá trị tuyệt đối, gồm hai biến thể bổ trợ nhau:

- Sai phân bậc một (first-order difference): x(t) − x(t−1), đo lượng thay đổi tuyệt đối giữa hai bước liên tiếp. Phù hợp khi đơn vị đo có ý nghĩa tuyệt đối ổn định (ví dụ số lượng request).
- Phần trăm thay đổi (percentage change): [x(t) − x(t−1)] / x(t−1), đo mức thay đổi tương đối. Phù hợp hơn khi metric có biên độ dao động tự nhiên thay đổi theo thời gian (ví dụ traffic ban đêm thấp hơn nhiều lần so với ban ngày — cùng một mức tăng tuyệt đối có ý nghĩa rất khác nhau ở hai thời điểm).

Nhóm feature này quan trọng vì hai lý do. Thứ nhất, nó giúp mô hình phát hiện các bước nhảy đột ngột (sudden jump/drop) — một trong những dạng bất thường phổ biến nhất ở metrics hệ thống — mà chỉ nhìn vào giá trị tuyệt đối có thể không lộ rõ nếu giá trị đó vẫn nằm trong biên độ lịch sử rộng. Thứ hai, vì sai phân loại bỏ thành phần xu hướng dài hạn (trend), nó giúp dữ liệu tiệm cận tính dừng (stationarity) hơn — một giả định quan trọng đối với nhiều mô hình thống kê và machine learning cổ điển.

Lưu ý kỹ thuật: percentage change cần xử lý cẩn thận trường hợp mẫu số gần 0 (ví dụ error_rate = 0 ở bước trước), vì có thể sinh ra giá trị vô cực hoặc chia cho 0. Cần áp dụng epsilon nhỏ hoặc capping giá trị để tránh làm hỏng pipeline downstream.

## 4. STL Residuals (tín hiệu đã khử trend + seasonal)

STL (Seasonal-Trend decomposition using Loess) phân rã một time-series thành ba thành phần cộng: Y(t) = Trend(t) + Seasonal(t) + Residual(t). Trong bối cảnh anomaly detection, residual (phần dư) chính là thành phần quan trọng nhất, vì nó đại diện cho phần không giải thích được sau khi đã loại bỏ xu hướng dài hạn và chu kỳ lặp lại có thể dự đoán được — và bất thường, theo định nghĩa, là thứ không khớp với mô hình hành vi thông thường.

Lý do đưa STL residual vào bộ feature, thay vì chỉ dùng giá trị thô:

- Một giá trị metric cao bất thường vào giờ cao điểm có thể hoàn toàn bình thường (được giải thích bởi thành phần seasonal), trong khi cùng giá trị đó vào lúc 3 giờ sáng lại là bất thường rõ rệt. Dùng giá trị thô khiến mô hình khó phân biệt hai trường hợp này; dùng residual đã loại trừ yếu tố chu kỳ giúp việc so sánh trở nên công bằng (apples-to-apples) giữa các thời điểm khác nhau trong ngày/tuần.
- Residual có xu hướng tiệm cận phân phối dừng quanh giá trị 0, cho phép áp dụng các ngưỡng thống kê đơn giản và nhất quán hơn (ví dụ khoảng tin cậy theo độ lệch chuẩn) so với việc đặt ngưỡng trên dữ liệu gốc vốn luôn dịch chuyển theo trend/seasonal.

Một số lưu ý quan trọng khi triển khai STL trong thực tế cần được ghi nhận rõ trong tài liệu kỹ thuật:

- STL là thuật toán dạng batch, đòi hỏi một cửa sổ dữ liệu đủ dài (thường vài chu kỳ mùa vụ) để ước lượng trend và seasonal đáng tin cậy; điều này gây khó khăn khi muốn tính residual theo thời gian thực cho dữ liệu mới chưa có đủ lịch sử.
- Nếu trong dữ liệu lịch sử dùng để fit STL đã tồn tại các anomaly tập thể (collective anomalies, tức bất thường kéo dài), các giá trị này có thể bị "hấp thụ" nhầm vào thành phần trend hoặc seasonal thay vì lộ ra ở residual, làm giảm độ nhạy phát hiện. Cần cân nhắc dùng dữ liệu huấn luyện đã được làm sạch tương đối, hoặc các biến thể robust hơn (như RobustSTL) nếu nguồn dữ liệu nhiều nhiễu.
- Cần xác định đúng chu kỳ mùa vụ (period) đầu vào cho STL — ví dụ 1440 phút cho chu kỳ ngày nếu lấy mẫu theo phút, hoặc 7 ngày cho chu kỳ tuần — vì chọn sai chu kỳ sẽ khiến cả thành phần trend và seasonal đều bị ước lượng sai, kéo theo residual sai lệch.

## 5. Cross-Metric Ratios (error_rate / request_rate, memory / cpu, v.v.)

Tỷ lệ chéo giữa hai metric khác nhau tạo ra một feature tổng hợp mang ý nghĩa nghiệp vụ mà từng metric đơn lẻ không thể hiện được. Ví dụ:

- error_rate / request_rate cho biết tỷ lệ lỗi trên mỗi đơn vị tải, tách biệt được trường hợp "số lỗi tăng vì traffic tăng" (bình thường) khỏi "số lỗi tăng dù traffic không đổi" (bất thường thực sự, có thể do bug hoặc dependency lỗi).
- memory / cpu hoặc các tỷ lệ tài nguyên tương tự giúp phát hiện các kiểu lệch tải bất thường, ví dụ memory tăng vọt trong khi cpu không đổi tương ứng — dấu hiệu kinh điển của memory leak thay vì tải tính toán tăng thông thường.

Giá trị cốt lõi của nhóm feature này là tính bất biến theo quy mô (scale invariance): hệ thống có thể trải qua biến động traffic tự nhiên rất lớn theo giờ trong ngày, nhưng tỷ lệ lỗi trên traffic (thay vì số lỗi tuyệt đối) mới là chỉ báo có ý nghĩa ổn định để giám sát. Việc đưa tỷ lệ chéo vào bộ feature giúp mô hình học được các mối quan hệ liên-metric (cross-metric relationship) mà nếu chỉ xử lý từng metric độc lập theo từng kênh riêng biệt sẽ bị bỏ sót hoàn toàn.

Lưu ý kỹ thuật tương tự rate-of-change: cần xử lý mẫu số bằng 0 hoặc gần 0 (ví dụ request_rate = 0 trong giai đoạn không có traffic) để tránh giá trị vô cực hoặc NaN làm hỏng các bước tính toán phía sau.

## 6. Time-of-Day và Day-of-Week Encoding

Encoding thời gian theo chu kỳ trong ngày (time-of-day) và trong tuần (day-of-week) cung cấp ngữ cảnh chu kỳ tường minh (explicit periodic context) cho mô hình, bổ trợ cho phần thành phần seasonal mà STL đã trích xuất.

Lý do cần thiết: nhiều hành vi hệ thống có tính chu kỳ rất mạnh và có thể dự đoán được — traffic ban ngày cao hơn ban đêm, ngày thường khác cuối tuần, giờ hành chính khác giờ nghỉ. Nếu không cung cấp ngữ cảnh này, mô hình anomaly detection dễ gắn nhãn sai các biến động hoàn toàn bình thường theo chu kỳ (ví dụ traffic giảm mạnh lúc nửa đêm) thành bất thường.

Một lưu ý kỹ thuật quan trọng: time-of-day và day-of-week là các biến có tính tuần hoàn (cyclical), nghĩa là giá trị 23h và 0h thực chất rất gần nhau về mặt thời gian nhưng nếu encode trực tiếp dưới dạng số nguyên (0–23 hoặc 0–6) thì mô hình sẽ hiểu sai khoảng cách giữa chúng là xa nhất. Cách xử lý chuẩn là dùng phép biến đổi sin/cos (cyclical encoding): sin(2π × giờ/24), cos(2π × giờ/24) — qua đó bảo toàn đúng tính liên tục vòng tròn của thời gian. Tương tự với day-of-week nếu cần biểu diễn dạng liên tục thay vì one-hot encoding rời rạc.

## Tổng kết

Các nhóm feature trên không độc lập mà bổ trợ lẫn nhau theo các trục ngữ cảnh khác nhau: lag và rate-of-change nắm bắt động lực học tức thời (immediate dynamics); rolling statistics nắm bắt hành vi cục bộ đa độ phân giải; STL residual và time encoding cùng xử lý yếu tố chu kỳ nhưng theo hai cách bổ trợ — một cách ngầm định qua phân rã thống kê, một cách tường minh qua đặc trưng lịch; cross-metric ratio bổ sung ngữ cảnh liên-metric mà các nhóm còn lại không thể cung cấp khi xử lý từng kênh độc lập. Kết quả của bước FE này là một không gian feature có kích thước lớn (với 5 lag × N metric × nhiều thống kê rolling × nhiều cửa sổ, số lượng cột có thể tăng theo cấp số nhân), do đó bước feature selection tiếp theo là cần thiết để loại bỏ dư thừa và giữ lại tập feature tinh gọn, có sức mạnh dự đoán cao.