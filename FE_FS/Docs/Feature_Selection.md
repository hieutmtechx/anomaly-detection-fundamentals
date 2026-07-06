# Feature Selection cho Anomaly Detection trên Time-Series Metrics

## Mục tiêu và bối cảnh

Sau bước feature engineering (FE) không gian feature thường phình to đáng kể: với nhiều metric gốc, mỗi metric lại sinh ra nhiều lag, nhiều thống kê rolling trên nhiều cửa sổ, cộng thêm rate-of-change, STL residual, cross-metric ratio và time encoding — tổng số cột có thể lên đến hàng trăm. Đưa thẳng toàn bộ tập feature này vào mô hình gây ra ba vấn đề: (1) nguy cơ overfitting do số chiều quá lớn so với số lượng mẫu bất thường thường rất hiếm (curse of dimensionality), (2) chi phí tính toán và độ trễ tăng, gây bất lợi cho hệ thống cần phát hiện gần thời gian thực, và (3) giảm khả năng diễn giải (interpretability) — khi một mô hình gắn cờ bất thường, đội vận hành cần biết feature nào là nguyên nhân, điều này khó khăn hơn nhiều khi có hàng trăm feature dư thừa, tương quan lẫn nhau.

Quy trình feature selection (FS) dưới đây áp dụng ba bộ lọc tuần tự (filter methods) — variance threshold, correlation filter, và mutual information — để thu gọn tập feature về một tập "curated" vừa tinh gọn vừa giữ được sức mạnh dự đoán. Đây là các phương pháp filter cổ điển: đánh giá feature dựa trên đặc tính thống kê nội tại hoặc mối quan hệ thống kê với biến mục tiêu, độc lập với bất kỳ thuật toán học máy cụ thể nào. Ưu điểm của cách tiếp cận filter là chi phí tính toán thấp, chạy nhanh trên không gian feature lớn, và không phụ thuộc vào mô hình cuối — phù hợp làm bước tiền xử lý trước khi thử nghiệm nhiều loại mô hình khác nhau (Isolation Forest, One-Class SVM, Autoencoder...). Nhược điểm cố hữu của filter method là không nắm bắt được tương tác phức tạp giữa các feature hoặc tương tác feature-mô hình cụ thể — đây là đánh đổi được chấp nhận để lấy tốc độ và tính đơn giản ở giai đoạn sàng lọc đầu.

## Bước 1: Variance Threshold — loại bỏ feature gần như không có phương sai

Variance threshold là bộ lọc đơn giản nhất và rẻ nhất về mặt tính toán, thường được áp dụng đầu tiên trong chuỗi xử lý filter. Nguyên lý: một feature có phương sai gần bằng 0 nghĩa là gần như không đổi trên toàn bộ tập dữ liệu, do đó không mang thông tin phân biệt (discriminative information) nào — nó không thể giúp phân biệt giữa trạng thái bình thường và bất thường, bất kể giá trị mục tiêu là gì.

Trong bối cảnh time-series metrics, feature gần như hằng số (near-constant) phát sinh khá phổ biến, ví dụ:

- Rolling std trên cửa sổ ngắn của một metric vốn rất ổn định (ví dụ một service ít traffic, ít dao động) có thể gần như bằng 0 trong phần lớn thời gian.
- Lag feature của một metric hiếm khi thay đổi giá trị (ví dụ một flag hoặc một counter ít biến động) sẽ tạo ra nhiều cột gần như trùng lặp giá trị hằng số.
- Cross-metric ratio khi một trong hai metric gốc hầu như không đổi cũng có thể sinh ra tỷ lệ gần như hằng số.

Về mặt thực hành, ngưỡng phương sai cần được chuẩn hóa trước khi so sánh (ví dụ qua hệ số biến thiên — coefficient of variation, hoặc chuẩn hóa dữ liệu về cùng thang đo trước khi áp ngưỡng), vì các metric gốc có đơn vị và biên độ rất khác nhau (ví dụ request_rate có thể dao động hàng nghìn trong khi error_rate dao động trong khoảng 0–1) — nếu áp một ngưỡng phương sai tuyệt đối chung cho mọi feature, các feature có đơn vị nhỏ sẽ bị loại bỏ một cách bất công dù chúng thực sự biến động (về mặt tương đối) không kém các feature khác.

Cần lưu ý hạn chế quan trọng của variance threshold: nó hoàn toàn không xét đến mối quan hệ giữa feature và biến mục tiêu (nhãn bất thường), cũng không xét quan hệ giữa các feature với nhau — nó chỉ đánh giá đặc tính nội tại của từng feature một cách độc lập. Do đó đây chỉ nên là bước lọc thô ban đầu (loại bỏ rác hiển nhiên), không phải bước quyết định cuối cùng về mức độ liên quan của feature.

## Bước 2: Correlation Filter — loại bỏ một feature trong mỗi cặp có |r| > 0.95

Sau khi loại bỏ các feature gần như hằng số, bước tiếp theo xử lý vấn đề dư thừa (redundancy) giữa các feature còn lại — một vấn đề đặc biệt nghiêm trọng trong bộ feature time-series vì bản chất các lag và rolling statistics liền kề nhau thường có tương quan tuyến tính rất cao.

Cách thực hiện: tính ma trận tương quan Pearson (hoặc Spearman nếu nghi ngờ quan hệ phi tuyến đơn điệu) giữa tất cả các cặp feature còn lại. Với mọi cặp có trị tuyệt đối hệ số tương quan |r| vượt ngưỡng 0.95, loại bỏ một trong hai feature của cặp đó, thường giữ lại feature có ý nghĩa diễn giải cao hơn hoặc tương quan mạnh hơn với biến mục tiêu (xem thêm bước 3).

Trong dữ liệu metrics time-series, các cặp tương quan cao điển hình bao gồm:

- Rolling mean trên các cửa sổ liền kề nhau (ví dụ rolling mean 5m và 15m) thường tương quan rất cao khi metric biến động chậm, vì hai cửa sổ chồng lấp nhiều dữ liệu chung.
- Lag t-1 và sai phân bậc một (first-order difference) có thể tương quan cao một cách gián tiếp qua giá trị hiện tại x(t), vì diff = x(t) − lag(t-1).
- Rolling min và rolling max trên cùng cửa sổ có thể tương quan cao với rolling mean nếu metric có biên độ dao động tương đối ổn định.
- STL residual và rate-of-change đôi khi nắm bắt thông tin trùng lặp đáng kể, vì cả hai đều cố gắng loại bỏ thành phần dễ dự đoán (trend/seasonal hoặc baseline) để lộ ra phần bất thường.

Lý do loại bỏ tương quan cao quan trọng hơn việc chỉ giảm số chiều: với các mô hình tuyến tính hoặc các phương pháp dựa trên khoảng cách (distance-based, như k-NN, hoặc một số biến thể Isolation Forest), đa cộng tuyến (multicollinearity) có thể làm méo trọng số ước lượng, khiến việc diễn giải mức độ quan trọng của từng feature trở nên không đáng tin cậy — hai feature tương quan cao có thể chia sẻ tầm quan trọng một cách tùy ý giữa chúng thay vì phản ánh đúng đóng góp thực sự. Bên cạnh đó, dư thừa làm tăng chi phí tính toán không cần thiết mà không bổ sung thông tin mới.

Một thực hành mở rộng đáng cân nhắc, kết hợp với phương pháp đo tương quan, là dùng thêm chỉ số VIF (Variance Inflation Factor) để định lượng mức độ đa cộng tuyến của từng feature so với tổ hợp tuyến tính của các feature còn lại, không chỉ xét theo từng cặp đơn lẻ — điều này giúp phát hiện trường hợp một feature tuy không tương quan cao với bất kỳ feature đơn lẻ nào nhưng lại gần như là tổ hợp tuyến tính của nhiều feature khác cộng lại.

## Bước 3: Mutual Information với Nhãn Anomaly — xếp hạng theo mức độ liên quan dự đoán

Sau hai bước lọc dựa trên đặc tính nội tại (variance) và dư thừa lẫn nhau (correlation), bước cuối cùng đánh giá trực tiếp mức độ liên quan giữa từng feature còn lại với biến mục tiêu — nhãn bất thường (anomaly label).

Mutual information (MI, thông tin tương hỗ) đo lượng thông tin mà việc biết giá trị của một feature giảm bớt được sự không chắc chắn (uncertainty) về nhãn mục tiêu. Khác với hệ số tương quan Pearson — vốn chỉ nắm bắt quan hệ tuyến tính — MI là một độ đo dựa trên lý thuyết thông tin (information theory) nên có thể nắm bắt cả các quan hệ phi tuyến phức tạp giữa feature và nhãn. Đây là lý do MI được ưu tiên hơn correlation ở bước đánh giá mức độ liên quan với target, vì trong bài toán anomaly detection, mối quan hệ giữa một feature (ví dụ STL residual) và việc một điểm có phải bất thường hay không thường mang tính ngưỡng (threshold-like) hoặc phi tuyến, chứ không hẳn là quan hệ tuyến tính đơn giản.

Quy trình thực hiện: tính điểm MI giữa từng feature và nhãn anomaly (dùng các hàm như mutual_info_classif khi nhãn là rời rạc/nhị phân, hoặc mutual_info_regression nếu dùng điểm bất thường liên tục làm proxy), sau đó xếp hạng toàn bộ feature theo điểm MI giảm dần. Từ bảng xếp hạng này, có thể chọn ra top-K feature, hoặc đặt một ngưỡng điểm MI tối thiểu để giữ lại.

Một số lưu ý quan trọng khi áp dụng MI trong bối cảnh dữ liệu anomaly detection:

- Nhãn anomaly thường rất mất cân bằng (highly imbalanced) — số điểm bất thường thường chỉ chiếm một tỷ lệ rất nhỏ so với điểm bình thường. Cần đảm bảo phương pháp ước lượng MI được dùng (thường dựa trên k-nearest-neighbor estimator) đủ ổn định với dữ liệu mất cân bằng, và nên kết hợp đánh giá chéo (cross-validation) qua nhiều lần lấy mẫu để điểm MI không bị nhiễu bởi cách chia tập dữ liệu ngẫu nhiên.
- MI là độ đo không âm và không có thang chuẩn hóa cố định (không giới hạn trong [0,1] như hệ số tương quan), nên việc so sánh điểm MI giữa các tập feature khác nhau, hoặc giữa các lần chạy khác nhau, cần thận trọng; nên ưu tiên so sánh thứ hạng tương đối (ranking) trong cùng một lần chạy hơn là so sánh giá trị tuyệt đối giữa các lần.
- Một số tài liệu chuyên sâu đề xuất kết hợp MI với hệ số tương quan trong cùng một framework đánh giá (ví dụ vừa xét MI để bắt quan hệ phi tuyến, vừa xét correlation để xác nhận hướng và độ mạnh quan hệ tuyến tính), giúp tăng độ tin cậy của việc xếp hạng so với chỉ dùng một độ đo duy nhất.

## Thứ tự thực hiện và lý do thiết kế pipeline tuần tự

Ba bộ lọc được áp dụng theo đúng thứ tự variance → correlation → mutual information, không phải ngẫu nhiên mà phản ánh nguyên tắc tăng dần về chi phí tính toán và mức độ phụ thuộc vào nhãn:

Variance threshold là bộ lọc đơn giản nhất, không cần nhãn, và loại bỏ rác hiển nhiên trước — giúp giảm số chiều đầu vào cho các bước sau, vốn tốn kém hơn về tính toán (đặc biệt correlation matrix có độ phức tạp tăng theo bình phương số lượng feature). Correlation filter chạy tiếp theo, vẫn không cần nhãn, xử lý dư thừa giữa các feature với nhau trước khi đánh giá độ liên quan với target — lý do là nếu đánh giá MI trước khi loại trùng lặp, có thể giữ lại nhiều feature gần như giống hệt nhau chỉ vì chúng đều có điểm MI cao (do mang cùng tín hiệu), gây lãng phí và làm bộ feature cuối cùng vẫn dư thừa dù từng feature riêng lẻ đều "hữu ích". Mutual information chạy sau cùng vì đây là bộ lọc duy nhất cần đến nhãn anomaly và tốn kém tính toán nhất trong ba bước, nên hợp lý nhất khi áp dụng trên tập feature đã được thu gọn đáng kể từ hai bước lọc trước.

## Output: Tập Feature Curated kèm Giải Trình

Kết quả cuối cùng của pipeline FS là một tập feature đã được tinh lọc qua ba lớp sàng lọc, đi kèm tài liệu giải trình (justification) cho từng feature được giữ lại hoặc loại bỏ. Một bộ giải trình đầy đủ nên bao gồm:

- Danh sách feature bị loại ở bước variance threshold, kèm giá trị phương sai (hoặc hệ số biến thiên) thực tế, để chứng minh đây thực sự là feature gần như hằng số chứ không phải loại bỏ nhầm do chọn ngưỡng sai.
- Danh sách các cặp feature bị loại ở bước correlation filter, kèm hệ số tương quan cụ thể và feature nào được giữ lại trong mỗi cặp cùng lý do lựa chọn (ví dụ giữ feature có điểm MI cao hơn, hoặc giữ feature dễ diễn giải hơn về mặt nghiệp vụ).
- Bảng xếp hạng đầy đủ điểm mutual information của tập feature cuối cùng, làm cơ sở định lượng cho việc một feature được xem là "quan trọng" — không chỉ dựa trên trực giác nghiệp vụ mà có bằng chứng thống kê đi kèm.
- Ghi chú về các feature được giữ lại dù điểm thống kê không nổi bật nhưng có giá trị diễn giải nghiệp vụ cao (ví dụ cross-metric ratio dễ giải thích cho đội vận hành khi gắn cờ bất thường, ngay cả khi điểm MI không phải cao nhất) — đây là điểm cần cân nhắc đánh đổi giữa tối ưu thuần thống kê và khả năng vận hành thực tế của hệ thống.

Việc ghi chép đầy đủ giải trình không chỉ phục vụ minh bạch và khả năng audit lại quyết định sau này, mà còn là cơ sở quan trọng để điều chỉnh lại pipeline khi nhãn anomaly hoặc đặc tính hệ thống thay đổi theo thời gian — bộ feature curated không nên được xem là cố định vĩnh viễn mà cần được đánh giá lại định kỳ khi phân phối dữ liệu (data distribution) hoặc hành vi hệ thống dịch chuyển.