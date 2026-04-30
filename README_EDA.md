# Khai phá Dữ liệu (EDA) - Tập dữ liệu nhận diện khối u (ISIC)

File `EDA data.ipynb` chứa các bước tiến hành Phân tích Dữ liệu Khám phá (Exploratory Data Analysis - EDA) cho tập dữ liệu hình ảnh bệnh lý và nhãn tương ứng (định dạng Pascal VOC XML). Mục đích của notebook này là rà soát, kiểm định chất lượng, và phát hiện các lỗi trong dữ liệu trước khi đưa vào quá trình huấn luyện mô hình (Training).

## 🛠️ Các thư viện sử dụng
- `os`, `collections`: Thao tác với hệ thống file, duyệt thư mục và lưu trữ dữ liệu tần suất.
- `cv2` (OpenCV): Đọc, xử lý ảnh và vẽ các Bounding Box.
- `matplotlib.pyplot`: Trực quan hóa hình ảnh và biểu đồ.
- `xml.etree.ElementTree` (ET): Đọc và phân tích cú pháp các file nhãn `.xml`.
- `numpy`: Xử lý tính toán ma trận, mảng dữ liệu.

---

## 🔍 Chi tiết các hàm và bước thực hiện trong Code

### 1. In cấu trúc thư mục dữ liệu (`print_tree`)
- **Mục đích:** Kiểm tra xem dữ liệu đã được tổ chức đúng cấu trúc hay chưa (ví dụ: chia thành các thư mục `train/`, `valid/`, `test/`).
- **Cách hoạt động:** Dùng `os.walk` lặp qua các thư mục con và in ra theo dạng cây (tree). Có giới hạn độ sâu (max_depth) và số file hiển thị để tránh bị trôi log.

### 2. Kiểm tra Rò rỉ dữ liệu (`check_leakage`)
- **Mục đích:** Đảm bảo tính độc lập của 3 tập dữ liệu (Training, Validation, Testing). Việc một ảnh xuất hiện ở cả tập Train và tập Test sẽ gây ra hiện tượng rò rỉ dữ liệu (Data Leakage), làm cho kết quả đánh giá mô hình không còn khách quan.
- **Cách hoạt động:** Lấy tên gốc của ảnh (bỏ đuôi định dạng và phần dư thừa `.rf...`) đưa vào một `defaultdict(set)`. Sau đó kiểm tra xem có tên ảnh nào thuộc về từ 2 tập (split) trở lên hay không.

### 3. Kiểm tra tính đồng bộ Ảnh và Nhãn (`check_dataset`)
- **Mục đích:** Đảm bảo mỗi một file ảnh (`.jpg`, `.png`) đều có chính xác một file nhãn `.xml` đi kèm và ngược lại.
- **Cách hoạt động:** Duyệt qua từng tập dữ liệu, tách tên file và đuôi file để chia vào 2 tập hợp (set) là `images` và `xmls`. So sánh hai tập hợp này bằng phép trừ (`images - xmls` và ngược lại) để tìm ra các file bị dư hoặc thiếu.

### 4. Kiểm tra tính hợp lệ của Bounding Box (`check_bbox`)
- **Mục đích:** Đảm bảo các tọa độ gán nhãn là hợp lệ về mặt toán học và vật lý (không vượt quá giới hạn của tấm ảnh).
- **Cách hoạt động:** Mở các file XML, đọc kích thước tấm ảnh (`width`, `height`) và các toạ độ hộp giới hạn (`xmin`, `ymin`, `xmax`, `ymax`). BBox sẽ bị đánh dấu là lỗi nếu:
  - Tọa độ max nhỏ hơn tọa độ min (`xmax <= xmin` hoặc `ymax <= ymin`).
  - Tọa độ vượt ra ngoài khung hình (`xmin < 0`, `ymin < 0` hoặc `xmax > width`, `ymax > height`).

### 5. In thông tin BBox bị lỗi (`show_invalid_samples`)
- **Mục đích:** Báo cáo chi tiết các file có chứa BBox lỗi để người dùng dễ dàng theo dõi.
- **Cách hoạt động:** In ra đường dẫn của file `.xml`, kích thước của ảnh và các toạ độ đang bị sai lệch.

### 6. Vẽ trực quan các BBox bị lỗi (`visualize_invalid`)
- **Mục đích:** Xem xét trực tiếp bằng mắt xem các BBox lỗi đó nằm ở đâu trên bức ảnh, mức độ sai lệch có nghiêm trọng hay không (Ví dụ: hộp giới hạn bị tràn ra khỏi ảnh 1 pixel).
- **Cách hoạt động:** Kết hợp `ET` để lấy tọa độ lỗi, dùng `cv2.rectangle` vẽ khung đỏ lên ảnh và `plt.imshow` để hiển thị trên notebook.

### 7. Phân tích Tỷ lệ diện tích Bounding Box (`bbox_area_ratio`)
- **Mục đích:** Đánh giá xem vùng tổn thương/khối u chiếm bao nhiêu phần trăm không gian của toàn bộ bức ảnh. 
- **Cách hoạt động:** Tính diện tích BBox `(xmax - xmin) * (ymax - ymin)` chia cho tổng diện tích ảnh `width * height`. Giá trị trung bình phản ánh kích thước tương đối của đối tượng cần nhận diện, giúp quyết định cấu hình của các Anchor Boxes khi train model.

### 8. Vẽ trực quan dữ liệu ngẫu nhiên (`visualize_samples`)
- **Mục đích:** Trực quan hóa tổng thể để xác nhận một lần cuối rằng ground-truth (nhãn) khớp với hình ảnh.
- **Cách hoạt động:** 
  - Đọc toàn bộ các class hiện có trong tập dữ liệu.
  - Sinh ngẫu nhiên mã màu (RGB) (`get_color_map`) cho từng class để dễ phân biệt.
  - Lấy ngẫu nhiên `num_samples` file XML để hiển thị.
  - Dùng `cv2` vẽ khung và gắn text (tên class) lên hình, sau đó hiển thị dạng lưới (grid) bằng `matplotlib`.

---

## 📝 Kết luận từ quá trình EDA
- **Về rò rỉ dữ liệu:** Tập dữ liệu hoàn toàn sạch, 0 ca rò rỉ.
- **Về tính đồng bộ:** 100% ảnh đều có nhãn đi kèm. (Tổng 4023 ảnh và 4023 file XML).
- **Về nhãn BBox:** Tồn tại một số lượng nhỏ BBox vượt quá kích thước ảnh (VD: tọa độ bằng 641 trên ảnh kích thước 640). Có thể tiến hành tiền xử lý bổ sung (cắt toạ độ - clip bounding box) trước khi train mô hình.

---

## 🔬 Các Phân tích Bổ sung (`EDA_supplementary.py`)

File `EDA_supplementary.py` chứa 4 phân tích bổ sung quan trọng chưa có trong notebook gốc.

### A. Phân tích Lớp (`class_analysis`)
- **Mục đích:** Đếm số lượng BBox của từng class (`Benign`, `Malignant`) trong từng tập train/valid/test để phát hiện hiện tượng **mất cân bằng lớp (Class Imbalance)**.
- **Cách hoạt động:** Dùng `collections.Counter` để đếm tần suất xuất hiện của từng class trong từng file XML. Vẽ biểu đồ cột nhóm (Grouped Bar Chart) so sánh phân bố class giữa 3 tập. In thêm tỷ lệ phần trăm để đánh giá mức độ mất cân bằng.
- **Ý nghĩa:** Nếu tỷ lệ giữa các class chênh lệch lớn (VD: 80% vs 20%), cần áp dụng kỹ thuật xử lý như **oversampling**, **undersampling** hoặc **class weighting** khi train model.

### B. Phân tích Hình học Bounding Box (`bbox_geometry_analysis`)
- **Mục đích:** Hiểu sâu hơn về kích thước và hình dạng của các BBox.
- **Cách hoạt động:** Thu thập chiều rộng (W), chiều cao (H), và tỷ lệ khung hình (Aspect Ratio = W/H) của toàn bộ BBox. Vẽ 3 loại biểu đồ:
  - **Histogram Width/Height:** Phân bố kích thước tuyệt đối.
  - **Histogram Aspect Ratio:** Nếu Ratio ≈ 1 → hình vuông; > 1 → rộng hơn cao; < 1 → cao hơn rộng.
  - **Scatter Width vs Height:** Quan sát hình dạng phân cụm của BBox → có thể dùng cho **K-Means Anchor Boxes** cho YOLO.
- **Ý nghĩa:** Hỗ trợ quyết định cấu hình **Anchor Boxes** cho các model họ YOLO, đặc biệt khi các BBox có tỷ lệ khung hình đặc thù.

### C. Phân tích Ảnh (`image_analysis`)
- **Mục đích:** Kiểm tra tính đồng nhất về kích thước ảnh và phân tích độ sáng (brightness) của toàn bộ tập dữ liệu.
- **Cách hoạt động:**
  - Đọc từng ảnh bằng `cv2`, kiểm tra kích thước `(width, height)` và dùng `Counter` để thống kê.
  - Tính **mean pixel value** của mỗi ảnh như một đại lượng đại diện cho độ sáng.
  - Vẽ histogram phân bố độ sáng với đường kẻ chỉ giá trị trung bình.
- **Ý nghĩa:**
  - Nếu tất cả ảnh cùng kích thước (VD: 640×640) → không cần resize khi train.
  - Phân bố độ sáng lệch về 1 phía → nên áp dụng **augmentation** (brightness jitter) để model tổng quát tốt hơn.

### D. Phân tích Số lượng BBox mỗi Ảnh (`bbox_count_per_image`)
- **Mục đích:** Hiểu được mỗi ảnh trong dataset có bao nhiêu vật thể cần nhận diện.
- **Cách hoạt động:** Đếm số lượng thẻ `<object>` trong mỗi file XML, tổng hợp phân bố và liệt kê các ảnh không có BBox nào (nếu có).
- **Ý nghĩa:**
  - Nếu hầu hết ảnh chỉ có **1 BBox** → đây là bài toán phát hiện vật thể đơn giản, có thể cân nhắc dùng classification thay vì detection.
  - Nếu có **ảnh không có BBox** → cần loại bỏ hoặc xử lý riêng trước khi train.
  - Phân bố nhiều BBox/ảnh → model cần xử lý được dense detection.

