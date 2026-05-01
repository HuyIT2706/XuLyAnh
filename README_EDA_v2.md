# Phân tích dữ liệu khám phá (EDA) - Dataset Phát hiện tổn thương da ISIC

Tài liệu này trình bày chi tiết quy trình phân tích và làm sạch dữ liệu (EDA) cho bài toán phát hiện vật thể (Object Detection) trên tập dữ liệu ISIC. Các bước thực hiện được thiết kế nhằm đảm bảo tính toàn vẹn của dữ liệu trước khi huấn luyện mô hình.

## 1. Tổng quan tập dữ liệu (Dataset Overview)

Tập dữ liệu bao gồm các hình ảnh lâm sàng về tổn thương da và nhãn tương ứng theo định dạng Pascal VOC (XML).

* **Tổng số mẫu:** 4.023 cặp (Ảnh .jpg và Nhãn .xml).
* **Kích thước ảnh chuẩn:** 640 x 640 pixels.
* **Phân chia dữ liệu (Data Splits):**
    * Huấn luyện (Train): 3.393 mẫu (84.3%)
    * Kiểm định (Valid): 420 mẫu (10.5%)
    * Kiểm tra (Test): 210 mẫu (5.2%)

## 2. Kiểm định tính toàn vẹn (Data Integrity)

* **Đồng bộ hóa:** 100% hình ảnh có tệp nhãn tương ứng. Không phát hiện tệp tin mồ côi.
* **Rò rỉ dữ liệu (Data Leakage):** Đã kiểm tra dựa trên ID ảnh gốc. Kết quả: **0 trường hợp trùng lặp** giữa các tập Train, Valid và Test.

## 3. Phân tích chất lượng nhãn (Annotation Quality)

Dữ liệu nhãn ban đầu chứa một số lỗi kỹ thuật quan trọng đã được xử lý:

### 3.1 Xử lý lỗi tọa độ Bounding Box
* **Trạng thái ban đầu:** 162 Bounding Box (BBox) vi phạm lỗi vượt khung hình (ví dụ: xmax = 641 trên ảnh 640) hoặc lỗi logic (xmin >= xmax).
* **Giải pháp xử lý:** Áp dụng kỹ thuật **Clipping**. Ép các tọa độ về giới hạn hợp lệ [0, Width] và [0, Height], đồng thời đảm bảo diện tích BBox luôn dương.
* **Trạng thái hiện tại:** **0 lỗi tọa độ**. Dữ liệu đã sẵn sàng về mặt toán học cho các framework huấn luyện.

### 3.2 Các trường hợp nghi vấn (Suspicious Cases)
Qua sàng lọc sâu, phát hiện 107 mẫu có đặc điểm hình học bất thường:
* **Mẫu quá lớn (Too Large):** 74 mẫu thuộc lớp `melanoma` chiếm trên 90% diện tích ảnh.
* **Quyết định kỹ thuật:** Giữ lại toàn bộ các mẫu này do tính chất quan trọng của lớp ác tính (Malignant) trong y tế, tránh làm trầm trọng thêm tình trạng mất cân bằng lớp.

## 4. Phân tích đặc trưng (Feature Analysis)

### 4.1 Phân phối lớp (Class Distribution)
Dữ liệu có sự **mất cân bằng nghiêm trọng**:
* Lớp áp đảo: `nevus` (lành tính).
* Lớp yếu: `melanoma`, `vascular lesion`, `squamous cell carcinoma`.
* *Lưu ý:* Cần áp dụng chiến lược Weighting hoặc Augmentation khi huấn luyện.

### 4.2 Đặc điểm hình học
* **Tỷ lệ diện tích BBox trung bình:** ~36% so với toàn bộ ảnh.
* **Kích thước trung bình:** Chiều rộng ~347px, Chiều cao ~390px.
* Các tổn thương thường nằm ở trung tâm ảnh và có kích thước lớn so với khung hình.

## 5. Kết luận và Hướng tiếp theo

Dữ liệu đã hoàn tất giai đoạn làm sạch kỹ thuật. Các bước tiếp theo bao gồm:
1.  Chuyển đổi nhãn từ XML sang định dạng YOLO (.txt).
2.  Thiết lập quy trình Tăng cường dữ liệu (Data Augmentation) tập trung vào các lớp thiểu số.
3.  Huấn luyện mô hình với hàm mất mát có trọng số (Weighted Loss).

---
*Báo cáo được thực hiện bởi Kỹ sư Data Science - Dự án ISIC Detection.*
