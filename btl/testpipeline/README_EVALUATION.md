# 📊 Báo Cáo Cấu Trúc & Quy Trình Kiểm Định Mô Hình (Model Validation Pipeline)

**Tệp thực thi chính:** `evaluate_test_set.ipynb`  
**Dự án:** Skin Disease Object Detection (Phát hiện và khoanh vùng tổn thương y tế)  
**Mô hình:** Faster R-CNN (ResNet-50 FPN)  

---

## 1. 🎯 Mục Tiêu Của Quy Trình Đánh Giá (Pipeline Objectives)

Trong bối cảnh y tế, việc chỉ dựa vào một con số độ chính xác (Accuracy/mAP) là chưa đủ để đưa AI vào ứng dụng thực tế. Pipeline đánh giá trong file `evaluate_test_set.ipynb` được thiết kế theo tiêu chuẩn công nghiệp nhằm:
1. **Kiểm định nghiêm ngặt độ tin cậy (Reliability Validation):** Đánh giá mô hình trên 100 hình ảnh hoàn toàn mới (Test Set).
2. **Cung cấp góc nhìn đa chiều (Multi-dimensional Analysis):** Kết hợp cả Đánh giá định lượng (chỉ số thống kê) và Đánh giá định tính (hình ảnh trực quan).
3. **Truy tìm nguyên nhân gốc rễ (Root Cause Analysis):** Không chỉ báo cáo số lượng lỗi, mà còn phân rã chi tiết các loại lỗi để đội ngũ AI biết chính xác cần tối ưu điều gì trong phiên bản tiếp theo.

---

## 2. ⚙️ Kiến Trúc Luồng Đánh Giá (Pipeline Workflow)

Pipeline được thực thi tự động qua các giai đoạn (Cells) logic sau:

### Giai Đoạn 1: Tiền Xử Lý & Khởi Tạo (Setup & Ingestion)
*   **Tham số hoá (Configuration):** Thiết lập `CONF_THRESH = 0.3` (ngưỡng tự tin tối thiểu) và `IoU Thresh = 0.5`.
*   **Xử lý dữ liệu (Data Loading):** Sử dụng `Albumentations` để đồng bộ hóa kích thước ảnh và chuẩn hóa tensor. Hệ thống tự động quét và loại bỏ các file nhãn bị lỗi (toạ độ bounding box không hợp lệ).
*   **Tải trọng số (Model Loading):** Đưa mô hình về trạng thái `.eval()` (tắt Dropout/cố định BatchNorm) để đảm bảo tính nhất quán của kết quả dự đoán.

### Giai Đoạn 2: Suy Luận Hàng Loạt (Batch Inference)
*   Thực hiện suy luận (predict) trên toàn bộ tập Test trong chế độ `torch.no_grad()` để tối ưu hóa bộ nhớ GPU và thời gian xử lý.
*   Toàn bộ kết quả (Predicted Boxes, Scores, Labels) và nhãn gốc (Ground Truths) được lưu trữ vào bộ nhớ đệm để phục vụ phân tích.

### Giai Đoạn 3: Báo Cáo Định Lượng (Quantitative Metrics)
*   **Tính toán mAP (Mean Average Precision):** Áp dụng thư viện `torchmetrics` chuẩn COCO. Tính toán mAP tổng thể và mAP chi tiết cho từng loại bệnh lý (Per-class mAP).
*   **Đường cong PR (Precision-Recall Curve):** Đánh giá hiệu suất của mô hình dọc theo nhiều ngưỡng Confidence khác nhau, giúp Business Team lựa chọn điểm cân bằng giữa việc "Báo động nhầm" và "Bỏ sót bệnh".
*   **F1-Score:** Tính toán tỷ lệ hài hòa giữa Precision và Recall.

### Giai Đoạn 4: Phân Rã Lỗi Chi Tiết (Detailed FP Breakdown) 🆕
*   Module này giải phẫu con số **False Positives (FP)** khổng lồ của mô hình bằng cách chia FP thành 3 nhóm nguyên nhân cốt lõi:
    *   **Lỗi định vị (Localization Error):** Mô hình nhận diện đúng loại bệnh, khoanh trúng vùng tổn thương nhưng vùng bounding box bị lệch (IoU < 0.5).
    *   **Lỗi phân loại (Classification Error):** Mô hình khoanh rất chuẩn vùng tổn thương (IoU >= 0.5) nhưng lại chẩn đoán nhầm tên bệnh.
    *   **Lỗi bối cảnh (Background Error):** Mô hình khoanh bừa vào vùng da bình thường hoặc nhiễu (thước đo, bóng mờ, lông/tóc).
*   *Output: Biểu đồ Pie Chart trực quan tỷ trọng các lỗi.*

### Giai Đoạn 5: Phân Tích Trực Quan (Visual Analytics) 🆕
*   Pipeline tự động trích xuất và hiển thị các trường hợp điển hình nhất dưới dạng Grid ảnh (2x5):
    *   **Top 10 True Positives:** Các ca mô hình dự đoán đúng với mức độ tự tin (Confidence) cao nhất.
    *   **Top 10 Worst False Positives:** Các ca mô hình sai nhưng lại "vô cùng tự tin". Đây là các ca nguy hiểm nhất, dễ gây hiểu lầm cho bác sĩ nhất.
    *   **Top 10 False Negatives:** Các vùng tổn thương bị mô hình bỏ sót hoàn toàn (được lọc dựa trên diện tích tổn thương lớn nhất).
*   *Output: 3 hình ảnh trực quan hỗ trợ bác sĩ và kỹ sư cùng "nghiệm thu" kết quả.*

### Giai Đoạn 6: Tổng Hợp Báo Cáo (Executive Summary)
*   In ra màn hình Terminal/Console một bảng báo cáo chốt hạ (Final Report) gọn gàng, chứa các key metrics (mAP, Precision, Recall) định dạng chuẩn mực để dễ dàng copy/paste vào tài liệu nghiệm thu.

---

## 3. 📂 Hướng Dẫn Đọc Hiểu File Đầu Ra (Artifacts Interpretation)

Sau khi chạy thành công Pipeline, hệ thống sẽ sinh ra các file phân tích sau:

| Tên File | Ý Nghĩa Chuyên Môn (Business Value) |
| :--- | :--- |
| `mAP_per_class.png` | Cho biết loại bệnh lý nào mô hình đang nhận diện tốt nhất, bệnh nào đang nhận diện kém nhất để có phương án tăng cường dữ liệu. |
| `pr_curve.png` | Hỗ trợ ra quyết định chọn ngưỡng (Threshold). Nếu hệ thống ưu tiên không bỏ sót bệnh (High Recall), ta phải chấp nhận hy sinh Precision dựa theo biểu đồ này. |
| `conf_dist.png` | Đo lường mức độ "phân vân" hay "tự tin" của AI trên diện rộng. |
| `fp_breakdown.png` | Định hướng kỹ thuật: Cần sửa hàm Loss (nếu Lỗi phân loại cao), cần sửa NMS (nếu Lỗi định vị cao), hay cần thêm dữ liệu ảnh nền (nếu Lỗi bối cảnh cao). |
| `tp_top10.png` | Ảnh minh họa những ca thành công nhất của mô hình. |
| `fp_top10.png` | Ảnh minh họa những ca báo động nhầm nguy hiểm nhất. |
| `fn_top10.png` | Ảnh minh họa những ca bỏ sót nghiêm trọng nhất. |

---

## 4. 🚀 Cách Khởi Chạy Pipeline

1. Đảm bảo bạn đã kích hoạt môi trường ảo (Virtual Environment) với các thư viện trong `requirements.txt`.
2. Mở file `evaluate_test_set.ipynb`.
3. Đảm bảo biến `MODEL_PATH` đang trỏ đúng đến file weights tốt nhất của bạn (VD: `best.pth`).
4. Bấm **Restart Kernel and Run All Cells**.
5. Đợi Pipeline xử lý xong 100 ảnh, toàn bộ chỉ số sẽ được in ra ở cuối file và các biểu đồ tự động lưu vào cùng thư mục `test/`.
