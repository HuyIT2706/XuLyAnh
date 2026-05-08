import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Chẩn đoán Da liễu AI", layout="wide")

st.title("🔍 Hệ thống nhận diện bệnh lý về da")
st.write("Tải ảnh lên để mô hình AI tự động phân tích vùng tổn thương.")

uploaded_file = st.file_uploader("Chọn một tấm ảnh (jpg, png...)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Hiển thị ảnh gốc
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc", use_column_width=True)

    # 2. Gọi API Backend
    if st.button("Bắt đầu phân tích"):
        with st.spinner('Đang xử lý...'):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post("http://localhost:8000/predict", files=files)
                predictions = response.json()

                if not predictions:
                    st.warning("Không tìm thấy dấu hiệu bệnh lý nào với ngưỡng tin cậy hiện tại.")
                else:
                    # Vẽ Bounding Box lên ảnh kết quả
                    for pred in predictions:
                        box = pred['box']
                        label = pred['label']
                        score = pred['confidence']
                        
                        # Vẽ bằng OpenCV (lưu ý: img_array là RGB)
                        cv2.rectangle(img_array, 
                                      (int(box[0]), int(box[1])), 
                                      (int(box[2]), int(box[3])), 
                                      (255, 0, 0), 5)
                        cv2.putText(img_array, f"{label}: {score:.2f}", 
                                    (int(box[0]), int(box[1]-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    with col2:
                        st.image(img_array, caption="Kết quả dự đoán", use_column_width=True)
                        st.success(f"Tìm thấy {len(predictions)} vùng nghi ngờ.")
                        st.json(predictions) # Hiển thị dữ liệu chi tiết
            except Exception as e:
                st.error(f"Lỗi kết nối Backend: {e}")
# cd D:\xu_li_anh\btl\demo> python -m backend.main
# cd D:\xu_li_anh\btl\demo streamlit run frontend.py