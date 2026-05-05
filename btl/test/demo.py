import streamlit as st
import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'nevus', 'melanoma', 'seborrheic keratosis', 'pigmented benign keratosis', 
    'vascular lesion', 'basal cell carcinoma', 'squamous cell carcinoma', 
    'dermatofibroma', 'actinic keratosis'
]

MODEL_PATH = r"D:\xu_li_anh\btl\checkpoin\models\best_model.pth"

# ================= FUNCTIONS =================
@st.cache_resource
def load_model():
    num_classes = len(CLASSES) + 1
    # Khởi tạo kiến trúc model
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load trọng số đã train
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def predict(image, model, threshold, iou_threshold=0.3):
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    # Tiền xử lý ảnh
    image_rgb = np.array(image)
    aug = transform(image=image_rgb)
    image_tensor = aug["image"].to(DEVICE)

    with torch.no_grad():
        # Model mong muốn đầu vào là list các tensor
        outputs = model([image_tensor])

    # Lấy kết quả
    boxes = outputs[0]["boxes"]
    scores = outputs[0]["scores"]
    labels = outputs[0]["labels"]

    # 1. Áp dụng NMS để lọc bỏ các box chồng chéo
    keep = nms(boxes, scores, iou_threshold)
    
    boxes = boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()

    # 2. Vẽ kết quả lên ảnh
    draw_img = image_rgb.copy()
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            class_name = CLASSES[label - 1]
            
            # Vẽ Bounding Box
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Tạo nhãn text
            label_txt = f"{class_name}: {score:.2f}"
            
            # Vẽ nền cho text để dễ đọc hơn
            (w, h), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(draw_img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            
            # Ghi chữ lên nền vừa vẽ
            cv2.putText(draw_img, label_txt, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
    return draw_img

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Skin Disease AI", layout="wide")

st.title("🩺 Hệ thống chẩn đoán bệnh lý da liễu qua hình ảnh")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Cấu hình")
conf_threshold = st.sidebar.slider("Độ tự tin tối thiểu (Confidence)", 0.1, 1.0, 0.5)
iou_threshold = st.sidebar.slider("Lọc chồng chéo (IoU NMS)", 0.1, 1.0, 0.3)

# Load model
with st.spinner('Đang khởi tạo mô hình AI...'):
    model = load_model()

# Giao diện chính
uploaded_file = st.file_uploader("📤 Tải ảnh nội soi da (Dermoscopy)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Ảnh đầu vào")
        st.image(image, use_container_width=True)

    if st.button("🔍 Bắt đầu phân tích"):
        with st.spinner('AI đang phân tích vùng tổn thương...'):
            result_img = predict(image, model, conf_threshold, iou_threshold)
            
            with col2:
                st.subheader("🖼️ Kết quả phân tích")
                st.image(result_img, use_container_width=True)
                st.success("Phân tích hoàn tất!")
                
                # Thêm chú thích bên dưới kết quả
else:
    st.info("Vui lòng tải ảnh lên để bắt đầu.")