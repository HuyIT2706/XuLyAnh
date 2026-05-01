import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.autonotebook import tqdm

# Nhúng Custom Dataset của bạn
from dataset import SkinCancerVOCDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def get_image_classes(xml_path):
    """Hàm bổ trợ để lấy danh sách class từ file XML"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return [obj.find('name').text.strip() for obj in root.iter('object')]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cấu hình đường dẫn
    TRAIN_PATH = r"D:\xu_li_anh\btl\data\train"
    TEST_PATH  = r"D:\xu_li_anh\btl\data\valid"
    OUTPUT_DIR = r"D:\xu_li_anh\btl\checkpoin"
    LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
    MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Định nghĩa CLASSES (Hãy đảm bảo danh sách này khớp với nhãn trong XML)
    # 2. Định nghĩa CLASSES chuẩn theo dữ liệu của bạn
    CLASSES = [
        'nevus', 'melanoma', 'seborrheic keratosis', 'pigmented benign keratosis', 
        'vascular lesion', 'basal cell carcinoma', 'squamous cell carcinoma', 
        'dermatofibroma', 'actinic keratosis'
    ]
    num_classes = len(CLASSES) + 1

  # Sửa lại phần AUGMENTATION trong file train_option1.py
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # THÊM DÒNG NÀY: Nó sẽ đưa ảnh về dạng float [0, 1]
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    test_transform = A.Compose([
        # CŨNG THÊM DÒNG NÀY CHO TẬP TEST
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    train_ds = SkinCancerVOCDataset(root_dir=TRAIN_PATH, classes=CLASSES, transform=train_transform)
    test_ds  = SkinCancerVOCDataset(root_dir=TEST_PATH, classes=CLASSES, transform=test_transform)

    # ==========================================
    # PHẦN KẾT HỢP: TÍNH TRỌNG SỐ CLASS (CLASS WEIGHTS)
    # ==========================================
    print("--- Đang thống kê class để cân bằng dữ liệu ---")
    # Lấy class đầu tiên của mỗi ảnh để đại diện tính weight
    image_classes = [get_image_classes(xml_path) for xml_path in train_ds.xml_paths]
    primary_classes = [classes[0] if len(classes) > 0 else '__empty__' for classes in image_classes]
    
    class_counts = Counter(primary_classes)
    # Tính trọng số nghịch đảo với căn bậc hai để cân bằng vừa phải
    class_weights = {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}
    sample_weights = torch.as_tensor([class_weights[cls] for cls in primary_classes], dtype=torch.double)

    # Khởi tạo Sampler thay vì dùng shuffle=True
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    print(f"Thống kê tập Train: {dict(class_counts)}")
    # ==========================================

    # 4. Khởi tạo Dataloader
    train_loader = DataLoader(train_ds, batch_size=4, sampler=train_sampler, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 5. Khởi tạo Mô hình
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, num_classes)
    model.to(device)

    # 6. Huấn luyện
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    writer = SummaryWriter(log_dir=LOG_DIR)
    map_metric = MeanAveragePrecision(iou_type="bbox")
    
    NUM_EPOCHS = 100
    best_map = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", colour="cyan")
        epoch_losses = []
        
        for images, targets in train_pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_losses.append(losses.item())
            train_pbar.set_postfix({"loss": f"{np.mean(epoch_losses):.4f}"})

        # Ghi log Loss trung bình epoch
        avg_loss = np.mean(epoch_losses)
        writer.add_scalar("Loss/Train", avg_loss, epoch)

        # Đánh giá mAP
        model.eval()
        map_metric.reset()
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating", leave=False):
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                preds = [{"boxes": o["boxes"].cpu(), "scores": o["scores"].cpu(), "labels": o["labels"].cpu()} for o in outputs]
                gts = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
                map_metric.update(preds, gts)
        
        result = map_metric.compute()
        current_map = result['map'].item()
        writer.add_scalar("mAP/Val", current_map, epoch)
        
        print(f"Epoch {epoch+1} done. mAP: {current_map:.4f}")

        # Lưu model tốt nhất
        if current_map > best_map:
            best_map = current_map
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))

    writer.close()
    print("Huấn luyện hoàn tất!")