import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class SkinDetector:
    def __init__(self, model_path, device):
        self.device = device
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        self.label_map = checkpoint['label_map']
        self.rev_label_map = {v: k for k, v in self.label_map.items()}
        
        # Init model
        num_classes = len(self.label_map)
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
        in_channels = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, num_classes)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

    def predict(self, image_bytes, threshold=0.4):
        # Giải mã ảnh từ bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(torch.float32) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(img_tensor)[0]

        # Lọc kết quả
        scores = prediction['scores'].cpu().numpy()
        keep = scores > threshold
        
        boxes = prediction['boxes'].cpu().numpy()[keep]
        labels = prediction['labels'].cpu().numpy()[keep]
        scores = scores[keep]

        results = []
        for box, label_id, score in zip(boxes, labels, scores):
            results.append({
                "label": self.rev_label_map.get(int(label_id), "Unknown"),
                "confidence": float(score),
                "box": box.tolist()
            })
        return results