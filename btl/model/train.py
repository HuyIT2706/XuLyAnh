import os
import torch
import torchvision
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ======================
# CONFIG
# ======================
DATASET_PATH = "bt1/Skin Cancer - ISIC-2019.v2i.voc"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

NUM_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.0001

# ======================
# DATASET CLASS (FIX)
# ======================
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root

        files = os.listdir(root)

        self.imgs = sorted([f for f in files if f.endswith(".jpg")])
        self.anns = sorted([f for f in files if f.endswith(".xml")])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        ann_path = os.path.join(self.root, self.anns[idx])

        img = Image.open(img_path).convert("RGB")
        img = torchvision.transforms.functional.to_tensor(img)

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target

    def __len__(self):
        return len(self.imgs)

# ======================
# COLLATE
# ======================
def collate_fn(batch):
    return tuple(zip(*batch))

# ======================
# LOAD DATA
# ======================
train_dataset = VOCDataset(os.path.join(DATASET_PATH, "train"))
valid_dataset = VOCDataset(os.path.join(DATASET_PATH, "valid"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ======================
# MODEL
# ======================
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model.to(DEVICE)

# ======================
# OPTIMIZER
# ======================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ======================
# TRAIN
# ======================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# ======================
# SAVE
# ======================
torch.save(model.state_dict(), "faster_rcnn_skin.pth")
print("✅ DONE TRAINING")