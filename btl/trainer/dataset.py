import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SkinCancerVOCDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # class mapping (0 = background)
        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}

        self.image_paths = []
        self.xml_paths = []

        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            xml_path = os.path.join(root_dir, os.path.splitext(fname)[0] + ".xml")

            if os.path.exists(xml_path):
                self.image_paths.append(os.path.join(root_dir, fname))
                self.xml_paths.append(xml_path)

        print(f"[Dataset] Loaded {len(self)} images | Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # =========================
        # 1. LOAD IMAGE
        # =========================
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)

        # =========================
        # 2. PARSE XML
        # =========================
        boxes, labels = [], []
        tree = ET.parse(self.xml_paths[idx])
        root = tree.getroot()

        for obj in root.iter("object"):
            name = obj.find("name").text.strip()

            if name not in self.class_to_idx:
                continue

            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

        # =========================
        # 3. AUGMENTATION
        # =========================
        if self.transform:
            transformed = self.transform(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed["image"]
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)
        else:
            img = ToTensorV2()(image=img)["image"]
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # =========================
        # 4. HANDLE EMPTY BOXES
        # =========================
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # =========================
        # 5. TARGET FORMAT (FRCNN)
        # =========================
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.tensor([])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }

        return img, target


# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# TEST RUN
# =========================
if __name__ == "__main__":

    TRAIN_DIR = r"D:\xu_li_anh\btl\data\train"

    CLASSES = [
        'nevus', 'melanoma', 'seborrheic keratosis',
        'pigmented benign keratosis', 'vascular lesion',
        'basal cell carcinoma', 'squamous cell carcinoma',
        'dermatofibroma', 'actinic keratosis'
    ]

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    dataset = SkinCancerVOCDataset(
        root_dir=TRAIN_DIR,
        classes=CLASSES,
        transform=train_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    images, targets = next(iter(loader))

    print("\n--- TEST ---")
    print("Batch size:", len(images))
    print("Image shape:", images[0].shape)
    print("Sample target:", targets[0])