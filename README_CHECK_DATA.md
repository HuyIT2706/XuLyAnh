# README - Training Pipeline va Kiem dinh du lieu cho `check_data.ipynb`

## 1. Muc dich tai lieu

Tai lieu nay mo ta day du vai tro, cau truc va cac thay doi da thuc hien trong notebook `check_data.ipynb`.

Notebook nay duoc xay dung cho bai toan **phat hien ton thuong da tren anh ISIC** theo huong **Object Detection**. Du lieu dau vao gom anh `.jpg` va nhan `.xml` theo dinh dang **Pascal VOC**. Model chinh duoc su dung la **Faster R-CNN ResNet50 FPN** tu `torchvision`.

Muc tieu cua notebook:

- Doc du lieu tu folder `btl/data`.
- Tao `Dataset` phu hop voi Faster R-CNN.
- Xu ly bbox, label va cac truong target can thiet.
- Ap dung augmentation dong bo giua anh va bbox.
- Giam anh huong mat can bang lop bang `WeightedRandomSampler`.
- Train model, tinh loss va cac chi so detection nhu `mAP`, `mAP@50`, `mAP@75`.
- Ve bieu do ket qua train.
- Ho tro visualize du lieu sau augmentation va visualize prediction.

---

## 2. Pham vi notebook

File lien quan:

- Notebook chinh: `check_data.ipynb`
- Du lieu: `btl/data`
- Train split: `btl/data/train`
- Validation split: `btl/data/valid`
- Test split: `btl/data/test`
- Model checkpoint output: `checkpoints/last.pth`, `checkpoints/best.pth`
- Bieu do metric output: `training_metrics.png`

Notebook tap trung vao **training pipeline** sau khi du lieu da duoc EDA va kiem tra chat luong o cac file README/EDA truoc do.

---

## 3. Tong quan du lieu hien tai

Dataset hien tai co cau truc:

```text
btl/data/
  train/
    *.jpg
    *.xml
  valid/
    *.jpg
    *.xml
  test/
    *.jpg
    *.xml
```

Thong ke da kiem tra trong qua trinh lam viec:

```text
Train: 3393 anh + 3393 XML
Valid: 420 anh + 420 XML
Test : 210 anh + 210 XML
```

Tat ca anh va XML co kich thuoc khop nhau, chu yeu la:

```text
640 x 640 pixels
```

So class:

```text
9 class ton thuong da
+ 1 background class do Faster R-CNN tu quan ly
```

Danh sach class:

```text
actinic keratosis
basal cell carcinoma
dermatofibroma
melanoma
nevus
pigmented benign keratosis
seborrheic keratosis
squamous cell carcinoma
vascular lesion
```

---

## 4. Bai toan business/ML

### 4.1 Bai toan can giai quyet

Muc tieu la train model co kha nang:

- Nhan dien vi tri ton thuong da tren anh.
- Ve bounding box quanh vung ton thuong.
- Gan dung loai benh/nhom ton thuong cho bbox.

Day khong phai chi la bai toan classification, vi model khong chi can noi "anh thuoc class nao", ma con phai tra loi:

```text
Ton thuong nam o dau tren anh?
Khung bbox co om dung vung ton thuong khong?
Class cua bbox la gi?
```

### 4.2 Ly do dung Object Detection

Dinh dang nhan hien tai la Pascal VOC XML, moi object co:

```text
class name
xmin
ymin
xmax
ymax
```

Viec nay phu hop truc tiep voi cac model object detection nhu:

- Faster R-CNN
- YOLO
- RetinaNet
- SSD

Trong notebook nay, Faster R-CNN duoc chon vi:

- Co san trong `torchvision`.
- Ho tro target dang dictionary gom `boxes`, `labels`, `image_id`, `area`, `iscrowd`.
- Phu hop voi bai toan detection co so luong object moi anh khong qua day dac.

---

## 5. Thu vien su dung

Notebook hien tai dung cac nhom thu vien sau:

### 5.1 Xu ly file va XML

```python
os
xml.etree.ElementTree as ET
```

Dung de doc folder, ghep cap anh/XML va parse annotation Pascal VOC.

### 5.2 Xu ly anh va truc quan hoa

```python
PIL.Image
matplotlib.pyplot
matplotlib.patches
numpy
```

Dung de doc anh, chuyen doi tensor, ve bbox va visualize ket qua.

### 5.3 Deep Learning

```python
torch
torchvision
DataLoader
Dataset
fasterrcnn_resnet50_fpn
```

Dung de tao dataset, loader, model Faster R-CNN va train pipeline.

### 5.4 Augmentation

```python
albumentations
ToTensorV2
```

Dung de bien doi anh va bbox cung luc.

### 5.5 Metric

```python
torchmetrics.detection.mean_ap.MeanAveragePrecision
```

Dung de tinh:

- `mAP`
- `mAP@50`
- `mAP@75`

### 5.6 Progress bar

```python
tqdm
```

Dung de hien thi tien trinh train, validation loss va mAP.

---

## 6. Cau hinh training hien tai

Cell cau hinh hien tai gom:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 1
BATCH_SIZE = 4
LR = 0.0005

train_path = r"btl/data/train"
valid_path = r"btl/data/valid"
test_path = r"btl/data/test"
```

Trong qua trinh kiem tra thiet bi, may hien co:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 6GB
RAM: ~16GB
CUDA: available
```

Khuyen nghi:

- De test pipeline: `NUM_EPOCHS = 1`, `BATCH_SIZE = 2` hoac `4`.
- De train nghiem tuc tren RTX 3050 6GB: `NUM_EPOCHS = 10`, `BATCH_SIZE = 2`, `LR = 0.0005`.
- Neu gap loi `CUDA out of memory`: giam `BATCH_SIZE` xuong `1` hoac `2`.
- Neu loss dao dong manh, mAP khong tang: thu `LR = 0.0001`.

---

## 7. Dataset class hien tai

Notebook su dung class:

```python
SkinCancerVOCDataset
```

### 7.1 Cac task class dataset da giai quyet

Class nay da giai quyet cac viec quan trong:

#### 1. Tim cap anh va XML

Dataset duyet folder dau vao, chi lay nhung anh co file XML cung ten.

Vi du:

```text
abc.jpg
abc.xml
```

Neu anh khong co XML, anh do khong duoc dua vao dataset.

#### 2. Doc anh

Anh duoc doc bang PIL va convert sang RGB:

```python
Image.open(...).convert("RGB")
```

#### 3. Parse XML Pascal VOC

Tu file XML, dataset doc:

```text
object/name
bndbox/xmin
bndbox/ymin
bndbox/xmax
bndbox/ymax
```

#### 4. Tao mapping class sang id

Class name duoc map thanh id bat dau tu `1`.

Ly do:

```text
0 duoc Faster R-CNN xem la background
```

#### 5. Clip bounding box

Neu bbox vuot bien anh, toa do duoc gioi han lai trong:

```text
x: [0, width]
y: [0, height]
```

Viec nay giup tranh loi training khi bbox nam ngoai anh.

#### 6. Bo bbox khong hop le

Dataset bo cac bbox co dien tich khong hop le:

```text
xmax <= xmin
ymax <= ymin
```

#### 7. Chuyen du lieu sang tensor

Target duoc chuyen sang tensor:

```python
boxes: torch.float32, shape (N, 4)
labels: torch.int64, shape (N,)
```

#### 8. Ho tro Albumentations

Dataset goi Albumentations theo dung format:

```python
transformed = self.transform(
    image=np.array(img),
    bboxes=boxes.tolist(),
    labels=labels.tolist()
)
```

Nho do anh, bbox va label duoc bien doi dong bo.

#### 9. Tao target dung format Faster R-CNN

Target hien tai gom:

```python
target = {
    "boxes": boxes,
    "labels": labels,
    "image_id": image_id,
    "area": area,
    "iscrowd": iscrowd
}
```

Cac truong nay can thiet cho Faster R-CNN va cac metric theo chuan COCO/mAP.

---

## 8. Transform va augmentation

### 8.1 Van de ban dau

Ban dau notebook dung `torchvision.transforms`:

```python
T.RandomHorizontalFlip
T.RandomVerticalFlip
T.ColorJitter
T.ToTensor
T.Normalize
```

Van de:

```text
torchvision.transforms kieu cu chi bien doi anh,
khong tu dong bien doi bbox.
```

Neu anh bi lat ngang/doc hoac xoay, bbox van giu toa do cu. Day la loi nguy hiem vi:

```text
Anh da thay doi vi tri object
nhung label/bbox van nam o vi tri cu
=> model hoc sai am tham
```

### 8.2 Giai phap da trien khai

Da thay transform bang Albumentations:

```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, border_mode=0, p=0.3),
    A.RandomBrightnessContrast(...),
    A.HueSaturationValue(...),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
    min_visibility=0.2
))
```

Y nghia:

- `HorizontalFlip`: lat ngang anh va bbox.
- `VerticalFlip`: lat doc anh va bbox.
- `Rotate`: xoay anh va bbox.
- `RandomBrightnessContrast`: thay doi do sang va tuong phan.
- `HueSaturationValue`: thay doi sac do/mau sac nhe.
- `A.ToFloat(max_value=255.0)`: dua pixel ve khoang `[0, 1]`.
- `ToTensorV2`: chuyen anh sang tensor PyTorch.

### 8.3 Transform cho validation/test

Validation va test khong nen augmentation.

Transform hien tai:

```python
test_transform = A.Compose([
    A.NoOp(p=1.0),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"]
))
```

Y nghia:

- Khong lam thay doi du lieu validation/test.
- Chi chuyen anh sang float tensor.

---

## 9. DataLoader va xu ly imbalance

### 9.1 Van de imbalance

Train set co phan bo object khong deu.

Thong ke train theo object:

```text
pigmented benign keratosis: 1144
melanoma: 533
basal cell carcinoma: 517
nevus: 425
squamous cell carcinoma: 310
vascular lesion: 228
actinic keratosis: 184
dermatofibroma: 131
seborrheic keratosis: 122
```

Class nhieu nhat gap class it nhat khoang:

```text
1144 / 122 ~= 9.4 lan
```

Neu train truc tiep, model co xu huong hoc tot class nhieu mau va yeu hon o class hiem.

### 9.2 Giai phap da trien khai

Notebook dung:

```python
torch.utils.data.WeightedRandomSampler
```

Trong so moi class:

```python
1.0 / sqrt(count)
```

Ly do khong dung `1 / count`:

- `1 / count` can bang qua manh.
- Class hiem co the bi lap lai qua nhieu.
- De gay overfit class it mau.

`1 / sqrt(count)` la can bang mem, phu hop hon voi dataset hien tai.

### 9.3 Loader hien tai

Notebook hien co:

```python
train_loader
valid_loader
test_loader
```

Trong do:

- `train_loader`: dung `WeightedRandomSampler`.
- `valid_loader`: `shuffle=False`.
- `test_loader`: `shuffle=False`.

Validation va test khong nen dung sampler vi can danh gia tren phan bo that cua du lieu.

---

## 10. Visualize du lieu sau augmentation

Notebook da them ham:

```python
show_augmented_samples(train_dataset, idx_to_class, num_samples=4)
```

Muc dich:

- Kiem tra anh sau augmentation.
- Xem bbox co con nam dung vung ton thuong khong.
- Phat hien loi transform/bbox som truoc khi train lau.

Day la buoc quan trong voi object detection vi chi can bbox sai sau augmentation, model se hoc sai.

Khuyen nghi khi chay:

- Chay cell visualize truoc khi train.
- Neu thay bbox lech, mat bbox bat thuong, hoac anh bien dang qua manh, can giam augmentation.

---

## 11. Model

Model hien tai:

```python
fasterrcnn_resnet50_fpn(weights="DEFAULT")
```

### 11.1 Thay head phan loai

Model pretrained mac dinh khong co so class giong dataset ISIC, nen notebook thay head:

```python
FastRCNNPredictor(in_features, num_classes)
```

Trong do:

```python
num_classes = 9 + 1 background = 10
```

### 11.2 Dong bang backbone

Notebook dong bang backbone:

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

Ly do:

- Giam VRAM.
- Train nhanh hon.
- Phu hop voi may RTX 3050 Laptop 6GB.
- Giai doan dau chi fine-tune detection head.

Luu y:

- Neu sau nay muon cai thien ket qua, co the mo khoa mot phan backbone de fine-tune sau khi head da on dinh.

---

## 12. Optimizer

Optimizer hien tai:

```python
optimizer = torch.optim.AdamW(
    params,
    lr=LR,
    weight_decay=0.003
)
```

Trong do `params` chi gom cac tham so co:

```python
requires_grad = True
```

Nghia la chi train cac phan chua bi dong bang.

---

## 13. Training loop

Training loop hien tai gom 3 pha moi epoch:

### 13.1 Train loss

Model duoc dat o:

```python
model.train()
```

Sau do tinh loss:

```python
loss_dict = model(images, targets)
losses = torch.stack(list(loss_dict.values())).sum()
```

Loss cua Faster R-CNN thuong gom:

```text
loss_classifier
loss_box_reg
loss_objectness
loss_rpn_box_reg
```

Tong loss duoc dung de backprop.

### 13.2 Validation loss

Notebook tinh validation loss tren `valid_loader`.

Luu y rieng voi Faster R-CNN:

- Khi `model.eval()`, model tra prediction.
- Khi `model.train()` va co `targets`, model tra loss.

Vi vay validation loss duoc tinh bang:

```python
model.train()
with torch.no_grad():
    loss_dict = model(images, targets)
```

Cach nay giup tinh loss ma khong update gradient.

### 13.3 mAP metrics

Sau khi tinh validation loss, notebook chuyen model sang:

```python
model.eval()
```

Va tinh prediction tren `valid_loader`:

```python
outputs = model(images)
```

Sau do cap nhat metric:

```python
MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    backend="faster_coco_eval"
)
```

Cac chi so duoc tinh:

```text
mAP
mAP@50
mAP@75
```

### 13.4 Log sau moi epoch

Notebook in:

```text
Train Loss
Val Loss
mAP
mAP@50
mAP@75
```

Vi du ket qua da chay thu 1 epoch:

```text
Train Loss: 0.3053
Val Loss: 0.2568
mAP: 0.1375
mAP@50: 0.3110
mAP@75: 0.0963
```

Ket qua nay moi chi la sau 1 epoch, chua phai ket qua cuoi cung.

---

## 14. Y nghia cac chi so danh gia

### 14.1 Train Loss

Cho biet model hoc tren train set co on khong.

Neu train loss giam:

```text
model dang hoc duoc tu train data
```

Nhung train loss thap khong dam bao model generalize tot.

### 14.2 Validation Loss

Cho biet model co phu hop voi valid set khong.

Neu train loss giam nhung validation loss tang:

```text
co kha nang overfitting
```

### 14.3 mAP

Metric tong quat cho object detection.

mAP danh gia dong thoi:

- bbox co dung vi tri khong.
- class co dung khong.
- score co hop ly khong.

### 14.4 mAP@50

Tinh AP tai IoU threshold 0.50.

Y nghia:

```text
Bbox chi can khop tuong doi voi bbox that.
```

Neu `mAP@50` cao nhung `mAP@75` thap:

```text
Model tim dung vung ton thuong, nhung bbox chua om sat.
```

### 14.5 mAP@75

Tinh AP tai IoU threshold 0.75.

Y nghia:

```text
Bbox phai khop chat hon voi ground truth.
```

Chi so nay kho hon `mAP@50`.

---

## 15. Luu checkpoint

Notebook luu:

```text
checkpoints/last.pth
checkpoints/best.pth
```

Trong do:

- `last.pth`: model o epoch gan nhat.
- `best.pth`: model co validation `mAP` tot nhat.

Ly do chon best theo `mAP`:

```text
Object detection nen uu tien metric detection hon la chi nhin loss.
```

---

## 16. Ve bieu do training

Notebook da them cell ve 2 nhom bieu do:

### 16.1 Loss curve

Gom:

```text
Train Loss
Val Loss
```

Dung de theo doi qua trinh hoc va phat hien overfitting.

### 16.2 Detection metrics curve

Gom:

```text
mAP
mAP@50
mAP@75
```

Dung de danh gia chat luong detection tren validation set.

Output:

```text
training_metrics.png
```

---

## 17. Predict va visualize prediction

Notebook co ham:

```python
predict_image(...)
visualize_prediction(...)
```

Muc dich:

- Chay model tren mot anh test.
- Ve bbox prediction len anh.
- Hien thi class va confidence score.

Luu y hien tai:

Cell chon anh test dang dung duong dan co dinh:

```python
image_path = r"btl/data\test\ISIC_0000021_downsampled_jpg.rf.8c49330bcfaf4afb3fd31b5b5f79ec11.jpg"
```

Duong dan nay da tung gay `FileNotFoundError` vi file khong ton tai trong folder hien tai.

Khuyen nghi thay bang:

```python
image_path = test_dataset.image_paths[0]
```

Hoac lay ngau nhien:

```python
image_path = random.choice(test_dataset.image_paths)
```

Day la diem can xu ly tiep de notebook chay tu dau den cuoi khong loi.

---

## 18. Cac thay doi da thuc hien tu dau toi hien tai

### 18.1 Doc va so sanh 2 dataset class trong notebook

Da doc `check_data.ipynb` va phat hien co 2 version dataset class.

Ket luan:

- Class 1 phu hop hon de phat trien tiep.
- Class 1 doc cap anh/XML an toan hon.
- Class 1 co xu ly clip bbox va bo bbox loi.
- Class 1 co logic scale bbox neu transform doi kich thuoc anh.

Van de phat hien:

- Neu dung `torchvision.transforms` flip anh thi bbox khong flip theo.
- Class 2 giu bbox khong doi neu anh bi resize, dan den bbox sai.

### 18.2 Chon Class 1 lam dataset chinh

Da giai thich va quyet dinh dung Class 1.

Class 1 da duoc cai thien:

- Them Albumentations.
- Them target fields `image_id`, `area`, `iscrowd`.
- Dam bao bbox rong co shape `(0, 4)`.
- Comment duoc lam sach de de doc.

### 18.3 Thay transform bang Albumentations

Da thay cell transform cu bang Albumentations de:

- Lat ngang anh va bbox dong bo.
- Lat doc anh va bbox dong bo.
- Xoay anh va bbox dong bo.
- Thay doi mau/sang/tuong phan.
- Chuyen anh ve float tensor.

### 18.4 Cai dat dependency

Da them:

```text
albumentations==1.4.18
opencv-python-headless==4.9.0.80
torchmetrics==1.3.2
faster-coco-eval==1.6.8
```

Ghi chu:

- `pycocotools` khong build duoc tren Windows do thieu Microsoft C++ Build Tools.
- Da chuyen sang backend `faster_coco_eval`.

### 18.5 Tach ro valid/test

Ban dau notebook dung `test_path` tro vao valid.

Da tach ro:

```python
valid_path = r"btl/data/valid"
test_path = r"btl/data/test"
```

Va tao rieng:

```python
valid_loader
test_loader
```

### 18.6 Them WeightedRandomSampler

Da them sampler cho train set de giam imbalance.

Logic:

```python
class_weights = 1 / sqrt(class_count)
sample_weights = class_weights[class_of_image]
```

`valid_loader` va `test_loader` giu `shuffle=False`.

### 18.7 Them visualize sau augmentation

Da them:

```python
show_augmented_samples(...)
```

Muc dich:

- Kiem tra bbox sau augmentation.
- Ho tro giai thich pipeline cho nguoi xem.

### 18.8 Them mAP metrics vao train loop

Da them:

- `mAP`
- `mAP@50`
- `mAP@75`

Tinh sau moi epoch tren `valid_loader`.

### 18.9 Doi cach luu best model

Ban dau best model dua vao `val_loss`.

Hien tai:

```text
best model duoc luu theo validation mAP cao nhat
```

Day la chuan hon cho object detection.

### 18.10 Them bieu do day du metric

Da them cell ve:

- Train Loss
- Val Loss
- mAP
- mAP@50
- mAP@75

Output:

```text
training_metrics.png
```

### 18.11 Don dep notebook

Da don:

- Import trung lap.
- Import khong dung nhu `pandas`, `seaborn`.
- Comment loi font trong dataset class.
- Cell `print(num_classes)` rieng le.
- Cell debug `loss_dict.items()` khong can thiet.
- Cell model khong con in nguyen kien truc Faster R-CNN dai.

---

## 19. Trang thai hien tai cua notebook

Notebook hien tai da co pipeline kha day du:

```text
Data loading -> Augmentation -> Sampling -> Model -> Training -> Validation metrics -> Plotting -> Prediction visualization
```

Da kiem tra:

- Code cells parse OK.
- Loader train/valid/test tao batch dung format.
- `MeanAveragePrecision` voi backend `faster_coco_eval` chay duoc.
- Anh dau vao sau transform la `torch.float32`, khoang `[0, 1]`.

Ket qua smoke test loader:

```text
train_loader: batch 4, image float32, target du keys
valid_loader: batch 1, image float32, target du keys
test_loader : batch 1, image float32, target du keys
```

---

## 20. Cac viec nen lam tiep truoc khi trinh bay/nop

### 20.1 Clear output notebook

Notebook van co output cu, gom ca loi `FileNotFoundError` o cell predict cuoi.

Nen lam:

```text
Kernel -> Restart & Clear Output
```

Sau do chay lai tung phan theo thu tu.

### 20.2 Sua cell predict cuoi

Thay duong dan co dinh bang:

```python
image_path = test_dataset.image_paths[0]
```

Hoac:

```python
image_path = random.choice(test_dataset.image_paths)
```

### 20.3 Chay thu voi 1 epoch

Truoc khi train dai:

```python
NUM_EPOCHS = 1
BATCH_SIZE = 2 hoac 4
```

Muc tieu:

- Dam bao train loop khong loi.
- Dam bao mAP tinh duoc.
- Dam bao checkpoint duoc luu.
- Dam bao bieu do metric ve duoc.

### 20.4 Train nghiem tuc

Sau khi smoke test on dinh:

```python
NUM_EPOCHS = 10
BATCH_SIZE = 2
LR = 0.0005
```

Neu VRAM on dinh, co the thu:

```python
BATCH_SIZE = 4
```

### 20.5 Theo doi metric

Can quan sat:

```text
Train Loss co giam khong?
Val Loss co tang bat thuong khong?
mAP co tang qua epoch khong?
mAP@50 va mAP@75 chenh nhau nhieu khong?
```

Neu:

```text
mAP@50 cao nhung mAP@75 thap
```

thi model da tim duoc vung ton thuong nhung bbox chua chinh xac.

### 20.6 Them per-class evaluation neu can

Vi du lieu imbalance, sau nay nen them:

```python
MeanAveragePrecision(class_metrics=True)
```

De xem class nao dang yeu.

### 20.7 Xem xet fixed class mapping

Hien dataset tu tao `class_to_idx` theo folder.

De production/san sang nop hon, nen tao mapping co dinh:

```python
CLASS_TO_IDX = {
    "actinic keratosis": 1,
    "basal cell carcinoma": 2,
    ...
}
```

Va dung chung cho train/valid/test.

---

## 21. Rủi ro va luu y ky thuat

### 21.1 Anh rong bbox

Sampler thong ke co:

```text
__empty__: 12
```

Nghia la co 12 anh train khong co object hop le sau parse/filter.

Nen xem xet:

- Kiem tra lai XML.
- Loai khoi train neu do la loi annotation.
- Hoac giu lai neu muon train negative samples.

### 21.2 Validation loss voi Faster R-CNN

Cach tinh validation loss hien tai dung:

```python
model.train()
with torch.no_grad()
```

Day la cach ky thuat de Faster R-CNN tra loss ma khong update gradient.

Khi tinh mAP, notebook da dung dung:

```python
model.eval()
outputs = model(images)
```

### 21.3 Normalization

Notebook hien khong dung `A.Normalize`, chi dung:

```python
A.ToFloat(max_value=255.0)
```

Ly do:

- Faster R-CNN cua `torchvision` da co normalize noi bo trong `GeneralizedRCNNTransform`.
- Neu normalize o dataset nua co the lam anh bi normalize hai lan.

---

## 22. Huong dan chay notebook

Thu tu chay khuyen nghi:

1. Chay import.
2. Chay config.
3. Chay transform.
4. Chay dataset class.
5. Chay dataset/loader.
6. Chay visualize augmentation.
7. Chay model.
8. Chay count parameters.
9. Chay optimizer.
10. Chay train loop.
11. Chay plot metrics.
12. Chay prediction visualization.

Neu chi muon kiem tra nhanh:

```python
NUM_EPOCHS = 1
BATCH_SIZE = 2
```

Neu muon train nghiem tuc tren RTX 3050 6GB:

```python
NUM_EPOCHS = 10
BATCH_SIZE = 2
LR = 0.0005
```

---

## 23. Ket luan

`check_data.ipynb` hien da duoc nang cap tu notebook kiem tra dataset thanh mot pipeline training kha hoan chinh cho object detection:

- Dataset class da doc Pascal VOC XML dung format.
- BBox duoc clip va validate.
- Albumentations da xu ly dung van de anh/bbox bien doi dong bo.
- Data imbalance duoc giam bang WeightedRandomSampler.
- Train loop co loss va metric detection chuan.
- Best checkpoint duoc luu theo validation mAP.
- Co visualize du lieu sau augmentation va visualize prediction.

Notebook hien da gan san sang de trinh bay/training, voi viec can lam tiep quan trong nhat la:

```text
Clear output
Sua cell predict cuoi khong dung duong dan co dinh
Chay lai 1 epoch de xac nhan pipeline
Train day du va bao cao metric
```

