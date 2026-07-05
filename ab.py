# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3.12 (ROCm)
#     language: python
#     name: py312-rocm
# ---

# %%
import pandas as pd

# %%
import torch
print(torch.__version__)           # 应包含 +rocm 字样
print(torch.cuda.is_available())   # ROCm 兼容 CUDA API，应返回 True
print(torch.cuda.device_count())   # 应显示你的 GPU 数量
print(torch.cuda.get_device_name(0))  # 显示 GPU 型号

# %%
df = pd.read_csv("F:/python/nih/BBox_List_2017.csv")
print(df.head())

# %% [markdown]
# finding lesion label

# %% jupyter={"outputs_hidden": true}
print(df.columns)
print(df['Finding Label'].value_counts())
####print(df['Finding Label'].unique())

# %%
nodule_df = df[df['Finding Label'] == 'Nodule']
print(nodule_df.head())
print(len(nodule_df))

# %% [markdown]
# 第九步：读取第一行数据

# %%
row = df.iloc[0]  ##above code, this will get the first row of the dataframe. You can change the index to get different rows.
print(row)

# %%
import os

img_dir = "F:/python/nih/images_001/images/"
files = os.listdir(img_dir)

print("total images:", len(files))


# %%
# only keep the rows in df where the 'Image Index' is in the list of files
df_subset = df[df['Image Index'].isin(files)]

print("CSV total images:", len(df))
print("images_001 available:", len(df_subset))

# %% [markdown]
#

# %%
print(df.columns)

# %% [markdown]
# 重命名列

# %%
df = df.rename(columns={
    'Bbox [x': 'Bbox_x',
    'y': 'Bbox_y',
    'w': 'Bbox_w',
    'h]': 'Bbox_h'
})

print(df.columns)

# %%
df_subset = df[df['Image Index'].isin(files)]
print("available lines:", len(df_subset))
row = df_subset.iloc[0]
print(row)

# %% [markdown]
# 打开图片

# %%
from PIL import Image
import os

img_name = row['Image Index']
img_path = os.path.join(img_dir, img_name)

print("imgpath:", img_path)

image = Image.open(img_path)
image

# %% [markdown]
# 画出 bounding box

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# get bbox
x = row['Bbox_x']
y = row['Bbox_y']
w = row['Bbox_w']
h = row['Bbox_h']

fig, ax = plt.subplots(1, figsize=(6,6))
ax.imshow(image, cmap='gray')

rect = patches.Rectangle((x, y), w, h,
                         linewidth=2,
                         edgecolor='red',
                         facecolor='none')

ax.add_patch(rect)
plt.show()

# %% [markdown]
# PyTorch Dataset。
#
# 第一步：准备标签编码
# 在建立 Dataset 之前，先把类别变成数字

# %%
# Get all unique classes from the 'Finding Label' column
classes = df_subset['Finding Label'].unique().tolist()
print("lesion categories:", classes)

# create a mapping from class names to integer labels
class_to_id = {c: i for i, c in enumerate(classes)}
print("lesion category mapping:", class_to_id)

# apply the mapping to the 'Finding Label' column
df_subset = df_subset.copy()
df_subset['label'] = df_subset['Finding Label'].map(class_to_id)

print(df_subset[['Image Index', 'Finding Label', 'label']].head())

# %% [markdown]
# second step: create a Dataset class

# %%
import torch ##it is a deep learning framework that provides a wide range of tools for building and training neural networks. It is widely used in both academia and industry for various machine learning tasks, including computer vision, natural language processing, and reinforcement learning.
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class NIHDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True) ## this is to reset the index of the dataframe
        self.img_dir = img_dir ## this is the directory where the images are stored
        self.transform = transform ## this is the transformation to be applied to the images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 读取图片
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        
        # 读取标签
        label = row['label']
        
        # 预处理
        if self.transform:
            image = self.transform(image)
        
        return image, label


# %% [markdown]
# define transform
#

# %%
transform = transforms.Compose([
    transforms.Resize((224, 224)),##piexls
    transforms.ToTensor(), ##tensor 是 PyTorch 里的基本数据结构，简单理解就是“多维数组”。 一维就是向量，二维就是矩阵，三维就是图像数据。


    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],##mean of the ImageNet dataset, which is commonly used for pretraining deep learning models. These values are used to normalize the pixel values of the input images to have zero mean and unit variance.
        std=[0.229, 0.224, 0.225]
    )
])

# %% [markdown]
# step 4: test a dataset object

# %%
dataset = NIHDataset(df_subset, img_dir, transform)

print("Dataset size:", len(dataset))

# 取第一个样本
image, label = dataset[0]

print("image shape:", image.shape)
print("label:", label)
print("lesion category:", classes[label])

# %% [markdown]
# 第五步：建立 DataLoader

# %%
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

# 测试一个 batch
images, labels = next(iter(dataloader))

print("Batch image shape:", images.shape)
print("Batch labels:", labels)

# %% [markdown]
# deploy CNN

# %%
import torch
import torch.nn as nn

# 检测 GPU（ROCm / CUDA）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 初始化模型
num_classes = len(classes)
model = SimpleCNN(num_classes=num_classes).to(device)
print(model)
print("Number of classes:", num_classes)

# %% [markdown]
# 第二步：定义 Loss 和 Optimizer

# %%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()#交叉熵损失，适用于分类任务；输入应是模型的原始 logits（未经过 softmax），标签是类别索引（LongTensor）。CrossEntropyLoss 内部做了 log-softmax + NLL。
optimizer = optim.Adam(model.parameters(), lr=0.001)#学习率为 0.001（可调

# %% [markdown]
# 第三步：训练循环

# %%
num_epochs = 10
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}')

print("Training complete!")

# plot loss curve
plt.plot(range(1, num_epochs+1), loss_history, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

# %% [markdown]
# 第四步：测试预测

# %%
model.eval()

# 前 10 个样本预测
first_predictions = []
with torch.no_grad():
    for idx in range(min(10, len(dataset))):
        image, label = dataset[idx]
        image = image.to(device)
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        first_predictions.append((idx, classes[label], classes[predicted.item()]))

print("First 10 predictions:")
for idx, true_label, pred_label in first_predictions:
    print(f"{idx}: true={true_label}, pred={pred_label}")

# 准确率
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")


# %% [markdown]
# dection dataset

# %%
class NIHDetectionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 读取图片
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        
        # 原始图片尺寸
        orig_w, orig_h = image.size
        
        # 读取标签
        label = int(row['label'])
        
        # 读取 bbox 并归一化到 0~1
        x = row['Bbox_x'] / orig_w
        y = row['Bbox_y'] / orig_h
        w = row['Bbox_w'] / orig_w
        h = row['Bbox_h'] / orig_h
        
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, bbox


# %% [markdown]
# create dataset

# %%
det_dataset = NIHDetectionDataset(df_subset, img_dir, transform)

image, label, bbox = det_dataset[0]

print("image shape:", image.shape)
print("labels:", label, "→", classes[label])
print("bboxes:", bbox)

# %% [markdown]
# check dataloader

# %%
det_dataloader = DataLoader(
    det_dataset,
    batch_size=8,
    shuffle=True
)

# 测试一个 batch
images, labels, bboxes = next(iter(det_dataloader))

print("image batch shape:", images.shape)
print("labels batch:", labels)
print("bboxes batch shape:", bboxes.shape)


# %%
class SimpleDetectionCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleDetectionCNN, self).__init__()
        
        # 特征提取（和之前一样）
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),#3 is the number of input channels RGB, 32 is the number of output channels, and 3x3 kernel_size is the size of the convolutional filter.
            nn.BatchNorm2d(32),#Batch normalization is a technique used to improve the training of deep neural networks by normalizing the inputs to each layer.
            nn.ReLU(),
            nn.MaxPool2d(2),#MaxPool2d(2) 后：[batch, 32, 111, 111]（下采样 /2，向下取整
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)#第二次 MaxPool2d(2) 后：[batch, 64, 54, 54]（109/2 = 54 向下取整
        )
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 256),#展平后全连接输入维度就是 64 * 54 * 54，这就是 nn.Linear(64 * 54 * 54, 256) 中的 645454 的来源
            nn.ReLU()
        )
        
        # 分类头
        self.cls_head = nn.Linear(256, num_classes)#输出维度是 num_classes，因为要输出每个类别的概率
        
        # bbox 回归头
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid()  # 输出 0~1 之间
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.shared(x)
        
        cls_out = self.cls_head(x)
        bbox_out = self.bbox_head(x)
        
        return cls_out, bbox_out


# %% [markdown]
# intialize model

# %%
print("使用设备:", device)

# 分类 Loss
cls_criterion = nn.CrossEntropyLoss()

# bbox 回归 Loss
bbox_criterion = nn.MSELoss()

# 初始化模型（移到 GPU）
model = SimpleDetectionCNN(num_classes=len(classes)).to(device)
print(model)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# traning model

# %%
num_epochs = 10
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, bboxes in det_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        cls_out, bbox_out = model(images)
        cls_loss = cls_criterion(cls_out, labels)
        bbox_loss = bbox_criterion(bbox_out, bboxes)
        total_loss = cls_loss + bbox_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(det_dataloader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Total Loss: {avg_loss:.4f} '
          f'Cls Loss: {cls_loss:.4f} '
          f'Bbox Loss: {bbox_loss:.4f}')

print("training complete!")

# %% [markdown]
# evluate model

# %%
model.eval()

with torch.no_grad():
    image, label, bbox = det_dataset[0]
    image = image.to(device)

    cls_out, bbox_out = model(image.unsqueeze(0))

    _, predicted = torch.max(cls_out, 1)

    print(f'detected class: {classes[predicted.item()]}')
    print(f'ground truth class: {classes[label]}')
    print(f'detected bbox: {bbox_out[0].cpu().numpy()}')
    print(f'actual bbox: {bbox.numpy()}')

# %%

# %%
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

model.eval()

with torch.no_grad():
    image_tensor, label, true_bbox = det_dataset[0]
    image_gpu = image_tensor.to(device)
    cls_out, bbox_out = model(image_gpu.unsqueeze(0))
    _, predicted = torch.max(cls_out, 1)
    pred_bbox = bbox_out[0].cpu().numpy()

# 把 tensor 转回图片
image_np = image_tensor.permute(1, 2, 0).numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

fig, ax = plt.subplots(1, figsize=(6,6))
ax.imshow(image_np)

# 真实框（绿色）
tx, ty = true_bbox[0]*224, true_bbox[1]*224
tw, th = true_bbox[2]*224, true_bbox[3]*224

true_rect = patches.Rectangle(
    (tx, ty), tw, th,
    linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth'
)

# 预测框（红色）
px, py = pred_bbox[0]*224, pred_bbox[1]*224
pw, ph = pred_bbox[2]*224, pred_bbox[3]*224

pred_rect = patches.Rectangle(
    (px, py), pw, ph,
    linewidth=2, edgecolor='red', facecolor='none', label='Prediction'
)

ax.add_patch(true_rect)
ax.add_patch(pred_rect)
ax.legend()
ax.set_title(f'Pred: {classes[predicted.item()]} | GT: {classes[label]}')
plt.show()

# %% [markdown]
# yolo

# %%
import os

# 创建文件夹
os.makedirs("F:/python/nih/yolo_dataset/images", exist_ok=True)
os.makedirs("F:/python/nih/yolo_dataset/labels", exist_ok=True)

# 转换
for _, row in df_subset.iterrows():
    img_name = row['Image Index']
    label_id = int(row['label'])
    
    # 归一化坐标
    x_center = (row['Bbox_x'] + row['Bbox_w'] / 2) / 1024
    y_center = (row['Bbox_y'] + row['Bbox_h'] / 2) / 1024
    w = row['Bbox_w'] / 1024
    h = row['Bbox_h'] / 1024
    
    # 写 txt
    label_file = img_name.replace('.png', '.txt')
    with open(f"F:/python/nih/yolo_dataset/labels/{label_file}", 'w') as f:
        f.write(f"{label_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("label transform complete!")
print(len(df_subset), "labels have been transformed!")

# %% [markdown]
# 复制图片到 YOLO 文件夹

# %%
import shutil

for img_name in df_subset['Image Index']:
    src = os.path.join(img_dir, img_name)
    dst = f"F:/python/nih/yolo_dataset/images/{img_name}"
    shutil.copy(src, dst)

print("images copied!")

# %% [markdown]
#  data.yaml

# %%
yaml_content = f"""
path: F:/python/nih/yolo_dataset
train: images
val: images

nc: {len(classes)} ##number of classes
names: {classes}  ##lesion categories
"""

with open("F:/python/nih/yolo_dataset/data.yaml", 'w') as f: ##this line opens the file "data.yaml" in write mode ('w') and assigns the file object to the variable f. If the file does not exist, it will be created. If it already exists, its contents will be overwritten.
    f.write(yaml_content)

print("data.yaml written!")
print(yaml_content)

# %% [markdown]
# start traning yolo

# %%
from ultralytics import YOLO

model = YOLO("F:/python/pytorch_cpuy/yolov8n.pt")

model.train(
    data="F:/python/nih/yolo_dataset/data.yaml",
    epochs=30,##number of epochs to train for
    imgsz=640,##image size for training
    batch=4,##number of samples per batch
    project="F:/python/nih/yolo_results",
    name="exp1",
    plots=False,       # disable plots during training
    save=True,         # save the model after training
    save_txt=False,
    save_conf=False,
    verbose=True
)

# %% [markdown]
# 读取训练指标

# %%
import pandas as pd

results_csv = "F:/python/nih/yolo_results/exp1/results.csv"

df_results = pd.read_csv(results_csv)


# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))


df_results.columns = df_results.columns.str.strip()

# 1. mAP50
axes[0,0].plot(df_results["metrics/mAP50(B)"])
axes[0,0].set_title("mAP50")
axes[0,0].set_xlabel("Epoch")

# 2. Precision
axes[0,1].plot(df_results["metrics/precision(B)"], color='green')
axes[0,1].set_title("Precision")
axes[0,1].set_xlabel("Epoch")

# 3. Recall
axes[0,2].plot(df_results["metrics/recall(B)"], color='orange')
axes[0,2].set_title("Recall")
axes[0,2].set_xlabel("Epoch")

# 4. Box Loss
axes[1,0].plot(df_results["train/box_loss"], color='red')
axes[1,0].set_title("Box Loss")
axes[1,0].set_xlabel("Epoch")

# 5. Cls Loss
axes[1,1].plot(df_results["train/cls_loss"], color='purple')
axes[1,1].set_title("Classification Loss")
axes[1,1].set_xlabel("Epoch")

# 6. DFL Loss
axes[1,2].plot(df_results["train/dfl_loss"], color='brown')
axes[1,2].set_title("DFL Loss")
axes[1,2].set_xlabel("Epoch")

plt.suptitle("YOLO Training Results", fontsize=14)
plt.tight_layout()
plt.show()

# %%
results = model.predict(
    source="F:/python/nih/yolo_dataset/images",
    conf=0.01,##confidence threshold for predictions. Predictions with confidence scores below this value will be discarded.
    save=False
)

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

# 选第一张
img_path = results[1].path
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, figsize=(6,6))
ax.imshow(img)

# 画预测框（红色）
for box in results[1].boxes.xyxy:
    x1, y1, x2, y2 = box.tolist()
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

# 再画真实框（绿色）
for _, row in df_subset.iterrows():
    if row["Image Index"] in img_path:
        x = row["Bbox_x"]
        y = row["Bbox_y"]
        w = row["Bbox_w"]
        h = row["Bbox_h"]

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='green',
            facecolor='none'
        )
        ax.add_patch(rect)

plt.title("Green = GT | Red = Prediction")
plt.show()

# %%
import pandas as pd
import os

df_entry = pd.read_csv("F:/python/nih/Data_Entry_2017.csv")

print("总行数:", len(df_entry))
print("列名:", df_entry.columns.tolist())
print(df_entry.head())

img_dir = "F:/python/nih/images_001/images/"
files = os.listdir(img_dir)

print("images_001 图片数量:", len(files))

df_local = df_entry[df_entry['Image Index'].isin(files)]
print("匹配到的图片数量:", len(df_local))

# %%
import matplotlib.pyplot as plt

df_local = df_local.copy()
df_local['label'] = df_local['Finding Labels'].apply(
    lambda x: x.split('|')[0]
)

label_counts = df_local['label'].value_counts()
print(label_counts)

label_counts.plot(kind='bar', figsize=(12,5))
plt.title('Disease Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
classes = df_local['label'].unique().tolist()
class_to_id = {c: i for i, c in enumerate(classes)}
df_local['label_id'] = df_local['label'].map(class_to_id)

print(f"类别数: {len(classes)}")
print(class_to_id)

# %% [markdown]
# data set

# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class NIHClassDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label_id'])
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = NIHClassDataset(df_local, img_dir, transform)
print("Dataset 大小:", len(dataset))

# 测试一个样本
image, label = dataset[0]
print("图片 shape:", image.shape)
print("标签:", label, "→", classes[label])

# %% [markdown]
# 划分 train/val

# %%
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False,
    num_workers=0
)

print("训练集:", len(train_dataset))
print("验证集:", len(val_dataset))

# %% [markdown]
# ResNet18

# %%
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

print("模型加载完成")
print(f"输出类别数: {len(classes)}")

# %% [markdown]
# train
# from tqdm import tqdm

# %%
loss_history = []
acc_history = []

num_epochs = 5

for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Batch [{i}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    # 验证
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    acc_history.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Loss: {avg_loss:.4f} '
          f'Val Accuracy: {val_acc:.2f}%')

print("训练完成!")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, num_epochs+1), loss_history, marker='o')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(range(1, num_epochs+1), acc_history, 
         marker='o', color='green')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()

# %% [markdown]
# re trian

# %% [markdown]
# optimiaze

# %%
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# %%
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# %%
# 去掉 No Finding
df_disease = df_local[df_local['label'] != 'No Finding'].copy()
print("去掉 No Finding 后:", len(df_disease))
print(df_disease['label'].value_counts())

# %%
# 重新标签编码
classes = df_disease['label'].unique().tolist()
class_to_id = {c: i for i, c in enumerate(classes)}
df_disease['label_id'] = df_disease['label'].map(class_to_id)

print(f"类别数: {len(classes)}")
print(f"总图片数: {len(df_disease)}")

# 重建 Dataset
train_dataset = NIHClassDataset(
    df_disease.iloc[:int(0.8*len(df_disease))],
    img_dir,
    transform_train
)

val_dataset = NIHClassDataset(
    df_disease.iloc[int(0.8*len(df_disease)):],
    img_dir,
    transform_val
)

# RX 6600M 8GB：batch 调到 64
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("训练集:", len(train_dataset))
print("验证集:", len(val_dataset))

# 重新加载模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# 训练
loss_history = []
acc_history = []
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                  f'Batch [{i}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    acc_history.append(val_acc)

    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Loss: {avg_loss:.4f} '
          f'Val Accuracy: {val_acc:.2f}%')

print("训练完成!")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, num_epochs+1), loss_history, marker='o')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(range(1, num_epochs+1), acc_history, 
         marker='o', color='green')
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()
