"""GPU migration script — modifies ab.ipynb cells to run on ROCm GPU with AMP."""
import json

with open("ab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def set_cell(nb, i, src):
    nb["cells"][i]["source"] = [src]

# ── Cell 30: SimpleCNN — add device + .to(device) ──
set_cell(nb, 30, '''import torch
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
print("Number of classes:", num_classes)''')

# ── Cell 34: SimpleCNN training — .to(device) + AMP ──
set_cell(nb, 34, '''from torch.cuda.amp import GradScaler, autocast

num_epochs = 10
loss_history = []
scaler = GradScaler()  # 混合精度（RX 6600M 8GB 加速约 1.5-2x）

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # forward pass（混合精度）
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
plt.show()''')

# ── Cell 36: SimpleCNN eval — .to(device) ──
set_cell(nb, 36, '''model.eval()

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
print(f"Accuracy: {accuracy:.2f}%")''')

# ── Cell 45: Detection model init — .to(device) + scaler ──
set_cell(nb, 45, '''from torch.cuda.amp import GradScaler, autocast

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

# 混合精度 scaler
scaler = GradScaler()''')

# ── Cell 47: Detection training — .to(device) + AMP ──
set_cell(nb, 47, '''num_epochs = 10
loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels, bboxes in det_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        bboxes = bboxes.to(device)

        # forward pass（混合精度）
        with autocast():
            cls_out, bbox_out = model(images)
            cls_loss = cls_criterion(cls_out, labels)
            bbox_loss = bbox_criterion(bbox_out, bboxes)
            total_loss = cls_loss + bbox_loss

        # backward pass
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(det_dataloader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Total Loss: {avg_loss:.4f} '
          f'Cls Loss: {cls_loss:.4f} '
          f'Bbox Loss: {bbox_loss:.4f}')

print("training complete!")''')

# ── Cell 49: Detection eval — .to(device) ──
set_cell(nb, 49, '''model.eval()

with torch.no_grad():
    image, label, bbox = det_dataset[0]
    image = image.to(device)

    cls_out, bbox_out = model(image.unsqueeze(0))

    _, predicted = torch.max(cls_out, 1)

    print(f'detected class: {classes[predicted.item()]}')
    print(f'ground truth class: {classes[label]}')
    print(f'detected bbox: {bbox_out[0].cpu().numpy()}')
    print(f'actual bbox: {bbox.numpy()}')''')

# ── Cell 51: Detection visualization — .to(device) + .cpu() ──
set_cell(nb, 51, '''import matplotlib.patches as patches
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
plt.show()''')

# ── Cell 73: ResNet18 — modern weights API + AMP import ──
set_cell(nb, 73, '''from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import ResNet18_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scaler = GradScaler()

print("模型加载完成")
print(f"输出类别数: {len(classes)}")''')

# ── Cell 75: ResNet18 training — AMP ──
set_cell(nb, 75, '''loss_history = []
acc_history = []

num_epochs = 5

for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

print("训练完成!")''')

# ── Cell 82: Optimized ResNet18 — batch=64 num_workers=2 pin_memory AMP ──
set_cell(nb, 82, '''# 重新标签编码
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

# RX 6600M 8GB：batch_size 调到 64
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
scaler = GradScaler()

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

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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

print("训练完成!")''')

# ── Save ──
with open("ab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done — all training cells now GPU-ready.")
