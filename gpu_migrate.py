"""Fix MIOpen error — remove autocast/GradScaler, keep pure GPU training."""
import json

with open("ab.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

def set_cell(nb, i, src):
    nb["cells"][i]["source"] = [src]

# ── Cell 34: SimpleCNN training — pure GPU, no AMP ──
set_cell(nb, 34, '''num_epochs = 10
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
plt.show()''')

# ── Cell 45: Detection model init — no GradScaler ──
set_cell(nb, 45, '''print("使用设备:", device)

# 分类 Loss
cls_criterion = nn.CrossEntropyLoss()

# bbox 回归 Loss
bbox_criterion = nn.MSELoss()

# 初始化模型（移到 GPU）
model = SimpleDetectionCNN(num_classes=len(classes)).to(device)
print(model)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)''')

# ── Cell 47: Detection training — pure GPU, no AMP ──
set_cell(nb, 47, '''num_epochs = 10
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

print("training complete!")''')

# ── Cell 73: ResNet18 — no GradScaler ──
set_cell(nb, 73, '''from torchvision import models
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
print(f"输出类别数: {len(classes)}")''')

# ── Cell 75: ResNet18 training — pure GPU, no AMP ──
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

print("训练完成!")''')

# ── Cell 82: Optimized ResNet18 — pure GPU, no AMP ──
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

print("训练完成!")''')

with open("ab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done — AMP removed, pure GPU training.")
