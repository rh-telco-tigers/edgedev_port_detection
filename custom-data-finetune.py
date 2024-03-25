import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize TensorBoard
writer = SummaryWriter()

# Hyperparameters
num_epochs = 20
batch_size = 4
learning_rate = 0.001
weight_decay = 1e-4

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset Class
class CocoFormatDataset(Dataset):
    def __init__(self, annotations, root_dir, transform=None):
        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations[idx]['image_id']
        img_name = os.path.join(self.root_dir, self.annotations[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')
        anns = [ann for ann in self.annotations if ann['image_id'] == img_id]

        labels = torch.zeros(len(class_names))
        for ann in anns:
            category_id = ann['category_id']
            labels[cat_id_to_seq_id[category_id]] = 1

        if self.transform:
            image = self.transform(image)

        return image, labels

# Load annotations and prepare dataset splits
with open('./dataset/custom-data/result.json', 'r') as f:
    data = json.load(f)
    annotations = data['annotations']
    images = data['images']

# Create mappings
image_id_to_file = {img['id']: img['file_name'] for img in images}
class_names = {cat['id']: cat['name'] for cat in data['categories']}
cat_id_to_seq_id = {cat_id: idx for idx, cat_id in enumerate(sorted(class_names.keys()))}

for ann in annotations:
    ann['file_name'] = image_id_to_file[ann['image_id']]

train_anns, test_anns = train_test_split(annotations, test_size=0.2, random_state=42)

train_dataset = CocoFormatDataset(train_anns, './dataset/custom-data', transform=train_transform)
test_dataset = CocoFormatDataset(test_anns, './dataset/custom-data', transform=test_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# Loss function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            writer.add_scalar('Training Loss', running_loss / 10, epoch * len(train_loader) + i)
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

writer.close()

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.sigmoid(outputs) > 0.5
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# Flattening the lists for evaluation
y_true = np.array(y_true).flatten()
y_pred = np.array(y_pred).flatten()

# Evaluation metrics
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
accuracy = (y_true == y_pred).mean()

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

print('Finished Training')
