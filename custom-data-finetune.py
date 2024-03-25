import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
import numpy as np


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
    def __init__(self, json_file, root_dir, transform=None, required_annotations=8):
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)

        # Filter images by the number of annotations
        image_id_to_anns = Counter(ann['image_id'] for ann in self.coco_data['annotations'])
        valid_image_ids = {id for id, count in image_id_to_anns.items() if count == required_annotations}

        self.images = [img for img in self.coco_data['images'] if img['id'] in valid_image_ids]
        self.annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] in valid_image_ids]
        self.root_dir = root_dir
        self.transform = transform
        self.cat_id_to_label = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        ann_ids = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        labels = torch.zeros(len(self.cat_id_to_label), dtype=torch.float32)
        categories = []

        for ann in ann_ids:
            cat_id = ann['category_id']
            label_index = self.cat_id_to_label[cat_id]
            labels[label_index] = 1
            for cat in self.coco_data['categories']:
                if cat['id'] == cat_id:
                    category_name = cat['name']
                    categories.append(category_name)
                    break

        if self.transform:
            image = self.transform(image)

        print(f"Fetching image_id: {img_info['id']}, file_name: {img_info['file_name']}")
        print(f"Identified Categories: {sorted(categories)}")
        print(f"Labels: {labels.numpy()}, Total Labels: {torch.sum(labels)}")

        if len(categories) != 8:
            print(f"Warning: Expected 8 categories for image_id {img_info['id']}, got {len(categories)}. Skipping.")
            return None

        return image, labels


# Load annotations and prepare dataset
json_file = './dataset/custom-data/result.json'
root_dir = './dataset/custom-data'  # Adjust if necessary

full_dataset = CocoFormatDataset(json_file=json_file, root_dir=root_dir, transform=train_transform)

# Split dataset
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# Apply the appropriate transform to the test dataset
test_dataset.dataset.transform = test_transform 

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.cat_id_to_label))

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
