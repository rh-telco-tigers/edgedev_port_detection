import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, random_split
import numpy as np
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pycocotools.coco import COCO
import os
import skimage.io as io

# Define evaluate_model function
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # Converting to numpy arrays for evaluation
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1

# Define visualize_image_annotations function
def visualize_image_annotations(coco_json, image_id, base_dir):
    coco = COCO(coco_json)

    # Load the image
    img = coco.loadImgs(image_id)[0]
    image_path = os.path.join(base_dir, img['file_name'])
    I = io.imread(image_path)

    # Load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    # Print the classes
    classes = [coco.cats[ann['category_id']]['name'] for ann in anns]
    print(f'Classes: {classes}')

    plt.show()


# Tensor board  default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist2')

# Hyperparameters
num_epochs = 1
batch_size = 8
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load COCO dataset
dataset = CocoDetection(root='/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8',
                        annFile='/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/result.json',
                        transform=transform)

# Get the maximum number of labels for padding
max_labels = max(len(ann) for _, ann in dataset)
print(f"\nmax_labels is {max_labels}")

# Load category information
with open('/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/result.json', 'r') as f:
    coco_info = json.load(f)

# Get the class names
class_names = {cat['id']: cat['name'] for cat in coco_info['categories']}

# Print the class names
print("Class Names:")
for class_id, class_name in class_names.items():
    print(f"Class ID: {class_id}, Class Name: {class_name}")

# Print one image
image, annotations = dataset[0]  # Change the index as needed
print(f"Image shape: {image.shape}, Annotations: {annotations}")

# Iterate through the dataset to find images with 8 labels
for idx, (image, annotations) in enumerate(dataset):
    if len(annotations) == 8:
        print(f"Image {idx + 1}:")
        print("Labels:")
        for annotation in annotations:
            label_id = annotation['category_id']
            label_name = class_names.get(label_id, f"Label ID: {label_id} (Unknown)")
            print(f"- {label_name}")
        print()

# Define a collate function to handle variable number of labels

def custom_collate_fn(batch):
    images, targets = zip(*batch)
    
    # Initialize lists to store one-hot encoded labels and labels present in each image
    labels_one_hot_list = []
    labels_present_list = []

    # Iterate over each target to extract labels present in each image
    for target in targets:
        # Extract category IDs present in the current target
        label_ids = [ann['category_id'] for ann in target]
        
        # Pad with zeros to ensure a total of 8 labels
        pad_length = 8 - len(label_ids)
        label_ids += [0] * pad_length
        
        # Append label IDs to the list of labels present in each image
        labels_present_list.append(label_ids)

    # Convert labels to one-hot encoded tensors
    for label_ids in labels_present_list:
        labels_one_hot = torch.zeros(8)  # Initialize one-hot encoded labels tensor with 8 labels
        for label_id in label_ids:
            if 0 <= label_id < 8:  # Ensure label is within the range [0, 7]
                labels_one_hot[label_id] = 1
        labels_one_hot_list.append(labels_one_hot)

    return torch.stack(images), torch.stack(labels_one_hot_list)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders with the filtered datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 54 * 54, 128)  # Adjust the input size according to your resized image dimensions
        self.fc2 = nn.Linear(128, num_classes)  # Output layer with the maximum number of labels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 54 * 54)  # Adjust the input size according to your resized image dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = CNN(num_classes=8)

# Define loss function and optimizer
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        print(f"\n outputs is ",outputs)
        print(f"\nlabels is ",labels)
        # Calculate loss
        loss = criterion(outputs, labels)  # labels here are already one-hot encoded

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate running accuracy
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels.argmax(dim=1)).sum().item()  # Compare with one-hot encoded labels

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}')

    # Log running loss and accuracy to TensorBoard
    writer.add_scalar('training loss', running_loss / total_steps, epoch)
    writer.add_scalar('training accuracy', running_correct / (total_steps * batch_size), epoch)

print('Finished Training')

# Evaluate the model on the test set after training
model.eval()
with torch.no_grad():
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')


# Visualize image annotations
image_id_to_visualize = 10  # Choose the image id you want to visualize
visualize_image_annotations('/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/result.json', image_id_to_visualize, '/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/')
