import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import numpy as np
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split  # Import random_split here


# Tensor board  default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist2')


# Hyperparameters
num_epochs = 100
batch_size = 6
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load COCO dataset
#train_dataset = CocoDetection(root='/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8', annFile='/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/result.json', transform=transform)

#test_dataset = CocoDetection(root='/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8', annFile='/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/result.json', transform=transform)

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

# Iterate through the dataset to find images with 6 labels
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
    
    # Extract category IDs as labels/classes from the target annotations
    labels = [[ann['category_id'] for ann in target] for target in targets]

    # Convert labels to one-hot encoded tensors
    labels_one_hot_list = []  # List to store one-hot encoded labels for each batch
    images_with_high_labels = []  # List to store images with labels > 21
    for image, label_batch in zip(images, labels):
        labels_batch_one_hot = torch.zeros(1, 22)  # Initialize one-hot encoded labels for the current batch with 22 classes
        for label in label_batch:
            if 0 <= label < 22:  # Ensure label is within the range [0, 21]
                labels_batch_one_hot[0, label] = 1
            else:  # If label is greater than 21, store the corresponding image
                images_with_high_labels.append(image)
        labels_one_hot_list.append(labels_batch_one_hot)

    # Display images with labels > 21
    for img in images_with_high_labels:
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))  # Display the image
        plt.axis('off')
        plt.show()

    return torch.stack(images), torch.cat(labels_one_hot_list, dim=0)


# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders with the filtered datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


# Create data loaders with the filtered datasets
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)


# Define CNN model
class CNN(nn.Module):
    def __init__(self,num_classes=22):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*54*54, 128)  # Adjust the input size according to your resized image dimensions
        self.fc2 = nn.Linear(128, num_classes)  # Output layer with the maximum number of labels

    def forward(self, x):
        x = F.relu(self.conv1(x))
 #       print(f"\nconv1 output shape is {x.shape}")
        x = F.max_pool2d(x, 2, 2)
#       print(f"\nmax pool output shape is {x.shape}")
        x = F.relu(self.conv2(x))
#       print(f"\nconv2 output shape is {x.shape}")
        x = F.max_pool2d(x, 2, 2)
#        print(f"\max pool2 output shape is {x.shape}")
        x = x.view(-1, 32*54*54)  # Adjust the input size according to your resized image dimensions
#        print(f"\adjustment shape before fc1 is {x.shape}")
        x = F.relu(self.fc1(x))
#        print(f"\ fc1 output shape is {x.shape}")
        x = self.fc2(x)
#        print(f"\ fc2 output shape is {x}")
        return x

# Instantiate the model
model = CNN(num_classes=22)

# Tensorboard graph generation 

writer.add_graph(model, torch.rand([1, 3, 224, 224])) 
#writer.close()
#sys.exit()

# Log weights and bias
#for name, param in model.named_parameters():
#    writer.add_histogram(name, param, bins='auto')

#for name, param in model.named_parameters():
#    if param.grad is not None:
#        writer.add_histogram(f'{name}.grad', param.grad, bins='auto')

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
#        print(f"\n labels is ",labels)
#        print(f"\n outputs is ",outputs) 
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

#        if (i+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}')
    # Log running loss and accuracy to TensorBoard
    writer.add_scalar('training loss', running_loss / total_steps, epoch)
    writer.add_scalar('training accuracy', running_correct / (total_steps * batch_size), epoch)

print('Finished Training')

# Evaluation
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
all_true_labels = []
all_predicted_labels = []

# Disable gradients during evaluation
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        test_loss += loss.item()

        # Convert predicted and true labels to class indices and append them to lists
        for idx in range(len(images)):
            # Get predicted labels
            predicted_idx = torch.where(outputs[idx] > 0.5)[0]  # Threshold at 0.5 for binary classification
            all_predicted_labels.append(predicted_idx.cpu().numpy())

            # Get true labels
            true_idx = torch.where(labels[idx] == 1)[0]
            all_true_labels.append(true_idx.cpu().numpy())

# Display true labels and predicted labels
for true_label, predicted_label in zip(all_true_labels, all_predicted_labels):
    print(f'True Label: {true_label}, Predicted Label: {predicted_label}')

# Calculate test loss (if needed)
print(f'Test Loss: {test_loss}')
