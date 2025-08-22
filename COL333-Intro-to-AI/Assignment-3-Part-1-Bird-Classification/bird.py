import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import csv
from torchcam.methods import SmoothGradCAMpp  # Import Grad-CAM from torchcam
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import to_pil_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Set random seed for reproducibility
torch.manual_seed(0)

# EarlyStopping class to monitor validation loss
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Model definition
class BirdClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(BirdClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Validation function
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_accuracy = 100 * correct / total
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')

    return val_loss / len(val_loader), val_accuracy, f1_macro, f1_micro

# Training function
def train(model, device, train_loader, val_loader, optimizer, criterion, scheduler, early_stopping, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_accuracy = 100 * correct / total
        val_loss, val_accuracy, val_f1_macro, val_f1_micro = validate(model, device, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%, Val F1 Macro: {val_f1_macro:.4f}, "
              f"Val F1 Micro: {val_f1_micro:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(torch.load(early_stopping.path))
            break

# Testing function with F1 score
def test_model(model, device, test_loader, output_csv='bird.csv'):
    model.eval()
    results = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.append(labels.item())
            y_pred.append(predicted.cpu().item())

            # Append to results list for CSV
            image_path = test_loader.dataset.samples[len(results)][0]
            image_name = os.path.basename(image_path)
            results.append([image_name, predicted.cpu().item()])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_Name", "Predicted_Label"])
        writer.writerows(results)

    print("Inference results saved to", output_csv)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    final_f1 = (f1_macro + f1_micro) / 2
    print(f"F1 Macro: {f1_macro}, F1 Micro: {f1_micro}, Final F1: {final_f1}")



def visualize_all_classes(model, device, loader, target_layers, num_images_per_class=2, output_dir='gradcam_outputs', num_classes=11):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    cam = GradCAM(model=model, target_layers=target_layers)

    for class_idx in range(num_classes):
        class_output_dir = os.path.join(output_dir, f'class_{class_idx}')
        os.makedirs(class_output_dir, exist_ok=True)
        print(f"Generating Grad-CAM for class {class_idx}")

        images_generated = 0
        for images, labels in loader:
            if images_generated >= num_images_per_class:
                break
            images = images.to(device)

            for j in range(images.size(0)):
                # Check if the image belongs to the current class
                if labels[j].item() == class_idx:
                    target = [ClassifierOutputTarget(class_idx)]
                    images[j].requires_grad = True

                    try:
                        # Generate Grad-CAM for the specified class
                        grayscale_cam = cam(input_tensor=images[j].unsqueeze(0), targets=target)
                        img = images[j].cpu().numpy().transpose(1, 2, 0)
                        img_normalized = (img - img.min()) / (img.max() - img.min())
                        cam_image = show_cam_on_image(img_normalized, grayscale_cam[0], use_rgb=True)
                        result = to_pil_image(cam_image)
                        result.save(os.path.join(class_output_dir, f'gradcam_class_{class_idx}_img_{images_generated + 1}.png'))
                        images_generated += 1
                        if images_generated >= num_images_per_class:
                            break
                    except Exception as e:
                        print(f"Failed for image {images_generated} in class {class_idx}: {e}")

# Main function
if __name__ == "__main__":
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "bird.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if trainStatus == "train":
        full_dataset = datasets.ImageFolder(root=os.path.join(dataPath, 'train'), transform=train_transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        class_counts = [0] * 10
        for _, label in full_dataset:
            class_counts[label] += 1
        class_weights = torch.tensor([1.0 / count for count in class_counts]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = BirdClassifier(num_classes=10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        early_stopping = EarlyStopping(patience=5, delta=0.01, path=modelPath)

        print("Starting training...")
        train(model, device, train_loader, val_loader, optimizer, criterion, scheduler, early_stopping, epochs=20)
        print(f"Model saved to {modelPath}")

    elif trainStatus == "test":
        test_dataset = datasets.ImageFolder(root=os.path.join(dataPath, 'test'), transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = BirdClassifier(num_classes=10).to(device)
        model.load_state_dict(torch.load(modelPath))
        model.eval()

        test_model(model, device, test_loader, output_csv='bird.csv')

        # Specify the last convolutional layer in BirdClassifier
        target_layers = [model.features[-1]]  # Confirm that this is your model’s final conv layer

        # Generate Grad-CAM for each class with two images each
        visualize_all_classes(model, device, test_loader, target_layers, num_images_per_class=2, output_dir='gradcam_outputs', num_classes=11)
