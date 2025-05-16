import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights
import time

def main():
    start_time = time.time()

    # Define image dimensions and batch size
    img_height, img_width = 180, 180
    batch_size = 32
    num_workers = 4  # Adjust depending on your CPU capabilities

    # Local path to dataset
    train_directory = 'C:/Users/maqil/Documents/UITM/CV/degree uitm/SEM 6/CSP650/DATA PADI/DATA PADI'  # Update this path to your local dataset folder
    test_directory = 'C:/Users/maqil/Documents/UITM/CV/degree uitm/SEM 6/CSP650/DATA PADI/DATA PADI'    # Update this path to your local dataset folder

    # Image augmentation with more aggressive augmentations
    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),  # More rotation
        transforms.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),  # Resize more aggressively
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Random shear and scaling
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet normalization
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(train_directory, transform=data_augmentation)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    # Create data loaders with pin_memory=True for faster data transfer to GPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Use ResNet50 as base model with updated weights argument
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Unfreeze the last few layers of the base model for fine-tuning
    for name, param in base_model.named_parameters():
        if 'layer4' in name or 'fc' in name:  # Unfreeze layer4 and fc layer
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Modify the final layer for binary classification (cultivated vs weedy)
    base_model.fc = nn.Sequential(
        nn.Linear(base_model.fc.in_features, 384),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(384, 1),  # Only 1 output for binary classification with BCE
    )

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = base_model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    # optimizer = optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=1e-3)  # Added weight decay for regularization
    # optimizer = optim.SGD(base_model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4) #SGD optimizer
    optimizer = optim.RMSprop(base_model.parameters(), lr=1e-5, weight_decay=1e-5) #RMSprop optimizer
    # optimizer = optim.AdamW(base_model.parameters(), lr=1e-3, weight_decay=1e-2) #AdamW optimizer
    # Learning rate scheduler without verbose
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training function with early stopping
    def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=100):
        history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            model.train()
            running_loss, running_corrects = 0.0, 0
            for inputs, labels in train_loader:
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)  # Adjust label shape for BCE
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.round(torch.sigmoid(outputs))  # Get predictions (0 or 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc.item())

            # Validation phase
            model.eval()
            val_loss, val_corrects = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move data to GPU
                    inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    preds = torch.round(torch.sigmoid(outputs))
                    val_corrects += torch.sum(preds == labels.data)

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.item())

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Adjust learning rate with the scheduler
            scheduler.step(val_loss)

        return history

    # Evaluate the model using additional metrics
    def evaluate_model(model, dataloader):
        model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds)

        # Metrics computation
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels).astype(int)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation F1-Score: {f1:.4f}")
        return accuracy, precision, f1

    # Plot training history
    def plot_training_history(history):
        epochs_range = range(len(history['train_acc']))

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_acc'], label="Training Accuracy")
        plt.plot(epochs_range, history['val_acc'], label="Validation Accuracy")
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_loss'], label="Training Loss")
        plt.plot(epochs_range, history['val_loss'], label="Validation Loss")
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Function to plot and save confusion matrix
    def plot_confusion_matrix(model, dataloader):
        model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.round(torch.sigmoid(outputs)).cpu().numpy()
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds)

        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cultivated', 'Weedy'], yticklabels=['Cultivated', 'Weedy'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig("confusion_matrix le-5 RMSprop.png")
        plt.show()

    # Train the model
    history = train_model(base_model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=100)

    # Plot and evaluate
    plot_training_history(history)
    evaluate_model(base_model, val_loader)

    # Plot and save the confusion matrix
    plot_confusion_matrix(base_model, val_loader)

    # Save the model
    print("Confusion matrix saved to 'confusion_matrixRMSProp3.png'")
    model_save_path = './Fyp_ResnetmodelRMSProp3.pth'  # Update this path if necessary
    torch.save(base_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Clear the GPU cache after training
    torch.cuda.empty_cache()
    time.sleep(5)  # Simulates delay

    # Record the end time
    end_time = time.time()

    # Calculate and print the runtime
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

# Ensure the main function is only called when the script is run directly
if __name__ == '__main__':
    main()

################################################################################################################
    # Function to show images
def show_images(images, title):
    plt.figure(figsize=(12, 6))
    for i in range(5):  # Show 5 images
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C) for display
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

# Load a few images before augmentation
sample_image, _ = train_dataset[0]  # Get the first image from the dataset (before augmentation)
original_images = [sample_image] * 5  # Duplicate the original image 5 times for comparison
show_images(original_images, "Original Images (Before Augmentation)")

# Apply augmentation manually (for visualizing the transformations)
augmented_images = []
for _ in range(5):
    augmented_image = data_augmentation(sample_image.unsqueeze(0))  # Apply the transformation
    augmented_images.append(augmented_image[0])  # Remove the batch dimension
show_images(augmented_images, "Augmented Images (After Augmentation)")