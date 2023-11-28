# training.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import RedAreasDataset
from resnet import resnet50, resnet34, resnext50_32x4d
from torchvision import transforms

# Configuration
learning_rate = 1e-4
weight_decay = 1e-5  # L2 regularization term
batch_size = 2
num_epochs = 50
early_stopping_patience = 20  # Number of epochs to wait after val loss stops improving
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare dataset and data loader
image_dir = 'data/train/'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
dataset = RedAreasDataset(image_dir=image_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Prepare the validation dataset and data loader
val_image_dir = 'data/val/'  # Path to the validation dataset
val_dataset = RedAreasDataset(image_dir=val_image_dir, transform=transform)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ResNet50_Fusion model
model = resnet34(num_classes=1)  # num_classes=1 for regression
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# For Early Stopping
best_val_loss = float('inf')
epochs_no_improve = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, _, severity_scores) in enumerate(data_loader):
        images = images.to(device).float()  # Convert images to float
        severity_scores = severity_scores.to(device).float().unsqueeze(1)  # Convert severity scores to float and ensure correct shape

        # Forward pass
        predictions = model(images)
        
        # Compute the loss
        loss = criterion(predictions, severity_scores)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Calculate and print the average loss over the epoch
    train_epoch_loss = running_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_epoch_loss:.4f}')

    # Validation phase
    model.eval()
    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        val_running_loss = 0.0
        for images, _, severity_scores in val_data_loader:
            images = images.to(device).float()
            severity_scores = severity_scores.to(device).float().unsqueeze(1)  # Ensure correct shape

            predictions = model(images)
            val_loss = criterion(predictions, severity_scores)
            val_running_loss += val_loss.item()

        val_epoch_loss = val_running_loss / len(val_data_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_epoch_loss:.4f}')

    # Early stopping check
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), 'models/bestmodel5a.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == early_stopping_patience:
            print(f'Early stopping triggered after epoch {epoch+1}!')
            break  # Stop training

print('Finished Training')
