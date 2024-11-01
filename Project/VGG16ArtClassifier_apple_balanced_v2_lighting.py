import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm


class ArtClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        # VGG16 feature extractor
        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        # Put entire VGG16 model in eval mode
        self.vgg16.eval()

        # Get feature extractor
        self.feature_extractor = self.vgg16.features
        # Double ensure feature extractor is in eval mode
        self.feature_extractor.eval()

        # Freeze all VGG16 parameters
        for param in self.vgg16.parameters():
            param.requires_grad = False

        # Calculate input size based on feature extractor output
        # VGG16 output is [batch_size, 512, 7, 7] for 224x224 input
        self.input_size = 512 * 7 * 7  # 25088

        # Print expected shapes for debugging
        print(f"Expected input size after flattening: {self.input_size}")

        # Rest of the architecture remains the same
        self.fc1 = nn.Linear(self.input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)

        self.residual_proj = nn.Linear(self.input_size, 1024)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x):
        # Add shape printing for debugging
        original_shape = x.shape

        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(x)

        feature_shape = features.shape
        # Flatten features
        x = features.view(features.size(0), -1)
        flattened_shape = x.shape

        # Store input for residual connection
        identity = self.residual_proj(x)

        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.selu(x)
        x = self.dropout1(x)

        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.selu(x)
        x = self.dropout2(x)

        # Residual connection
        x = x + identity

        # Output block
        x = self.fc3(x)
        x = self.dropout3(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)
        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.val_losses.append(loss.item())
        self.val_accuracies.append(acc.item())
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=3e-3,
            weight_decay=0.05,
            betas=(0.9, 0.999)
        )

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_losses) / len(self.train_losses)
        self.log('train_epoch_loss', avg_loss)
        self.train_losses = []

    def on_validation_epoch_end(self):
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        avg_acc = sum(self.val_accuracies) / len(self.val_accuracies)
        self.log('val_epoch_loss', avg_loss)
        self.log('val_epoch_acc', avg_acc)
        self.val_losses = []
        self.val_accuracies = []


if __name__ == '__main__':
    # Same dataset loading and transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load and process dataset (same as original code)
    images = torchvision.datasets.ImageFolder("./reduced_wikiart/", transform=transform)
    print("Dataset loaded, size:", len(images))

    # Balance dataset (same as original code)
    min_class = min([sum([1 for t in images.targets if t == i])
                     for i in range(len(images.classes))])
    balanced_indices = []
    for i in range(len(images.classes)):
        class_indices = [j for j in range(len(images.targets)) if images.targets[j] == i]
        balanced_indices.extend(class_indices[:min_class])
    images = Subset(images, balanced_indices)

    # Split dataset (same as original code)
    targets = [images.dataset.targets[i] for i in images.indices]
    train_idx, temp_idx = train_test_split(
        range(len(images)),
        train_size=0.8,
        stratify=targets,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        stratify=[targets[i] for i in temp_idx],
        random_state=42
    )

    # Create data subsets (same as original code)
    train_data = Subset(images, train_idx)
    val_data = Subset(images, val_idx)
    test_data = Subset(images, test_idx)

    # Create dataloaders (same as original code)
    train_loader = DataLoader(
        train_data,
        batch_size=64,
        shuffle=True,
        num_workers=9,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_data,
        batch_size=64,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_data,
        batch_size=64,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Initialize model and trainer
    model = ArtClassifier(num_classes=len(images.dataset.classes))

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='mps',
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save model
    model_name = 'ArtClassifier' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.pth'
    torch.save(model.state_dict(), model_name)

    # Test model
    model.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(len(images.dataset.classes), len(images.dataset.classes))

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels, predicted):
                confusion_matrix[t.long(), p.long()] += 1

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')

    # Plot and save results (same as original code)
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.colorbar()
    plt.savefig('confusion_matrix' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
    plt.close()