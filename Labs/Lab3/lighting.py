import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class LightningConvNet(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 49 * 49, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 49 * 49)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)
        return loss


# Training setup code
def train_model(train_data, val_data, test_data, num_classes, batch_size=32, max_epochs=100):
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=19, pin_memory=True,
                              drop_last=True, prefetch_factor=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=19, pin_memory=True, drop_last=True,
                            prefetch_factor=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=19, pin_memory=True, drop_last=True,
                             prefetch_factor=2)

    # Initialize model
    model = LightningConvNet(num_classes=num_classes)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='convnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',  # Automatically detect GPU/CPU
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

    return model, trainer


# Usage example
if __name__ == "__main__":
    # Define transforms (you can modify these based on your needs)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # First resize the image
            transforms.ToTensor(),  # Then convert to tensor (must come before normalize)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Finally normalize
        ]
    )

    # set the Lab3/Lab3 Dataset/Lab3_Gestures_Summer/ as the ImageFolder root
    images = torchvision.datasets.ImageFolder(
        "./Lab3 Dataset/Lab3_Gestures_Summer/", transform=transform
    )
    num_classes = len(images.classes)

    # divide the data into training, validation and test sets
    # 80% training, 10% validation, 10% test
    n = len(images)
    n_train = int(0.8 * n)
    n_val = n_test = (n - n_train) // 2
    train_data, val_data, test_data = torch.utils.data.random_split(
        images, [n_train, n_val, n_test]
    )


    model, trainer = train_model(train_data, val_data, test_data, num_classes)

    # Save the model
    torch.save(model.state_dict(), "model.pth")