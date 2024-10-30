import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os


class LightningConvNet(pl.LightningModule):
    def __init__(self, trial, num_classes):
        super().__init__()
        self.save_hyperparameters()

        # Tunable hyperparameters
        self.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        # Tunable convolutional layers
        conv1_features = trial.suggest_int("conv1_features", 4, 16)
        conv2_features = trial.suggest_int("conv2_features", 8, 32)
        conv3_features = trial.suggest_int("conv3_features", 16, 64)

        # Network architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_features, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=conv1_features, out_channels=conv2_features, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=conv2_features, out_channels=conv3_features, kernel_size=5)

        # Calculate input features for first FC layer
        self.fc1_input_features = conv3_features * 49 * 49

        # Tunable fully connected layers
        fc1_units = trial.suggest_int("fc1_units", 64, 256)
        fc2_units = trial.suggest_int("fc2_units", 32, 128)

        self.fc1 = nn.Linear(self.fc1_input_features, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, num_classes)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc1_input_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
        acc = self.train_acc(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('hp_metric', acc, on_step=False, on_epoch=True)
        return loss


class GestureDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def setup(self, stage: str = None):
        # Load dataset
        full_dataset = torchvision.datasets.ImageFolder(
            self.data_dir,
            transform=self.transform
        )

        # Calculate lengths
        n = len(full_dataset)
        n_train = int(0.8 * n)
        n_val = n_test = (n - n_train) // 2

        # Split dataset
        self.train_data, self.val_data, self.test_data = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                          num_workers=19, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,
                          num_workers=19, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,
                          num_workers=19, pin_memory=True)


def objective(trial: optuna.trial.Trial) -> float:
    # Initialize data module
    datamodule = GestureDataModule(
        data_dir="./Lab3 Dataset/Lab3_Gestures_Summer/",
        batch_size=32
    )

    # Get number of classes
    datamodule.setup()
    num_classes = len(datamodule.train_data.dataset.classes)

    # Initialize model
    model = LightningConvNet(trial, num_classes=num_classes)

    # Initialize trainer
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=50,
        accelerator="auto",
        devices=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    # Log hyperparameters
    hyperparameters = dict(trial.params)
    trainer.logger.log_hyperparams(hyperparameters)

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Return the best validation accuracy
    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    # Create study
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Optimize
    study.optimize(objective, n_trials=20, timeout=3600)  # 1 hour timeout

    # Print results
    print("Number of finished trials: {}".format(len(study.trials)))
    print("\nBest trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save best model
    best_model = LightningConvNet(trial, num_classes=5)  # Adjust num_classes as needed
    torch.save(best_model.state_dict(), "best_model.pth")