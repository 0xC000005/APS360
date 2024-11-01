import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device)
vgg16.features = vgg16.features.to(device)
vgg16.eval()


class ArtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtClassifier, self).__init__()
        self.input_size = 512 * 7 * 7  # 25088 (VGG16 feature size)

        # Increased network width
        self.fc1 = nn.Linear(self.input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        # Batch normalization (more effective than layer norm for this case)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)

        # Smoother dropout progression
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.1)

        # Residual projection for skip connection
        self.residual_proj = nn.Linear(self.input_size, 1024)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: [N, 512, 7, 7] -> [N, 25088]

        # Store input for residual connection
        identity = self.residual_proj(x)

        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.selu(x)  # SELU for self-normalizing properties
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


def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device):
    # Initialize history tensors on GPU
    train_losses = torch.zeros(num_epochs, device=device)
    val_losses = torch.zeros(num_epochs, device=device)
    val_accuracies = torch.zeros(num_epochs, device=device)

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        # Training phase
        model.train()
        epoch_train_loss = torch.tensor(0.0, device=device)
        batch_count = torch.tensor(0, device=device)

        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc='Training', leave=False)
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = vgg16.features(images).detach()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss on GPU
            epoch_train_loss += loss
            batch_count += 1

            # Update progress bar with current loss
            train_pbar.set_postfix({'loss': loss.item()})

        # Calculate average training loss for epoch
        avg_train_loss = epoch_train_loss / batch_count
        train_losses[epoch] = avg_train_loss

        # Validation phase
        model.eval()
        epoch_val_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)
        val_batch_count = torch.tensor(0, device=device)

        # Progress bar for validation batches
        val_pbar = tqdm(val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)

                features = vgg16.features(images).detach()
                outputs = model(features)
                loss = loss_fn(outputs, labels)

                epoch_val_loss += loss
                val_batch_count += 1

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': loss.item()})

        # Calculate epoch metrics
        avg_val_loss = epoch_val_loss / val_batch_count
        accuracy = correct.float() / total

        # Store metrics
        val_losses[epoch] = avg_val_loss
        val_accuracies[epoch] = accuracy

        # Print epoch summary
        tqdm.write(f'Epoch {epoch}:')
        tqdm.write(f'Training Loss: {avg_train_loss.item():.4f}')
        tqdm.write(f'Validation Loss: {avg_val_loss.item():.4f}')
        tqdm.write(f'Validation Accuracy: {accuracy.item():.4f}')
        tqdm.write('-' * 50)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

    return model, history


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Larger initial size for better cropping
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load dataset
    images = torchvision.datasets.ImageFolder(
        "./reduced_wikiart/", transform=transform
    )
    print("Dataset loaded, size:", len(images))

    num_classes = len(images.classes)

    # Print initial class distribution
    print("\nInitial class distribution:")
    for i in range(len(images.classes)):
        count = sum([1 for t in images.targets if t == i])
        print(f'{images.classes[i]}: {count} images ({100 * count / len(images.targets):.2f}%)')

    # Get the minimal number of images in a class
    min_class = min([sum([1 for t in images.targets if t == i]) for i in range(len(images.classes))])
    print(f"\nMinimum class size: {min_class}")

    # Create balanced dataset by selecting min_class samples from each class
    balanced_indices = []
    for i in range(len(images.classes)):
        class_indices = [j for j in range(len(images.targets)) if images.targets[j] == i]
        balanced_indices.extend(class_indices[:min_class])

    # Create balanced subset
    images = Subset(images, balanced_indices)
    print(f"Balanced dataset size: {len(images)}")


    # Create a function to get targets from Subset
    def get_targets_from_subset(subset):
        if hasattr(subset.dataset, 'targets'):
            return [subset.dataset.targets[i] for i in subset.indices]
        else:
            return [subset.dataset.dataset.targets[i] for i in subset.indices]


    # Get targets from the balanced dataset
    targets = get_targets_from_subset(images)

    # Print balanced distribution
    print("\nBalanced class distribution:")
    for i in range(len(images.dataset.classes)):
        count = sum([1 for t in targets if t == i])
        print(f'{images.dataset.classes[i]}: {count} images ({100 * count / len(targets):.2f}%)')

    # Create indices for splitting
    indices = list(range(len(images)))

    # # First split: 20% sampled, 80% ignored
    # sampled_idx, _ = train_test_split(
    #     indices,
    #     train_size=0.20,
    #     stratify=targets,
    #     random_state=42
    # )

    sampled_idx = indices

    print("Dataset sampled, size:", len(sampled_idx))

    # First split: 80% train, 20% temp from the sampled indices
    train_idx, temp_idx = train_test_split(
        sampled_idx,
        train_size=0.8,
        stratify=[targets[i] for i in sampled_idx],
        random_state=42
    )
    print("Dataset split, train size:", len(train_idx), "temp size:", len(temp_idx))

    # Second split: split the remaining 20% into two equal parts (10% each)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,  # 0.5 of 20% is 10% of total
        stratify=[targets[i] for i in temp_idx],
        random_state=42
    )
    print("Dataset split, val size:", len(val_idx), "test size:", len(test_idx))

    # print out the number of classes in the training, validation, and test sets
    print(f'Number of classes in training set: {len(set([targets[i] for i in train_idx]))}')
    print(f'Number of classes in validation set: {len(set([targets[i] for i in val_idx]))}')
    print(f'Number of classes in test set: {len(set([targets[i] for i in test_idx]))}')

    # Create the final subsets
    train_data = Subset(images, train_idx)
    val_data = Subset(images, val_idx)
    test_data = Subset(images, test_idx)

    # create data loaders
    # Optimized DataLoader configuration
    train_loader = DataLoader(
        train_data,
        batch_size=64,  # Reduced batch size
        shuffle=True,
        num_workers=9,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Reduce prefetching
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

    # create the model, loss function and optimizer
    model = ArtClassifier(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Optimizer with higher initial learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )


    # train the model
    model, history = train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=30, device=device)

    # save the model
    model_name = 'ArtClassifier' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.pth'
    torch.save(model.state_dict(), model_name)

    # free the file workers from the train and validation loaders
    del train_loader, val_loader

    # load the model
    model.load_state_dict(torch.load(model_name))

    # Evaluate the model on the test set
    # Plot the confusion matrix
    model.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            features = vgg16.features(images).detach()
            outputs = model(features)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += int((predicted == labels).sum())
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, interpolation='nearest')
    plt.colorbar()
    # Save the confusion matrix plot
    plt.savefig('confusion_matrix' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')

    # Move tensors to CPU and convert to numpy arrays
    train_losses = history['train_losses'].detach().cpu().numpy()
    val_losses = history['val_losses'].detach().cpu().numpy()
    val_accuracies = history['val_accuracies'].detach().cpu().numpy()

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.grid(True)
    # Save the plot
    plt.savefig('losses' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
    plt.close()

    # Plot the validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.grid(True)
    # Save the plot
    plt.savefig('accuracy' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
    plt.close()

    # Save the logs as csv
    logs = pd.DataFrame({'train_losses': train_losses, 'val_losses': val_losses, 'val_accuracies': val_accuracies})
    logs.to_csv('logs' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv', index=False)
