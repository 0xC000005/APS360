# download the pretrain resnet152 model from torchvision.models
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


device = torch.device('cpu')

# vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device)
# vgg16.features = vgg16.features.to(device)
# vgg16.eval()  # Important: prevents unnecessary computations


class ArtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtClassifier, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        self.features = self.vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Freeze early layers (first few convolutional blocks)
        for i, param in enumerate(self.features[:24].parameters()):  # First 3 blocks
            param.requires_grad = False

        # Only fine-tune the last convolutional block
        for param in self.features[24:].parameters():  # Last 2 blocks
            param.requires_grad = True

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs):
    # train the model
    for epoch in range(num_epochs):
        model.train()
        mini_batch_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # features = vgg16.features(images).detach()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # print mini-batch loss
            print(f'Epoch: {epoch}, Mini-batch: {mini_batch_count}, Loss: {loss.item()}')
            mini_batch_count += 1

        # validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                # features = vgg16.features(images).detach()
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += int((predicted == labels).sum())
                # calculate the validation loss
        accuracy = correct / total
        print(f'Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}')

    return model


if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # First resize the image
            transforms.ToTensor(),  # Then convert to tensor (must come before normalize)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), # Use the VGG normalization
        ]
    )

    # Load dataset
    images = torchvision.datasets.ImageFolder(
        "./processed_wikiart/", transform=transform
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

    print("\nDataset sampled, size:", len(sampled_idx))

    # Print sampled distribution
    sampled_targets = [targets[i] for i in sampled_idx]
    print("\nSampled data distribution:")
    for i in range(len(images.dataset.classes)):
        count = sum([1 for t in sampled_targets if t == i])
        print(f'{images.dataset.classes[i]}: {count} images ({100 * count / len(sampled_targets):.2f}%)')

    # Split into train and temp
    train_idx, temp_idx = train_test_split(
        sampled_idx,
        train_size=0.8,
        stratify=sampled_targets,
        random_state=42
    )
    print("\nDataset split, train size:", len(train_idx), "temp size:", len(temp_idx))

    # Print training distribution
    train_targets = [targets[i] for i in train_idx]
    print("\nTraining data distribution:")
    for i in range(len(images.dataset.classes)):
        count = sum([1 for t in train_targets if t == i])
        print(f'{images.dataset.classes[i]}: {count} images ({100 * count / len(train_targets):.2f}%)')

    # Split temp into val and test
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=0.5,
        stratify=[targets[i] for i in temp_idx],
        random_state=42
    )
    print("\nDataset split, val size:", len(val_idx), "test size:", len(test_idx))

    # Print validation distribution
    val_targets = [targets[i] for i in val_idx]
    print("\nValidation data distribution:")
    for i in range(len(images.dataset.classes)):
        count = sum([1 for t in val_targets if t == i])
        print(f'{images.dataset.classes[i]}: {count} images ({100 * count / len(val_targets):.2f}%)')

    # Create final datasets
    train_data = Subset(images, train_idx)
    val_data = Subset(images, val_idx)
    test_data = Subset(images, test_idx)


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

    # create the model, loss function and optimizer
    model = ArtClassifier(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # train the model
    model = train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=30)

    # save the model
    torch.save(model.state_dict(), 'ArtClassifier.pth')




