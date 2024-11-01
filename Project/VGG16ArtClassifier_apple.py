import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device('mps')

vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).to(device)
vgg16.features = vgg16.features.to(device)
vgg16.eval()


class ArtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtClassifier, self).__init__()

        # Input will be [N, 512, 7, 7] -> flattened to [N, 25088]
        input_size = 512 * 7 * 7  # 25088

        self.fc1 = nn.Linear(input_size, 1024)  # 25088 -> 1024
        self.fc2 = nn.Linear(1024, 512)  # 1024 -> 512
        self.fc3 = nn.Linear(512, num_classes)  # 512 -> num_classes

        # Batch Norm layers
        self.bn1 = nn.BatchNorm1d(1024)  # Shape: [N, 1024]
        self.bn2 = nn.BatchNorm1d(512)  # Shape: [N, 512]

        # Dropout with decreasing rates
        self.dropout1 = nn.Dropout(0.4)  # 40% dropout
        self.dropout2 = nn.Dropout(0.4 * 0.8)  # 32% dropout
        self.dropout3 = nn.Dropout(0.4 * 0.8 * 0.6)  # 19.2% dropout

    def forward(self, x):
        # Input shape: [N, 512, 7, 7]
        x = x.view(x.size(0), -1)  # Shape: [N, 25088]

        x = self.fc1(x)  # Shape: [N, 1024]
        x = self.bn1(x)  # Shape: [N, 1024]
        x = torch.relu(x)  # Shape: [N, 1024]
        x = self.dropout1(x)  # Shape: [N, 1024]

        x = self.fc2(x)  # Shape: [N, 512]
        x = self.bn2(x)  # Shape: [N, 512]
        x = torch.relu(x)  # Shape: [N, 512]
        x = self.dropout2(x)  # Shape: [N, 512]

        x = self.fc3(x)  # Shape: [N, num_classes]
        x = self.dropout3(x)  # Shape: [N, num_classes]

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
            features = vgg16.features(images).detach()
            outputs = model(features)
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
                features = vgg16.features(images).detach()
                outputs = model(features)
                loss = loss_fn(outputs, labels)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += int((predicted == labels).sum())
        accuracy = correct / total
        print(f'Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}')

    return model


if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # First resize the image
            transforms.ToTensor(),  # Then convert to tensor (must come before normalize)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),  # Use the VGG normalization
        ]
    )

    # set the Lab3/Lab3 Dataset/Lab3_Gestures_Summer/ as the ImageFolder root
    images = torchvision.datasets.ImageFolder(
        "./reduced_wikiart/", transform=transform
    )

    # Print initial class distribution
    print("\nInitial class distribution:")
    for i in range(len(images.classes)):
        count = sum([1 for t in images.targets if t == i])
        print(f'{images.classes[i]}: {count} images ({100 * count / len(images.targets):.2f}%)')

    num_classes = len(images.classes)

    # Get targets for stratification
    targets = images.targets

    # Create indices for splitting
    indices = list(range(len(images)))

    # First split: 90% sampled, 10% ignored
    sampled_idx, _ = train_test_split(
        indices,
        train_size=0.5,
        stratify=[targets[i] for i in indices],
        random_state=42
    )
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
        batch_size=512,  # Reduced batch size
        shuffle=True,
        num_workers=9,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Reduce prefetching
    )

    val_loader = DataLoader(
        val_data,
        batch_size=256,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_data,
        batch_size=256,
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
    model = train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=2)

    # save the model
    torch.save(model.state_dict(), 'ArtClassifier.pth')

    # free the file workers from the train and validation loaders
    del train_loader, val_loader

    # load the model
    model.load_state_dict(torch.load('ArtClassifier.pth'))

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
    plt.savefig('confusion_matrix.png')