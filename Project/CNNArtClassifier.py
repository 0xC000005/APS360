# download the pretrain resnet152 model from torchvision.models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device('mps')


class ArtClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ArtClassifier, self).__init__()
        # First Convolutional Layer
        # Input: batch x 3 x 224 x 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # Output: batch x 6 x 220 x 220 ((224 - 5 + 0)/1 + 1 = 220)

        # First Max Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: batch x 6 x 110 x 110 (220/2 = 110)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # Output: batch x 16 x 106 x 106 ((110 - 5 + 0)/1 + 1 = 106)

        # Second Max Pooling Layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: batch x 16 x 53 x 53 (106/2 = 53)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        # Output: batch x 32 x 49 x 49 ((53 - 5 + 0)/1 + 1 = 49)

        # Calculate the input features for the first fully connected layer
        self.fc1 = nn.Linear(32 * 49 * 49, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # Replace num_classes with actual number

    def forward(self, x):
        # Input shape: batch x 3 x 224 x 224
        x = self.pool1(F.relu(self.conv1(x)))
        # Shape: batch x 6 x 110 x 110

        x = self.pool2(F.relu(self.conv2(x)))
        # Shape: batch x 16 x 53 x 53

        x = F.relu(self.conv3(x))
        # Shape: batch x 32 x 49 x 49

        # Flatten the tensor
        x = x.view(-1, 32 * 49 * 49)
        # Shape: batch x 76,832 (32 * 49 * 49 = 76,832)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
                outputs = model(images)
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
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),  # Normalize the image
        ]
    )

    # set the Lab3/Lab3 Dataset/Lab3_Gestures_Summer/ as the ImageFolder root
    images = torchvision.datasets.ImageFolder(
        "./processed_wikiart/", transform=transform
    )

    num_classes = len(images.classes)

    # Get targets for stratification
    targets = images.targets

    # Create indices for splitting
    indices = list(range(len(images)))

    # First split: 50% sampled, 50% ignored
    sampled_idx, _ = train_test_split(
        indices,
        train_size=0.5,
        stratify=[targets[i] for i in indices],
        random_state=42
    )
    print("Dataset sampled, size:", len(sampled_idx))

    # First split: 50% train, 50% temp from the sampled indices
    train_idx, temp_idx = train_test_split(
        sampled_idx,
        train_size=0.5,
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

    # # train the model
    # model = train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=30)
    #
    # # save the model
    # torch.save(model.state_dict(), 'CNNArtClassifier.pth')

    # free the file workers from the train and validation loaders to prevent the too many open files error
    del train_loader, val_loader

    # load the model
    model.load_state_dict(torch.load('CNNArtClassifier.pth'))

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
            outputs = model(images)
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
    plt.savefig('CNNconfusion_matrix.png')