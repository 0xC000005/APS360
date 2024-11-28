import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import vit_b_16
from tqdm import tqdm
from datasets import load_dataset
from datasets import DatasetDict
import datetime



def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    num_epochs,
    device,
):
    # Initialize history tensors on GPU
    train_losses = torch.zeros(num_epochs, device=device)
    val_losses = torch.zeros(num_epochs, device=device)
    val_accuracies = torch.zeros(num_epochs, device=device)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        # Training phase
        model.train()
        epoch_train_loss = torch.tensor(0.0, device=device)
        batch_count = torch.tensor(0, device=device)

        # Progress bar for training batches
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in train_pbar:
            # Unpack the dictionary batch
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate loss on GPU
            epoch_train_loss += loss
            batch_count += 1

            # Update progress bar with current loss
            train_pbar.set_postfix({"loss": loss.item()})

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
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                # Unpack the dictionary batch
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(images)
                loss = loss_fn(outputs, labels)

                epoch_val_loss += loss
                val_batch_count += 1

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

                # Update progress bar with current loss
                val_pbar.set_postfix({"loss": loss.item()})

        # Calculate epoch metrics
        avg_val_loss = epoch_val_loss / val_batch_count
        accuracy = correct.float() / total

        # Store metrics
        val_losses[epoch] = avg_val_loss
        val_accuracies[epoch] = accuracy

        # Print epoch summary
        tqdm.write(f"Epoch {epoch}:")
        tqdm.write(f"Training Loss: {avg_train_loss.item():.4f}")
        tqdm.write(f"Validation Loss: {avg_val_loss.item():.4f}")
        tqdm.write(f"Validation Accuracy: {accuracy.item():.4f}")
        tqdm.write("-" * 50)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

    return model, history


# Function to transform images
train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )



from torchvision.datasets import CIFAR10

class ArtBench10(CIFAR10):

    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }


if __name__ == "__main__":
    device = torch.device("mps")

    # Load the artbench dataset
    train_dataset = ArtBench10(root="data", train=True, download=True, transform=train_transform)
    test_dataset = ArtBench10(root="data", train=False, download=True, transform=val_transform)
    val_dataset = test_dataset
    
    num_classes = 10
    classes = train_dataset.classes
    print(classes)

    # Keep the class 0, 1, 2, 3, 5
    selected_classes = [0, 1, 2, 3, 5]

    # Create class-specific indices for both training and testing
    train_indices_by_class = {i: [] for i in range(len(selected_classes))}
    test_indices_by_class = {i: [] for i in range(len(selected_classes))}

    # Process training dataset
    for i, label in enumerate(train_dataset.targets):
        if label in selected_classes:
            new_label = selected_classes.index(label)  # Remap label to 0-4
            train_dataset.targets[i] = new_label
            if len(train_indices_by_class[new_label]) < 1000:  # Only take first 1000 per class
                train_indices_by_class[new_label].append(i)

    # Process test dataset
    for i, label in enumerate(test_dataset.targets):
        if label in selected_classes:
            new_label = selected_classes.index(label)  # Remap label to 0-4
            test_dataset.targets[i] = new_label
            if len(test_indices_by_class[new_label]) < 100:  # Only take first 100 per class
                test_indices_by_class[new_label].append(i)

    # Combine indices for all classes
    train_indices = []
    test_indices = []
    for class_idx in range(len(selected_classes)):
        train_indices.extend(train_indices_by_class[class_idx])
        test_indices.extend(test_indices_by_class[class_idx])

    # Create subsets with selected classes
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    val_dataset = test_dataset

    # Print original class names and their new indices
    original_classes = classes
    print("\nOriginal classes selected and their new indices:")
    for new_idx, old_idx in enumerate(selected_classes):
        print(f"Original class {old_idx} ({original_classes[old_idx]}) -> New class {new_idx}")

    # Print class distribution
    def count_samples_per_class(dataset, num_classes):
        counts = [0] * num_classes
        for i in tqdm(range(len(dataset))):
            _, label = dataset[i]
            counts[label] += 1
        return counts

    # print("\nSamples per class in training set:")
    # train_counts = count_samples_per_class(train_dataset, len(selected_classes))
    # for i, count in enumerate(train_counts):
    #     print(f"Class {i}: {count} samples")

    # print("\nSamples per class in test set:")
    # test_counts = count_samples_per_class(test_dataset, len(selected_classes))
    # for i, count in enumerate(test_counts):
    #     print(f"Class {i}: {count} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Reduced batch size
        shuffle=True,
        num_workers=9,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Reduce prefetching
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    vit = vit_b_16(weights='IMAGENET1K_V1').to(device)

    # alexnet.classifier[6] = nn.Linear(4096, num_classes).to(device)
    vit.head = nn.Linear(768, num_classes).to(device)

    # Freeze all layers except the head
    for param in vit.parameters():
        param.requires_grad = False

    # Unfreeze the head
    vit.head.requires_grad = True


    for param in vit.encoder.layers[-1].parameters():
        param.requires_grad = True


    # Add dropout to head
    vit.head = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(768, num_classes)
    ).to(device)


    num_epochs = 30
    num_training_steps = num_epochs * len(train_loader)

    # Create the model, loss function and optimizer
    model = vit
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Better configuration would be:
    initial_lr = 0.0001
    
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    # train the model
    model, history = train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # save the model
    model_name = (
        "efficientnet_ArtClassifier"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + ".pth"
    )
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
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += int((predicted == labels).sum())
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(confusion_matrix, interpolation="nearest")
    plt.colorbar()
    
    # Add numbers to the plot
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, int(confusion_matrix[i, j]),
                    ha="center", va="center")
            
    # Save the confusion matrix plot
    plt.savefig(
        "efficientnet_confusion_matrix"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + ".png"
    )
    plt.close()

    # Move tensors to CPU and convert to numpy arrays
    train_losses = history["train_losses"].detach().cpu().numpy()
    val_losses = history["val_losses"].detach().cpu().numpy()
    val_accuracies = history["val_accuracies"].detach().cpu().numpy()

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.grid(True)
    # Save the plot
    plt.savefig(
        "efficientnet_losses"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + ".png"
    )
    plt.close()

    # Plot the validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.grid(True)
    # Save the plot
    plt.savefig(
        "efficientnet_accuracy"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + ".png"
    )
    plt.close()

    # Save the logs as csv
    logs = pd.DataFrame(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        }
    )
    logs.to_csv(
        "efficientnet_logs" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".csv",
        index=False,
    )

