import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm
from datasets import load_dataset
from datasets import DatasetDict
import datetime
from datasets import ClassLabel
from datasets import concatenate_datasets


def train(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    lr_scheduler,
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
            images = batch["image"].to(device)
            labels = batch["style"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

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
                images = batch["image"].to(device)
                labels = batch["style"].to(device)

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
def train_transform_func(data):
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
    data["image"] = train_transform(data["image"])
    return data


def val_transform_func(data):
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    data["image"] = val_transform(data["image"])
    return data


if __name__ == "__main__":
    device = torch.device("mps")

    # Load dataset and verify its size
    ds = load_dataset("huggan/wikiart")
    ds = ds["train"]  # get the train split

    # Drop all columns except for image and style
    ds = ds.remove_columns(
        [col for col in ds.column_names if col not in ["image", "style"]]
    )

    # Print total size
    total_samples = len(ds)
    print(f"Total dataset size: {total_samples}")

    # Show style distribution
    style_counts = pd.Series(ds["style"]).value_counts()
    print("Style distribution:")
    print(style_counts)
    print(f"Total samples from style counts: {style_counts.sum()}")

    # Print out all style mappings
    print("Style mappings:")
    print(ds.features["style"].names)

    # Get the top 5 styles with the most samples
    top_styles = style_counts.head(5).index.tolist()
    # Print the top styles with their corresponding string names
    print("Top styles with their mappings:")
    for style in top_styles:
        print(style, ds.features["style"].names[style])
    
    # Get the minial number of samples for the top 5 styles
    min_samples = style_counts[top_styles].min()

    # Remove all other styles except for the top 5
    ds = ds.filter(lambda x: x["style"] in top_styles, num_proc=9, cache_file_name="./filtered.arrow", load_from_cache_file=True)

    # Keep only the first 1000 samples for each style using select
    import pandas as pd

    # Convert to pandas DataFrame
    df = pd.DataFrame({'index': range(len(ds)), 'style': ds['style']})

    # Group by style and take first 1000 samples from each group
    balanced_indices = df.groupby('style').head(1000)['index'].tolist()

    # Select these indices from the dataset
    ds = ds.select(balanced_indices)
        
    # Verify the balanced distribution
    print("\nDistribution after balancing (1000 samples per style):")
    print(pd.Series(ds["style"]).value_counts())

    # Create a new ClassLabel feature with only the top 5 styles
    new_style_names = [ds.features["style"].names[i] for i in top_styles]
    new_style_feature = ClassLabel(names=new_style_names)

    # Create a mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(top_styles)}

    # Update the dataset with new style indices
    def update_style_index(example):
        example["style"] = old_to_new[example["style"]]
        return example

    # Apply the mapping and set the new feature schema
    ds = ds.map(
        update_style_index,
        num_proc=9
    )
    ds = ds.cast_column("style", new_style_feature)

    # Verify the updated feature names
    print("Updated style mappings:")
    print(ds.features["style"].names)


    # Verify the new dataset distribution
    style_counts = pd.Series(ds["style"]).value_counts()
    print("Style distribution after filtering:")
    print(style_counts)

    num_classes = 5

    efficientnet = efficientnet_v2_s(weights='IMAGENET1K_V1')
    efficientnet.classifier[1] = nn.Linear(1280, num_classes)
    efficientnet = efficientnet.to(device)

    # Split with proper proportions
    splits = ds.train_test_split(train_size=0.8, seed=42)
    train_data = splits["train"]
    remaining = splits["test"].train_test_split(train_size=0.5, seed=42)

    # Create DatasetDict
    ds = DatasetDict(
        {
            "train": train_data,
            "validation": remaining["train"],
            "test": remaining["test"],
        }
    )

    # print data distribution in train
    print("Train data distribution:")
    print(pd.Series(ds["train"]["style"]).value_counts())

    # print data distribution in validation
    print("Validation data distribution:")
    print(pd.Series(ds["validation"]["style"]).value_counts())

    # print data distribution in test
    print("Test data distribution:")
    print(pd.Series(ds["test"]["style"]).value_counts())


    # Apply transformations
    train_dataset = (
        ds["train"]
        .map(
            train_transform_func,
            num_proc=16
        )
        .with_format("torch")
    )
    val_dataset = (
        ds["validation"]
        .map(
            val_transform_func,
            num_proc=16
        )
        .with_format("torch")
    )
    test_dataset = (
        ds["test"]
        .map(
            val_transform_func,
            num_proc=16
        )
        .with_format("torch")
    )

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

    num_epochs = 30
    num_training_steps = num_epochs * len(train_loader)

    # create the model, loss function and optimizer
    model = efficientnet
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Better configuration would be:
    initial_lr = 1e-5
    max_lr = 3e-4  # or find using lr_finder
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, max_lr, 0)

    # train the model
    model, history = train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        lr_scheduler,
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
            images = batch["image"].to(device)
            labels = batch["style"].to(device)
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

