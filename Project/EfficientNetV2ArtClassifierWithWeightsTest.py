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

    # Drop all columns except for image and genre
    ds = ds.remove_columns(
        [col for col in ds.column_names if col not in ["image", "genre"]]
    )

    # Print total size
    total_samples = len(ds)
    print(f"Total dataset size: {total_samples}")

    if total_samples != 81444:
        print(
            f"Warning: Dataset size ({total_samples}) differs from documentation (11,320)"
        )

    # Show genre distribution
    genre_counts = pd.Series(ds["genre"]).value_counts()
    print("\nGenre distribution:")
    print(genre_counts)
    print(f"Total samples from genre counts: {genre_counts.sum()}")

    # Print out all genre mappings
    print("Genre mappings:")
    print(ds.features["genre"].names)

    num_classes = len(ds.features["genre"].names)

    efficientnet = efficientnet_v2_s(weights='IMAGENET1K_V1')
    efficientnet.classifier[1] = nn.Linear(1280, num_classes)
    efficientnet = efficientnet.to(device)
    # alexnet.classifier[6] = nn.Linear(4096, num_classes).to(device)

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

    print("\nSplit sizes:")
    print(f"Training: {len(ds['train'])} ({len(ds['train'])/total_samples:.1%})")
    print(
        f"Validation: {len(ds['validation'])} ({len(ds['validation'])/total_samples:.1%})"
    )
    print(f"Test: {len(ds['test'])} ({len(ds['test'])/total_samples:.1%})")

    # Verify no data loss
    total_after_split = len(ds["train"]) + len(ds["validation"]) + len(ds["test"])
    print(f"\nVerification:")
    print(f"Sum of all splits: {total_after_split}")
    print(f"Original total: {total_samples}")
    assert total_after_split == total_samples, "Data loss during splitting!"

    # print a image in the training set
    # print(ds['train'][0])
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1382x1888 at 0x16D7C3250>, 'genre': 6}

    test_dataset = (
        ds["test"]
        .map(
            val_transform_func,
            num_proc=16,
            cache_file_name="./cache_test.arrow",
        )
        .with_format("torch")
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # create the model, loss function and optimizer
    model = efficientnet
   
    # load the model
    model_name = "efficientnet_ArtClassifier20241124081624.pth"
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
            labels = batch["genre"].to(device)
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


# Calculate per-class metrics
metrics = {}
for i in range(num_classes):
    # Get true positives, false positives, false negatives
    tp = confusion_matrix[i, i]
    fp = confusion_matrix[:, i].sum() - tp
    fn = confusion_matrix[i, :].sum() - tp
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = tp / confusion_matrix[i, :].sum() if confusion_matrix[i, :].sum() > 0 else 0
    
    metrics[i] = {
        'genre': ds.features["genre"].names[i],
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

    # Convert to DataFrame for better display
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df[['genre', 'accuracy', 'precision', 'recall', 'f1']]
    metrics_df = metrics_df.round(4)
    
    # Print results
    print("\nPer-class Performance Metrics:")
    print(metrics_df)
    
    # Save metrics to CSV
    metrics_df.to_csv(
        "efficientnet_per_class_metrics" 
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S") 
        + ".csv",
        index=False
    )
    
    # Calculate and display average metrics
    avg_metrics = metrics_df[['accuracy', 'precision', 'recall', 'f1']].mean()
    print("\nAverage Metrics:")
    print(avg_metrics.round(4))
    
    # Plot heatmap-style visualization using basic matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(metrics_df[['accuracy', 'precision', 'recall', 'f1']].values, aspect='auto')
    plt.colorbar()
    plt.xticks(range(4), ['Accuracy', 'Precision', 'Recall', 'F1'])
    plt.yticks(range(len(metrics_df)), metrics_df['genre'])
    
    # Add text annotations to the heatmap
    for i in range(len(metrics_df)):
        for j, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            plt.text(j, i, f'{metrics_df[metric].iloc[i]:.3f}', 
                    ha='center', va='center')
    
    plt.title('Per-class Performance Metrics')
    plt.tight_layout()
    plt.savefig(
        "efficientnet_metrics_heatmap"
        + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + ".png"
    )
    plt.close()
    
