import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import alexnet
from tqdm import tqdm
from datasets import load_dataset
from datasets import DatasetDict


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
            outputs = model(images)
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

                outputs = model(images)
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


def transform_images(examples):
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    # For training data
    if 'train' in examples:
        examples['image'] = [test_transform(image.convert('RGB')) for image in examples['image']]
    # For validation and test data
    else:
        examples['image'] = [val_transform(image.convert('RGB')) for image in examples['image']]
    return examples


if __name__ == '__main__':
    device = torch.device('mps')

    alexnet = alexnet(weights=None).to(device)

    # Load dataset and verify its size
    ds = load_dataset("huggan/wikiart")
    ds = ds['train']  # get the train split

    # Drop all columns except for image and genre
    ds = ds.remove_columns([col for col in ds.column_names if col not in ['image', 'genre']])

    # Print total size
    total_samples = len(ds)
    print(f"Total dataset size: {total_samples}")
    
    if total_samples != 81444:
        print(f"Warning: Dataset size ({total_samples}) differs from documentation (11,320)")
    
    # Show genre distribution
    genre_counts = pd.Series(ds['genre']).value_counts()
    print("\nGenre distribution:")
    print(genre_counts)
    print(f"Total samples from genre counts: {genre_counts.sum()}")


    # Print out all genre mappings
    print("Genre mappings:")
    print(ds.features['genre'].names)
    
    
    # Split with proper proportions
    splits = ds.train_test_split(train_size=0.8, seed=42)
    train_data = splits['train']
    remaining = splits['test'].train_test_split(train_size=0.5, seed=42)
    
    # Create DatasetDict
    ds = DatasetDict({
        'train': train_data,
        'validation': remaining['train'],
        'test': remaining['test']
    })
    
    print("\nSplit sizes:")
    print(f"Training: {len(ds['train'])} ({len(ds['train'])/total_samples:.1%})")
    print(f"Validation: {len(ds['validation'])} ({len(ds['validation'])/total_samples:.1%})")
    print(f"Test: {len(ds['test'])} ({len(ds['test'])/total_samples:.1%})")
    
    # Verify no data loss
    total_after_split = len(ds['train']) + len(ds['validation']) + len(ds['test'])
    print(f"\nVerification:")
    print(f"Sum of all splits: {total_after_split}")
    print(f"Original total: {total_samples}")
    assert total_after_split == total_samples, "Data loss during splitting!"

    # Apply transformations to images 
    ds['train'] = ds['train'].with_transform(transform_images)
    ds['validation'] = ds['validation'].with_transform(transform_images)
    ds['test'] = ds['test'].with_transform(transform_images)
