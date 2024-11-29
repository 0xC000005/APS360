import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import vit_b_16, efficientnet_v2_s
from tqdm import tqdm
from datasets import load_dataset
from datasets import DatasetDict
import datetime


def val_transform_func(data):
    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    data['image'] = val_transform(data['image'])
    return data


if __name__ == '__main__':
    device = torch.device('mps')

    vit = vit_b_16(weights='IMAGENET1K_V1').to(device)
    efficientnet = efficientnet_v2_s(weights='IMAGENET21K').to(device)

    # keep the classifier and attention layers unfrozen
    for name, param in vit.named_parameters():
        if 'head' in name or 'attn' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


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

    num_classes = len(ds.features['genre'].names)

    vit.head = nn.Linear(768, num_classes).to(device)
    efficientnet.classifier = nn.Linear(1280, num_classes).to(device)
    
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

    # print a image in the training set
    # print(ds['train'][0])
    # {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1382x1888 at 0x16D7C3250>, 'genre': 6}

    # Apply transformations
    test_dataset = ds['test'].map(val_transform_func, num_proc=16, cache_file_name="./cache_test.arrow", load_from_cache_file=True).with_format('torch')

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=9,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # create the model, loss function and optimizer
    model1 = vit
    model2 = efficientnet

    model1_name = 'VitArtTorchClassifierWithWeight.pth'
    model2_name = 'EfficientNetArtTorchClassifierWithWeight.pth'
    

    # load the model
    model1.load_state_dict(torch.load(model1_name))
    model2.load_state_dict(torch.load(model2_name))

    # Evaluate the model on the test set
    # Plot the confusion matrix
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['genre'].to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)
            # _, predicted = torch.max(outputs, dim=1)
            # instead of using the model with the highest accuracy, we will use the ensemble method
            # convert the outputs to probabilities
            prob1 = torch.nn.functional.softmax(outputs1, dim=1)
            print(prob1)
            prob2 = torch.nn.functional.softmax(outputs2, dim=1)
            print(prob2)
            # combine the probabilities using the weighted average method
            # model1 has 0.67/(0.68+0.67), model2 has 0.68/(0.68+0.67)
            prob = 0.67 * prob1 + 0.68 * prob2
            print(prob)
            # sample from the probabilities using  multinomial distribution
            predicted = torch.multinomial(prob, 1).view(-1)
            print(predicted)
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
    plt.savefig('esemmble_model_confusion_matrix' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
