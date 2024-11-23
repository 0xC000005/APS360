import pandas as pd
import torch
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoImageProcessor, AutoModelForImageClassification

if __name__ == '__main__':
    device = torch.device('mps')


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

    processor = AutoImageProcessor.from_pretrained("oschamp/vit-artworkclassifier")
    model = AutoModelForImageClassification.from_pretrained("oschamp/vit-artworkclassifier", num_labels=num_classes, ignore_mismatched_sizes=True)
    
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
    # train_dataset = ds['train'].map(processor, num_proc=16, cache_file_name="~/.cache/huggingface/cached_train_transform.arrow").with_format('torch')
    # val_dataset = ds['validation'].map(processor, num_proc=16, cache_file_name="~/.cache/huggingface/cached_val_transform.arrow").with_format('torch')
    # Apply transformations
    def preprocess_images(examples):
        images = examples['image']
        inputs = processor(images=images)
        return inputs
    
    # train_dataset = ds['train'].map(preprocess_images, batched=True, num_proc=16, cache_file_name="~/.cache/huggingface/cached_train_transform.arrow")
    # val_dataset = ds['validation'].map(preprocess_images, batched=True, num_proc=16, cache_file_name="~/.cache/huggingface/cached_val_transform.arrow")
    test_dataset = ds['test'].map(preprocess_images, batched=True, num_proc=16, cache_file_name=".cache/huggingface/cached_train_transform.arrow")
