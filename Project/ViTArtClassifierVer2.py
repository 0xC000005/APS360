import pandas as pd
import torch
from datasets import load_dataset
from datasets import DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification
import numpy as np
import evaluate
from transformers import TrainingArguments
from transformers import Trainer

processor = AutoImageProcessor.from_pretrained("oschamp/vit-artworkclassifier")


def transforms(batch):
    batch['image'] = [x.convert('RGB') for x in batch['image']]
    inputs = processor(batch['image'], return_tensors='pt')
    inputs['labels']=batch['labels']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]).to(device),
        'labels': torch.tensor([x['labels'] for x in batch]).to(device)
    }


accuracy = evaluate.load('accuracy')
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits,axis=1)
    score = accuracy.compute(predictions=predictions, references=labels)
    return score


if __name__ == '__main__':
    device = torch.device('mps')
    ds = load_dataset("huggan/wikiart")
    ds = ds.rename_column('genre','labels')
    ds = ds.remove_columns(['artist','style'])
    labels = ds['train'].features['labels'].names
    label2id = {name: i for i, name in enumerate(labels)}
    id2label = {i: name for i, name in enumerate(labels)}

    split_dataset = ds['train'].train_test_split(test_size=0.2) # 80% train, 20% evaluation
    eval_dataset = split_dataset['test'].train_test_split(test_size=0.5) # 50% validation, 50% test

    # recombining the splits using a DatasetDict
    ds = DatasetDict({
        'train': split_dataset['train'],
        'validation': eval_dataset['train'],
        'test': eval_dataset['test']
    })

    # print the length of each split
    print({split: len(ds[split]) for split in ds})

    # Print example before transform
    print("Before transform:", ds['train'][0])

    # Apply transform using map instead of with_transform
    ds = ds.map(
        transforms,
        batched=True,
        batch_size=64,
        remove_columns=ds['train'].column_names,
        num_proc=16,
        cache_file_names={
            'train': './cache_train.arrow',
            'validation': './cache_val.arrow',
            'test': './cache_test.arrow'
        },
        load_from_cache_file=True 
    )

    # Set format to PyTorch tensors after mapping
    ds = ds.with_format("torch")

    # Print example after transform
    print("After transform:", ds['train'][0])

    # This will work:
    print(ds['train'][0]['pixel_values'].shape)  # Should be something like (3, 224, 224)

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    for name, p in model.named_parameters():
        if not name.startswith('classifier'):
            p.requires_grad = False

    num_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"{num_params = :,} | {trainable_params = :,}")

    training_args = TrainingArguments(
        output_dir="./vit-base-wikiart",
        per_device_train_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=30,
        learning_rate=3e-4,
        save_total_limit=2,
        remove_unused_columns=False,  # Changed to False
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=processor
    )

    trainer.train()
    trainer.evaluate(ds['test'])
    model.save_pretrained("ViTArtClassifierVer2")
    
