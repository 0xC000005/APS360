import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_distributed():
    """Initialize distributed training if available"""
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            return True, local_rank
    return False, 0


def main():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
        cpu=False
    )

    device = get_device()
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset("Artificio/WikiArt")

    # Get unique style labels
    labels = dataset["train"].unique("style")
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    num_labels = len(labels)

    # Load model and processor
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    # Optimize model memory usage
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing

    # Define transforms with GPU acceleration
    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

    class WikiArtDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transforms, label2id):
            self.dataset = dataset
            self.transforms = transforms
            self.label2id = label2id

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = self.transforms(item["image"].convert('RGB'))
            label = self.label2id[item["style"]]
            return {"pixel_values": image, "labels": label}

    # Create datasets
    train_dataset = WikiArtDataset(dataset["train"], transforms, label2id)

    # Create data loaders with pinned memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Pin memory for faster data transfer to GPU
        prefetch_factor=2  # Prefetch batches
    )

    # Training arguments optimized for GPU
    training_args = TrainingArguments(
        output_dir="art_style_classifier",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,
        logging_steps=10,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Optimizer settings
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_steps=500,
    )

    # Initialize trainer with accelerator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,  # Add validation dataset if needed
        data_collator=None,  # Default collator handles the batching
    )

    # Prepare for distributed training
    model, train_loader, optimizer = accelerator.prepare(
        model, train_loader, trainer.optimizer
    )

    # Train with accelerator
    with accelerator.main_process_first():
        trainer.train()

    # Save the model
    if accelerator.is_main_process:
        trainer.save_model("art_style_classifier")


@torch.inference_mode()
def predict_artwork_style(image_path, model, transforms, id2label, device):
    """Optimized prediction function"""
    from PIL import Image

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    # Batch processing for efficiency
    with torch.cuda.amp.autocast():  # Use mixed precision for inference
        inputs = transforms(image).unsqueeze(0).to(device)
        outputs = model(inputs)
        predictions = outputs.logits.softmax(dim=-1)
        predicted_idx = predictions.argmax().item()

    predicted_style = id2label[predicted_idx]
    confidence = predictions[0][predicted_idx].item()

    return predicted_style, confidence


if __name__ == '__main__':
    main()