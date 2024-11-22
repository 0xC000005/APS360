import datetime
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.models import alexnet
from tqdm import tqdm
from datasets import load_dataset



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


if __name__ == '__main__':
    device = torch.device('mps')

    alexnet = alexnet(weights=None).to(device)

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

    # load the dataset, print the number of samples in each unique genre
    ds = load_dataset("huggan/wikiart")
    print(ds['train'].unique('genre'))
    
    
    # # create data loaders
    # # Optimized DataLoader configuration
    # train_loader = DataLoader(
    #     train_data,
    #     batch_size=64,  # Reduced batch size
    #     shuffle=True,
    #     num_workers=9,
    #     pin_memory=True,
    #     drop_last=True,
    #     persistent_workers=True,  # Keep workers alive between epochs
    #     prefetch_factor=2  # Reduce prefetching
    # )

    # val_loader = DataLoader(
    #     val_data,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=9,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=2
    # )

    # test_loader = DataLoader(
    #     test_data,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=9,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=2
    # )

    # # create the model, loss function and optimizer
    # model = alexnet
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # # train the model
    # model, history = train(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=90, device=device)

    # # save the model
    # model_name = 'alexnet_ArtClassifier' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.pth'
    # torch.save(model.state_dict(), model_name)

    # # free the file workers from the train and validation loaders
    # del train_loader, val_loader

    # # load the model
    # model.load_state_dict(torch.load(model_name))

    # # Evaluate the model on the test set
    # # Plot the confusion matrix
    # model.eval()
    # correct = 0
    # total = 0
    # confusion_matrix = torch.zeros(num_classes, num_classes)
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, dim=1)
    #         total += labels.size(0)
    #         correct += int((predicted == labels).sum())
    #         for t, p in zip(labels.view(-1), predicted.view(-1)):
    #             confusion_matrix[t.long(), p.long()] += 1
    # accuracy = correct / total
    # print(f'Test Accuracy: {accuracy}')

    # # Plot the confusion matrix
    # plt.figure(figsize=(10, 10))
    # plt.imshow(confusion_matrix, interpolation='nearest')
    # plt.colorbar()
    # # Save the confusion matrix plot
    # plt.savefig('alexnet_confusion_matrix' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')

    # # Move tensors to CPU and convert to numpy arrays
    # train_losses = history['train_losses'].detach().cpu().numpy()
    # val_losses = history['val_losses'].detach().cpu().numpy()
    # val_accuracies = history['val_accuracies'].detach().cpu().numpy()

    # # Plot training and validation losses
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Losses')
    # plt.grid(True)
    # # Save the plot
    # plt.savefig('alexnet_losses' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
    # plt.close()

    # # Plot the validation accuracy
    # plt.figure(figsize=(10, 5))
    # plt.plot(val_accuracies, label='Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Validation Accuracy')
    # plt.grid(True)
    # # Save the plot
    # plt.savefig('alexnet_accuracy' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png')
    # plt.close()

    # # Save the logs as csv
    # logs = pd.DataFrame({'train_losses': train_losses, 'val_losses': val_losses, 'val_accuracies': val_accuracies})
    # logs.to_csv('alexnex_logs' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv', index=False)
