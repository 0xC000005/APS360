import matplotlib.pyplot as plt
import re
import numpy as np

def parse_log_file(file_path):
    # Initialize lists to store metrics
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Read the log file
    with open(file_path, 'r') as f:
        log_content = f.read()
    
    # Extract epoch information using regex
    epoch_pattern = r"Epoch (\d+\.0): val_loss=([\d.]+), val_acc=([\d.]+)"
    epoch_matches = re.finditer(epoch_pattern, log_content)
    
    for match in epoch_matches:
        epoch = float(match.group(1))
        val_loss = float(match.group(2))
        val_acc = float(match.group(3))
        
        epochs.append(epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    # Extract training loss information
    train_pattern = r"Step \d+: training_loss=([\d.]+)"
    train_matches = re.finditer(train_pattern, log_content)
    
    current_epoch_losses = []
    current_step = 0
    steps_per_epoch = len(re.findall(train_pattern, log_content)) // len(epochs)
    
    for match in train_matches:
        train_loss = float(match.group(1))
        current_epoch_losses.append(train_loss)
        current_step += 1
        
        if current_step % steps_per_epoch == 0:
            train_losses.append(np.mean(current_epoch_losses))
            current_epoch_losses = []
    
    return epochs, train_losses, val_losses, val_accuracies

def plot_training_metrics(epochs, train_losses, val_losses, val_accuracies):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss over Epochs')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy over Epochs')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('training_metrics.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # # Save the log content to a file
    # log_content = """[Your log content here]"""  # Replace with actual log content
    
    # with open('training.log', 'w') as f:
    #     f.write(log_content)

    log_name = "training_20241124_220649.log"

    # open the log file
    with open(log_name, 'r') as f:
        log_content = f.read()

        
    
    # Parse and plot
    epochs, train_losses, val_losses, val_accuracies = parse_log_file(log_name)
    plot_training_metrics(epochs, train_losses, val_losses, val_accuracies)
    
    # Print final metrics
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")