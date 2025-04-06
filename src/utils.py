import torch
import pandas as pd

def get_device():
    """
    Determine the available device for computation.
    Returns 'cuda' for NVIDIA GPU, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def save_training_results(learning_rates, train_losses, val_losses, train_accuracies, val_accuracies, filename='training_results.csv'):
    """
    Save training results to a CSV file.
    
    Args:
    - learning_rates: List of learning rates for each epoch
    - train_losses: List of training losses for each epoch
    - val_losses: List of validation losses for each epoch
    - train_accuracies: List of training accuracies for each epoch
    - val_accuracies: List of validation accuracies for each epoch
    - filename: Name of the output CSV file
    """
    results = pd.DataFrame({
        'epoch': range(1, len(learning_rates) + 1),
        'learning_rate': learning_rates,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    results.set_index('epoch', inplace=True)
    results.to_csv(filename)
    print(f"Training results saved to {filename}")
