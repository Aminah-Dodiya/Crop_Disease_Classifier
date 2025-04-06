import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import get_device
from model import get_model, get_loss_function, get_optimizer

def train_epoch(model, optimizer, loss_fn, data_loader, device=None):
    """Train the model for one epoch."""
    if device is None:
        device = get_device()
    training_loss = 0.0
    model.train()

    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)

    return training_loss / len(data_loader.dataset)

def predict(model, data_loader, device=None):
    """Generate predictions for the given data loader."""
    if device is None:
        device = get_device()
    all_probs = torch.tensor([]).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)
            probs = F.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs

def score(model, data_loader, loss_fn, device=None):
    """Compute the loss and accuracy for the given data loader."""
    if device is None:
        device = get_device()
    total_loss, total_correct = 0, 0

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Scoring", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            total_loss += loss.data.item() * inputs.size(0)
            total_correct += torch.sum(torch.eq(torch.argmax(output, dim=1), targets)).item()

    n_observations = len(data_loader.dataset)
    return total_loss / n_observations, total_correct / n_observations

def early_stopping(validation_loss, best_val_loss, counter, patience=3):
    """Implement early stopping mechanism."""
    stop = False
    if validation_loss < best_val_loss:
        best_val_loss, counter = validation_loss, 0
    else:
        counter += 1
    if counter >= patience:
        stop = True
    return best_val_loss, counter, stop

def checkpointing(validation_loss, best_val_loss, model, optimizer, path):
    """Save model checkpoint if validation loss improves."""
    if validation_loss < best_val_loss:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path} with validation loss {validation_loss:.4f}")

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device=None,
          scheduler=None, checkpoint_path=None, early_stopping=None):
    """Main training loop."""
    if device is None:
        device = get_device()
    
    train_losses, train_accuracies, val_losses, val_accuracies, learning_rates = [], [], [], [], []
    best_val_loss, early_stopping_counter = float("inf"), 0

    for epoch in range(1, epochs + 1):
        print(f"\nStarting epoch {epoch}/{epochs}")

        train_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)
        train_loss, train_accuracy = score(model, train_loader, loss_fn, device)
        validation_loss, validation_accuracy = score(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch}")
        print(f"Training loss: {train_loss:.4f}, accuracy: {train_accuracy*100:.4f}%")
        print(f"Validation loss: {validation_loss:.4f}, accuracy: {validation_accuracy*100:.4f}%")

        lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(lr)

        if scheduler:
            scheduler.step()

        if checkpoint_path:
            checkpointing(validation_loss, best_val_loss, model, optimizer, checkpoint_path)

        if early_stopping:
            best_val_loss, early_stopping_counter, stop = early_stopping(
                validation_loss, best_val_loss, early_stopping_counter
            )
            if stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        best_val_loss = min(best_val_loss, validation_loss)

    print(f"Training completed after {epoch} epochs.")
    return learning_rates, train_losses, val_losses, train_accuracies, val_accuracies, epoch

if __name__ == "__main__":
    # Test the training process
    from torch.utils.data import DataLoader
    from torchvision.datasets import Data
    from torchvision.transforms import ToTensor

    # Create data for testing
    train_data = FakeData(size=1000, image_size=(3, 224, 224), num_classes=5, transform=ToTensor())
    val_data = FakeData(size=200, image_size=(3, 224, 224), num_classes=5, transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = get_device()
    model = get_model(num_classes=5, device=device)
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model)

    # Run a test training loop
    results = train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device=device)
    
    print("Test training completed successfully.")
    print(f"Final training accuracy: {results[3][-1]*100:.2f}%")
    print(f"Final validation accuracy: {results[4][-1]*100:.2f}%")
