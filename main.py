import torch
import numpy as np
from data_preprocessing import prepare_data
from model import get_model, get_loss_function, get_optimizer
from train import train, predict, early_stopping
from utils import get_device, save_training_results
from torch.optim.lr_scheduler import StepLR
import argparse

def main(args):
    # Set up device (CPU/GPU)
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Preparing data...")
    train_loader, val_loader, _, _ = prepare_data(args.data_dir, args.batch_size)
    print("Data preprocessing completed.")

    # Initialize model, loss function, optimizer, and scheduler
    print("Initializing model...")
    model = get_model(num_classes=5, device=device)
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model, args.learning_rate)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.2)

    # Train the model
    print("Starting training...")
    results = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        checkpoint_path=args.checkpoint_path,
        early_stopping=early_stopping
    )
    
    # Process and display training results
    learning_rates, train_losses, val_losses, train_accuracies, val_accuracies, epochs_run = results
    print(f"\nTraining completed after {epochs_run} epochs")
    print(f"Final training accuracy: {train_accuracies[-1]*100:.2f}%")
    print(f"Final validation accuracy: {val_accuracies[-1]*100:.2f}%")
    
    # Save training history
    save_training_results(learning_rates, train_losses, val_losses, train_accuracies, val_accuracies, 'training_results.csv')

    # Load best model and make predictions
    best_model = get_model(num_classes=5, device=device)
    best_model.load_state_dict(torch.load(args.checkpoint_path)['model_state_dict'])
    val_probs = predict(best_model, val_loader, device)
    val_preds = torch.argmax(val_probs, dim=1).cpu().numpy()

    # Get true labels and save results
    val_true = []
    for _, labels in val_loader:
        val_true.extend(labels.numpy())
    val_true = np.array(val_true)

    np.save('val_true.npy', val_true)
    np.save('val_preds.npy', val_preds)
    print("Validation predictions and true labels saved.")

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Train Cassava Disease Classifier")
    parser.add_argument("--data_dir", type=str, default="data_p2/train", help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--checkpoint_path", type=str, default="best_model.pth", help="Path to save the best model")
    
    args = parser.parse_args()
    main(args)
