import os
import torch
from torch.utils.data import random_split, Dataset
from torchvision import transforms, datasets
import numpy as np
from PIL import Image
from tqdm import tqdm

class GrayscaleToRGB:
    """Convert grayscale images to RGB."""
    def __call__(self, image):
        if image.mode != 'RGB':
            return image.convert('RGB')
        return image

def prepare_data(data_dir, batch_size=32):
    """
    Prepare and load the dataset for training and validation.
    
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Number of samples per batch.
    
    Returns:
        tuple: train_loader, val_loader, train_dataset, val_dataset
    """
    # Define image transformations
    transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and split the dataset
    print("Loading dataset...")
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    print("Splitting dataset...")
    g = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=g)

    # Print dataset statistics
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Train data: {100 * len(train_dataset) / len(full_dataset):.2f}% of full data")
    print(f"Validation data: {100 * len(val_dataset) / len(full_dataset):.2f}% of full data")

    # Create data loaders
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset

if __name__ == "__main__":
    data_dir = "data_p2/train"  
    train_loader, val_loader, train_dataset, val_dataset = prepare_data(data_dir)
    print("Data preprocessing completed.")
