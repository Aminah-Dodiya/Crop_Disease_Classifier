import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchinfo import summary
from utils import get_device

class CassavaClassifier(nn.Module):
    """
    Custom CNN model for Cassava leaf disease classification.
    Uses a pre-trained ResNet50 as the base model with a custom classifier.
    """
    def __init__(self, num_classes=5, use_pretrained=True):
        super(CassavaClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)
        
        # Freeze the pre-trained model weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the final fully connected layer for our specific task
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, **kwargs):
        return self.model(x)

def get_model(num_classes=5, device=None):
    """Initialize and return the CassavaClassifier model."""
    if device is None:
        device = get_device()
    model = CassavaClassifier(num_classes=num_classes)
    return model.to(device)

def get_loss_function():
    """Return the loss function for the model."""
    return nn.CrossEntropyLoss()

def get_optimizer(model, learning_rate=0.001):
    """Initialize and return the optimizer for the model."""
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

def print_model_summary(model, batch_size=1, height=224, width=224):
    """Print a summary of the model architecture."""
    print(model)
    summary(model, input_size=(batch_size, 3, height, width), device=get_device())

if __name__ == "__main__":
    # Test and print model details
    device = get_device()
    model = get_model(device=device)
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    print("Model:")
    print(model)
    print("\nLoss Function:")
    print(loss_fn)
    print("\nOptimizer:")
    print(optimizer)
    print("\nModel Device:")
    print(next(model.parameters()).device)
    print("\nModel Summary:")
    print_model_summary(model)
    print("\nScheduler:")
    print(scheduler)
