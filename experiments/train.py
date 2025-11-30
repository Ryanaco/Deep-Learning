"""
Training script for experiments.
"""
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FullModel
from utils import get_data_loaders, get_dataset_info, Trainer


def train_model(dataset_name, classifier_type, latent_dim, num_epochs,
                learning_rate, weight_decay, batch_size, num_workers,
                classifier_kwargs=None, checkpoint_dir=None):
    """
    Train a model with specified configuration.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        classifier_type: Type of classifier ('dotproduct', 'rbf', etc.)
        latent_dim: Dimension of latent space
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        classifier_kwargs: Additional kwargs for classifier
        checkpoint_dir: Directory to save checkpoints
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    num_classes = dataset_info['num_classes']
    in_channels = dataset_info['in_channels']
    
    print(f"\nDataset: {dataset_name}")
    print(f"Classifier: {classifier_type}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, test_loader = get_data_loaders(
        dataset_name, batch_size, num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    if classifier_kwargs is None:
        classifier_kwargs = {}
    
    model = FullModel(
        dataset_name=dataset_name,
        classifier_type=classifier_type,
        latent_dim=latent_dim,
        num_classes=num_classes,
        in_channels=in_channels,
        classifier_kwargs=classifier_kwargs
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Train
    history = trainer.train(num_epochs)
    
    # Save checkpoint
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{dataset_name}_{classifier_type}_dim{latent_dim}.pt"
        )
        trainer.save_checkpoint(checkpoint_path)
    
    return model, history


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--classifier', type=str, default='dotproduct',
                       choices=['dotproduct', 'rbf', 'cosface', 'arcface', 
                               'hybrid', 'adaptive_rbf', 'mahalanobis',
                               'gaussian', 'laplacian', 'polynomial', 'sigmoid', 'gaussian_norm'])
    parser.add_argument('--latent_dim', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        dataset_name=args.dataset,
        classifier_type=args.classifier,
        latent_dim=args.latent_dim,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        num_workers=4,
        checkpoint_dir=args.checkpoint_dir
    )
