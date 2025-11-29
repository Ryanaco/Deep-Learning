"""
Evaluation script for trained models.
"""
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FullModel
from utils import (
    get_data_loaders,
    get_dataset_info,
    evaluate_model,
    visualize_results,
    plot_training_curves
)


def evaluate_trained_model(checkpoint_path, output_dir, use_tsne=True):
    """
    Evaluate a trained model and generate visualizations.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save evaluation results
        use_tsne: Whether to use t-SNE for visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model info from checkpoint path
    # Format: {dataset}_{classifier}_dim{latent_dim}.pt
    # filename = os.path.basename(checkpoint_path)
    # parts = filename.replace('.pt', '').split('_')
    # dataset_name = parts[0]
    # classifier_type = parts[1]
    # latent_dim = int(parts[2].replace('dim', ''))

    filename = os.path.basename(checkpoint_path).replace('.pt', '')
    parts = filename.split('_')
    # dataset 名字永远在第一个
    dataset_name = parts[0]
    # latent_dim 在最后一个字段，例如 dim2
    dim_part = parts[-1]
    if dim_part.startswith("dim"):
        latent_dim = int(dim_part.replace("dim", ""))
    else:
        raise ValueError(f"Cannot parse latent dim from: {filename}")
    # classifier 是中间所有部分合起来
    # 例如: ['mnist', 'adaptive', 'rbf', 'dim2'] → 'adaptive_rbf'
    classifier_type = "_".join(parts[1:-1])


    print(f"Dataset: {dataset_name}")
    print(f"Classifier: {classifier_type}")
    print(f"Latent dimension: {latent_dim}")
    
    # Get dataset info
    dataset_info = get_dataset_info(dataset_name)
    num_classes = dataset_info['num_classes']
    in_channels = dataset_info['in_channels']
    
    # Create model
    print("\nCreating model...")
    model = FullModel(
        dataset_name=dataset_name,
        classifier_type=classifier_type,
        latent_dim=latent_dim,
        num_classes=num_classes,
        in_channels=in_channels
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load data
    print("\nLoading test data...")
    _, test_loader = get_data_loaders(dataset_name, batch_size=128, num_workers=4)
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Expected Calibration Error (ECE): {results['ece']:.4f}")
    print("=" * 60)
    
    # Get prototypes
    prototypes = model.get_prototypes()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_results(
        features=results['features'],
        labels=results['labels'],
        prototypes=prototypes,
        calibration_bins=results['calibration_bins'],
        output_dir=output_dir,
        classifier_name=f"{classifier_type}_dim{latent_dim}",
        use_tsne=(latent_dim > 2 and use_tsne)
    )
    
    # Plot training curves if available
    if 'history' in checkpoint:
        history = checkpoint['history']
        plot_training_curves(
            train_losses=history['train_loss'],
            train_accs=history['train_acc'],
            val_accs=history['test_acc'],
            save_path=os.path.join(output_dir, f"{classifier_type}_dim{latent_dim}_training.png")
        )
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--use_tsne', action='store_true',
                       help='Use t-SNE for high-dimensional visualization')
    
    args = parser.parse_args()
    
    # Evaluate model
    results = evaluate_trained_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        use_tsne=args.use_tsne
    )
