"""
Main script to run experiments with different configurations.
"""
import argparse
import os
import sys
import json
from datetime import datetime

from experiments import train_model, evaluate_trained_model
from configs import mnist_config, cifar10_config


def run_single_experiment(dataset_name, classifier_type, latent_dim, config):
    """Run a single experiment with specified configuration."""
    
    print("\n" + "=" * 80)
    print(f"Running experiment: {dataset_name} | {classifier_type} | dim={latent_dim}")
    print("=" * 80)
    
    # Get classifier kwargs
    classifier_kwargs = config.CLASSIFIERS.get(classifier_type, {})
    
    # Train model
    model, history = train_model(
        dataset_name=dataset_name,
        classifier_type=classifier_type,
        latent_dim=latent_dim,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        classifier_kwargs=classifier_kwargs,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    
    # Evaluate model
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{dataset_name}_{classifier_type}_dim{latent_dim}.pt"
    )
    
    output_dir = os.path.join(
        config.OUTPUT_DIR,
        f"{classifier_type}_dim{latent_dim}"
    )
    
    results = evaluate_trained_model(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        use_tsne=(latent_dim > 2)
    )
    
    return model, history, results


def run_comparison_experiment(dataset_name, classifiers, latent_dim):
    """Run comparison experiments across multiple classifiers."""
    
    # Select config
    if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
        config = mnist_config
    elif dataset_name == 'cifar10':
        config = cifar10_config
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    results_summary = []
    
    for classifier_type in classifiers:
        try:
            model, history, results = run_single_experiment(
                dataset_name=dataset_name,
                classifier_type=classifier_type,
                latent_dim=latent_dim,
                config=config
            )
            
            # Save results summary
            results_summary.append({
                'classifier': classifier_type,
                'latent_dim': latent_dim,
                'test_accuracy': results['accuracy'],
                'test_ece': results['ece'],
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1],
            })
            
        except Exception as e:
            print(f"\nError in experiment {classifier_type}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(config.OUTPUT_DIR, 'comparison_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Classifier':<20} {'Latent Dim':<12} {'Test Acc':<12} {'ECE':<12}")
    print("-" * 80)
    for result in results_summary:
        print(f"{result['classifier']:<20} {result['latent_dim']:<12} "
              f"{result['test_accuracy']*100:<12.2f} {result['test_ece']:<12.4f}")
    print("=" * 80)
    
    return results_summary


def run_latent_dim_experiment(dataset_name, classifier_type, latent_dims):
    """Run experiments with different latent dimensions."""
    
    # Select config
    if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
        config = mnist_config
    elif dataset_name == 'cifar10':
        config = cifar10_config
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    results_summary = []
    
    for latent_dim in latent_dims:
        try:
            model, history, results = run_single_experiment(
                dataset_name=dataset_name,
                classifier_type=classifier_type,
                latent_dim=latent_dim,
                config=config
            )
            
            # Save results summary
            results_summary.append({
                'classifier': classifier_type,
                'latent_dim': latent_dim,
                'test_accuracy': results['accuracy'],
                'test_ece': results['ece'],
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1],
            })
            
        except Exception as e:
            print(f"\nError in experiment dim={latent_dim}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(
        config.OUTPUT_DIR,
        f'{classifier_type}_latent_dim_comparison.json'
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\n" + "=" * 80)
    print(f"LATENT DIMENSION COMPARISON - {classifier_type}")
    print("=" * 80)
    print(f"{'Latent Dim':<12} {'Test Acc':<12} {'ECE':<12}")
    print("-" * 80)
    for result in results_summary:
        print(f"{result['latent_dim']:<12} "
              f"{result['test_accuracy']*100:<12.2f} {result['test_ece']:<12.4f}")
    print("=" * 80)
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Run similarity function comparison experiments'
    )
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'fashion_mnist', 'cifar10'],
                       help='Dataset to use')
    
    parser.add_argument('--classifier', type=str, default='gaussian_norm',
                       choices=['dotproduct', 'rbf', 'cosface', 'arcface',
                               'hybrid', 'adaptive_rbf', 'mahalanobis',
                               'gaussian', 'laplacian', 'polynomial', 'sigmoid', 'gaussian_norm','dnc'],
                       help='Classifier type')
    
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent space dimension')
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'compare_classifiers', 'compare_dims'],
                       help='Experiment mode')
    
    parser.add_argument('--classifiers', nargs='+',
                       default=['dotproduct', 'rbf', 'cosface', 'arcface'],
                       help='Classifiers to compare (for compare_classifiers mode)')
    
    parser.add_argument('--latent_dims', nargs='+', type=int,
                       default=[2, 5, 10, 20],
                       help='Latent dimensions to compare (for compare_dims mode)')
    
    args = parser.parse_args()
    
    # Select config
    if args.dataset in ['mnist', 'fashion_mnist']:
        config = mnist_config
    else:
        config = cifar10_config
    
    if args.mode == 'single':
        # Run single experiment
        run_single_experiment(
            dataset_name=args.dataset,
            classifier_type=args.classifier,
            latent_dim=args.latent_dim,
            config=config
        )
    
    elif args.mode == 'compare_classifiers':
        # Compare different classifiers
        run_comparison_experiment(
            dataset_name=args.dataset,
            classifiers=args.classifiers,
            latent_dim=args.latent_dim
        )
    
    elif args.mode == 'compare_dims':
        # Compare different latent dimensions
        run_latent_dim_experiment(
            dataset_name=args.dataset,
            classifier_type=args.classifier,
            latent_dims=args.latent_dims
        )


if __name__ == "__main__":
    main()
