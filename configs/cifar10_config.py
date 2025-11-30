"""
Configuration for CIFAR-10 experiments.
"""

# Dataset settings
DATASET_NAME = 'cifar10'
BATCH_SIZE = 128
NUM_WORKERS = 4

# Model settings
LATENT_DIMS = [64, 128, 256, 512]  # Different latent dimensions to try
DEFAULT_LATENT_DIM = 128

# Classifier settings
CLASSIFIERS = {
    'dotproduct': {},
    'rbf': {'gamma': 35.0,"s":16},
    'cosface': {'s': 30.0, 'm': 0.35},
    'arcface': {'s': 30.0, 'm': 0.5},
    'hybrid': {'alpha': 0.5, 'gamma': 0.1},
    'adaptive_rbf': {},
    'mahalanobis': {},
    "dnc":{ 'K':5, 'alpha':40}
}

# Training settings
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Output settings
OUTPUT_DIR = './results/cifar10'
CHECKPOINT_DIR = './checkpoints/cifar10'
