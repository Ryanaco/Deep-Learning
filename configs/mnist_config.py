"""
Configuration for MNIST experiments.
"""

# Dataset settings
DATASET_NAME = 'mnist'
BATCH_SIZE = 128
NUM_WORKERS = 4

# Model settings
LATENT_DIMS = [2, 5, 10, 20, 50]  # Different latent dimensions to try
DEFAULT_LATENT_DIM = 10

# Classifier settings
CLASSIFIERS = {
    'dotproduct': {},
    'rbf': {'gamma': 20.0},
    'cosface': {'s': 30.0, 'm': 0.35},
    'arcface': {'s': 30.0, 'm': 0.5},
    'hybrid': {'alpha': 0.5, 'gamma': 1.0},
    'adaptive_rbf': {'s':1},
    'mahalanobis': {},
    "dnc":{ 'K':3, 'alpha':10}
}

# Training settings
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Output settings
OUTPUT_DIR = './results/mnist'
CHECKPOINT_DIR = './checkpoints/mnist'
