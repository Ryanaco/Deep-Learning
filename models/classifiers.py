"""
Different classifier heads based on various similarity functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductClassifier(nn.Module):
    """Standard softmax classifier using dot product similarity."""
    
    def __init__(self, in_features, num_classes):
        super(DotProductClassifier, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        # Standard linear layer: logits = x @ W^T
        logits = F.linear(x, self.weight)
        return logits


# class RBFClassifier(nn.Module):
#     """RBF-based classifier using radial basis function kernel."""
    
#     def __init__(self, in_features, num_classes, gamma=1.0):
#         super(RBFClassifier, self).__init__()
#         # Learnable class prototypes
#         self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
#         nn.init.xavier_uniform_(self.prototypes)
#         self.gamma = gamma
        
#     def forward(self, x):
#         # Compute squared Euclidean distances
#         # ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x, c>
#         x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)  # (batch, 1)
#         p_norm_sq = (self.prototypes ** 2).sum(dim=1)  # (num_classes,)
        
#         # Compute distances
#         distances_sq = x_norm_sq + p_norm_sq - 2 * (x @ self.prototypes.T)
        
#         # RBF kernel: exp(-gamma * distance^2)
#         # Use negative distance as logits for numerical stability
#         logits = -self.gamma * distances_sq
#         return logits
    
class RBFClassifier(nn.Module):
    """RBF-based classifier using radial basis function kernel."""
    
    def __init__(self, in_features, num_classes, gamma=1.0, s=2.0, 
                 normalize=False, mode='rbf_kernel'):
        super().__init__()
        
        # Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)

        self.gamma = gamma
        self.s = s
        self.normalize = normalize
        self.mode = mode
        
    def forward(self, x):
        # Optional normalization
        if self.normalize:
            x = F.normalize(x, dim=1)
            prototypes = F.normalize(self.prototypes, dim=1)
        else:
            prototypes = self.prototypes
        
        # Squared Euclidean distance
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        p_norm = (prototypes ** 2).sum(dim=1)
        d = x_norm + p_norm - 2 * (x @ prototypes.T)

        if self.mode == 'rbf_kernel':
            # paper RBF-Softmax
            rbf = torch.exp(-d / self.gamma)
            logits = self.s * rbf
        else:
            # stable
            logits = -d / self.gamma

        return logits



class CosFaceClassifier(nn.Module):
    """CosFace: cosine similarity with additive margin."""
    
    def __init__(self, in_features, num_classes, s=30.0, m=0.35):
        super(CosFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s  # Scale factor
        self.m = m  # Margin
        
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, labels=None):
        # Normalize features and weights
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        
        # Apply margin during training
        if self.training and labels is not None:
            # Create one-hot labels
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            
            # Add margin to target class
            cosine = cosine - one_hot * self.m
        
        # Scale
        logits = self.s * cosine
        return logits


class ArcFaceClassifier(nn.Module):
    """ArcFace: cosine similarity with angular margin."""
    
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super(ArcFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # For numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, x, labels=None):
        # Normalize features and weights
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        
        # Apply angular margin during training
        if self.training and labels is not None:
            # Calculate cos(theta + m)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            
            # Avoid numerical issues
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            # Create one-hot labels
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            
            # Apply margin to target class
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        else:
            output = self.s * cosine
            
        return output


class HybridClassifier(nn.Module):
    """Hybrid classifier combining distance and angular similarity."""
    
    def __init__(self, in_features, num_classes, alpha=0.5, gamma=1.0):
        super(HybridClassifier, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        self.alpha = alpha  # Balance between cosine and distance
        self.gamma = gamma  # RBF bandwidth
        
    def forward(self, x):
        # L2 normalize for cosine similarity
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        
        # Cosine similarity
        cosine_sim = F.linear(x_norm, p_norm)
        
        # Euclidean distance
        x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        p_norm_sq = (self.prototypes ** 2).sum(dim=1)
        distances_sq = x_norm_sq + p_norm_sq - 2 * (x @ self.prototypes.T)
        
        # Combine both similarities
        # Higher cosine = better, lower distance = better
        logits = self.alpha * cosine_sim - (1 - self.alpha) * self.gamma * distances_sq
        return logits


# class AdaptiveRBFClassifier(nn.Module):
#     """RBF classifier with learnable bandwidth per class."""
    
#     def __init__(self, in_features, num_classes):
#         super(AdaptiveRBFClassifier, self).__init__()
#         self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
#         nn.init.xavier_uniform_(self.prototypes)
        
#         # Learnable log-bandwidth for each class (ensures positive values)
#         self.log_gamma = nn.Parameter(torch.zeros(num_classes))
        
#     def forward(self, x):
#         # Compute squared distances
#         x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
#         p_norm_sq = (self.prototypes ** 2).sum(dim=1)
#         distances_sq = x_norm_sq + p_norm_sq - 2 * (x @ self.prototypes.T)
        
#         # Apply learnable bandwidth per class
#         gamma = torch.exp(self.log_gamma)  # Ensure positive
#         logits = -(gamma.unsqueeze(0) * distances_sq)
#         return logits

class AdaptiveRBFClassifier(nn.Module):
    """RBF classifier with a learnable bandwidth (gamma) for each class.

    The implementation style is based on the simple RBFLogits from test.py.
    Each class prototype has its own gamma, allowing the model to learn
    different feature space sensitivities for each class.

    Args:
        in_features: Input feature dimension.
        num_classes: Number of classes.
        s: Scale parameter for logits.
    """

    def __init__(self, in_features, num_classes, s=20.0):
        super(AdaptiveRBFClassifier, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s

        # Use log_gamma for stability, ensuring gamma is positive.
        # Initialize with a value around log(10.0) with noise to provide a
        # good starting point and break symmetry for adaptive learning.
        initial_log_gamma = math.log(6.0)
        noise = torch.randn(num_classes) * 0.1
        self.log_gamma = nn.Parameter(
            torch.full((num_classes,), initial_log_gamma) + noise
        )

    def forward(self, x):
        """Calculates Adaptive RBF logits.

        Args:
            x: Input features, shape (batch_size, in_features).

        Returns:
            Logits, shape (batch_size, num_classes).
        """
        # (batch, 1, features) - (1, classes, features) -> (batch, classes, features)
        diff = x.unsqueeze(1) - self.weight.unsqueeze(0)

        # (batch, classes)
        dist_sq = torch.sum(diff ** 2, dim=-1)

        # Ensure gamma is positive
        gamma = torch.exp(self.log_gamma)  # shape: (num_classes)

        # Apply RBF kernel with per-class gamma.
        # Unsqueeze gamma to (1, num_classes) for broadcasting over the batch.
        kernel = torch.exp(-dist_sq / gamma.unsqueeze(0))

        # Scale to get logits
        logits = self.s * kernel
        return logits
class DNCClassifier(nn.Module):
    """
    Simplified Deep Nearest Centroids Classifier (ICLR 2023)
    For each class j, maintain K learnable prototypes c_{j,k}.
    Logits are computed via soft-min over distances.
    """

    def __init__(self, in_features, num_classes, K=3, alpha=10.0):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        self.alpha = alpha  # soft-min sharpness

        # Learnable prototypes: (num_classes, K, in_features)
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, K, in_features)
        )
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        """
        x: (batch, in_features)
        returns: logits (batch, num_classes)
        """
        # Expand dims for broadcasting
        # x: (B, 1, 1, D)
        x_expanded = x.unsqueeze(1).unsqueeze(1)

        # prototypes: (1, C, K, D)
        p = self.prototypes.unsqueeze(0)

        # squared Euclidean distance: (B, C, K)
        dist_sq = torch.sum((x_expanded - p) ** 2, dim=-1)

        # soft-min over K prototypes
        # logsumexp(-alpha * dist)
        logits = -torch.logsumexp(-self.alpha * dist_sq, dim=-1)

        return logits

class MahalanobisClassifier(nn.Module):
    """Mahalanobis distance-based classifier."""
    
    def __init__(self, in_features, num_classes):
        super(MahalanobisClassifier, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Learn precision matrix (inverse covariance) for each class
        # Use Cholesky decomposition for positive definiteness
        self.log_diag = nn.Parameter(torch.zeros(num_classes, in_features))
        
    def forward(self, x):
        batch_size = x.size(0)
        num_classes = self.prototypes.size(0)
        
        # Compute difference vectors
        x_expanded = x.unsqueeze(1).expand(-1, num_classes, -1)  # (batch, classes, features)
        diff = x_expanded - self.prototypes.unsqueeze(0)  # (batch, classes, features)
        
        # Apply learned scaling (simplified precision matrix - diagonal)
        precision_diag = torch.exp(self.log_diag)  # (classes, features)
        weighted_diff = diff * precision_diag.unsqueeze(0)  # (batch, classes, features)
        
        # Mahalanobis distance
        mahalanobis_sq = (weighted_diff * diff).sum(dim=2)  # (batch, classes)
        
        logits = -mahalanobis_sq
        return logits
    
# New similarity-based classifiers (Gaussian/Laplacian/Polynomial/Sigmoid kernels)

class GaussianKernelClassifier(nn.Module):
    """Gaussian RBF kernel classifier.
    logits = s * exp(-0.5 * d / sigma^2) where d is squared Euclidean distance.
    gamma can be used instead of sigma: gamma = 1 / (2 * sigma^2).
    """
    def __init__(self, in_features, num_classes, sigma=1.0, s=10.0, normalize=False):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        self.sigma = float(sigma)
        self.s = s
        self.normalize = normalize

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, dim=1)
            prototypes = F.normalize(self.prototypes, dim=1)
        else:
            prototypes = self.prototypes

        x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        p_norm_sq = (prototypes ** 2).sum(dim=1)
        d_sq = x_norm_sq + p_norm_sq - 2 * (x @ prototypes.T)  # (B, C)

        # numerical safety
        d_sq = torch.clamp(d_sq, min=0.0)

        gamma = 1.0 / (2.0 * (self.sigma ** 2) + 1e-12)
        rbf = torch.exp(-0.5 * d_sq * (1.0 / (self.sigma ** 2 + 1e-12)))  # equivalent
        logits = self.s * rbf
        return logits


class LaplacianRBFClassifier(nn.Module):
    """Laplacian RBF kernel: exp(-||x-c|| / b). Uses Euclidean distance (not squared)."""
    def __init__(self, in_features, num_classes, b=1.0, s=10.0, normalize=False):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        self.b = float(b)
        self.s = s
        self.normalize = normalize

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, dim=1)
            prototypes = F.normalize(self.prototypes, dim=1)
        else:
            prototypes = self.prototypes

        x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        p_norm_sq = (prototypes ** 2).sum(dim=1)
        d_sq = x_norm_sq + p_norm_sq - 2 * (x @ prototypes.T)
        d_sq = torch.clamp(d_sq, min=0.0)
        d = torch.sqrt(d_sq + 1e-12)
        lap = torch.exp(-d / (self.b + 1e-12))
        logits = self.s * lap
        return logits


class PolynomialKernelClassifier(nn.Module):
    """Polynomial kernel-like classifier:
    logits = s * (alpha * <x, w> + c)^degree
    This can express non-linear separations in feature space.
    """
    def __init__(self, in_features, num_classes, degree=2, alpha=1.0, c=1.0, s=1.0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.c = float(c)
        self.s = s
        self.bias = bias
        if bias:
            self.class_bias = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter('class_bias', None)

    def forward(self, x):
        linear = F.linear(x, self.weight)  # (B, C)
        poly = self.alpha * linear + self.c
        # allow negative values if degree is odd; clamp for even degree to avoid NaN gradients if desired
        if self.degree % 2 == 0:
            poly = torch.clamp(poly, min=-1e6)  # avoid overflow for even powers
        logits = self.s * (poly.pow(self.degree))
        if self.bias:
            logits = logits + self.class_bias.unsqueeze(0)
        return logits


class SigmoidKernelClassifier(nn.Module):
    """Sigmoid (tanh) kernel-like classifier:
    logits = s * tanh(alpha * <x, w> + c)
    """
    def __init__(self, in_features, num_classes, alpha=0.1, c=0.0, s=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.alpha = float(alpha)
        self.c = float(c)
        self.s = s

    def forward(self, x):
        inner = F.linear(x, self.weight)  # (B, C)
        out = torch.tanh(self.alpha * inner + self.c)
        logits = self.s * out
        return logits
    
class NormalizedGaussianClassifier(GaussianKernelClassifier):
    """Gaussian RBF that additionally L2-normalizes the RBF responses across classes (softness)."""
    def forward(self, x):
        rbf_logits = super().forward(x)  # (B, C)
        # normalize per-sample across classes so that responses form a soft distribution
        normalized = F.normalize(rbf_logits + 1e-12, p=1, dim=1)
        # optionally rescale to original s-range
        return normalized * self.s


def get_classifier(classifier_type, in_features, num_classes, **kwargs):
    """Factory function to get the appropriate classifier."""
    classifiers = {
        'dotproduct': DotProductClassifier,
        'rbf': RBFClassifier,
        'cosface': CosFaceClassifier,
        'arcface': ArcFaceClassifier,
        'hybrid': HybridClassifier,
        'adaptive_rbf': AdaptiveRBFClassifier,
        'mahalanobis': MahalanobisClassifier,
        # new kernels
        'gaussian': GaussianKernelClassifier,
        'laplacian': LaplacianRBFClassifier,
        'polynomial': PolynomialKernelClassifier,
        'sigmoid': SigmoidKernelClassifier,
        'gaussian_norm': NormalizedGaussianClassifier,
        'dnc': DNCClassifier,
    }
    
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    return classifiers[classifier_type](in_features, num_classes, **kwargs)
