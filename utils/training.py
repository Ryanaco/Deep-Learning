"""
Training utilities and trainer class.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, train_loader, test_loader, device, 
                 learning_rate=0.001, weight_decay=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'test_ece': [],
        }
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            if self.model.needs_labels:
                outputs, _ = self.model(images, labels)
            else:
                outputs, _ = self.model(images)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on test set."""
        from .metrics import evaluate_model
        
        results = evaluate_model(self.model, self.test_loader, self.device)
        return results
    
    def train(self, num_epochs):
        """Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        
        Returns:
            history: Dictionary with training history
        """
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 60)
        
        best_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            eval_results = self.evaluate()
            test_acc = eval_results['accuracy'] * 100
            test_ece = eval_results['ece']
            
            # Update scheduler
            self.scheduler.step(test_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)
            self.history['test_ece'].append(test_ece)
            
            # Print summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Acc: {test_acc:.2f}% | Test ECE: {test_ece:.4f}")
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                print(f"New best accuracy: {best_acc:.2f}%")
        
        print("\n" + "=" * 60)
        print(f"Training completed! Best test accuracy: {best_acc:.2f}%")
        print("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")
