import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class RBFLogits(nn.Module):
    def __init__(self, feature_dim, class_num, scale=20, gamma=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        self.scale = scale
        self.gamma = gamma
        nn.init.xavier_uniform_(self.weight)
        self.debug_count = 0
    
    def forward(self, feat, labels=None):
        diff = feat.unsqueeze(1) - self.weight.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        kernel = torch.exp(-dist_sq / self.gamma)
        logits = self.scale * kernel
        
        # 计算center loss（类内距离）
        center_loss = None
        if labels is not None:
            batch_size = feat.size(0)
            intra_dist = dist_sq[torch.arange(batch_size, device=feat.device), labels]
            center_loss = intra_dist.mean()
        
        if self.training and self.debug_count % 200 == 0:
            with torch.no_grad():
                feat_norm = feat.norm(dim=1).mean()
                proto_norm = self.weight.norm(dim=1).mean()
                cl = center_loss.item() if center_loss is not None else 0
                print(f"[DEBUG] feat={feat_norm:.2f}, proto={proto_norm:.2f}, "
                      f"center_loss={cl:.3f}, kernel=[{kernel.min():.4f},{kernel.max():.4f}]")
        self.debug_count += 1
        
        return logits, center_loss

class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=2, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(256, feature_dim)
        self.classifier = RBFLogits(feature_dim, num_classes, scale=20, gamma=1.0)
        
    def forward(self, x, labels=None, return_feature=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.prelu(self.fc1(x))
        feature = self.fc2(x)
        
        if return_feature:
            return feature
        
        logits, center_loss = self.classifier(feature, labels)
        return logits, center_loss

def train(model, train_loader, optimizer, epoch, center_weight=0.5):
    model.train()
    total_loss = 0
    total_ce = 0
    total_center = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        
        logits, center_loss = model(data, labels=target)
        ce_loss = F.cross_entropy(logits, target)
        
        # 总loss = CE + center_weight * center_loss
        loss = ce_loss + center_weight * center_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_center += center_loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    acc = 100. * correct / total
    n = len(train_loader)
    print(f'Epoch {epoch}: Loss={total_loss/n:.4f} (CE={total_ce/n:.4f}, Center={total_center/n:.4f}), Acc={acc:.2f}%')
    return acc

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            logits, _ = model(data, labels=target)
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    print(f'Test Acc: {acc:.2f}%')
    return acc

def visualize_features(model, test_loader, epoch, save_path='feature_vis.png'):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda()
            feat = model(data, return_feature=True)
            features.append(feat.cpu().numpy())
            labels.append(target.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    prototypes = model.classifier.weight.detach().cpu().numpy()
    
    # 统计
    print("\n" + "="*60)
    total_offset = 0
    for i in range(10):
        mask = labels == i
        center = features[mask].mean(axis=0)
        proto = prototypes[i]
        dist = np.sqrt(((center - proto)**2).sum())
        total_offset += dist
        print(f"  类{i}: 中心=({center[0]:+.2f},{center[1]:+.2f}) "
              f"Proto=({proto[0]:+.2f},{proto[1]:+.2f}) 偏差={dist:.3f}")
    print(f"  >>> 平均偏差: {total_offset/10:.3f} <<<")
    print("="*60)
    
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(range(10))
    
    for i in range(10):
        mask = labels == i
        plt.scatter(features[mask, 0], features[mask, 1], 
                   c=[colors[i]], s=5, alpha=0.5, label=str(i))
        plt.scatter(prototypes[i, 0], prototypes[i, 1], 
                   c=[colors[i]], s=200, edgecolors='black', linewidths=2)
    
    plt.legend(loc='upper right')
    plt.title(f'RBF-Softmax + CenterLoss (Epoch {epoch})')
    plt.axis('equal')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    model = SimpleCNN(feature_dim=2, num_classes=10).cuda()
    
    # 分组学习率
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = torch.optim.SGD([
        {'params': backbone_params, 'lr': 0.01},
        {'params': classifier_params, 'lr': 0.1}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    for epoch in range(1, 51):
        # center_weight: 控制center loss的权重
        train(model, train_loader, optimizer, epoch, center_weight=0.5)
        scheduler.step()
        
        if epoch % 10 == 0:
            test(model, test_loader)
            visualize_features(model, test_loader, epoch, f'vis_epoch_{epoch}.png')

if __name__ == '__main__':
    main()