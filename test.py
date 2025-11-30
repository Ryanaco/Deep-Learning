import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ============ 1. RBF-Softmax 层 (修正参数) ============
class RBFLogits(nn.Module):
    def __init__(self, feature_dim, class_num, scale=20, gamma=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        self.scale = scale
        self.gamma = gamma
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, feat):
        diff = feat.unsqueeze(1) - self.weight.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        kernel = torch.exp(-dist_sq / self.gamma)
        logits = self.scale * kernel
        return logits

# ============ 2. 简单CNN (用PReLU，论文设置) ============
class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=2, num_classes=10):
        super().__init__()
        # 论文说用6层CNN + PReLU
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
        self.fc2 = nn.Linear(256, feature_dim)  # 2维特征
        
        # RBF分类器 - 关键参数！
        self.classifier = RBFLogits(feature_dim, num_classes, scale=1, gamma=10.0)
        
    def forward(self, x, return_feature=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.prelu(self.fc1(x))
        feature = self.fc2(x)
        
        if return_feature:
            return feature
        
        logits = self.classifier(feature)
        return logits

# ============ 3. 训练 ============
def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    acc = 100. * correct / total
    print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%')
    return acc

# ============ 4. 测试 ============
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    acc = 100. * correct / total
    print(f'Test Acc: {acc:.2f}%')
    return acc

# ============ 5. 可视化 ============
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
    
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(range(10))
    
    for i in range(10):
        mask = labels == i
        plt.scatter(features[mask, 0], features[mask, 1], 
                   c=[colors[i]], s=5, alpha=0.5, label=str(i))
        plt.scatter(prototypes[i, 0], prototypes[i, 1], 
                   c=[colors[i]], s=200, edgecolors='black', linewidths=2)
    
    plt.legend(loc='upper right')
    plt.title(f'RBF-Softmax 2D Features (Epoch {epoch})')
    plt.axis('equal')  # 保持x、y比例一致
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

# ============ 6. 主函数 ============
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 模型 - 关键参数: scale=20, gamma=10
    model = SimpleCNN(feature_dim=2, num_classes=10).cuda()
    
    # SGD + momentum，和论文一致
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    for epoch in range(1, 100):
        train(model, train_loader, optimizer, epoch)
        scheduler.step()
        
        if epoch % 5 == 0:
            test(model, test_loader)
            visualize_features(model, test_loader, epoch, f'vis_epoch_{epoch}.png')

if __name__ == '__main__':
    main()