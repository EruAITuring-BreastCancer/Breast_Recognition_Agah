import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
import timm
from sklearn.metrics import f1_score
import numpy as np
import os


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)  # Modelin doğruluk olasılığı
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def get_dataloaders(batch_size, data_dir='./data'):
    # Transformlar (Burası senin veri setine göre özelleşebilir)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # ConvNext genelde 224x224 sever
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Klasörden veriyi oku (ImageFolder yapısında olduğunu varsayıyoruz)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

    # Train/Val Split (%80 Train, %20 Val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Validation için transformu güncelle (Data leakage olmasın diye kopyalamak gerekir ama basitlik için geçiyoruz)
    # train_dataset.dataset.transform = train_transforms ... (İleri seviye detay)

    targets = [s[1] for s in train_dataset]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts

    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, len(class_counts)


def train_evaluate_model(params):
    lr = params['lr']
    batch_size = int(params['batch_size'])
    beta1 = params['beta1']
    beta2 = params['beta2']
    l1_reg = params['l1_reg']
    l2_reg = params['l2_reg']
    dropout = params['dropout']
    gamma = params['gamma']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = get_dataloaders(batch_size, data_dir='./dataset')

    model = timm.create_model('convnext_nano', pretrained=True, num_classes=num_classes, drop_rate=dropout)
    model = model.to(device)

    criterion = FocalLoss(gamma=gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=l2_reg)

    epochs = 3

    best_val_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if l1_reg > 0:
                l1_loss = 0
                for param in model.parameters():
                    l1_loss += torch.sum(torch.abs(param))
                loss += l1_reg * l1_loss

            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    return {
        'loss_to_minimize': 1 - best_val_f1,
        'f1_score': best_val_f1,
        'model_name': 'ConvNext_Nano'
    }