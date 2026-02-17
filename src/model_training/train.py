"""
Model eğitimi, validasyon ve test fonksiyonları.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns


class Trainer:
    """Model eğitimi için ana sınıf."""

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_classes: int,
            class_weights: Optional[torch.Tensor] = None,
            device: str = 'cuda',
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            output_dir: str = 'outputs'
    ):
        """
        Args:
            model: Eğitilecek PyTorch modeli
            train_loader: Eğitim DataLoader
            val_loader: Validation DataLoader
            num_classes: Sınıf sayısı
            class_weights: Dengesiz veri için sınıf ağırlıkları
            device: 'cuda' veya 'cpu'
            learning_rate: Başlangıç öğrenme oranı
            weight_decay: L2 regularization
            output_dir: Model ve sonuçların kaydedileceği dizin
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss fonksiyonu (class_weights ile)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Eğitim geçmişi
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }

        # En iyi model bilgileri
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0

        print(f"\n=== Trainer Başlatıldı ===")
        print(f"Device: {device}")
        print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Class weights kullanılıyor: {class_weights is not None}")

    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """
        Bir epoch eğitim.

        Args:
            epoch: Epoch numarası

        Returns:
            (ortalama loss, accuracy, f1_score)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        epoch_f1 = f1_score(all_targets, all_preds, average='macro') * 100

        return epoch_loss, epoch_acc, epoch_f1

    def validate(self, epoch: int) -> Tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1} [Val]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        epoch_f1 = f1_score(all_targets, all_preds, average='macro') * 100

        return epoch_loss, epoch_acc, epoch_f1

    def train(self, num_epochs: int, save_best: bool = True) -> Dict:
        print(f"\n=== Eğitim Başlıyor ({num_epochs} epoch) ===\n")

        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)

            val_loss, val_acc, val_f1 = self.validate(epoch)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rates'].append(current_lr)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val F1:   {val_f1:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f"  ✓ Yeni en iyi model! (Val F1: {val_f1:.2f}%, Val Acc: {val_acc:.2f}%)")

                if save_best:
                    self.save_checkpoint(
                        epoch,
                        filename='best_model.pth',
                        is_best=True
                    )

            print("-" * 60)

        print(f"\n=== Eğitim Tamamlandı ===")
        print(f"En iyi validation F1-score: {self.best_val_f1:.2f}% (Epoch {self.best_epoch + 1})")
        print(f"En iyi validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch + 1})")

        self.save_checkpoint(num_epochs - 1, filename='last_model.pth')

        self.plot_history()

        return self.history

    def save_checkpoint(self, epoch: int, filename: str = 'checkpoint.pth',
                        is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }

        filepath = self.output_dir / filename
        torch.save(checkpoint, filepath)
        print(f"{'  ' if not is_best else ''}Model kaydedildi: {filepath}")

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.history = checkpoint['history']

        print(f"Checkpoint yüklendi: {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"  Best Val F1: {self.best_val_f1:.2f}%")

    def plot_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss Grafiği', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Accuracy Grafiği', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.history['train_f1'], label='Train F1', linewidth=2, color='green')
        axes[1, 0].plot(self.history['val_f1'], label='Val F1', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12)
        axes[1, 0].set_title('F1-Score Grafiği (Macro Average)', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.history['learning_rates'], linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Grafiği', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Eğitim grafikleri kaydedildi: {save_path}")
        plt.close()


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        class_names: Optional[List[str]] = None,
        output_dir: str = 'outputs'
) -> Dict:

    model.eval()
    model = model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    print("\n=== Test Başlıyor ===")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    accuracy = 100. * (all_preds == all_targets).sum() / len(all_targets)

    f1_macro = f1_score(all_targets, all_preds, average='macro') * 100
    f1_weighted = f1_score(all_targets, all_preds, average='weighted') * 100
    f1_per_class = f1_score(all_targets, all_preds, average=None) * 100

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(all_targets)))]

    report = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4
    )

    print("\n=== Test Sonuçları ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1-Score (Macro):    {f1_macro:.2f}%")
    print(f"F1-Score (Weighted): {f1_weighted:.2f}%")
    print("\nSınıf Bazında F1-Scores:")
    for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
        print(f"  {name}: {f1:.2f}%")

    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(all_targets, all_preds)

    # Confusion matrix görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cm_path = output_path / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix kaydedildi: {cm_path}")
    plt.close()

    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': {name: float(f1) for name, f1 in zip(class_names, f1_per_class)},
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist(),
        'probabilities': all_probs.tolist()
    }

    results_path = output_path / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test sonuçları kaydedildi: {results_path}")

    return results