import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class Trainer:

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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        self.best_val_acc = 0.0
        self.best_epoch = 0

        print(f"\n=== Trainer Başlatıldı ===")
        print(f"Device: {device}")
        print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Class weights kullanılıyor: {class_weights is not None}")

    def train_epoch(self, epoch: int) -> Tuple[float, float]:

        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)


            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)


            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch: int) -> Tuple[float, float]:

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

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

                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs: int, save_best: bool = True) -> Dict:

        print(f"\n=== Eğitim Başlıyor ({num_epochs} epoch) ===\n")

        for epoch in range(num_epochs):
            # Eğitim
            train_loss, train_acc = self.train_epoch(epoch)

            # Validation
            val_loss, val_acc = self.validate(epoch)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f"  ✓ Yeni en iyi model! (Val Acc: {val_acc:.2f}%)")

                if save_best:
                    self.save_checkpoint(
                        epoch,
                        filename='best_model.pth',
                        is_best=True
                    )

            print("-" * 60)

        print(f"\n=== Eğitim Tamamlandı ===")
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
        self.history = checkpoint['history']

        print(f"Checkpoint yüklendi: {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")

    def plot_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Grafiği')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Grafiği')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.history['learning_rates'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Grafiği')
        axes[2].set_yscale('log')
        axes[2].grid(True)

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

    # Bu f1 olacak, accuracy de ölçelim yine de
    accuracy = 100. * (all_preds == all_targets).sum() / len(all_targets)

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