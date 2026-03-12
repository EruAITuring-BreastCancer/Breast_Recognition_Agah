import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict
from collections import Counter
from sklearn.model_selection import train_test_split
import os
import pandas as pd



class CustomImageDataset(Dataset):

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Görüntüyü okumayı dene
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            # Eğer dosya bozuksa, terminale uyarısını bas ve bir sonraki resme atla
            print(f"\n[!] Uyarı: Bozuk dosya atlandı -> {self.image_paths[idx]}")
            new_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(new_idx)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transforms(image_size: int = 224) -> transforms.Compose:

    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),


        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:

    class_counts = Counter(labels)
    total_samples = len(labels)

    weights = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
        else:
            weights[class_idx] = 0.0

    return weights


def get_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:

    class_counts = Counter(labels)
    num_samples = len(labels)

    sample_weights = []
    for label in labels:
        weight = 1.0 / class_counts[label]
        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    # Sampler oluştur
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True  # Tekrar örneklemeye izin ver
    )

    return sampler


def create_dataloaders(
        train_image_paths: List[str],
        train_labels: List[int],
        val_image_paths: List[str],
        val_labels: List[int],
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        use_weighted_sampler: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Datasets
    train_dataset = CustomImageDataset(train_image_paths, train_labels, train_transform)
    val_dataset = CustomImageDataset(val_image_paths, val_labels, val_transform)

    num_classes = max(max(train_labels), max(val_labels)) + 1
    train_class_counts = Counter(train_labels)
    val_class_counts = Counter(val_labels)

    print("\n=== Veri Seti İstatistikleri ===")
    print(f"Toplam sınıf sayısı: {num_classes}")
    print(f"\nEğitim seti: {len(train_labels)} örnek")
    for class_idx in range(num_classes):
        count = train_class_counts.get(class_idx, 0)
        percentage = (count / len(train_labels)) * 100
        print(f"  Sınıf {class_idx}: {count} örnek ({percentage:.2f}%)")

    print(f"\nValidation seti: {len(val_labels)} örnek")
    for class_idx in range(num_classes):
        count = val_class_counts.get(class_idx, 0)
        percentage = (count / len(val_labels)) * 100
        print(f"  Sınıf {class_idx}: {count} örnek ({percentage:.2f}%)")

    # Sınıf ağırlıklarını hesapla (loss için)
    # class_weights = calculate_class_weights(train_labels, num_classes)
    class_weights = None
    # DataLoaders
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        print("\n✓ WeightedRandomSampler aktif - Her batch'te dengeli sınıf dağılımı sağlanacak")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        print("\n✗ WeightedRandomSampler kullanılmıyor - Standart shuffle aktif")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    info = {
        'num_classes': num_classes,
        'class_weights': class_weights,
        'train_size': len(train_labels),
        'val_size': len(val_labels),
        'train_class_counts': train_class_counts,
        'val_class_counts': val_class_counts
    }

    return train_loader, val_loader, info


def prepare_and_split_data(csv_path: str, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
    """
    CSV dosyasını okur, yolları doğrular ve veri setini 3'e böler.
    Artık etiketler (0,1,2) dönüştürülmüş olarak geldiği için doğrudan okunur.
    """
    print(f"CSV okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)

    image_paths = []
    labels = []
    missing_count = 0

    for index, row in df.iterrows():
        # YENİ HALİ: Artık yeni CSV'deki 'image_path' ve 'label' sütunlarını doğrudan çekiyoruz
        img_path = str(row['image_path'])
        raw_label = int(row['label'])

        # Harici diskte dosya gerçekten var mı kontrolü
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(raw_label)
        else:
            missing_count += 1

    print(f"✓ Toplam {len(image_paths)} geçerli görüntü eşleşti.")
    if missing_count > 0:
        print(f"✗ Uyarı: CSV'de olup diskte bulunamayan {missing_count} görüntü var.")

    # 1. Aşama: Önce Test setini ayır
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # 2. Aşama: Kalanı Train ve Val olarak ayır
    val_ratio_adjusted = val_size / (1.0 - test_size)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_ratio_adjusted, random_state=random_state,
        stratify=train_val_labels
    )

    print(f"Veri Dağılımı: Train({len(train_paths)}) | Val({len(val_paths)}) | Test({len(test_paths)})")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels