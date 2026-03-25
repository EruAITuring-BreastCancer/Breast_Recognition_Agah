import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import os


class FeatureDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df['image_path'].tolist()
        self.labels = df['mapped_label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, path
        except:
            return torch.zeros((3, 224, 224)), -1, path


def extract_features(df, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = models.resnet50(weights=None)
    weight_path = '/app/medical_weights/ResNet50.pt'
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
        model.load_state_dict(state_dict, strict=False)
        print("RadImageNet ResNet50 ağırlıkları başarıyla yüklendi.")
    else:
        print(f"Uyarı: {weight_path} bulunamadı!")

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FeatureDataset(df, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []
    all_paths = []

    print("Derin özellikler (embeddings) çıkarılıyor...")
    with torch.no_grad():
        for images, labels, paths in tqdm(loader):
            images = images.to(device)
            features = model(images)

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    all_features = np.vstack(all_features)
    return all_features, np.array(all_labels), np.array(all_paths)


def remove_outliers_4class_strict(csv_path, output_csv, drop_ratio=0.20):
    print(f"Teknofest CSV Okunuyor ve 4 Sınıf İçin Agresif Yüzdelik Kesim (%{int(drop_ratio * 100)}) Uygulanıyor...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # BI-RADS 1, 2, 4, 5 etiketlerini sırasıyla 0, 1, 2, 3'e eşleştiriyoruz
    label_map = {1: 0, 2: 1, 4: 2, 5: 3}

    # Eğer orijinal Excel'inde sütun adı 'label' veya farklıysa burayı ona göre güncellemelisin.
    # Varsayılan olarak 'birads_label' sütununu arıyoruz.
    target_column = 'BI_RADS' if 'BI_RADS' in df.columns else 'label'

    df = df[df[target_column].isin(label_map.keys())].copy()
    df['mapped_label'] = df[target_column].map(label_map)

    # Fiziksel olarak var olan dosyaları filtrele
    df = df[df['image_path'].apply(lambda x: os.path.exists(str(x)))]

    features, labels, paths = extract_features(df)
    keep_indices = []

    # 4 sınıfın her biri için aykırı değerleri kes
    for current_class in [0, 1, 2, 3]:
        class_idx = np.where(labels == current_class)[0]

        # Orijinal BI-RADS etiketini konsolda göstermek için ters eşleştirme (reverse mapping)
        original_birads = list(label_map.keys())[list(label_map.values()).index(current_class)]

        print(
            f"\nSınıf {current_class} (BI-RADS {original_birads}) analiz ediliyor (En az benzeyen %{int(drop_ratio * 100)} KESİN silinecek)...")

        class_features = features[class_idx]

        # 1. Sınıfın ideal merkezini bul
        class_center = np.mean(class_features, axis=0, keepdims=True)
        # 2. Her resmin bu merkeze olan uzaklığını ölç
        distances = cdist(class_features, class_center, metric='cosine').flatten()

        # 3. Barajı (Threshold) belirle
        threshold = np.percentile(distances, 100 - (drop_ratio * 100))

        # 4. Barajın altında (merkeze yakın) olanları tut
        valid_idx = class_idx[distances <= threshold]
        keep_indices.extend(valid_idx)

        removed_count = len(class_idx) - len(valid_idx)
        print(f"-> BI-RADS {original_birads} için {len(class_idx)} resimden {removed_count} resim atıldı.")

    clean_paths = paths[keep_indices]
    clean_labels = labels[keep_indices]

    # Orijinal BI-RADS etiketlerini geri yükleyerek yeni CSV'yi kaydet
    clean_df = pd.DataFrame({
        'image_path': clean_paths,
        'birads_label': [list(label_map.keys())[list(label_map.values()).index(l)] for l in clean_labels]
    })

    clean_df.to_csv(output_csv, index=False)
    print(f"\nİşlem tamam! Dört sınıflı agresif temizlenmiş veri seti kaydedildi: {output_csv}")


if __name__ == '__main__':
    # Dosya yolları (Teknofest veri setinin bulunduğu orijinal CSV dosyasını belirt)
    INPUT_CSV = "/media/agah/Sata/Breast_veriler/Etiketler/kirpilmis_etiketler.csv"  # Orijinal dosyanın adını buraya gir
    OUTPUT_CSV = "/media/agah/Sata/Breast_veriler/Etiketler/teknofest_4class_homojen.csv"

    # %20 kesim (drop_ratio=0.20)
    remove_outliers_4class_strict(INPUT_CSV, OUTPUT_CSV, drop_ratio=0.20)