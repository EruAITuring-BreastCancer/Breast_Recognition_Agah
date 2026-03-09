import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def apply_clahe_to_dataset():
    source_base = Path("/media/agah/Sata/Breast_veriler/yolo_dataset")
    target_base = Path("/media/agah/Sata/Breast_veriler/yolo_dataset_clahe")

    # Kendi dataset.py dosyasındaki birebir aynı parametreler
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Tüm PNG dosyalarını bul
    image_paths = list(source_base.rglob("*.png"))

    print(f"Toplam {len(image_paths)} görüntüye CLAHE filtresi uygulanıyor...")

    for img_path in tqdm(image_paths):
        # Hedef klasör yapısını (train/0, val/1 vb.) aynı şekilde koruyarak oluştur
        relative_path = img_path.relative_to(source_base)
        target_path = target_base / relative_path

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Görüntüyü oku
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # LAB renk uzayına çevir ve sadece L (Lightness) kanalına CLAHE uygula
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Filtrelenmiş yeni görüntüyü SATA diskine kaydet
        cv2.imwrite(str(target_path), final_img)

    print(f"\nİşlem tamamlandı! Yeni veri seti burada: {target_base}")


if __name__ == '__main__':
    apply_clahe_to_dataset()