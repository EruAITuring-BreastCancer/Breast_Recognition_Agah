from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
from sklearn.metrics import f1_score, accuracy_score, classification_report

# 1. Hatalı resimleri çökmek yerine siyah matrisle atlayan yama (Monkey Patch)
original_cvtColor = cv2.cvtColor


def safe_cvtColor(src, code, dst=None, dstCn=0):
    try:
        if src is None or not isinstance(src, np.ndarray) or src.size == 0:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return original_cvtColor(src, code, dst, dstCn)
    except cv2.error:
        return np.zeros((224, 224, 3), dtype=np.uint8)


cv2.cvtColor = safe_cvtColor


def main():
    # 2. Daha büyük model seçimi: Medium (~10.4 Milyon Parametre)
    model = YOLO('yolo26l-cls.pt')

    print("Eğitim başlatılıyor...")
    results = model.train(
        data='/media/agah/Sata/Breast_veriler/yolo_dataset',
        epochs=50,
        imgsz=224,
        batch=64,  # 8GB VRAM'i verimli kullanmak için
        device='cuda',
        workers=4  # Sata diski boğmamak için düşük tutuyoruz
    )

    # 3. Eğitim bittikten sonra kapsamlı F1 Skoru hesaplaması
    print("\n" + "=" * 50)
    print("EĞİTİM BİTTİ - TEST VE F1 SKORU HESAPLANIYOR...")
    print("=" * 50)

    # Eğitilen en iyi modelin ağırlıklarını yükle
    best_model_path = str(results.save_dir) + '/weights/best.pt'
    best_model = YOLO(best_model_path)

    val_dir = '/media/agah/Sata/Breast_veriler/yolo_dataset/val'
    classes = sorted(os.listdir(val_dir))

    y_true = []
    y_pred = []

    print("Doğrulama (Validation) seti üzerinden tahminler alınıyor...")
    for idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(val_dir, cls_name)
        img_paths = glob.glob(os.path.join(cls_dir, '*.png'))

        if len(img_paths) > 0:
            # Toplu tahmin (hızlı olması için)
            preds = best_model(img_paths, verbose=False)
            for p in preds:
                y_pred.append(p.probs.top1)
                y_true.append(idx)

    # Metrikleri hesapla
    acc = accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
    weighted_f1 = f1_score(y_true, y_pred, average='weighted') * 100

    print(f"\nModel: YOLO11 Medium")
    print(f"Validation Accuracy: {acc:.2f}%")
    print(f"Validation F1-Score (Macro): {macro_f1:.2f}%")
    print(f"Validation F1-Score (Weighted): {weighted_f1:.2f}%\n")

    print("Detaylı Sınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=classes))


if __name__ == '__main__':
    main()