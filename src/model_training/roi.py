import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def forced_grayscale_crop():
    model_path = "/app/data/best.pt"
    source_dir = Path("/media/agah/Sata/Breast_veriler/Teknofest_Breast_PNG")
    target_dir = Path("/media/agah/Sata/Breast_veriler/Teknofest_Breast_PNG_Kirpilmis")

    print("YOLO tespit modeli yükleniyor...")
    model = YOLO(model_path)

    image_paths = list(source_dir.rglob("*.png"))
    print(f"Toplam {len(image_paths)} görüntü tek kanala (Grayscale) zorlanarak kırpılacak...")

    for img_path in tqdm(image_paths):
        relative_path = img_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. YOLO tespiti için standart 3 kanallı okuma (YOLO 3 kanal bekler)
        img_color = cv2.imread(str(img_path))
        if img_color is None:
            continue

        results = model(img_color, verbose=False)

        # 2. KAYIT İÇİN ZORUNLU TEK KANAL (GRAYSCALE) OKUMA
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            h, w = img_gray.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 3. Kırpma işlemini tek kanallı, hafif resim üzerinden yap
            cropped_img = img_gray[y1:y2, x1:x2]

            # Maksimum sıkıştırma ile kaydet
            cv2.imwrite(str(target_path), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(target_path), img_gray, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    print(f"\nKırpma bitti! Saf tek kanallı hafif veriler burada: {target_dir}")


if __name__ == '__main__':
    forced_grayscale_crop()