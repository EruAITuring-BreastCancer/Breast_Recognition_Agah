import os
from ultralytics import YOLO
import cv2
if __name__ == '__main__':



    # modelin yolunu buraya girdim
    model = YOLO(r"C:\Users\Agah\PycharmProjects\dicomPng\BreastRecognition\runs\detect\train32\weights\best.pt")

    # Test görüntülerini buraya girdim
    test_image_folder = r"C:\Users\Agah\Desktop\BI-RADS_5"
    output_cropped_folder = r"C:\Users\Agah\Desktop\BI-RADS_5_kirpilmis"

    os.makedirs(output_cropped_folder, exist_ok=True)

    # test görüntüleri üzerinde ilgi alanı kırp
    for image_name in os.listdir(test_image_folder):
        image_path = os.path.join(test_image_folder, image_name)
        results = model.predict(image_path, save=False, conf=0.8)  # %50 güven eşiği

        # Tahmin edilen bounding box'lardan ilgi alanını kırp
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        for i, box in enumerate(results[0].boxes.xyxy):  # x_min, y_min, x_max, y_max formatı
            x_min, y_min, x_max, y_max = map(int, box.tolist())
            cropped_image = image[y_min:y_max, x_min:x_max]  # İlgi alanını kırp

            # Kırpılan görüntüyü kaydet
            output_path = os.path.join(output_cropped_folder, f"{os.path.splitext(image_name)[0]}_crop{i}.png")
            cv2.imwrite(output_path, cropped_image)
            print(f"Kırpılan görüntü kaydedildi: {output_path}")

    """

    model = YOLO("yolov8n.pt")  # YOLOv5 Nano modelini kullan
    model.train(
        data = "C:\\Users\\Agah\\Desktop\\dataset2\\data.yaml",
        epochs=100,  # bunu arttırmam lazım
        workers=4,
        imgsz=640,
        batch=8,
        device='0'  # GPU (0) veya CPU (cpu) kullanımı
        # name="meme_modeli1

    )
"""
