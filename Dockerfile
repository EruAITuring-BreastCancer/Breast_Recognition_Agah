# 1. Taban İmaj: PyTorch ve CUDA yüklü resmi imajı kullanıyoruz.
# Deep Learning projeleri için "slim" python yerine bunu öneririm.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. Sistem Güncellemeleri:
# Eğer oyun ortamın (env) görselleştirme için ekstra kütüphane isterse (örn: OpenGL) buraya ekleriz.
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3. Çalışma Klasörü: Konteyner içindeki "evimiz".
WORKDIR /app

# 4. Kütüphanelerin Kurulumu:
# Önce sadece requirements'ı kopyalayıp kuruyoruz ki kod değişse bile bu adım önbellekten (cache) gelsin.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Proje Kodlarının Kopyalanması:
COPY . .

# 6. Başlangıç Komutu: Konteyner açılınca ne yapsın?
# Örneğin eğitimi başlatan komut:
CMD ["python", "train.py"]