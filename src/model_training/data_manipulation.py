import pandas as pd
import os

# 1. Klasör yollarını senin belirttiğin yapıya göre tanımlıyoruz
mapping = {
    'inbreast': '/media/agah/Sata/Breast_veriler/INBreast_Cropped/INbreast_Cropped-20260225T105615Z-1-001/INbreast_Cropped/',
    'rsna': '/media/agah/Sata/Breast_veriler/RSNA_Cropped/RSNA_Cropped/',
    'vindr': '/media/agah/Sata/Breast_veriler/Vindr/VinDR_Cropped/VinDr_Cropped/'
}


def update_csv_paths():
    base_dir = "/media/agah/Sata/Breast_veriler/Etiketler/"
    csv_input = os.path.join(base_dir, "master_dataset.csv")
    csv_output = os.path.join(base_dir, "master_dataset_local.csv")

    # CSV'yi oku
    df = pd.read_csv(csv_input)
    df.columns = df.columns.str.strip()

    def construct_new_path(row):
        source = str(row['dataset_source']).lower().strip()
        # Eski Colab yolundan sadece dosya ismini al (Örn: MG_123.png)
        filename = os.path.basename(row['image_path'])

        # İlgili kaynağın SATA üzerindeki yolunu al
        new_base = mapping.get(source)

        if new_base:
            return os.path.join(new_base, filename)
        return row['image_path']  # Eşleşme yoksa (hata olmaması için) dokunma

    # Yolları güncelle
    df['image_path'] = df.apply(construct_new_path, axis=1)

    # Yeni dosyayı kaydet
    df.to_csv(csv_output, index=False)
    print(f"✅ İşlem tamamlandı! Yeni dosya: {csv_output}")

    # Örnek bir dosyanın diskte gerçekten olup olmadığını test edelim
    sample_path = df['image_path'].iloc[0]
    if os.path.exists(sample_path):
        print(f"🔍 Doğrulama: İlk dosya diskte bulundu! -> {sample_path}")
    else:
        print(f"⚠️ Uyarı: İlk dosya bulunamadı, klasör isimlerini kontrol et: {sample_path}")


if __name__ == "__main__":
    update_csv_paths()