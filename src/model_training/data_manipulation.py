import pandas as pd
import os
import re
import glob


def create_3class_dataset():
    # 1. Dosya Yolları
    excel1_path = "/media/agah/Sata/Breast_veriler/Etiketler/Supplementary_TRAIN1.xlsx"
    excel2_path = "/media/agah/Sata/Breast_veriler/Etiketler/Supplementary_TRAIN2.xlsx"
    yeni_csv = "/media/agah/Sata/Breast_veriler/Etiketler/uclu_siniflandirma_etiketleri.csv"
    yeni_base_dir = "/media/agah/Sata/Breast_veriler/Teknofest_Breast_PNG_Kirpilmis/PNG_Görüntüler/"

    print("Excel dosyaları okunuyor...")
    df1 = pd.read_excel(excel1_path)
    df2 = pd.read_excel(excel2_path)
    df = pd.concat([df1, df2], ignore_index=True)

    # 2. Akıllı Üçlü Sınıflandırma (İçindeki sayıyı ayıklar)
    def map_birads(birads_val):
        match = re.search(r'\d+', str(birads_val))
        if match:
            val = int(match.group())
            if val == 0:
                return 0
            elif val in [1, 2]:
                return 1
            elif val in [4, 5]:
                return 2
        return None

    df['label'] = df['BIRADS CATEGORY'].apply(map_birads)
    df = df.dropna(subset=['label'])

    # 3. Hızlı arama için CASENUMBER -> Label sözlüğü oluştur
    # (Hasta numaralarını string yapıyoruz ki metin eşleştirmesinde hata çıkmasın)
    label_dict = dict(zip(df['CASENUMBER'].astype(str), df['label'].astype(int)))

    print(f"Excel'den {len(label_dict)} benzersiz hasta numarası (CASENUMBER) etiketlendi.")
    print("\nKırpılmış resim klasörü taranıyor ve hastalarla eşleştiriliyor...")

    # 4. Gerçek Resimleri Tarayıp Eşleştir
    resim_yollari = glob.glob(os.path.join(yeni_base_dir, "*.png"))
    eslesen_veriler = []

    for yol in resim_yollari:
        dosya_adi = os.path.basename(yol)

        # Dosya adının içindeki 7 veya daha uzun basamaklı o ana CASENUMBER'ı bulur
        # Örn: MG_EGITIM_1_825898305_LMLO.png -> 825898305
        match = re.search(r'(\d{7,})', dosya_adi)

        if match:
            casenumber = match.group(1)

            # Eğer resimdeki hasta numarası Excel'de varsa, CSV'ye eklenecek listeye koy
            if casenumber in label_dict:
                eslesen_veriler.append({
                    'image_path': yol,
                    'label': label_dict[casenumber]
                })

    # 5. Yeni CSV'yi Oluştur
    son_df = pd.DataFrame(eslesen_veriler)
    son_df.to_csv(yeni_csv, index=False)

    print(f"\nİşlem başarıyla tamamlandı!")
    print(f"Klasördeki {len(son_df)} adet gerçek kırpılmış resim '{yeni_csv}' dosyasına kaydedildi.")
    print("\nYeni Sınıf Dağılımı:")
    print(son_df['label'].value_counts().sort_index())


if __name__ == '__main__':
    create_3class_dataset()