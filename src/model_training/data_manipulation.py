import pandas as pd
import os
import re

# Yolları Tanımlayalım
base_dir = "/media/agah/Sata/Breast_veriler/"
image_folder = os.path.join(base_dir, "Teknofest_Breast_PNG/PNG_Görüntüler/")
input_csv = os.path.join(base_dir, "Etiketler/nihai_egitim_verisi.csv")  # Elindeki CSV
output_csv = os.path.join(base_dir, "Etiketler/teknofest_final_master.csv")


def create_exact_path_csv():
    if not os.path.exists(input_csv):
        print(f"❌ CSV dosyası bulunamadı: {input_csv}")
        return

    # 1. CSV'yi oku ve CaseNumber'ı temizle (822670189.0 -> 822670189)
    df_labels = pd.read_csv(input_csv)
    df_labels.columns = df_labels.columns.str.strip()
    df_labels['CaseNumber'] = pd.to_numeric(df_labels['CaseNumber'], errors='coerce').fillna(0).astype(int).astype(str)

    # 2. Klasördeki gerçek dosya isimlerini al (Sadece isimleri okur, resimleri açmaz)
    print("🔍 Gerçek dosya isimleri taranıyor...")
    all_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]

    file_data = []
    # Dosya isminin içindeki 7-12 haneli vaka numarasını bulacak kural
    pattern = re.compile(r'(\d{7,12})')

    for filename in all_files:
        match = pattern.search(filename)
        if match:
            case_id = match.group(1)
            file_data.append({
                'CaseNumber': str(case_id),
                'image_path': os.path.join(image_folder, filename)  # Dosya ismi neyse onu kullanır
            })

    df_files = pd.DataFrame(file_data)

    if df_files.empty:
        print("❌ Klasörde hiçbir PNG dosyası bulunamadı veya ID'ler okunamadı.")
        return

    # 3. CSV'deki etiketlerle, klasördeki gerçek yolları birleştir
    final_df = pd.merge(df_files, df_labels, on='CaseNumber', how='inner')

    # 4. İhtiyacımız olan sütunları seç ve kaydet
    final_df = final_df[['image_path', 'BI_RADS', 'CaseNumber']]
    final_df.to_csv(output_csv, index=False)

    print("\n--- İŞLEM SONUCU ---")
    print(f"📊 Toplam eşleşen ve oluşturulan satır sayısı: {len(final_df)}")

    if not final_df.empty:
        print(f"💾 Kayıt yeri: {output_csv}")
        print("\nÖrnek Çıktılar (Gerçek dosya yolları):")
        for path in final_df['image_path'].head(3):
            print(path)
    else:
        print("⚠️ HATA: Dosya isimleri tarandı ama CSV'deki numaralarla eşleşmedi.")


if __name__ == "__main__":
    create_exact_path_csv()