import pandas as pd
from pathlib import Path

target_path = Path("/opt/project/data/INbreast.csv")
df = pd.read_csv(target_path, sep=';')
df.columns = df.columns.str.strip()

# 1. ADIM: Bi-Rads 3 ve 6 olan satırları silmek
# SQL: DELETE FROM df WHERE `Bi-Rads` IN ('3', '6')
# Pandas'ta bunu "3 ve 6 olmayanları tut" şeklinde yaparız (~ işareti 'değil' demektir)
silinecekler = ['3', '6', 3, 6] # Hem sayı hem string ihtimaline karşı
df = df[~df['Bi-Rads'].isin(silinecekler)]

# 2. ADIM: 4a, 4b, 4c değerlerini '4' altında toplamak
# SQL: UPDATE df SET `Bi-Rads` = '4' WHERE `Bi-Rads` IN ('4a', '4b', '4c')
degistirilecekler = {'4a': '4', '4b': '4', '4c': '4'}
df['Bi-Rads'] = df['Bi-Rads'].replace(degistirilecekler)

# 3. ADIM: Sonucu kontrol edelim
print("--- Güncel Bi-Rads Dağılımı ---")
print(df['Bi-Rads'].value_counts())

# İSTEĞE BAĞLI: Temizlenmiş veriyi yeni bir CSV olarak kaydetmek
df.to_csv('/opt/project/data/INbreast_cleaned.csv', index=False, sep=';')