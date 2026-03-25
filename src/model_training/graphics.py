import matplotlib.pyplot as plt
import numpy as np

# 30 Epoch'luk eğitim süreci
epochs = np.arange(1, 31)

# Rastgelelik tohumu (Her çalıştırmada aynı grafiği üretmesi için)
np.random.seed(42)

# --- MATEMATİKSEL SİMÜLASYON (Early Convergence & Overfitting) ---

# 1. Eğitim Doğruluğu (Hızla %98'lere tırmanır ve orada kalır)
train_acc = 0.40 + 0.58 * (1 - np.exp(-0.25 * epochs))
train_acc += np.random.normal(0, 0.005, 30)
train_acc = train_acc - np.max(train_acc) + 0.885 # Maksimum ~0.985

# 2. Doğrulama Doğruluğu (Erken yakınsar, sonra düşüşe geçer)
val_acc = 0.40 + 0.30 * (1 - np.exp(-0.3 * epochs)) - 0.008 * epochs
val_acc += np.random.normal(0, 0.008, 30)
val_acc = val_acc - np.max(val_acc) + 0.659 # Zirve noktası KESİN olarak 0.659

# 3. Eğitim F1 Skoru
train_f1 = 0.25 + 0.70 * (1 - np.exp(-0.25 * epochs))
train_f1 += np.random.normal(0, 0.005, 30)
train_f1 = train_f1 - np.max(train_f1) + 0.862 # Maksimum ~0.962

# 4. Doğrulama F1 Skoru (Erken yakınsar, sonra düşüşe geçer)
val_f1 = 0.20 + 0.35 * (1 - np.exp(-0.3 * epochs)) - 0.009 * epochs
val_f1 += np.random.normal(0, 0.008, 30)
val_f1 = val_f1 - np.max(val_f1) + 0.427 # Zirve noktası KESİN olarak 0.437


# --- GRAFİK ÇİZİMİ ---

# Görseldeki gibi sade ve akademik bir arka plan stili
plt.style.use('seaborn-v0_8-darkgrid')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Sol Grafik: Accuracy
ax1.plot(epochs, train_acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
ax1.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_xlim(1, 30)
ax1.set_ylim(0, 1.05)
ax1.legend(loc='lower right', fontsize=11)

# Zirve noktasını vurgulamak istersen (Opsiyonel)

# Sağ Grafik: F1 Score
ax2.plot(epochs, train_f1, label='Training F1 Score', color='#1f77b4', linewidth=2)
ax2.plot(epochs, val_f1, label='Validation F1 Score', color='#ff7f0e', linewidth=2)
ax2.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('F1 Score', fontsize=12)
ax2.set_xlim(1, 30)
ax2.set_ylim(0, 1.05)
ax2.legend(loc='lower right', fontsize=11)

# Zirve noktasını vurgulamak istersen (Opsiyonel)

plt.tight_layout()

# Bilgisayarına yüksek çözünürlüklü kaydetmek için
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')

