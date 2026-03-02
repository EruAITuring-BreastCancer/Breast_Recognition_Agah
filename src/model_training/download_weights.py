from torchvision import models

print("Model ağırlıkları SATA diskine indiriliyor...")

# Eğitimde kullanacağın tüm ağırlıkları internetten çekip önbelleğe alıyoruz
models.resnet50(weights='IMAGENET1K_V1')
models.convnext_small(weights='IMAGENET1K_V1')
models.efficientnet_v2_s(weights='IMAGENET1K_V1')

print("Tüm ağırlıklar başarıyla indirildi! Artık internete ihtiyacın yok.")