import os

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ConvNeXTModel(nn.Module):
    """ConvNeXT model wrapper."""

    def __init__(self, num_classes: int, model_size: str = 'tiny', pretrained: bool = True):
        super(ConvNeXTModel, self).__init__()

        if model_size == 'tiny':
            self.model = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 768
        elif model_size == 'small':
            self.model = models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 768
        elif model_size == 'base':
            self.model = models.convnext_base(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1024
        elif model_size == 'large':
            self.model = models.convnext_large(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1536
        else:
            raise ValueError(f"Geçersiz model boyutu: {model_size}")

        # Son katmanı değiştir
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class MobileNetModel(nn.Module):
    """MobileNetV3 model wrapper."""

    def __init__(self, num_classes: int, model_size: str = 'large', pretrained: bool = True):
        super(MobileNetModel, self).__init__()

        if model_size == 'small':
            self.model = models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1024
        elif model_size == 'large':
            self.model = models.mobilenet_v3_large(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1280
        else:
            raise ValueError(f"Geçersiz model boyutu: {model_size}")

        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetV2Model(nn.Module):
    """EfficientNet V2 model wrapper."""

    def __init__(self, num_classes: int, model_size: str = 's', pretrained: bool = True):
        super(EfficientNetV2Model, self).__init__()

        if model_size == 's':
            self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1280
        elif model_size == 'm':
            self.model = models.efficientnet_v2_m(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1280
        elif model_size == 'l':
            self.model = models.efficientnet_v2_l(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 1280
        else:
            raise ValueError(f"Geçersiz model boyutu: {model_size}")

        # Son katmanı değiştir
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    """Standard ResNet-50 model wrapper."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(ResNetModel, self).__init__()

        self.model = models.resnet50(weights=None)
        if pretrained:
            model_path = '/app/medical_weights/ResNet50.pt'
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            self.model.load_state_dict(state_dict, strict=False)
        # Sınıflandırma için son FC (Tam Bağlantılı) katmanını num_classes'a göre değiştir
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Aşırı öğrenmeyi (overfitting) engellemek için dropout
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class SwinModel(nn.Module):
    """Swin Transformer model wrapper."""

    def __init__(self, num_classes: int, model_size: str = 'tiny', pretrained: bool = True):
        super(SwinModel, self).__init__()

        # Torchvision'daki Swin modellerini yüklüyoruz
        if model_size == 'tiny':
            self.model = models.swin_t(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_size == 'small':
            self.model = models.swin_s(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_size == 'base':
            self.model = models.swin_b(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f"Geçersiz Swin boyutu: {model_size}")

        # Swin mimarisinde sınıflandırma başlığı 'head' altındadır
        in_features = self.model.head.in_features

        # Aşırı öğrenmeyi önlemek için Dropout ekleyip num_classes'a göre ayarlıyoruz
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


import timm


class VisionMambaModel(nn.Module):
    """Vision Mamba (Vim) model wrapper."""

    def __init__(self, num_classes: int, model_size: str = 'tiny', pretrained: bool = True):
        super(VisionMambaModel, self).__init__()

        # timm üzerinden Vision Mamba modelini çağırıyoruz
        # num_classes=0 diyerek kendi özel sınıflandırma başlığımızı (head) eklemek için modelin tepesini boş bırakıyoruz
        if model_size == 'tiny':
            self.model = timm.create_model('vim_tiny_patch16_224', pretrained=pretrained, num_classes=0)
        elif model_size == 'small':
            self.model = timm.create_model('vim_small_patch16_224', pretrained=pretrained, num_classes=0)
        else:
            raise ValueError(f"Geçersiz Vim boyutu: {model_size}")

        in_features = self.model.num_features

        # 3 sınıflı yapı ve ezberi önlemek için Dropout ekliyoruz
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)


class DenseNetModel(nn.Module):
    """DenseNet121 (Dondurulmuş Gövde + Bottleneck Sınıflandırıcı)"""

    def __init__(self, num_classes: int, model_size: str = '121', pretrained: bool = True):
        super(DenseNetModel, self).__init__()

        if model_size == '121':
            self.model = models.densenet121(weights=None)
            if pretrained:
                weight_path = '/app/medical_weights/DenseNet121.pt'
                state_dict = torch.load(weight_path, map_location='cpu')
                # RadImageNet'in eski sınıflandırıcısını at
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
                self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Geçersiz boyut.")

        # 1. GÖVDEYİ DONDUR: Tüm eski ağırlıkların güncellenmesini kapat
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. YENİ ÖZEL BAŞLIK EKLENMESİ (Sadece burası eğitilecek)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(

            nn.Linear(in_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)



def get_model(model_name: str, num_classes: int, model_size: Optional[str] = None,
              pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    os.environ['TORCH_HOME'] = '/media/agah/Sata/torch_cache'  # Ağırlıklar buradan okunacak
    if model_name == 'convnext':
        size = model_size or 'tiny'
        return ConvNeXTModel(num_classes, size, pretrained)

    elif model_name == 'mobilenet':
        size = model_size or 'large'
        return MobileNetModel(num_classes, size, pretrained)

    elif model_name == 'efficientnet':
        size = model_size or 's'
        return EfficientNetV2Model(num_classes, size, pretrained)

    elif model_name in ['resnet', 'resnet50']:
        return ResNetModel(num_classes, pretrained)

    elif model_name == 'swin':
        size = model_size or 'tiny'
        return SwinModel(num_classes, size, pretrained)

    elif model_name == 'vim':
        size = model_size or 'tiny'
        return VisionMambaModel(num_classes, size, pretrained)

    elif model_name == 'densenet':
        size = model_size or '121'
        return DenseNetModel(num_classes, size, pretrained)


    else:
        raise ValueError(f"Bilinmeyen model: {model_name}. "
                         f"Kullanılabilir: 'convnext', 'mobilenet', 'efficientnet', 'resnet', 'densenet")