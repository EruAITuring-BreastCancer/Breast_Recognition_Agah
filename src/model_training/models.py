import torch.nn as nn
from torchvision import models
from typing import Optional
import torch


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



class HybridResNetTransformer(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(HybridResNetTransformer, self).__init__()

        # ResNet-50 Omurgası
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        in_features = 2048

        # Transformer Attention Katmanı
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features, nhead=8, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, C, H, W = features.shape
        tokens = features.view(B, C, -1).permute(0, 2, 1)

        attended_tokens = self.transformer(tokens)

        features_attended = attended_tokens.permute(0, 2, 1).view(B, C, H, W)
        pooled = self.gap(features_attended).flatten(1)
        out = self.classifier(pooled)
        return out


class PaperHybridResNetTransformer(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        # 2.3.1 Backbone Network: ResNet-50
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # 2.3.2 Transformer-based Feature Fusion
        self.projection = nn.Conv2d(2048, 128, kernel_size=1)
        self.num_patches = 7 * 7
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, 128))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=512,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 2.3.4 Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)

        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


def get_model(model_name: str, num_classes: int, model_size: Optional[str] = None,
              pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    import os
    os.environ['TORCH_HOME'] = '/media/agah/Sata/torch_cache'  # PyTorch önbellek dizini
    if model_name == 'convnext':
        size = model_size or 'tiny'
        return ConvNeXTModel(num_classes, size, pretrained)  # type: ignore (eğer yukarıda tanımlıysa)

    elif model_name == 'mobilenet':
        size = model_size or 'large'
        return MobileNetModel(num_classes, size, pretrained)  # type: ignore

    elif model_name == 'efficientnet':
        size = model_size or 's'
        return EfficientNetV2Model(num_classes, size, pretrained)  # type: ignore

    elif model_name == 'resnet':
        # Yeni Hybrid Mimariyi çağırıyoruz
        return HybridResNetTransformer(num_classes, pretrained)

    elif model_name == 'paper_hybrid':
        return PaperHybridResNetTransformer(num_classes, pretrained)

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")