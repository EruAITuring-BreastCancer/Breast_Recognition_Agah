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
    """ResNet model wrapper."""

    def __init__(self, num_classes: int, model_size: str = '50', pretrained: bool = True):
        super(ResNetModel, self).__init__()

        if model_size == '18':
            self.model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 512
        elif model_size == '34':
            self.model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 512
        elif model_size == '50':
            self.model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            in_features = 2048
        else:
            raise ValueError(f"Geçersiz model boyutu: {model_size}. (18, 34 veya 50 kullanın)")

        # Son katmanı değiştir (Dropout eklendi)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_model(model_name: str, num_classes: int, model_size: Optional[str] = None,
              pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()

    if model_name == 'convnext':
        size = model_size or 'tiny'
        return ConvNeXTModel(num_classes, size, pretrained)

    elif model_name == 'mobilenet':
        size = model_size or 'large'
        return MobileNetModel(num_classes, size, pretrained)

    elif model_name == 'efficientnet':
        size = model_size or 's'
        return EfficientNetV2Model(num_classes, size, pretrained)

    elif model_name == 'resnet':
        size = model_size or '50'
        return ResNetModel(num_classes, size, pretrained)

    else:
        raise ValueError(f"Bilinmeyen model: {model_name}. "
                         f"Kullanılabilir: 'convnext', 'mobilenet', 'efficientnet', 'resnet'")