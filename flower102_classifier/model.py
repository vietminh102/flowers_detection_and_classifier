import torch.nn as nn
from torchvision import models


def build_model(model_name: str = "convnext_tiny", num_classes: int = 102,
                dropout: float = 0.3, pretrained: bool = True) -> nn.Module:
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        return model

    elif model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        return model

    else:
        raise ValueError(f"Không hỗ trợ kiến trúc: {model_name}")


def set_backbone_trainable(model: nn.Module, trainable: bool):
    """Đóng băng (Stage 1) hoặc mở băng (Stage 2) toàn bộ backbone, luôn giữ head mở."""
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def get_param_groups(model: nn.Module, backbone_lr: float, head_lr: float):
    """Tách Learning Rate nhỏ cho backbone và lớn hơn cho classifier head."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]