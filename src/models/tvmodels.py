# Copyright (c) EEEM071, University of Surrey

import torch.nn as nn
import torchvision.models as tvmodels
import timm

__all__ = ["mobilenet_v3_small", "vgg16", "vit_b_16", "swin_v2_tiny_patch4_window7_224"]


class TorchVisionModel(nn.Module):
    def __init__(self, name, num_classes, loss, pretrained, **kwargs):
        super().__init__()

        self.loss = loss
        if 'timm:' in name:
            self.backbone = timm.create_model(name.replace('timm:', ''), pretrained=pretrained, **kwargs)
            # Adjust how we extract the feature dimension and reset the classifier
            if hasattr(self.backbone, 'get_classifier'):
                self.feature_dim = self.backbone.get_classifier().in_features
                self.backbone.reset_classifier(0, nn.Identity())
            else:
                raise Exception("Classifier not found in the timm model")
        else:
            self.backbone = tvmodels.__dict__[name](pretrained=pretrained)
            if 'densenet' in name:
               self.feature_dim = self.backbone.classifier.in_features
               self.backbone.classifier = nn.Identity()  # Remove final classification layer
            elif 'vgg' in name or 'mobilenet' in name:
               self.feature_dim = self.backbone.classifier[0].in_features
               self.backbone.classifier = nn.Identity()
            elif 'vit' in name:
              self.feature_dim = self.backbone.heads[0].in_features
              self.backbone.heads = nn.Identity()

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        v = self.backbone(x)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


def vgg16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vgg16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


def mobilenet_v3_small(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "mobilenet_v3_small",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model


# Define any models supported by torchvision bellow
# https://pytorch.org/vision/0.11/models.html

def resnext50_32x4d(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "resnext50_32x4d",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

def vit_b_16(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "vit_b_16",
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model

def swin_v2_tiny_patch4_window7_224(num_classes, loss={"xent"}, pretrained=True, **kwargs):
    model = TorchVisionModel(
        "timm:swin_v2_tiny_patch4_window7_224",  # Correct model name for Swin Transformer V2 Tiny from timm
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        **kwargs,
    )
    return model
