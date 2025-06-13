# Copyright (c) EEEM071, University of Surrey

from .resnet import (
    resnet18,
    resnet18_fc512,
    resnet34,
    resnet34_fc512,
    resnet50,
    resnet50_fc512,
    resnext50_32x4d,
)
from .tvmodels import mobilenet_v3_small, vgg16, vit_b_16, swin_v2_tiny_patch4_window7_224


__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet18_fc512": resnet18_fc512,
    "resnet34": resnet34,
    "resnet34_fc512": resnet34_fc512,
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
    "mobilenet_v3_small": mobilenet_v3_small,
    "vgg16": vgg16,
    "resnext50_32x4d": resnext50_32x4d,
    "vit_b_16": vit_b_16,
    "swin_v2_tiny_patch4_window7_224": swin_v2_tiny_patch4_window7_224,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
