import logging

import torch
import torch.nn as nn
from torchvision import models

from metrics.registry import BACKBONE
from networks.base_backbone import AbstractBackbone

logger = logging.getLogger(__name__)


class ResNet50(nn.Module, AbstractBackbone):

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.resnet(data_dict)

    def feature_params(self):
        return self.resnet.parameters()

    def classifier_params(self):
        yield self.fc.parameters()

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.fc(features)

    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features  # 2048 for ResNet50
        self.resnet.fc = nn.Sequential()  # 2 outputs: fake (0) or real (1)
        self.fc = nn.Linear(num_features, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@BACKBONE.register_module("resnet50")
def resnet50(config):
    return ResNet50(num_classes=config["num_classes"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
