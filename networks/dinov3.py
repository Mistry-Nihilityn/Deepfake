import logging

import torch
import torch.nn as nn

from metrics.registry import BACKBONE
from networks.base_backbone import AbstractBackbone
from transformers import AutoModel

logger = logging.getLogger(__name__)


class DINOv3(nn.Module, AbstractBackbone):

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.dino(data_dict)

    def feature_params(self):
        return self.dino.parameters()

    def classifier_params(self):
        return self.fc.parameters()

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.dino = AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                device_map="auto",
            )
        else:
            self.dino = AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m",
                state_dict={},
                device_map="auto",
            )
        self.fc = nn.Linear(1024, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@BACKBONE.register_module("dinov3")
def dinov3(config):
    return DINOv3(num_classes=config["num_classes"], pretrained=config["pretrained"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
