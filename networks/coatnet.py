import logging

import timm
import torch
import torch.nn as nn
from metrics.registry import BACKBONE
from networks.base_backbone import AbstractBackbone

logger = logging.getLogger(__name__)


class CoAtNet2(nn.Module, AbstractBackbone):

    def features(self, data_dict: dict) -> torch.Tensor:
        return self.coatnet(data_dict)

    def feature_params(self):
        return self.coatnet.parameters()

    def classifier_params(self):
        return self.fc.parameters()

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.coatnet = timm.create_model('coatnet_2_rw_224.sw_in12k', checkpoint_path="./pretrained_weights/coatnet.safetensors")
        else:
            self.coatnet = timm.create_model('coatnet_2_rw_224.sw_in12k', pretrained=False)
        # print(self.coatnet)
        num_features = self.coatnet.head.fc.in_features  # 2048 for ResNet50
        self.coatnet.head.fc = nn.Sequential()  # 2 outputs: fake (0) or real (1)
        self.fc = nn.Linear(num_features, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@BACKBONE.register_module("coatnet_2")
def coatnet_2(config):
    return CoAtNet2(num_classes=config["num_classes"], pretrained=config["pretrained"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
