import logging

import torch

from detectors.base_detector import AbstractDetector
from metrics.base_metrics_class import calculate_metrics_for_train
from metrics.registry import BACKBONE, DETECTOR, LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='resnet')
class ResNetDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        backbone = backbone_class(config)
        # if donot load the pretrained weights, fail to get good results
        if 'pretrained' in config and isinstance(config['pretrained'], str):
            state_dict = torch.load(config['pretrained'])
            for name, weights in state_dict.items():
                if 'pointwise' in name:
                    state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            backbone.load_state_dict(state_dict, False)
            logger.info('Load pretrained model successfully!')
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class(config)
        return loss_func

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.features(x)  # 32,3,256,256

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(features)

    def get_losses(self, label: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        return self.loss_func(pred, label)

    def get_train_metrics(self, data: list, pred: torch.Tensor) -> dict:
        x, label = data
        auc, eer, acc, ap = calculate_metrics_for_train(label.cpu(), pred.cpu())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, x: torch.Tensor, inference=False) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def feature_params(self):
        return self.backbone.feature_params()

    def classifier_params(self):
        return self.backbone.classifier_params()

