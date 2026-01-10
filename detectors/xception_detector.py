r'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XceptionDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{rossler2019faceforensics++,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1--11},
  year={2019}
}
'''

import logging
from typing import Iterable

import torch
from torch.nn import Parameter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='xception')
class XceptionDetector(AbstractDetector):

    def classifier_params(self) -> Iterable[Parameter]:
        return self.backbone.classifier_params()

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
        if 'pretrained' in config:
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