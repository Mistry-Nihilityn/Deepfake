import torch
import torch.nn as nn

from metrics.registry import LOSSFUNC
from .abstract_loss_func import AbstractLossClass


@LOSSFUNC.register_module(module_name="cross_entropy")
class CrossEntropyLoss(AbstractLossClass):
    def __init__(self, config):
        super().__init__()
        if "weight" in config["dataset"]:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config["dataset"]["weight"]), reduction="none")
        else:
            self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inputs, targets, reduction="mean"):
        """
        Computes the cross-entropy loss.

        Args:
            inputs: A PyTorch tensor of size (batch_size, num_classes) containing the predicted scores.
            targets: A PyTorch tensor of size (batch_size) containing the ground-truth class indices.

        Returns:
            A scalar tensor representing the cross-entropy loss.
        """
        # Compute the cross-entropy loss
        loss = self.loss_fn(inputs, targets)
        if reduction == "mean":
            return loss.mean()
        return loss