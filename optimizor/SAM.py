from typing import Optional, Callable

import torch
from torch import nn


class SAM(torch.optim.Optimizer):
    def __init__(self, feature_params, classifier_params, base_optimizer_class, rho=0.05, affect_classifier=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, **kwargs)
        self.affect_classifier = affect_classifier
        feature_params = [{"params": params, "type": "feature"} for params in feature_params]
        classifier_params = [{"params": params, "type": "classifier"} for params in classifier_params]
        self.rho = rho
        super(SAM, self).__init__(feature_params + classifier_params, defaults)
        self.base_optimizer = base_optimizer_class(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            if not self.affect_classifier and group["type"] == "classifier":
                continue
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            if not self.affect_classifier and group["type"] == "classifier":
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, **kwargs):
        self.base_optimizer.step()

    def _grad_norm(self):
        params = [group for group in self.param_groups if not (group["type"] == "classifier" and not self.affect_classifier)]
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in params for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
