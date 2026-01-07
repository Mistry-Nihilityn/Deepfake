import torch
from torch import nn


class SAM(torch.optim.Optimizer):
    def __init__(self, feature_params, classifier_params, base_optimizer, rho=0.05, affect_classifier=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, **kwargs)
        self.affect_classifier = affect_classifier
        self.feature_params = [{"params": params} for params in feature_params]
        self.all_params = [{"params": params} for params in list(feature_params) + list(classifier_params)]
        self.rho = rho
        super(SAM, self).__init__(self.all_params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        params = self.all_params if self.affect_classifier else self.feature_params
        for group in params:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # 向梯度方向扰动参数
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        params = self.all_params if self.affect_classifier else self.feature_params
        for group in params:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
