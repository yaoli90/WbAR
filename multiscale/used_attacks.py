import torch
import torch.nn as nn
import numpy as np

class regression_PGD:
    def __init__(self, model, lb, ub, eps=0.1, eta=0.02, steps=20, loss=nn.L1Loss()):
        self.model = model
        self.eps = eps
        self.eta = eta
        self.steps = steps
        self.loss = loss
        self.lb = lb
        self.ub = ub
        self.device = next(model.parameters()).device
    def attack(self, samples):
        if torch.is_tensor(samples) != True:
            samples = torch.from_numpy(samples.copy()).float().to(self.device)
        else:
            samples = samples.clone().detach()
        adv_samples = samples.clone().detach()

        adv_samples += torch.empty_like(adv_samples).uniform_(-self.eps, self.eps)
        for i in range(len(self.lb)):
            adv_samples[:,i] = torch.clamp(adv_samples[:,i], min=self.lb[i], max=self.ub[i])
        adv_samples = adv_samples.detach()

        for _ in range(self.steps):
            adv_samples.requires_grad = True
            f = self.model.function(adv_samples)
            cost = self.loss(f, torch.zeros(f.shape).to(self.device))

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_samples,
                                       retain_graph=False, create_graph=False)[0]

            adv_samples = adv_samples.detach() + self.eta*grad.sign()
            delta = torch.clamp(adv_samples - samples, min=-self.eps, max=self.eps)
            adv_samples = samples + delta
            for i in range(len(self.lb)):
                adv_samples[:,i] = torch.clamp(adv_samples[:,i], min=self.lb[i], max=self.ub[i])
            adv_samples = adv_samples.detach()

        return adv_samples
