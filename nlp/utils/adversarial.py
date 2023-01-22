# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 9:51
import torch

__all__ = [
    'FGM',
    'FGSM_Local',
    'FGSM',
    'PGD',
    'FreeAT_Local',
    'FreeAT'
]





class FGM():
    def __init__(self, model, emb_name='embedding'):
        self.model = model
        self.backup = {}
        self.emb_name = emb_name

    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model, emb_name='embedding'):
        self.emb_name = emb_name
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FGSM_Local(): #局部Embedding
    def __init__(self, model):
        self.model = model

    def attack(self, is_first_attack=False, epsilon=1., alpha = 0.3, delta=None):
        # emb_name这个参数要换成你模型中embedding的参数名
        if is_first_attack:
            delta.uniform_(-epsilon, epsilon)
            delta.requires_grad = True
        else:
            grad = delta.grad.detach()
            norm = torch.norm(grad)
            if norm != 0 and not torch.isnan(norm):
                delta.data = torch.clamp(delta + alpha * grad / norm, torch.tensor((-epsilon)).cuda(), torch.tensor((epsilon)).cuda())
                delta = delta.detach()
        return delta 
    
class FGSM(): #全局Embedding
    def __init__(self, model, emb_name='embedding'):
        self.emb_name = emb_name
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., alpha = 0.3, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.backup[name] = param.data.clone()
                    delta = torch.zeros_like(param.data).cuda()
                    delta.uniform_(-epsilon, epsilon)
                    param.data.add_(delta)
                else:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        delta = torch.clamp(alpha * param.grad / norm, torch.tensor(-epsilon).cuda(), torch.tensor(epsilon).cuda())
                        param.data.add_(delta)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FreeAT(): #全局Embedding
    def __init__(self, model, emb_name='embedding'):
        self.emb_name = emb_name
        self.model = model

    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = torch.clamp(epsilon * param.grad / norm, torch.tensor((-epsilon)).cuda(), torch.tensor((epsilon)).cuda())
                    param.data.add_(r_at)

class FreeAT_Local(): #局部Embedding
    def __init__(self, model):
        self.model = model

    def attack(self, epsilon=1., delta=None):
        grad = delta.grad.detach()
        norm = torch.norm(grad)
        if norm != 0 and not torch.isnan(norm):
            delta.data = torch.clamp(delta + epsilon * grad / norm, torch.tensor((-epsilon)).cuda(), torch.tensor((epsilon)).cuda())
        return delta



AdversarialMethods = {
    "pgd": PGD,
    "fgm": FGM,
    "fgsm": FGSM,
    "fgsm_local": FGSM_Local,  # 扰动计算方式不一样
    "free": FreeAT,
    "free_local": FreeAT_Local,
}