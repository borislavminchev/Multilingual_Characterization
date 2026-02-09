"""
Loss Functions for Hierarchical Entity Role Classification

This module contains all loss functions used in the project:
- FocalLoss: Standard focal loss for single-label classification
- ClassBalancedCrossEntropy: Class-balanced CE (Cui et al., CVPR 2019)
- MultiLabelFocalLoss: Focal loss for multi-label classification
- AsymmetricLoss: ASL for multi-label (Ben-Baruch et al., 2021)
- AsymmetricLossOptimized: ASL with class weighting and entropy regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in single-label classification.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, samples_per_cls=None, beta=0.9999, alpha=None, gamma=2.0, 
                 reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                self.alpha = alpha
        elif samples_per_cls is not None:
            self.alpha = self._calculate_alpha(samples_per_cls, beta)
        else:
            self.alpha = None

    def _calculate_alpha(self, samples_per_cls, beta):
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        if beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - beta) / eff_num
        weights = weights / (weights.mean() + 1e-12)
        return weights
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            else:
                focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedCrossEntropy(nn.Module):
    """
    Class-balanced cross-entropy (Cui et al., CVPR 2019).
    """
    def __init__(
        self,
        samples_per_cls: Optional[Sequence[int]] = None,
        class_weights: Optional[Sequence[float]] = None,
        beta: float = 0.9999,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        device: str = "cpu",
        normalize: bool = True,
    ):
        super().__init__()
        self.beta = float(beta)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        self.device = device
        self.normalize = normalize

        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32, device=device)
            self.register_buffer("_weight", w)
        elif samples_per_cls is not None:
            weights = self._calculate_weight(samples_per_cls)
            self.register_buffer("_weight", weights)
        else:
            self.register_buffer("_weight", None)

    @property
    def weight(self):
        return self._weight
    
    def _calculate_weight(self, samples_per_cls: Sequence[int]):
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        if self.beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(self.beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - self.beta) / eff_num

        if self.normalize:
            weights = weights / (weights.mean() + 1e-12)
        return weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError("logits must be shape (N, C)")
        if targets.dim() != 1:
            targets = targets.view(-1)

        if self.weight is not None:
            weight = self.weight.to(logits.device)
        else:
            weight = None

        if self.label_smoothing > 0.0:
            loss = F.cross_entropy(logits, targets, weight=weight, reduction="none", 
                                   label_smoothing=self.label_smoothing)
        else:
            loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label Focal Loss for handling class imbalance in multi-label classification.
    """
    def __init__(self, samples_per_cls=None, beta=0.9999, alpha=None, gamma=2.0, 
                 reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                self.alpha = alpha
        elif samples_per_cls is not None:
            self.alpha = self._calculate_alpha(samples_per_cls, beta)
        else:
            self.alpha = None
    
    def _calculate_alpha(self, samples_per_cls, beta):
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        counts = torch.clamp(counts, min=1)
        
        if beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - beta) / eff_num
        
        weights = weights / (weights.mean() + 1e-12)
        return weights
    
    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            alpha_weight = self.alpha.unsqueeze(0).expand_as(focal_loss)
            focal_loss = alpha_weight * focal_loss
        
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            valid_count = mask.float().sum()
            if valid_count > 0:
                if self.reduction == 'mean':
                    return focal_loss.sum() / valid_count
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification with class imbalance.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)
    https://arxiv.org/abs/2009.14119
    """
    
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8,
                 reduction='mean', disable_torch_grad_focal_loss=True):
        super().__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        
        logger.info(f"Initialized AsymmetricLoss with gamma_neg={gamma_neg}, "
                   f"gamma_pos={gamma_pos}, clip={clip}")
    
    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        loss_pos = -torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = -torch.log(probs_neg.clamp(min=self.eps))
        
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
                focal_weight_neg = probs_pos ** self.gamma_neg
        else:
            focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
            focal_weight_neg = probs_pos ** self.gamma_neg
        
        loss_pos = focal_weight_pos * loss_pos
        loss_neg = focal_weight_neg * loss_neg
        
        loss = targets * loss_pos + (1 - targets) * loss_neg
        
        if mask is not None:
            loss = loss * mask.float()
            valid_count = mask.float().sum()
            if valid_count > 0:
                if self.reduction == 'mean':
                    return loss.sum() / valid_count
                elif self.reduction == 'sum':
                    return loss.sum()
                else:
                    return loss
            else:
                return loss.sum() * 0
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLossOptimized(nn.Module):
    """
    Optimized version of Asymmetric Loss with class weighting and entropy regularization.
    """
    
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8,
                 samples_per_cls=None, beta=0.9999,
                 reduction='mean', disable_torch_grad_focal_loss=True, device='cpu',
                 entropy_weight=0.1):
        super().__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.device = device
        self.entropy_weight = entropy_weight
        
        if samples_per_cls is not None:
            self.class_weights = self._calculate_weights(samples_per_cls, beta)
        else:
            self.class_weights = None
        
        logger.info(f"Initialized AsymmetricLossOptimized with gamma_neg={gamma_neg}, "
                   f"gamma_pos={gamma_pos}, clip={clip}, entropy_weight={entropy_weight}, "
                   f"class_weights={'enabled' if self.class_weights is not None else 'disabled'}")
    
    def _calculate_weights(self, samples_per_cls, beta):
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        counts = torch.clamp(counts, min=1)
        
        if beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - beta) / eff_num
        
        weights = weights / (weights.mean() + 1e-12)
        return weights
    
    def _compute_entropy_regularization(self, probs, mask=None):
        if mask is not None:
            probs_masked = probs.clone()
            probs_masked[~mask] = 0.5
        else:
            probs_masked = probs
        
        p = probs_masked.clamp(min=self.eps, max=1-self.eps)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        
        if mask is not None:
            entropy = entropy * mask.float()
            valid_count = mask.float().sum()
            if valid_count > 0:
                return entropy.sum() / valid_count
            else:
                return entropy.sum() * 0
        
        return entropy.mean()
    
    def forward(self, logits, targets, mask=None):
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        loss_pos = -torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = -torch.log(probs_neg.clamp(min=self.eps))
        
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
                focal_weight_neg = probs_pos ** self.gamma_neg
        else:
            focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
            focal_weight_neg = probs_pos ** self.gamma_neg
        
        loss_pos = focal_weight_pos * loss_pos
        loss_neg = focal_weight_neg * loss_neg
        
        loss = targets * loss_pos + (1 - targets) * loss_neg
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = loss * weights.unsqueeze(0)
        
        if mask is not None:
            loss = loss * mask.float()
            valid_count = mask.float().sum()
            if valid_count > 0:
                if self.reduction == 'mean':
                    base_loss = loss.sum() / valid_count
                elif self.reduction == 'sum':
                    base_loss = loss.sum()
                else:
                    base_loss = loss
            else:
                base_loss = loss.sum() * 0
        else:
            if self.reduction == 'mean':
                base_loss = loss.mean()
            elif self.reduction == 'sum':
                base_loss = loss.sum()
            else:
                base_loss = loss
        
        if self.entropy_weight > 0:
            entropy_loss = self._compute_entropy_regularization(probs, mask)
            total_loss = base_loss + self.entropy_weight * entropy_loss
            return total_loss
        
        return base_loss


class CardinalityRegularizer(nn.Module):
    """
    Regularizer that penalizes deviation from expected label count.
    
    Encourages predictions to have approximately the target number of positive labels.
    """
    
    def __init__(self, target_cardinality=1.5, weight=0.3):
        super().__init__()
        self.target_cardinality = target_cardinality
        self.weight = weight
        logger.info(f"Initialized CardinalityRegularizer: target={target_cardinality}, weight={weight}")
    
    def forward(self, probs, mask=None):
        if mask is not None:
            masked_probs = probs * mask.float()
            predicted_count = masked_probs.sum(dim=1)
        else:
            predicted_count = probs.sum(dim=1)
        
        cardinality_loss = ((predicted_count - self.target_cardinality) ** 2).mean()
        
        return self.weight * cardinality_loss