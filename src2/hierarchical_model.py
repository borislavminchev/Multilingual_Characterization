import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


from data_utils import ENTITY_END_TOKEN, ENTITY_START_TOKEN
from taxonomy_manager import TaxonomyManager
from datasets import coarse_label2id, fine_label2id, coarse_id2label, fine_id2label
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self,samples_per_cls=None, beta=0.9999, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        # alpha can be None (no class weighting), a scalar, or a tensor of per-class weights
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                # Scalar alpha - will be broadcasted
                self.alpha = alpha
        elif samples_per_cls is not None:
            # Compute per-class alpha using effective number of samples
            self.alpha = self.calculate_alpha(samples_per_cls, beta)
        else:
            self.alpha = None

    def calculate_alpha(self, samples_per_cls, beta):
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        if beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - beta) / eff_num

        # Normalize to sum to number of classes
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
        
        # Apply per-class alpha weighting if available
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Per-class alpha: index by target class
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
            else:
                # Scalar alpha
                focal_loss = self.alpha * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

class ClassBalancedCrossEntropy(nn.Module):
    """
    Class-balanced cross-entropy (Cui et al., CVPR 2019).

    Weight per class: w_i = (1 - beta) / (1 - beta^{n_i}), where n_i is number of
    samples for class i. Optionally normalized so that sum(w) == num_classes.

    Args:
        samples_per_cls: sequence of length C with sample counts for each class.
                         If None, you can pass explicit class_weights at init or later.
        class_weights: optional tensor/sequence (len C) of weights to use directly.
        beta: float close to 1.0 (e.g., 0.9999). If beta==0 -> weights uniform.
        reduction: 'mean' | 'sum' | 'none'
        label_smoothing: optional float in [0,1), requires PyTorch supporting label_smoothing in F.cross_entropy.
        device: device for internal tensors.
        normalize: whether to normalize weights so that mean(weight) == 1 (or sum==C).
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
            weights = self.calculate_weight(samples_per_cls)

            self.register_buffer("_weight", weights)
        else:
            # will handle None at forward (no weighting)
            self.register_buffer("_weight", None)

    @property
    def weight(self):
        return self._weight
    
    def calculate_weight(self, samples_per_cls: Sequence[int]):
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
        """
        Args:
            logits: (N, C) raw logits
            targets: (N,) long labels
        Returns:
            scalar loss (if reduction != 'none') or per-sample losses (if reduction == 'none')
        """
        if logits.dim() != 2:
            raise ValueError("logits must be shape (N, C)")

        if targets.dim() != 1:
            targets = targets.view(-1)

        # Use PyTorch's cross_entropy with weight and optional label smoothing
        # Note: label_smoothing requires PyTorch >= 1.10 (most modern versions support it)
        if self.weight is not None:
            # ensure weight on same device and dtype
            weight = self.weight.to(logits.device)
        else:
            weight = None

        # F.cross_entropy handles reduction for us
        # If label_smoothing unsupported in the installed torch, it will raise; then set to 0.0.
        if self.label_smoothing > 0.0:
            # Safe call: F.cross_entropy takes label_smoothing param in modern PyTorch
            loss = F.cross_entropy(logits, targets, weight=weight, reduction="none", label_smoothing=self.label_smoothing)
        else:
            loss = F.cross_entropy(logits, targets, weight=weight, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss



class SemanticSimilarityHead(nn.Module):
    
    def __init__(self, model, tokenizer, input_dim, parent_label=None, freeze_anchors=True, similarity_metric='cosine', device='cpu'):
        super().__init__()
        
    
        self.taxonomy_manager = TaxonomyManager(model, tokenizer, device=device)
        self.parent_label = parent_label
        self.input_dim = input_dim
        self.similarity_metric = similarity_metric
        
        # Get anchor vectors from taxonomy manager
        anchor_vectors_dict = self.taxonomy_manager.get_parent_anchors() if parent_label is None else self.taxonomy_manager.get_subtype_anchors(parent_label)
        
        # Select appropriate label2id mapping based on parent_label
        label2id = coarse_label2id if parent_label is None else fine_label2id
        
        # Order anchor vectors by label IDs to ensure consistent ordering
        anchor_vectors_list = [anchor_vectors_dict[label] for label in sorted(label2id.keys(), key=lambda x: label2id[x])]
        self.num_classes = len(anchor_vectors_list)

        # Register anchor vectors (optionally frozen)
        if freeze_anchors:
            self.register_buffer('anchor_vectors', torch.tensor(anchor_vectors_list).detach())
        else:
            self.anchor_vectors = nn.Parameter(torch.tensor(anchor_vectors_list), requires_grad=True)
        
        # Projection layer to align entity embeddings with semantic space
        self.projection = nn.Linear(input_dim, input_dim)
        
        # Optional: scaling factor for similarity scores
        self.temperature = 15.0 #nn.Parameter(torch.tensor(10.0))
        
        logger.info(f"Initialized SemanticSimilarityHead with {self.num_classes} classes using {similarity_metric} similarity.")

    def forward(self, entity_vectors):
        projected = self.projection(entity_vectors)
        
        if self.similarity_metric == 'cosine':
            projected_norm = F.normalize(projected, p=2, dim=1)
            anchors_norm = F.normalize(self.anchor_vectors, p=2, dim=1)
            
            # Cosine similarity: values in range [-1, 1]
            logits = torch.matmul(projected_norm, anchors_norm.t())
            
            # Scale by a factor to make logits more discriminative
            # (cosine similarity values are small, need scaling for softmax)
            logits = logits * self.temperature  # Temperature scaling to increase contrast
        else:  # dot_product
            logits = torch.matmul(projected, self.anchor_vectors.t())
        
       
        # Return raw logits without activation - let caller decide
        # (softmax for classification, sigmoid for multi-label)
        return logits
    

def entity_span_pooling(
    hidden_states,
    input_ids,
    entity_start_id,
    entity_end_id
):
    """
    hidden_states: (B, T, H)
    input_ids: (B, T)
    Returns: (B, H) pooled entity representations
    """
    batch_size, seq_len, hidden_dim = hidden_states.size()
    pooled_outputs = []

    for i in range(batch_size):
        ids = input_ids[i]

        start_positions = (ids == entity_start_id).nonzero(as_tuple=True)[0]
        end_positions = (ids == entity_end_id).nonzero(as_tuple=True)[0]

        # Fallback: no entity markers found
        if len(start_positions) == 0 or len(end_positions) == 0:
            pooled_outputs.append(hidden_states[i, 0])  # [CLS]
            continue

        start = start_positions[0].item() + 1
        end = end_positions[0].item()

        if start >= end:
            pooled_outputs.append(hidden_states[i, 0])
            continue

        span_embeddings = hidden_states[i, start:end]
        pooled_outputs.append(span_embeddings.mean(dim=0))

    return torch.stack(pooled_outputs)

    
class CoarseRoleClassifier(nn.Module):
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine', 
                 device='cpu', num_unfrozen_layers=2,
                #    class_weights=None, 
                   class_counts=None):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Freeze all encoder layers except the last num_unfrozen_layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Also freeze embeddings to save memory
        # for param in base_model.embeddings.parameters():
        #     param.requires_grad = False
        
        self.semantic_head = SemanticSimilarityHead(
            model=base_model,
            tokenizer=tokenizer,
            input_dim=base_model.config.hidden_size,
            freeze_anchors=freeze_anchors,
            similarity_metric=similarity_metric,
            device=device
        )
        
        # Compute per-class alpha weights for Focal Loss
        # If class_weights provided, use inverse frequency weighting
        # if class_weights is not None:
        #     alpha = torch.tensor(class_weights, dtype=torch.float32, device=device)
        # else:
        #     # Default: equal weight per class (no class balancing)
        #     num_classes = self.semantic_head.num_classes
        #     alpha = torch.ones(num_classes, dtype=torch.float32, device=device)
        
        self.loss_fn = FocalLoss(samples_per_cls=class_counts, beta=0.9999, gamma=2.0, reduction='sum', device=device)
        # self.loss_fn = ClassBalancedCrossEntropy(
        #     samples_per_cls=class_counts,   # e.g. computed once from the dataframe
        #     beta=0.9999,
        #     reduction='sum',
        #     label_smoothing=0.0,
        #     device=device,
        #     normalize=True
        # )
        self.to(device)

    def forward(self, input_ids, attention_mask, coarse_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # entity_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )

        logits = self.semantic_head(entity_vectors)
        
        loss = None
        if coarse_labels is not None:
            loss = self.loss_fn(logits, coarse_labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )


class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label Focal Loss for handling class imbalance in multi-label classification.
    
    Applies sigmoid to logits and computes binary focal loss for each label independently.
    Supports masking to ignore certain labels in the loss computation (for hierarchical classification).
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, samples_per_cls=None, beta=0.9999, alpha=None, gamma=2.0, 
                 reduction='mean', device='cpu'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        
        # alpha can be None (no class weighting), a scalar, or a tensor of per-class weights
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
            elif isinstance(alpha, torch.Tensor):
                self.alpha = alpha.to(device)
            else:
                self.alpha = alpha
        elif samples_per_cls is not None:
            self.alpha = self.calculate_alpha(samples_per_cls, beta)
        else:
            self.alpha = None
    
    def calculate_alpha(self, samples_per_cls, beta):
        """Calculate class-balanced weights using effective number of samples."""
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        # Handle zero counts by setting minimum to 1
        counts = torch.clamp(counts, min=1)
        
        if beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - beta) / eff_num
        
        # Normalize weights
        weights = weights / (weights.mean() + 1e-12)
        return weights
    
    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (N, C) raw logits for each class
            targets: (N, C) binary targets (multi-hot encoding)
            mask: (N, C) optional boolean mask - True for valid labels to include in loss
        Returns:
            Scalar loss value
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy components
        # For positive samples: -log(p)
        # For negative samples: -log(1-p)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        
        # Focal weight: (1 - p_t)^gamma
        # p_t = p for y=1, (1-p) for y=0
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        # Apply per-class alpha weighting if available
        if self.alpha is not None:
            # alpha is (C,), broadcast to (N, C)
            alpha_weight = self.alpha.unsqueeze(0).expand_as(focal_loss)
            focal_loss = alpha_weight * focal_loss
        
        # Apply mask if provided (for hierarchical classification)
        if mask is not None:
            # Only compute loss for valid labels
            focal_loss = focal_loss * mask.float()
            # Adjust for number of valid labels
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
    
    Key idea: Use different focusing parameters (gamma) for positive and negative samples.
    - gamma_neg (high, e.g., 4): Down-weight easy negatives more aggressively
    - gamma_pos (low, e.g., 0-1): Don't down-weight easy positives as much
    
    Additionally applies probability shifting to hard threshold negatives.
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)
    https://arxiv.org/abs/2009.14119
    
    Args:
        gamma_neg: Focusing parameter for negative samples (higher = more focus on hard negatives)
        gamma_pos: Focusing parameter for positive samples (lower = preserve easy positives)
        clip: Probability margin for hard thresholding negatives (shifts probability)
        eps: Small constant for numerical stability
        reduction: 'mean' | 'sum' | 'none'
        disable_torch_grad_focal_loss: Performance optimization flag
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
        
        logger.info(f"Initialized AsymmetricLoss with gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}")
    
    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (N, C) raw logits for each class
            targets: (N, C) binary targets (multi-hot encoding)
            mask: (N, C) optional boolean mask - True for valid labels to include in loss
        Returns:
            Scalar loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Asymmetric Clipping (Probability Shifting)
        # Shift negative probabilities by margin to create hard threshold
        # This helps ignore very easy negatives (prob < clip)
        if self.clip > 0:
            # Shift probabilities: max(p - clip, 0)
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # Compute losses separately for positive and negative samples
        # Positive loss: -log(p) * (1-p)^gamma_pos
        # Negative loss: -log(1-p) * p^gamma_neg
        
        # Basic BCE components
        loss_pos = -torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = -torch.log(probs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            # Don't compute gradients through focal weights (performance optimization)
            with torch.no_grad():
                focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
                focal_weight_neg = probs_pos ** self.gamma_neg
        else:
            focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
            focal_weight_neg = probs_pos ** self.gamma_neg
        
        # Apply focal weights
        loss_pos = focal_weight_pos * loss_pos
        loss_neg = focal_weight_neg * loss_neg
        
        # Combine: positive samples contribute loss_pos, negative samples contribute loss_neg
        # targets=1 -> loss_pos, targets=0 -> loss_neg
        loss = targets * loss_pos + (1 - targets) * loss_neg
        
        # Apply mask if provided (for hierarchical classification)
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
                return loss.sum() * 0  # Return 0 loss if no valid labels
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLossOptimized(nn.Module):
    """
    Optimized version of Asymmetric Loss with additional features.
    
    Adds:
    - Per-class weighting based on class frequency
    - Entropy regularization to encourage sharp (peaky) predictions
    - Configurable probability shifting per class
    
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)
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
        self.entropy_weight = entropy_weight  # Weight for entropy regularization
        
        # Optional class-balanced weighting
        if samples_per_cls is not None:
            self.class_weights = self._calculate_weights(samples_per_cls, beta)
        else:
            self.class_weights = None
        
        logger.info(f"Initialized AsymmetricLossOptimized with gamma_neg={gamma_neg}, "
                   f"gamma_pos={gamma_pos}, clip={clip}, entropy_weight={entropy_weight}, "
                   f"class_weights={'enabled' if self.class_weights is not None else 'disabled'}")
    
    def _calculate_weights(self, samples_per_cls, beta):
        """Calculate class-balanced weights using effective number of samples."""
        counts = torch.tensor(samples_per_cls, dtype=torch.float32, device=self.device)
        counts = torch.clamp(counts, min=1)
        
        if beta <= 0:
            weights = torch.ones_like(counts)
        else:
            eff_num = 1.0 - torch.pow(beta, counts)
            eff_num = torch.clamp(eff_num, min=1e-8)
            weights = (1.0 - beta) / eff_num
        
        # Normalize weights
        weights = weights / (weights.mean() + 1e-12)
        return weights
    
    def _compute_entropy_regularization(self, probs, mask=None):
        """
        Compute entropy of probability distribution per sample.
        
        Low entropy = sharp/peaky distribution (good - model is confident)
        High entropy = uniform distribution (bad - model is uncertain)
        
        We want to MINIMIZE entropy to encourage confident predictions.
        
        Args:
            probs: (N, C) sigmoid probabilities
            mask: (N, C) optional boolean mask for valid labels
        Returns:
            Scalar entropy loss (higher = more uniform, lower = more peaky)
        """
        if mask is not None:
            # Only compute entropy over valid labels
            # Set invalid probs to 0.5 (neutral, won't affect entropy direction)
            probs_masked = probs.clone()
            probs_masked[~mask] = 0.5
        else:
            probs_masked = probs
        
        # Binary entropy for each position: -p*log(p) - (1-p)*log(1-p)
        # This is maximized when p=0.5 (most uncertain)
        # This is minimized when p=0 or p=1 (most confident)
        p = probs_masked.clamp(min=self.eps, max=1-self.eps)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        
        if mask is not None:
            # Average entropy only over valid positions
            entropy = entropy * mask.float()
            valid_count = mask.float().sum()
            if valid_count > 0:
                return entropy.sum() / valid_count
            else:
                return entropy.sum() * 0
        
        return entropy.mean()
    
    def forward(self, logits, targets, mask=None):
        """
        Args:
            logits: (N, C) raw logits for each class
            targets: (N, C) binary targets (multi-hot encoding)
            mask: (N, C) optional boolean mask - True for valid labels to include in loss
        Returns:
            Scalar loss value
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs
        
        # Asymmetric Clipping (Probability Shifting)
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)
        
        # BCE components
        loss_pos = -torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = -torch.log(probs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            with torch.no_grad():
                focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
                focal_weight_neg = probs_pos ** self.gamma_neg
        else:
            focal_weight_pos = (1 - probs_pos) ** self.gamma_pos
            focal_weight_neg = probs_pos ** self.gamma_neg
        
        # Apply focal weights
        loss_pos = focal_weight_pos * loss_pos
        loss_neg = focal_weight_neg * loss_neg
        
        # Combine losses
        loss = targets * loss_pos + (1 - targets) * loss_neg
        
        # Apply class weights if available
        if self.class_weights is not None:
            # Ensure weights are on the same device
            weights = self.class_weights.to(logits.device)
            # Expand weights to match loss shape: (C,) -> (1, C) -> (N, C)
            loss = loss * weights.unsqueeze(0)
        
        # Compute base loss
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
        
        # Add entropy regularization to encourage sharper predictions
        if self.entropy_weight > 0:
            entropy_loss = self._compute_entropy_regularization(probs, mask)
            total_loss = base_loss + self.entropy_weight * entropy_loss
            return total_loss
        
        return base_loss


class FineSemanticSimilarityHead(nn.Module):
    """
    Semantic Similarity Head for fine-grained multi-label classification.
    
    Uses ALL fine label anchors but supports masking based on coarse predictions
    to ensure only valid fine labels are considered for each sample.
    """
    
    def __init__(self, model, tokenizer, input_dim, freeze_anchors=True, 
                 similarity_metric='cosine', device='cpu'):
        super().__init__()
        
        self.taxonomy_manager = TaxonomyManager(model, tokenizer, device=device)
        self.input_dim = input_dim
        self.similarity_metric = similarity_metric
        self.device = device
        
        # Get ALL fine label anchors
        anchor_vectors_dict = self.taxonomy_manager.get_all_fine_anchors()
        
        # Order anchor vectors by fine_label2id to ensure consistent ordering
        anchor_vectors_list = [
            anchor_vectors_dict[label] 
            for label in sorted(fine_label2id.keys(), key=lambda x: fine_label2id[x])
        ]
        self.num_classes = len(anchor_vectors_list)
        
        # Register anchor vectors (optionally frozen)
        if freeze_anchors:
            self.register_buffer('anchor_vectors', torch.tensor(anchor_vectors_list).detach())
        else:
            self.anchor_vectors = nn.Parameter(torch.tensor(anchor_vectors_list), requires_grad=True)
        
        # Projection layer to align entity embeddings with semantic space
        self.projection = nn.Linear(input_dim, input_dim)
        
        # Temperature scaling for similarity scores
        self.temperature = 15.0
        
        # Store coarse-to-fine mask for inference
        self.register_buffer(
            'coarse_to_fine_mask', 
            self.taxonomy_manager.get_coarse_to_fine_mask(device=device)
        )
        
        logger.info(f"Initialized FineSemanticSimilarityHead with {self.num_classes} fine classes.")
    
    def forward(self, entity_vectors, coarse_labels=None):
        """
        Args:
            entity_vectors: (B, H) entity representations
            coarse_labels: (B,) optional coarse label indices for masking
        Returns:
            logits: (B, num_fine_classes) similarity scores
            mask: (B, num_fine_classes) boolean mask of valid fine labels
        """
        projected = self.projection(entity_vectors)
        
        if self.similarity_metric == 'cosine':
            projected_norm = F.normalize(projected, p=2, dim=1)
            anchors_norm = F.normalize(self.anchor_vectors, p=2, dim=1)
            logits = torch.matmul(projected_norm, anchors_norm.t())
            logits = logits * self.temperature
        else:  # dot_product
            logits = torch.matmul(projected, self.anchor_vectors.t())
        
        # Generate mask based on coarse labels
        if coarse_labels is not None:
            # coarse_labels: (B,) -> mask: (B, num_fine_classes)
            mask = self.coarse_to_fine_mask[coarse_labels]
        else:
            # No mask - all labels valid
            mask = torch.ones(
                entity_vectors.size(0), self.num_classes, 
                dtype=torch.bool, device=entity_vectors.device
            )
        
        return logits, mask


class FineRoleClassifier(nn.Module):
    """
    Unified Fine Role Classifier for multi-label classification.
    
    Uses a single model with all 22 fine label anchors.
    Supports hierarchical masking based on coarse predictions.
    
    Supports multiple loss functions:
    - 'focal': MultiLabelFocalLoss (standard focal loss)
    - 'asl': AsymmetricLoss (asymmetric loss without class weighting)
    - 'asl_optimized': AsymmetricLossOptimized (ASL with class-balanced weighting)
    """
    
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine',
                 device='cpu', num_unfrozen_layers=2, class_counts=None, threshold=0.5,
                 loss_type='asl_optimized', gamma_neg=4.0, gamma_pos=1.0, clip=0.05,
                 entropy_weight=0.1):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.loss_type = loss_type
        
        # Freeze encoder layers except the last num_unfrozen_layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        self.semantic_head = FineSemanticSimilarityHead(
            model=base_model,
            tokenizer=tokenizer,
            input_dim=base_model.config.hidden_size,
            freeze_anchors=freeze_anchors,
            similarity_metric=similarity_metric,
            device=device
        )
        
        # Select loss function based on loss_type
        if loss_type == 'focal':
            self.loss_fn = MultiLabelFocalLoss(
                samples_per_cls=class_counts,
                beta=0.9999,
                gamma=2.0,
                reduction='mean',
                device=device
            )
            logger.info("Using MultiLabelFocalLoss for fine classification")
        elif loss_type == 'asl':
            self.loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip,
                reduction='mean'
            )
            logger.info(f"Using AsymmetricLoss (gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip})")
        elif loss_type == 'asl_optimized':
            self.loss_fn = AsymmetricLossOptimized(
                gamma_neg=gamma_neg,
                gamma_pos=gamma_pos,
                clip=clip,
                samples_per_cls=class_counts,
                beta=0.9999,
                reduction='mean',
                device=device,
                entropy_weight=entropy_weight
            )
            logger.info(f"Using AsymmetricLossOptimized (gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}, entropy_weight={entropy_weight}, class_weighted=True)")
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'focal', 'asl', 'asl_optimized'")
        
        self.to(device)
    
    def forward(self, input_ids, attention_mask, coarse_labels=None, fine_labels=None, **kwargs):
        """
        Args:
            input_ids: (B, T) input token IDs
            attention_mask: (B, T) attention mask
            coarse_labels: (B,) coarse label indices (for masking)
            fine_labels: (B, num_fine_classes) multi-hot fine label targets
        Returns:
            SequenceClassifierOutput with loss and logits
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        # Get logits and mask based on coarse labels
        logits, mask = self.semantic_head(entity_vectors, coarse_labels)
        
        # Apply mask to logits for inference (set invalid to large negative)
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        
        loss = None
        if fine_labels is not None:
            # Compute loss only on valid (masked) labels
            loss = self.loss_fn(logits, fine_labels, mask=mask)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=masked_logits  # Return masked logits for inference
        )
    
    def predict(self, input_ids, attention_mask, coarse_labels):
        """
        Predict fine labels given coarse labels.
        
        Args:
            input_ids: (B, T) input token IDs
            attention_mask: (B, T) attention mask
            coarse_labels: (B,) predicted coarse label indices
        Returns:
            predictions: (B, num_fine_classes) binary predictions
            probabilities: (B, num_fine_classes) sigmoid probabilities (masked)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, coarse_labels=coarse_labels)
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= self.threshold).float()
        return predictions, probabilities
