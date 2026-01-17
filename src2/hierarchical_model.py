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
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
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
        else:
            self.alpha = None
    
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
        self.temperature = nn.Parameter(torch.tensor(10.0))
        
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
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine', device='cpu', num_unfrozen_layers=2, class_weights=None):
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
        if class_weights is not None:
            alpha = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            # Default: equal weight per class (no class balancing)
            num_classes = self.semantic_head.num_classes
            alpha = torch.ones(num_classes, dtype=torch.float32, device=device)
        
        self.loss_fn = FocalLoss(alpha=alpha, gamma=2.0, device=device)
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