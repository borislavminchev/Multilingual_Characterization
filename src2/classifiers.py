"""
Classifier Models for Hierarchical Entity Role Classification

This module contains:
- CoarseRoleClassifier: Single-label classification (3 classes)
- FineRoleClassifier: Multi-label classification with hard masking (22 labels)
- SoftConditionedFineClassifier: Multi-label with soft hierarchy conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from data_utils import ENTITY_START_TOKEN, ENTITY_END_TOKEN
from taxonomy_manager import TaxonomyManager
from datasets import coarse_label2id, fine_label2id, coarse_id2label, fine_id2label
from transformers.modeling_outputs import SequenceClassifierOutput

from losses import (
    FocalLoss, MultiLabelFocalLoss, 
    AsymmetricLoss, AsymmetricLossOptimized,
    CardinalityRegularizer
)

logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def entity_span_pooling(hidden_states, input_ids, entity_start_id, entity_end_id):
    """
    Pool entity representations from hidden states.
    
    Args:
        hidden_states: (B, T, H) transformer hidden states
        input_ids: (B, T) input token IDs
        entity_start_id: Token ID for [ENTITY] marker
        entity_end_id: Token ID for [/ENTITY] marker
    Returns:
        (B, H) pooled entity representations
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


# =============================================================================
# SEMANTIC SIMILARITY HEADS
# =============================================================================

class SemanticSimilarityHead(nn.Module):
    """Semantic similarity head for coarse-level classification."""
    
    def __init__(self, model, tokenizer, input_dim, parent_label=None, 
                 freeze_anchors=True, similarity_metric='cosine', device='cpu',
                 dropout=0.0):
        super().__init__()
        
        self.taxonomy_manager = TaxonomyManager(model, tokenizer, device=device)
        self.parent_label = parent_label
        self.input_dim = input_dim
        self.similarity_metric = similarity_metric
        
        anchor_vectors_dict = (
            self.taxonomy_manager.get_parent_anchors() 
            if parent_label is None 
            else self.taxonomy_manager.get_subtype_anchors(parent_label)
        )
        
        label2id = coarse_label2id if parent_label is None else fine_label2id
        anchor_vectors_list = [
            anchor_vectors_dict[label] 
            for label in sorted(label2id.keys(), key=lambda x: label2id[x])
        ]
        self.num_classes = len(anchor_vectors_list)

        if freeze_anchors:
            self.register_buffer('anchor_vectors', torch.tensor(anchor_vectors_list).detach())
        else:
            self.anchor_vectors = nn.Parameter(torch.tensor(anchor_vectors_list), requires_grad=True)
        
        # Projection with dropout for regularization
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout)
        )
        self.temperature = 15.0
        
        logger.info(f"Initialized SemanticSimilarityHead with {self.num_classes} classes, dropout={dropout}")

    def forward(self, entity_vectors):
        projected = self.projection(entity_vectors)
        
        if self.similarity_metric == 'cosine':
            projected_norm = F.normalize(projected, p=2, dim=1)
            anchors_norm = F.normalize(self.anchor_vectors, p=2, dim=1)
            logits = torch.matmul(projected_norm, anchors_norm.t())
            logits = logits * self.temperature
        else:
            logits = torch.matmul(projected, self.anchor_vectors.t())
        
        return logits


class FineSemanticSimilarityHead(nn.Module):
    """Semantic similarity head for fine-grained multi-label classification."""
    
    def __init__(self, model, tokenizer, input_dim, freeze_anchors=True, 
                 similarity_metric='cosine', device='cpu'):
        super().__init__()
        
        self.taxonomy_manager = TaxonomyManager(model, tokenizer, device=device)
        self.input_dim = input_dim
        self.similarity_metric = similarity_metric
        self.device = device
        
        anchor_vectors_dict = self.taxonomy_manager.get_all_fine_anchors()
        anchor_vectors_list = [
            anchor_vectors_dict[label] 
            for label in sorted(fine_label2id.keys(), key=lambda x: fine_label2id[x])
        ]
        self.num_classes = len(anchor_vectors_list)
        
        if freeze_anchors:
            self.register_buffer('anchor_vectors', torch.tensor(anchor_vectors_list).detach())
        else:
            self.anchor_vectors = nn.Parameter(torch.tensor(anchor_vectors_list), requires_grad=True)
        
        self.projection = nn.Linear(input_dim, input_dim)
        self.temperature = 15.0
        
        self.register_buffer(
            'coarse_to_fine_mask', 
            self.taxonomy_manager.get_coarse_to_fine_mask(device=device)
        )
        
        logger.info(f"Initialized FineSemanticSimilarityHead with {self.num_classes} fine classes")
    
    def forward(self, entity_vectors, coarse_labels=None):
        projected = self.projection(entity_vectors)
        
        if self.similarity_metric == 'cosine':
            projected_norm = F.normalize(projected, p=2, dim=1)
            anchors_norm = F.normalize(self.anchor_vectors, p=2, dim=1)
            logits = torch.matmul(projected_norm, anchors_norm.t())
            logits = logits * self.temperature
        else:
            logits = torch.matmul(projected, self.anchor_vectors.t())
        
        if coarse_labels is not None:
            mask = self.coarse_to_fine_mask[coarse_labels]
        else:
            mask = torch.ones(
                entity_vectors.size(0), self.num_classes, 
                dtype=torch.bool, device=entity_vectors.device
            )
        
        return logits, mask


# =============================================================================
# HIERARCHY AFFINITY LAYER (FOR SOFT CONDITIONING)
# =============================================================================

class HierarchyAffinityLayer(nn.Module):
    """
    Learnable affinity layer that maps coarse probabilities to fine label priors.
    
    Instead of hard masking, this layer learns soft relationships between
    coarse categories and fine labels.
    """
    
    def __init__(self, num_coarse=3, num_fine=22, init_from_taxonomy=True, device='cpu'):
        super().__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.device = device
        
        self.affinity = nn.Parameter(torch.zeros(num_coarse, num_fine, device=device))
        
        if init_from_taxonomy:
            self._init_from_taxonomy()
        else:
            nn.init.xavier_uniform_(self.affinity)
        
        logger.info(f"Initialized HierarchyAffinityLayer: ({num_coarse}, {num_fine})")
    
    def _init_from_taxonomy(self):
        """Initialize affinity matrix based on known taxonomy hierarchy."""
        init_valid = 2.0
        init_invalid = -2.0
        
        self.affinity.data.fill_(init_invalid)
        
        # Protagonist subtypes
        protagonist_fine = ['Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous']
        for fine_name in protagonist_fine:
            if fine_name in fine_label2id:
                self.affinity.data[0, fine_label2id[fine_name]] = init_valid
        
        # Antagonist subtypes
        antagonist_fine = ['Instigator', 'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor',
                          'Spy', 'Saboteur', 'Corrupt', 'Incompetent', 'Terrorist', 'Deceiver', 'Bigot']
        for fine_name in antagonist_fine:
            if fine_name in fine_label2id:
                self.affinity.data[1, fine_label2id[fine_name]] = init_valid
        
        # Innocent subtypes
        innocent_fine = ['Forgotten', 'Exploited', 'Victim', 'Scapegoat']
        for fine_name in innocent_fine:
            if fine_name in fine_label2id:
                self.affinity.data[2, fine_label2id[fine_name]] = init_valid
        
        logger.info("Initialized affinity matrix from taxonomy hierarchy")
    
    def forward(self, coarse_probs):
        """
        Compute soft fine label prior from coarse probabilities.
        
        Args:
            coarse_probs: (B, num_coarse) softmax probabilities for coarse labels
        Returns:
            fine_prior: (B, num_fine) soft prior over fine labels
        """
        fine_prior = torch.matmul(coarse_probs, self.affinity)
        return fine_prior


# =============================================================================
# CLASSIFIER MODELS
# =============================================================================

class CoarseRoleClassifier(nn.Module):
    """
    Coarse Role Classifier for single-label classification (3 classes).
    
    Uses semantic similarity with coarse label anchors.
    """
    
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine', 
                 device='cpu', num_unfrozen_layers=2, class_counts=None, dropout=0.0,
                 focal_gamma=2.0, beta=0.9999):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Freeze encoder layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        logger.info(f"CoarseRoleClassifier: Frozen {total_layers - num_unfrozen_layers}/{total_layers} layers")
        
        self.semantic_head = SemanticSimilarityHead(
            model=base_model,
            tokenizer=tokenizer,
            input_dim=base_model.config.hidden_size,
            freeze_anchors=freeze_anchors,
            similarity_metric=similarity_metric,
            device=device,
            dropout=dropout
        )
        
        self.loss_fn = FocalLoss(
            samples_per_cls=class_counts, 
            beta=beta, 
            gamma=focal_gamma, 
            reduction='mean',  # Changed from 'sum' to 'mean' for better stability
            device=device
        )
        self.to(device)

    def forward(self, input_ids, attention_mask, coarse_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
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
        
        return SequenceClassifierOutput(loss=loss, logits=logits)


class FineRoleClassifier(nn.Module):
    """
    Fine Role Classifier for multi-label classification with hard masking.
    
    Uses semantic similarity with fine label anchors and masks invalid
    labels based on coarse predictions.
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
        
        # Freeze encoder layers
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
        
        # Select loss function
        if loss_type == 'focal':
            self.loss_fn = MultiLabelFocalLoss(
                samples_per_cls=class_counts, beta=0.9999, gamma=2.0,
                reduction='mean', device=device
            )
        elif loss_type == 'asl':
            self.loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, reduction='mean'
            )
        elif loss_type == 'asl_optimized':
            self.loss_fn = AsymmetricLossOptimized(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
                samples_per_cls=class_counts, beta=0.9999, reduction='mean',
                device=device, entropy_weight=entropy_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        logger.info(f"Initialized FineRoleClassifier with {loss_type} loss")
        self.to(device)
    
    def forward(self, input_ids, attention_mask, coarse_labels=None, fine_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        logits, mask = self.semantic_head(entity_vectors, coarse_labels)
        
        # Apply mask for inference
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        
        loss = None
        if fine_labels is not None:
            loss = self.loss_fn(logits, fine_labels, mask=mask)
        
        return SequenceClassifierOutput(loss=loss, logits=masked_logits)
    
    def predict(self, input_ids, attention_mask, coarse_labels):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, coarse_labels=coarse_labels)
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= self.threshold).float()
        return predictions, probabilities


class SoftConditionedFineClassifier(nn.Module):
    """
    Soft Conditioned Fine Role Classifier combining:
    1. Soft hierarchy conditioning (coarse probs → fine prior via learnable affinity)
    2. Cardinality regularization (constrain number of predictions)
    3. Asymmetric Loss (handle class imbalance)
    
    Key differences from FineRoleClassifier:
    - Uses coarse probabilities (soft) instead of coarse labels (hard)
    - Learns hierarchy relationship rather than hard masking
    - Adds cardinality constraint to prevent over/under-prediction
    """
    
    def __init__(self, base_model, tokenizer, freeze_anchors=True, similarity_metric='cosine',
                 device='cpu', num_unfrozen_layers=2, class_counts=None, threshold=0.5,
                 loss_type='asl_optimized', gamma_neg=4.0, gamma_pos=1.0, clip=0.05,
                 entropy_weight=0.1, cardinality_weight=0.3, target_cardinality=1.5,
                 num_coarse=3, num_fine=22):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold
        self.loss_type = loss_type
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        
        # Freeze encoder layers
        total_layers = len(base_model.encoder.layer)
        for idx, layer in enumerate(base_model.encoder.layer):
            if idx < total_layers - num_unfrozen_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # Entity to fine logits projection
        hidden_size = base_model.config.hidden_size
        self.entity_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_fine)
        )
        
        # Hierarchy affinity layer
        self.hierarchy_affinity = HierarchyAffinityLayer(
            num_coarse=num_coarse,
            num_fine=num_fine,
            init_from_taxonomy=True,
            device=device
        )
        
        # Cardinality regularizer
        self.cardinality_reg = CardinalityRegularizer(
            target_cardinality=target_cardinality,
            weight=cardinality_weight
        )
        
        # Base loss function
        if loss_type == 'focal':
            self.loss_fn = MultiLabelFocalLoss(
                samples_per_cls=class_counts, beta=0.9999, gamma=2.0,
                reduction='mean', device=device
            )
        elif loss_type == 'asl':
            self.loss_fn = AsymmetricLoss(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip, reduction='mean'
            )
        elif loss_type == 'asl_optimized':
            self.loss_fn = AsymmetricLossOptimized(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip,
                samples_per_cls=class_counts, beta=0.9999, reduction='mean',
                device=device, entropy_weight=entropy_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        logger.info(f"Initialized SoftConditionedFineClassifier: "
                   f"cardinality_weight={cardinality_weight}, target={target_cardinality}")
        
        self.to(device)
    
    def forward(self, input_ids, attention_mask, coarse_probs=None, coarse_labels=None, 
                fine_labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        entity_vectors = entity_span_pooling(
            hidden_states=outputs.last_hidden_state,
            input_ids=input_ids,
            entity_start_id=self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN),
            entity_end_id=self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        )
        
        entity_logits = self.entity_projection(entity_vectors)
        
        # Compute hierarchy prior
        if coarse_probs is not None:
            hierarchy_prior = self.hierarchy_affinity(coarse_probs)
        elif coarse_labels is not None:
            coarse_one_hot = F.one_hot(coarse_labels, num_classes=self.num_coarse).float()
            hierarchy_prior = self.hierarchy_affinity(coarse_one_hot)
        else:
            batch_size = entity_vectors.size(0)
            hierarchy_prior = torch.zeros(batch_size, self.num_fine, device=self.device)
        
        fine_logits = entity_logits + hierarchy_prior
        
        loss = None
        if fine_labels is not None:
            base_loss = self.loss_fn(fine_logits, fine_labels, mask=None)
            probs = torch.sigmoid(fine_logits)
            cardinality_loss = self.cardinality_reg(probs, mask=None)
            loss = base_loss + cardinality_loss
        
        return SequenceClassifierOutput(loss=loss, logits=fine_logits)
    
    def predict(self, input_ids, attention_mask, coarse_probs=None, coarse_labels=None):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids, attention_mask, 
                coarse_probs=coarse_probs, 
                coarse_labels=coarse_labels
            )
            probabilities = torch.sigmoid(outputs.logits)
            predictions = (probabilities >= self.threshold).float()
        return predictions, probabilities